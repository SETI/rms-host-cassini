##########################################################################################
# host_cassini/oops/uvis.py
##########################################################################################

import numbers
import re

import numpy as np
import julian

from filecache import FCPath

import oops
from oops.host import HostFormatError, HostIndexError
from ._cassini import (_Cassini, _cassini_obs_id_code, _CMATRIX_ROTATION,
                       _pds3_options_with_defaults)

__all__ = ['CassiniUVIS']

_DEBUG = False      # True to assert that the data array must have null values outside the
                    # active windows
_ARRAY_NULL = 65535  # Incorrectly -1 in many labels

# TARGET_NAME in a UVIS label -> the target body; "NONE" if no body was targeted
_TARGET_NAME_REPAIRS = {
    'AG'        : 'AEGAEON',
    'ATLAS:'    : 'ATLAS',
    'CALIB'     : 'NONE',
    'DA'        : 'DAPHNIS',
    'IPH'       : 'NONE',
    'ME'        : 'METHONE',
    'N/A'       : 'NONE',
    'PL'        : 'PALLENE',
    'PO'        : 'POLYDEUCES',
    'SKY'       : 'NONE',
    'SOLAR WIND': 'NONE',
    'STAR'      : 'NONE',
    'UNK'       : 'NONE',
}

# These are repairs to star names embedded in the OBSERVATION_ID
# A second value is an alternative name.
_OBS_CODE_REPAIRS = {
    '205839'    : ('b Cen', 'SAO 205839'),
    'ALPCRUSA'  : 'ALPCRU',
    'BETCENSA'  : 'BETCEN',
    'CWLEON'    : 'CW Leo',
    'SAKAPCEN'  : 'KAPCEN',
    'SPICARAST' : 'SPICA',
    'ZETAORI'   : 'ZETORI',
}

# Map NAIF names to (detector, resolution)
_ABBREVS = {
    'CASSINI_UVIS_FUV_HI' : ('FUV', 'HIGH_RESOLUTION'),
    'CASSINI_UVIS_FUV_LO' : ('FUV', 'LOW_RESOLUTION'),
    'CASSINI_UVIS_FUV_OCC': ('FUV', 'OCCULTATION'),
    'CASSINI_UVIS_EUV_HI' : ('EUV', 'HIGH_RESOLUTION'),
    'CASSINI_UVIS_EUV_LO' : ('EUV', 'LOW_RESOLUTION'),
    'CASSINI_UVIS_EUV_OCC': ('EUV', 'OCCULTATION'),
    'CASSINI_UVIS_SOLAR'  : ('SOLAR',   ''),
    'CASSINI_UVIS_SOL_OFF': ('SOL_OFF', ''),
    'CASSINI_UVIS_HSP'    : ('HSP',     ''),
    'CASSINI_UVIS_HDAC'   : ('HDAC',    ''),
}

# Map detector to SPICE frame name
_FRAME_NAMES = {
    'FUV'    : 'CASSINI_UVIS_FUV',
    'EUV'    : 'CASSINI_UVIS_EUV',
    'SOLAR'  : 'CASSINI_UVIS_SOLAR',
    'SOL_OFF': 'CASSINI_UVIS_SOL_OFF',
    'HSP'    : 'CASSINI_UVIS_HSP',
    'HDAC'   : 'CASSINI_UVIS_HDAC',
}

# Map SPICE frame name to its frame ID in the Cassini frame kernel
_FRAME_IDS = {
    'CASSINI_UVIS_FUV'    : -82840,
    'CASSINI_UVIS_EUV'    : -82842,
    'CASSINI_UVIS_SOLAR'  : -82843,
    'CASSINI_UVIS_HSP'    : -82844,
    'CASSINI_UVIS_HDAC'   : -82845,
    'CASSINI_UVIS_SOL_OFF': -82849,
}

# Map NAIF body names to the instrument codes that key them in the instrument kernel.
# SOL_OFF uses the SOLAR code, -82848: the instrument kernels before v06 do not define
# -82849, and the two FOVs have the same size.
_INSTRUMENT_IDS = {
    'CASSINI_UVIS_FUV_HI' : -82840,
    'CASSINI_UVIS_FUV_LO' : -82841,
    'CASSINI_UVIS_FUV_OCC': -82842,
    'CASSINI_UVIS_EUV_HI' : -82843,
    'CASSINI_UVIS_EUV_LO' : -82844,
    'CASSINI_UVIS_EUV_OCC': -82845,
    'CASSINI_UVIS_HSP'    : -82846,
    'CASSINI_UVIS_HDAC'   : -82847,
    'CASSINI_UVIS_SOLAR'  : -82848,
    'CASSINI_UVIS_SOL_OFF': -82848,
}


class CassiniUVIS(oops.Host):
    """The Cassini UVIS host: Observation constructors for UVIS data products."""

    NAME = 'Cassini UVIS'

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, enclose=False, pds3_options=None, select=None,
                  astrometry=False, target=None, lightsource=None, path=None, frame=None,
                  cadence=None, fov=None, calibrations=None, timeshift=None,
                  navigation=None, tracker=None, parallel=None, **kwargs):
        """One or more Observations based on the label of a Cassini UVIS file.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label): The path to the PDS3 label
                of a UVIS data file, or the parsed label.
            enclose (bool, optional): True to return a single observation, regardless of
                how many windows are defined. If multiple windows are used, then the
                observation (and the optional data array) are defined by the enclosing
                limits in line and band, and the binning is assumed to be 1. If False and
                multiple windows are used, the function returns a tuple of observations
                rather than a single observation.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to the returned observations,
                treating a single observation as a tuple of one. For example, if the file
                defines two windows but `select=1`, only the second window's observation
                is returned.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observations.
            target (str | oops.Body, optional): Override for the target Body of every
                observation, which is otherwise derived from TARGET_NAME in the label.
                Use "NONE" for inertial pointing, indicating that no Solar System body
                was tracked.
            lightsource (oops.LightSource, optional): Override for the LightSource of
                every observation, which is otherwise the Sun.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
            cadence (oops.Cadence, optional): Override for the Cadence of the observation.
            fov (oops.FOV, optional): Override for the full FOV of the detector, before
                any windowing and binning.
            calibrations (oops.Calibration | list[oops.Calibration], optional): Override
                for the calibration or list of calibrations.
            timeshift (tuple[float, str], optional): Assign a Fittable time shift to one
                or more attributes of each Observation. The first item is the initial time
                shift in seconds. The second item is a string indicating which attributes
                are to be time-shifted: "T" for the start/stop time, "P" for the path,
                and/or "F" for the frame. If multiple letters are given, the time shifts
                are coupled to whichever attribute is listed first.
            navigation (tuple[float, ...], optional): Two or three rotation angles in
                radians to apply to the frame in order to align the calculated geometry
                with the observed geometry. In this case, a fittable Navigation frame
                wraps the default frame. Specify three angles to include a rotation about
                the optic axis or two for a pointing offset without rotation. Use (0,0) or
                (0,0,0) as the input if you have no starting guess.
            tracker (str, optional): Assign a TrackerFrame to each Observation, to ensure
                that the target body remains at a fixed position within the FOV. Use one
                of "start", "midtime", and "end", indicating the time within the
                Observation at which the target body's calculated position within the FOV
                is accurate.
                Note that no more than one of `parallel` and `tracker` can be specified.
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which each Observation has a
                fixed offset.
            **kwargs: Additional keyword arguments; they are accepted and ignored.

        Returns:
            Observation | tuple[Observation, ...]: A :class:`~oops.observation.Pixel` for
            a SPECTRUM, a TIME_SERIES, or a one-line QUBE, or a
            :class:`~oops.observation.TimedImage` for a multi-line QUBE. A QUBE with
            multiple windows returns a tuple of observations, one per window, unless
            `enclose` is True. With `select`, the observation chosen by an index, or a
            tuple of the observations chosen by a slice or a tuple.

            Each observation has subfields `dict`, `host`, `instrument`, `detector`,
            `product_type`, `line_window`, `line_bin`, `band_window`, `band_bin`,
            `samples`, `label_target`, `observation_id`, `filepath` and `basename`;
            `sampling` unless it is a TIME_SERIES; and `data` unless `astrometry` is True.

        Raises:
            IndexError: If an index in `select` is out of range.
            TypeError: If `select` has an invalid type.
            ValueError: If `select` is a name; UVIS observations have none.
            OopsValueError: If a TIME_SERIES is from neither the HSP nor the HDAC.
            oops.host.HostFormatError: If some aspect of the label or file is not valid.
        """

        CassiniUVIS.initialize()  # define everything the first time through; use
                                  # defaults unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        # Read the label
        fileinfo = oops.Host._read_fileinfo(fileinfo, formats='P', astrometry=astrometry,
                                            pds3_options=pds3_options)
        filepath = FCPath(fileinfo.filepath)
        label = fileinfo.as_dict()

        # Construct and return the Observation(s)
        overrides = {
            'target'      : target,
            'lightsource' : lightsource,
            'path'        : path,
            'frame'       : frame,
            'cadence'     : cadence,
            'fov'         : fov,
            'calibrations': calibrations,
            'timeshift'   : timeshift,
            'navigation'  : navigation,
            'tracker'     : tracker,
            'parallel'    : parallel,
        }

        return CassiniUVIS._make_observations(label, filepath, astrometry=astrometry,
                                              enclose=enclose, overrides=overrides,
                                              select=select)

    @staticmethod
    def from_index(filepath, *, enclose=False, supplement=None, label=None,
                   row_dicts=None, pds3_options=None, filter=None, select=None,
                   target=None, lightsource=None, path=None, frame=None, cadence=None,
                   fov=None, calibrations=None, timeshift=None, navigation=None,
                   tracker=None, parallel=None, **kwargs):
        """A list of Observations, one for each row of a data file index.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            enclose (bool, optional): True to return a single observation for each row,
                regardless of how many windows are defined. If multiple windows are used,
                then the observation (and the optional data array) are defined by the
                enclosing limits in line and band, and the binning is assumed to be 1. If
                False and multiple windows are used, the function returns a tuple of
                observations rather than a single observation.
            supplement (str | pathlib.Path | FCPath, optional): The path to the PDS label
                of a supplemental index file. This file must have the same records in the
                same order as the table described by `filepath`. If not specified, the
                directory of `filepath` will be searched for a file whose name matches
                except that "_index" is replaced by "_supplemental_index" in upper or
                lower case.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (list[dict[str, Any]], optional): A list of dictionaries, one per
                row of the index, if `filepath` and `supplement` have already been read.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included in the returned list.
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to the observations of each row,
                treating a single observation as a tuple of one. For example, if a row
                defines two windows but `select=1`, only the second window's observation
                is returned.
            target (str | oops.Body, optional): Override for the default target Body of
                every observation. Use "NONE" for inertial pointing, indicating that no
                Solar System body was tracked.
            lightsource (oops.LightSource, optional): Override for the default LightSource
                of every observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
            cadence (oops.Cadence, optional): Override for the Cadence of the observation.
            fov (oops.FOV, optional): Override for the default FOV of the observing
                instrument.
            calibrations (oops.Calibration | list[oops.Calibration], optional): Override
                for the calibration or list of calibrations.
            timeshift (tuple[float, str], optional): Assign a Fittable time shift to one
                or more attributes of the Observation. The first item is the initial time
                shift in seconds. The second item is a string indicating which attributes
                are to be time-shifted: "T" for the start/stop time, "P" for the path,
                and/or "F" for the frame. If multiple letters are given, the time shifts
                are coupled to whichever attribute is listed first.
            navigation (tuple[float, ...], optional): Two or three rotation angles in
                radians to apply to the frame in order to align the calculated geometry
                with the observed geometry. In this case, a fittable Navigation frame
                wraps the default frame. Specify three angles to include a rotation about
                the optic axis or two for a pointing offset without rotation. Use (0,0) or
                (0,0,0) as the input if you have no starting guess.
            tracker (str, optional): Assign a TrackerFrame to the Observation, to ensure
                that the target body remains at a fixed position within the FOV. Use one
                of "start", "midtime", and "end", indicating the time within the
                Observation at which the target body's calculated position within the FOV
                is accurate.
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which this Observation has a
                fixed offset.
            **kwargs: Additional keyword arguments, ignored here except as noted below.

        Returns:
            list[Observation | ObsArray]: The list of Observations or arrays of
            Observations, one for each filtered row of the index file.

        Raises:
            oops.host.HostIndexError: If no supplemental index was found or merged into
                `row_dicts`, or if the supplement does not match the index row for row.
        """

        CassiniUVIS.initialize()

        # Read and filter the index table
        if row_dicts is None:
            supplement = supplement or oops.Host.SUPPLEMENT     # supplement is required
            row_dicts = oops.Host._read_index_rows(filepath, supplement=supplement,
                                                   validate=CassiniUVIS._validate_index,
                                                   pds3_options=pds3_options)

        # The file path and most of the observation's parameters come from the supplement
        if row_dicts and 'FILE_SPECIFICATION_NAME' not in row_dicts[0]:
            raise HostIndexError(f'Cassini UVIS needs a supplemental index: {filepath}')

        row_dicts = oops.Host._filter_index_rows(row_dicts, filter)

        overrides = {
            'target'      : target,
            'lightsource' : lightsource,
            'path'        : path,
            'frame'       : frame,
            'cadence'     : cadence,
            'fov'         : fov,
            'calibrations': calibrations,
            'timeshift'   : timeshift,
            'navigation'  : navigation,
            'tracker'     : tracker,
            'parallel'    : parallel,
        }

        # Create the list of Observations
        obslist = []
        for row_dict in row_dicts:
            fpath = row_dict['VOLUME_ID'] + '/' + row_dict['FILE_SPECIFICATION_NAME']
            item = CassiniUVIS._make_observations(row_dict, fpath, astrometry=True,
                                                  enclose=enclose, overrides=overrides,
                                                  select=select)
            obslist.append(item)

        return obslist

    @staticmethod
    def _validate_index(indx_dict, supp_dict):
        """Raise an exception if the index row and supplement row do not match."""

        full_path = indx_dict['FILE_NAME']

        if 'VOLUME_ID' not in supp_dict:
            raise HostIndexError('missing VOLUME_ID in supplemental index')
        volume_id = supp_dict['VOLUME_ID']
        if volume_id not in full_path:
            raise HostIndexError(f'VOLUME_IDs do not match: "{volume_id}", '
                                 f'"{full_path}"')

        if 'FILE_SPECIFICATION_NAME' not in supp_dict:
            raise HostIndexError('missing FILE_SPECIFICATION_NAME in supplemental index')
        file_spec = supp_dict['FILE_SPECIFICATION_NAME']
        if file_spec not in full_path:
            raise HostIndexError(f'file paths do not match: "{file_spec}", '
                                 f'"{full_path}"')

        return True

    @staticmethod
    def _make_observations(label, filepath, *, astrometry, enclose, overrides,
                           select=None):
        """The observation(s) described by a UVIS label.

        Parameters:
            label (dict): The label content from a detached PDS3 label.
            filepath (FCPath): The path to the cube or to its detached PDS3 label.
            astrometry (bool): True to omit the data arrays.
            enclose (bool): True to return a single observation for each row, regardless
                of how many windows are defined. If multiple windows are used, then the
                observation (and the optional data array) are defined by the enclosing
                limits in line and band, and the binning is assumed to be 1. If False and
                multiple windows are used, the function returns a tuple of observations
                rather than a single observation.
            overrides (dict): The standard override options of :meth:`from_file`.
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to the returned observations,
                treating a single observation as a tuple of one.

        Returns:
            Observation | tuple[Observation, ...]: Observation(s) from `filepath` after
            `select` has been applied.

        Raises:
            oops.host.HostFormatError: If some aspect of the label or file is not valid.
        """

        # Load any needed SPICE kernels
        tstart = julian.tdb_from_iso(label['START_TIME'])
        tstop = julian.tdb_from_iso(label['STOP_TIME'])
        _Cassini.load_spks(tstart, tstop)
        if overrides['frame'] is None:
            _Cassini.load_cks(tstart, tstop)
            CassiniUVIS._define_frames()

        # Figure out the PDS object class and get the observation(s)
        if 'QUBE' in label:
            result = CassiniUVIS._get_qube(filepath, label, enclose,
                                           astrometry=astrometry, overrides=overrides)
        elif 'TIME_SERIES' in label:
            result = CassiniUVIS._get_time_series(filepath, label, astrometry=astrometry,
                                                  overrides=overrides)
        else:
            result = CassiniUVIS._get_spectrum(filepath, label, astrometry=astrometry,
                                               overrides=overrides)

        if select is None:
            return result
        if not isinstance(result, tuple):
            result = (result,)
        return oops.Host._apply_select(result, select)

    @staticmethod
    def _get_qube(filepath, label, enclose, *, astrometry, overrides):
        """The observation object given that it is a QUBE.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            enclose (bool): True to combine multiple windows into a single observation
                defined by their enclosing limits in line and band; False to return one
                observation per window.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Observation | tuple[Observation, ...]: A :class:`~oops.observation.Pixel` for
            a one-line QUBE or a :class:`~oops.observation.TimedImage` otherwise; a tuple
            of observations, one per window, if multiple windows are defined and
            `enclose` is False.
        """

        info = label['QUBE']
        (bands, lines, samples) = info['CORE_ITEMS']

        # Load the data array if necessary
        if astrometry:
            array = None
        else:
            if (info['CORE_ITEM_TYPE'] != 'MSB_UNSIGNED_INTEGER'
                    or info['CORE_ITEM_BYTES'] != 2):
                raise HostFormatError(f'unexpected QUBE data type in {filepath}')
            if info['SUFFIX_ITEMS'] != [0, 0, 0]:
                raise HostFormatError(f'unexpected QUBE SUFFIX_ITEMS in {filepath}')

            array = CassiniUVIS._load_data(filepath, label, 'QUBE')

            # Re-shape into something sensible
            # Note that the axis order in the label is first-index-fastest
            if lines > 1:
                array = array.reshape((samples, lines, bands))
                array = array.swapaxes(0, 1)
            else:
                array = array.reshape((samples, bands))

        # Identify the window(s) used
        # Note that these are either integers or lists of integers
        line0 = info['UL_CORNER_LINE']
        line1 = info['LR_CORNER_LINE']
        line_bin = info['LINE_BIN']

        band0 = info['UL_CORNER_BAND']
        band1 = info['LR_CORNER_BAND']
        band_bin = info['BAND_BIN']

        # Check the outer periphery of the data array in DEBUG mode
        if _DEBUG and array is not None:
            assert np.all(array[:min(line0),    ...] == _ARRAY_NULL)
            assert np.all(array[ max(line1)+1:, ...] == _ARRAY_NULL)
            assert np.all(array[..., :min(band0)   ] == _ARRAY_NULL)
            assert np.all(array[...,  max(band1)+1:] == _ARRAY_NULL)

        # One window
        if isinstance(line0, numbers.Integral):
            return CassiniUVIS._get_one_qube(filepath, label, array, line0=line0,
                                             line1=line1+1, line_bin=line_bin,
                                             band0=band0, band1=band1+1,
                                             band_bin=band_bin, rebin=True,
                                             overrides=overrides)

        # Multiple windows combined into one enclosure
        if enclose:
            return CassiniUVIS._get_one_qube(filepath, label, array, line0=min(line0),
                                             line1=max(line1)+1, line_bin=min(line_bin),
                                             band0=min(band0), band1=max(band1)+1,
                                             band_bin=min(band_bin), rebin=False,
                                             overrides=overrides)

        # Separate windows
        return tuple(CassiniUVIS._get_one_qube(filepath, label, array, line0=line0[w],
                                               line1=line1[w]+1, line_bin=line_bin[w],
                                               band0=band0[w], band1=band1[w]+1,
                                               band_bin=band_bin[w], rebin=True,
                                               overrides=overrides)
                     for w in range(len(line0)))

    @staticmethod
    def _get_one_qube(filepath, label, array, *, line0, line1, line_bin, band0,
                      band1, band_bin, rebin, overrides):
        """A single Observation object for the identified window of the UVIS qube.

        The line window and binning apply only to a qube with more than one line. A
        one-line qube was binned along the slit on board, so its FOV is the full slit.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            array (numpy.ndarray | None): The full data array, or None if the data are not
                loaded.
            line0 (int): The first line of the window.
            line1 (int): One past the last line of the window.
            line_bin (int): The binning factor along the line axis.
            band0 (int): The first band of the window.
            band1 (int): One past the last band of the window.
            band_bin (int): The binning factor along the band axis.
            rebin (bool): True to apply the binning factors to the field of view, the
                shape and the data array; False to ignore them.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Observation: A :class:`~oops.observation.Pixel` for a one-line qube or a
            :class:`~oops.observation.TimedImage` otherwise.
        """

        # Determine the detector and mode
        detector = label['PRODUCT_ID'][:3]
        if detector not in {'EUV', 'FUV'}:
            raise HostFormatError(f'invalid instrument {detector} for QUBE in '
                                  f'{filepath}')

        resolution = label['SLIT_STATE']

        # Get array shape
        info = label['QUBE']
        (bands, lines, samples) = info['CORE_ITEMS']
        if lines not in {1, 64}:
            raise HostFormatError(f'invalid QUBE lines {lines} in {filepath}')

        if lines == 1:
            shape = (samples, bands)
        else:
            shape = (lines, samples, bands)

        # Define the instrument frame
        frame_name = _FRAME_NAMES[detector]

        # Define the cadence
        tstart = julian.tdb_from_iso(label['START_TIME'])
        texp = label['INTEGRATION_DURATION']
        cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

        # Define the full FOV
        fov = CassiniUVIS._FOVS[(detector, resolution, lines)]

        # Trim the lines
        dline = line1 - line0
        if lines > 1 and (line0, line1) != (0, lines):
            fov = oops.fov.SliceFOV(fov, (0, line0), (1, dline))
            shape = (dline,) + shape[1:]

            if array is not None:
                array = array[line0:line1, :]

        # Trim the bands
        dband = band1 - band0
        if (band0, band1) != (0, bands):
            shape = shape[:-1] + (dband,)

            if array is not None:
                array = array[..., band0:band1]

        # Bin the lines
        if rebin and lines > 1 and line_bin > 1:
            assert dline % line_bin == 0
            fov = oops.fov.SubsampledFOV(fov, (1, line_bin))
            dline_binned = dline // line_bin
            shape = (dline_binned,) + shape[1:]

            if array is not None:
                if _DEBUG:
                    assert np.all(array[dline_binned:, ...] == _ARRAY_NULL)
                array = array[:dline_binned]

        # Bin the bands
        if rebin and band_bin > 1:
            if _DEBUG:
                assert dband % band_bin == 0    # seen to fail occasionally

            dband_binned = dband // band_bin
            shape = shape[:-1] + (dband_binned,)

            if array is not None:
                if _DEBUG:
                    assert np.all(array[..., dband_binned:] == _ARRAY_NULL)
                array = array[..., :dband_binned]

        # Define the Observation parameters
        params = CassiniUVIS._default_params(label, cadence, fov, frame_name)
        oops.Host._apply_overrides(params, filepath, **overrides)

        # Construct the Observation
        if lines == 1:
            obs = oops.observation.Pixel(('t', 'b'), **params)
        else:
            obs = oops.observation.TimedImage(('v', 'ut', 'b'), **params)

        if array is not None:
            obs.insert_subfield('data', array)

        obs.insert_subfield('dict', label)
        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('sampling', resolution)
        obs.insert_subfield('product_type', 'QUBE')
        obs.insert_subfield('label_target', label['TARGET_NAME'])
        obs.insert_subfield('observation_id', label['OBSERVATION_ID'])

        obs.insert_subfield('line_window', (line0, line1))
        obs.insert_subfield('line_bin', line_bin)
        obs.insert_subfield('band_window', (band0, band1))
        obs.insert_subfield('band_bin', band_bin)
        obs.insert_subfield('samples', samples)

        kernels = _Cassini.used_kernels(obs.time, 'uvis', ck=(overrides['frame'] is None),
                                        return_all_planets=False)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
        obs.insert_subfield('spice_frame_name', frame_name)
        obs.insert_subfield('spice_frame_id', _FRAME_IDS[frame_name])

        # Update the observation shape
        obs.shape = shape
        return obs

    @staticmethod
    def _get_time_series(filepath, label, *, astrometry, overrides):
        """The observation object given that it is a TIME_SERIES.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Pixel: The observation.

        Raises:
            OopsValueError: If the product is neither HSP nor HDAC.
        """

        # Determine the detector
        product_id = label['PRODUCT_ID']
        if product_id.startswith('HSP'):
            detector = 'HSP'
        elif product_id.startswith('HDAC'):
            detector = 'HDAC'
        else:
            raise oops.OopsValueError(f'Time series is neither HSP nor HDAC: {filepath}')

        # Get the array shape
        info = label['TIME_SERIES']
        samples = info['ROWS']

        # Define the instrument frame
        frame_name = _FRAME_NAMES[detector]

        # Define the cadence
        if info['COLUMNS'] != 1:
            raise HostFormatError(f'TIME_SERIES with COLUMNS != 1 in {filepath}')
        if not info['SAMPLING_PARAMETER_UNIT'].startswith('MILLISECOND'):
            raise HostFormatError(f'TIME_SERIES with unit not ms in {filepath}')
        tstart = julian.tdb_from_iso(label['START_TIME'])
        texp = info['SAMPLING_PARAMETER_INTERVAL'] * 0.001
        cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

        # Define the Observation parameters
        fov = CassiniUVIS._FOVS[(detector, '', 1)]
        params = CassiniUVIS._default_params(label, cadence, fov, frame_name)
        oops.Host._apply_overrides(params, filepath, **overrides)

        # Construct the Observation
        obs = oops.observation.Pixel(('t',), **params)

        # Load the data array if necessary
        if not astrometry:
            column = info['PHOTOMETER_COUNTS']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in TIME_SERIES: '
                                      f'{filepath}')
            array = CassiniUVIS._load_data(filepath, label, 'TIME_SERIES')
            obs.insert_subfield('data', array)

        obs.insert_subfield('dict', label)
        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('product_type', 'TIME_SERIES')
        obs.insert_subfield('label_target', label['TARGET_NAME'])
        obs.insert_subfield('observation_id', label['OBSERVATION_ID'])

        obs.insert_subfield('line_window', None)
        obs.insert_subfield('line_bin', None)
        obs.insert_subfield('band_window', None)
        obs.insert_subfield('band_bin', None)
        obs.insert_subfield('samples', samples)

        kernels = _Cassini.used_kernels(obs.time, 'uvis', ck=(overrides['frame'] is None),
                                        return_all_planets=False)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
        obs.insert_subfield('spice_frame_name', frame_name)
        obs.insert_subfield('spice_frame_id', _FRAME_IDS[frame_name])

        # Update the observation shape
        obs.shape = (samples,)
        return obs

    @staticmethod
    def _get_spectrum(filepath, label, *, astrometry, overrides):
        """The observation object given that it is a SPECTRUM.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Pixel: The observation.
        """

        # Determine the detector
        detector = label['PRODUCT_ID'][:3]
        if detector not in {'EUV', 'FUV'}:
            raise HostFormatError(f'invalid instrument {detector} for SPECTRUM in '
                                  f'{filepath}')

        # Get array shape
        info = label['SPECTRUM']
        bands = info['ROWS']

        # Define the instrument frame
        frame_name = _FRAME_NAMES[detector]

        # Define the cadence
        if info['COLUMNS'] != 1:
            raise HostFormatError(f'SPECTRUM has multiple COLUMNS in {filepath}')
        tstart = julian.tdb_from_iso(label['START_TIME'])
        texp = label['INTEGRATION_DURATION']
        cadence = oops.cadence.Metronome(tstart, texp, texp, 1)

        # Define the FOV
        resolution = label['SLIT_STATE']
        fov = CassiniUVIS._FOVS[(detector, resolution, 64)]

        # Define the window
        line0 = info['UL_CORNER_SPATIAL']
        line1 = info['LR_CORNER_SPATIAL'] + 1
        if (line0, line1) != (0, 64):
            fov = oops.fov.SliceFOV(fov, (0, line0), (1, line1-line0))

        line_bin = info['BIN_SPATIAL']
        if line_bin != 1:
            fov = oops.fov.SubsampledFOV(fov, (1, line_bin))

        band0 = info['UL_CORNER_SPECTRAL']
        band1 = info['LR_CORNER_SPECTRAL'] + 1
        band_bin = info['BIN_SPECTRAL']

        # Define the observation
        params = CassiniUVIS._default_params(label, cadence, fov, frame_name)
        oops.Host._apply_overrides(params, filepath, **overrides)
        obs = oops.observation.Pixel(('b',), **params)

        # Load the data array if necessary
        if not astrometry:
            column = info['SPECTRUM']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in SPECTRUM: {filepath}')
            array = CassiniUVIS._load_data(filepath, label, 'SPECTRUM')
            obs.insert_subfield('data', array)

        obs.insert_subfield('dict', label)
        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('product_type', 'SPECTRUM')
        obs.insert_subfield('label_target', label['TARGET_NAME'])
        obs.insert_subfield('observation_id', label['OBSERVATION_ID'])
        obs.insert_subfield('sampling', resolution)
        obs.insert_subfield('line_window', (line0, line1))
        obs.insert_subfield('line_bin', line_bin)
        obs.insert_subfield('band_window', (band0, band1))
        obs.insert_subfield('band_bin', band_bin)
        obs.insert_subfield('samples', 1)

        kernels = _Cassini.used_kernels(obs.time, 'uvis', ck=(overrides['frame'] is None),
                                        return_all_planets=False)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
        obs.insert_subfield('spice_frame_name', frame_name)
        obs.insert_subfield('spice_frame_id', _FRAME_IDS[frame_name])

        # Update the observation shape
        obs.shape = (bands,)
        return obs

    @staticmethod
    def _default_params(label, cadence, fov, frame_name):
        """The Observation constructor inputs before any overrides.

        Parameters:
            label (dict): The PDS label as a dictionary.
            cadence (Cadence): The cadence of the samples.
            fov (FOV): The field of view.
            frame_name (str): The SPICE name of the instrument frame.

        Returns:
            dict: The inputs "cadence", "fov", "path", "frame", "target", "lightsource"
            and "calibrations", with the target derived from TARGET_NAME in `label` and
            the Sun as the light source.
        """

        target = _TARGET_NAME_REPAIRS.get(label['TARGET_NAME'], label['TARGET_NAME'])
        lightsource = CassiniUVIS._star_from_obs_id(label['OBSERVATION_ID']) or 'SUN'

        return {
            'target'      : target,
            'lightsource' : lightsource,
            'path'        : 'CASSINI',
            'frame'       : frame_name,
            'cadence'     : cadence,
            'fov'         : fov,
            'calibrations': [],
        }

    @staticmethod
    def _load_data(filepath, label, key):
        """The contents of a UVIS data file as a NumPy array.

        If the data file named in the label does not exist, the same name in lower case
        is tried.

        Parameters:
            filepath (FCPath): Path to the label file.
            label (dict[str, Any]): The PDS3 label content.
            key (str): The label key identifying the data file name, one of "QUBE",
                "TIME_SERIES", or "SPECTRUM".

        Returns:
            numpy.ndarray: The values read from the file, as MSB 2-byte unsigned
            integers; the caller checks that the label describes the data that way.
        """

        basename = label['^' + key]
        data_filepath = FCPath(filepath).with_name(basename)

        try:
            local_path = data_filepath.retrieve()
        except FileNotFoundError:
            data_filepath = data_filepath.with_name(basename.lower())
            local_path = data_filepath.retrieve()

        return np.fromfile(local_path, sep='', dtype='>u2')

    _OBS_CODE_CLEANUP = re.compile(r'(?:UR|)(.*?)(?:OCC|TI|V1|RDI|TK[ABC]|)$')

    @staticmethod
    def _star_from_obs_id(obs_id):
        """The StarSource from a Cassini Observation ID; None otherwise.

        If the star is identified by a different name in the OBSERVATION ID, add this as
        an `alt_name`.
        """

        # Extract the code, strip "UR", "OCC", etc.
        obs_code = _cassini_obs_id_code(obs_id)
        match = CassiniUVIS._OBS_CODE_CLEANUP.match(obs_code)
        if not match:
            return None
        obs_code = match.group(1)

        # Apply repairs for known issues
        repaired = _OBS_CODE_REPAIRS.get(obs_code, obs_code)
        if isinstance(repaired, tuple):
            star_code, alt_name = repaired
        else:
            star_code, alt_name = (repaired, '')

        # Construct the StarSource if possible
        try:
            star = oops.lightsource.star_lookup(star_code)
        except KeyError:
            return None

        # Add an alternative name if any
        if alt_name:
            star.add_name(alt_name)

        return star

    ######################################################################################
    # Detectors
    ######################################################################################

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes data from Cassini UVIS.

        Parameters:
            label (Pds3Label | dict): A parsed PDS3 label.

        Returns:
            bool: True if `label` describes Cassini UVIS data.
        """

        return (label.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and label.get('INSTRUMENT_ID', '').startswith('UVIS'))

    ######################################################################################
    # Initialization
    ######################################################################################

    @staticmethod
    def initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                   gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the UVIS instrument.

        Optional. Call it before the first :meth:`~from_file` or :meth:`~from_index` to
        override the kernel defaults. Once it has run, explicitly or implicitly, later
        calls are ignored.

        Parameters:
            ck (str, optional): The set of C kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no C kernels automatically,
                leaving their handling to the caller.
            planets (list, optional): A list of planets to pass to
                :meth:`~oops.Body.define_solar_system`. None or 0 means all.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.

        Raises:
            OopsValueError: If the instrument kernel gives an unrecognized FOV_SHAPE.
        """

        # Quick exit after first call
        if CassiniUVIS._initialized:
            return

        _Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                            mst_pck=mst_pck, irregulars=irregulars)
        _Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        CassiniUVIS._INSTRUMENT_KERNEL = _Cassini.spice_instrument_kernel('UVIS')[0]

        # The instrument kernel keys each detector by its instrument code, not its name
        ins = CassiniUVIS._INSTRUMENT_KERNEL['INS']

        # Construct a flat FOV for each detector
        fovs = {}
        for (key, (detector, resolution)) in _ABBREVS.items():

            # Get the FOV angles
            info = ins[_INSTRUMENT_IDS[key]]

            if info['FOV_SHAPE'] == 'RECTANGLE':
                u_angle = 2. * info['FOV_CROSS_ANGLE'] * oops.RPD
                v_angle = 2. * info['FOV_REF_ANGLE'] * oops.RPD
            elif info['FOV_SHAPE'] == 'CIRCLE':
                u_angle = 2. * info['FOV_REF_ANGLE'] * oops.RPD
                v_angle = u_angle
            else:
                raise oops.OopsValueError('Unrecognized FOV_SHAPE: ' + info['FOV_SHAPE'])

            # Define the FOV for 1 or 64 lines
            # Not every combination is really used but that doesn't matter
            for lines in (1, 64):
                fov = oops.fov.FlatFOV((u_angle, v_angle/lines), (1,lines))
                fovs[(detector, resolution, lines)] = fov

        CassiniUVIS._FOVS = fovs
        CassiniUVIS._initialized = True

    @staticmethod
    def _define_frames():
        """Register the SPICE frame of each UVIS detector.

        Built lazily (and only once) so that observations using a custom frame never
        construct or depend on the SPICE frames.
        """

        # Each frame is tested against the Frame registry, because registering a second
        # frame under an existing ID renames it rather than failing.
        for frame_name in _FRAME_NAMES.values():
            if not oops.Frame.frame_id_exists(frame_name):
                oops.frame.SpiceFrame(frame_name, frame_id=frame_name)

    @staticmethod
    def _reset():
        """Reset the internal Cassini UVIS parameters.

        Can be useful for debugging.
        """

        CassiniUVIS._INSTRUMENT_KERNEL = None
        CassiniUVIS._FOVS = {}
        CassiniUVIS._initialized = False
        _Cassini.reset()


CassiniUVIS._register()

##########################################################################################
