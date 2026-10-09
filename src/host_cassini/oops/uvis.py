##########################################################################################
# host_cassini/oops/uvis.py
##########################################################################################

import numbers

import numpy as np
import julian

from filecache import FCPath

import oops
from ._cassini import _apply_select, _build_observation, _Cassini

__all__ = ['CassiniUVIS']

_DEBUG = False      # True to assert that the data array must have null values outside the
                    # active windows

_ARRAY_NULL = 65535 # Incorrectly -1 in many labels

# TARGET_NAME in a UVIS label -> the target body; "NONE" if no body was targeted
_TARGET_NAME_REPAIRS = {
    'N/A' : 'NONE',
    'STAR': 'NONE',
    'SKY' : 'NONE',
    'UNK' : 'NONE',
}

# Map NAIF body names to (detector, resolution)
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

# Map detector to NAIF frame ID
_FRAME_IDS = {
    'FUV'    : 'CASSINI_UVIS_FUV',
    'EUV'    : 'CASSINI_UVIS_EUV',
    'SOLAR'  : 'CASSINI_UVIS_SOLAR',
    'SOL_OFF': 'CASSINI_UVIS_SOL_OFF',
    'HSP'    : 'CASSINI_UVIS_HSP',
    'HDAC'   : 'CASSINI_UVIS_HDAC',
}

# Map NAIF body names to the NAIF IDs that key them in the instrument kernel
_NAIF_IDS = {
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
    def from_file(fileinfo, *, enclose=False, pds3_method='fast', select=None,
                  astrometry=False, target=None, lightsource=None, path=None, frame=None,
                  fov=None, calibrations=None, timeshift=None, navigation=None,
                  parallel=None, tracker=None, **kwargs):
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
            pds3_method (str, optional): The method of parsing a PDS3 label. One of:

                * "strict" performs strict parsing, which requires that the label conform
                  to the full PDS3 standard.
                * "loose" is similar to the above, but tolerates some common syntax
                  errors.
                * "compound" is similar to "loose", but it parses a "compound" label,
                  i.e., one that might contain more than one "END" statement. This option
                  is not supported for attached labels.
                * "fast": uses a different parser, which executes ~30x faster than the
                  above and handles all the most common aspects of the PDS3 standard.
                  However, it is not guaranteed to provide an accurate parsing under all
                  circumstances.

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
            lightsource (oops.Lightsource, optional): Override for the Lightsource of
                every observation, which is otherwise the Sun.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument.
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
            parallel (Observation, optional): A parallel Observation (same origin and
                time, different frame and FOV) relative to which each Observation has a
                fixed offset.
            tracker (str, optional): Assign a TrackerFrame to each Observation, to ensure
                that the target body remains at a fixed position within the FOV. Use one
                of "start", "midtime", and "end", indicating the time within the
                Observation at which the target body's calculated position within the FOV
                is accurate.
                Note that no more than one of `parallel` and `tracker` can be specified.
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
            `samples`, `label_target`, `filepath` and `basename`; `sampling` unless it is
            a TIME_SERIES; and `data` unless `astrometry` is True.

        Raises:
            IndexError: If an index in `select` is out of range.
            TypeError: If `select` has an invalid type.
            ValueError: If `select` is a name; UVIS observations have none.
            OopsValueError: If a TIME_SERIES is from neither the HSP nor the HDAC.
        """

        CassiniUVIS._initialize()  # define everything the first time through; use
                                   # defaults unless _initialize() was called explicitly.

        fileinfo = CassiniUVIS._read_fileinfo(fileinfo, formats='P',
                                              astrometry=astrometry,
                                              pds3_method=pds3_method)
        filepath = FCPath(fileinfo.filepath)
        label = fileinfo.dict

        # Load any needed SPICE kernels
        tstart = julian.tdb_from_iso(label['START_TIME'])
        tstop  = julian.tdb_from_iso(label['STOP_TIME'])
        _Cassini.load_spks(tstart, tstop)
        if frame is None:
            _Cassini.load_cks(tstart, tstop)
            CassiniUVIS._define_frames()

        overrides = {
            'target'      : target,
            'lightsource' : lightsource,
            'path'        : path,
            'frame'       : frame,
            'calibrations': calibrations,
            'timeshift'   : timeshift,
            'navigation'  : navigation,
            'parallel'    : parallel,
            'tracker'     : tracker,
        }

        # Figure out the PDS object class and get the observation(s)
        data = not astrometry
        if 'QUBE' in label:
            result = CassiniUVIS._get_qube(filepath, tstart, label, data, enclose,
                                           fov=fov, overrides=overrides)
        elif 'TIME_SERIES' in label:
            result = CassiniUVIS._get_time_series(filepath, tstart, label, data,
                                                  fov=fov, overrides=overrides)
        else:
            result = CassiniUVIS._get_spectrum(filepath, tstart, label, data,
                                               fov=fov, overrides=overrides)

        if select is None:
            return result
        if not isinstance(result, tuple):
            result = (result,)
        return _apply_select(result, select)

    @staticmethod
    def _get_qube(filepath, tstart, label, data, enclose, *, fov, overrides):
        """The observation object given that it is a QUBE.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            tstart (float): The start time of the observation in seconds TDB.
            label (dict): The PDS label as a dictionary.
            data (bool): True to include the data array.
            enclose (bool): True to combine multiple windows into a single observation
                defined by their enclosing limits in line and band; False to return one
                observation per window.
            fov (FOV | None): The full FOV of the detector; None for the default.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Observation | tuple[Observation, ...]: A :class:`~oops.observation.Pixel` for
            a one-line QUBE or a :class:`~oops.observation.TimedImage` otherwise; a tuple
            of observations, one per window, if multiple windows are defined and
            `enclose` is False.
        """

        # Determine the detector and mode
        detector = label['PRODUCT_ID'][:3]
        assert detector in ('EUV', 'FUV')

        resolution = label['SLIT_STATE']

        # Define the instrument frame
        frame_id = _FRAME_IDS[detector]

        # Get array shape
        info = label['QUBE']
        (bands,lines,samples) = info['CORE_ITEMS']
        assert lines in (1,64), f'invalid lines = {lines}'

        if lines == 1:
            shape = (samples, bands)
        else:
            shape = (lines, samples, bands)

        # Define the cadence
        texp = label['INTEGRATION_DURATION']
        cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

        # Define the full FOV
        if fov is None:
            fov = CassiniUVIS._FOVS[(detector, label['SLIT_STATE'], lines)]

        # Load the data array if necessary
        assert info['CORE_ITEM_TYPE'] == 'MSB_UNSIGNED_INTEGER'
        assert info['CORE_ITEM_BYTES'] == 2
        assert info['SUFFIX_ITEMS'] == [0,0,0]

        if data:
            array = CassiniUVIS._load_data(filepath, label['^QUBE'], '>u2')

            # Re-shape into something sensible
            # Note that the axis order in the label is first-index-fastest
            if lines > 1:
                array = array.reshape((samples,lines,bands))
                array = array.swapaxes(0,1)
            else:
                array = array.reshape(shape)
        else:
            array = None

        # Identify the window(s) used
        # Note that these are either integers or lists of integers
        line0 = info['UL_CORNER_LINE']
        line1 = info['LR_CORNER_LINE']
        line_bin = info['LINE_BIN']

        band0 = info['UL_CORNER_BAND']
        band1 = info['LR_CORNER_BAND']
        band_bin = info['BAND_BIN']

        # Check the outer periphery of the data array in DEBUG mode
        if _DEBUG and data:
            assert np.all(array[:min(line0),    ...] == _ARRAY_NULL)
            assert np.all(array[ max(line1)+1:, ...] == _ARRAY_NULL)

            assert np.all(array[..., :min(band0)   ] == _ARRAY_NULL)
            assert np.all(array[...,  max(band1)+1:] == _ARRAY_NULL)

        common = {'label': label, 'detector': detector, 'resolution': resolution,
                  'fov': fov, 'cadence': cadence, 'frame_id': frame_id, 'shape': shape,
                  'array': array, 'samples': samples, 'lines': lines, 'bands': bands,
                  'filepath': filepath, 'overrides': overrides}

        # One window
        if isinstance(line0, numbers.Integral):
            return CassiniUVIS._get_one_qube(line0=line0, line1=line1+1,
                                             line_bin=line_bin, band0=band0,
                                             band1=band1+1, band_bin=band_bin,
                                             rebin=True, **common)

        # Multiple windows combined into one enclosure
        if enclose:
            return CassiniUVIS._get_one_qube(line0=min(line0), line1=max(line1)+1,
                                             line_bin=min(line_bin), band0=min(band0),
                                             band1=max(band1)+1, band_bin=min(band_bin),
                                             rebin=False, **common)

        # Separate windows
        return tuple(CassiniUVIS._get_one_qube(line0=line0[w], line1=line1[w]+1,
                                               line_bin=line_bin[w], band0=band0[w],
                                               band1=band1[w]+1, band_bin=band_bin[w],
                                               rebin=True, **common)
                     for w in range(len(line0)))

    @staticmethod
    def _get_one_qube(*, label, detector, resolution, fov, cadence, frame_id, shape,
                      array, samples, lines, line0, line1, line_bin, bands, band0, band1,
                      band_bin, rebin, filepath, overrides):
        """A single Observation object for the identified window of the UVIS qube.

        The line window and binning apply only to a qube with more than one line. A
        one-line qube was binned along the slit on board, so its FOV is the full slit.

        Parameters:
            label (dict): The PDS label as a dictionary.
            detector (str): The detector name, 'EUV' or 'FUV'.
            resolution (str): The slit state, which defines the spatial resolution.
            fov (FOV): The full field of view of the detector.
            cadence (Cadence): The cadence of the samples.
            frame_id (str): The ID of the instrument frame.
            shape (tuple[int, ...]): The shape of the full observation, (lines, samples,
                bands) or (samples, bands) for a one-line qube.
            array (numpy.ndarray | None): The full data array, or None if the data are not
                loaded.
            samples (int): The number of samples along the time axis.
            lines (int): The number of lines in the full qube.
            line0 (int): The first line of the window.
            line1 (int): One past the last line of the window.
            line_bin (int): The binning factor along the line axis.
            bands (int): The number of bands in the full qube.
            band0 (int): The first band of the window.
            band1 (int): One past the last band of the window.
            band_bin (int): The binning factor along the band axis.
            rebin (bool): True to apply the binning factors to the field of view, the
                shape and the data array; False to ignore them.
            filepath (FCPath): The full path to the PDS label.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Observation: A :class:`~oops.observation.Pixel` for a one-line qube or a
            :class:`~oops.observation.TimedImage` otherwise.
        """

        # Trim the lines
        dline = line1 - line0
        if lines > 1 and (line0,line1) != (0,lines):
            fov = oops.fov.SliceFOV(fov, (0,line0), (1,dline))
            shape = (dline,) + shape[1:]

            if array is not None:
                array = array[line0:line1, :]

        # Trim the bands
        dband = band1 - band0
        if (band0,band1) != (0,bands):
            shape = shape[:-1] + (dband,)

            if array is not None:
                array = array[..., band0:band1]

        # Bin the lines
        if rebin and lines > 1 and line_bin > 1:
            assert dline % line_bin == 0
            fov = oops.fov.SubsampledFOV(fov, (1,line_bin))
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

        # Create the Observation
        params = CassiniUVIS._default_params(label, cadence, fov, frame_id)
        if lines == 1:
            obs = _build_observation(oops.observation.Pixel, ('t','b'), params, filepath,
                                     overrides)
        else:
            obs = _build_observation(oops.observation.TimedImage, ('v','ut','b'), params,
                                     filepath, overrides)

        CassiniUVIS._insert_subfields(obs, label, detector, product_type='QUBE',
                                      sampling=resolution, line_window=(line0,line1),
                                      line_bin=line_bin, band_window=(band0,band1),
                                      band_bin=band_bin, samples=samples, array=array)

        # Update the observation shape
        obs.shape = shape

        return obs

    @staticmethod
    def _get_time_series(filepath, tstart, label, data, *, fov, overrides):
        """The observation object given that it is a TIME_SERIES.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            tstart (float): The start time of the observation in seconds TDB.
            label (dict): The PDS label as a dictionary.
            data (bool): True to include the data array.
            fov (FOV | None): The FOV of the detector; None for the default.
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

        # Define the cadence
        assert info['COLUMNS'] == 1
        assert (info['SAMPLING_PARAMETER_UNIT'] == 'MILLISECOND' or
                info['SAMPLING_PARAMETER_UNIT'] == 'MILLISECONDS')
        texp = info['SAMPLING_PARAMETER_INTERVAL'] * 0.001

        # Define the observation
        if fov is None:
            fov = CassiniUVIS._FOVS[(detector, '', 1)]
        cadence = oops.cadence.Metronome(tstart, texp, texp, samples)
        params = CassiniUVIS._default_params(label, cadence, fov, _FRAME_IDS[detector])
        obs = _build_observation(oops.observation.Pixel, ('t',), params, filepath,
                                 overrides)

        # Load the data array if necessary
        array = None
        if data:
            column = info['PHOTOMETER_COUNTS']
            assert column['DATA_TYPE'] == 'MSB_UNSIGNED_INTEGER'
            assert column['BYTES'] == 2

            array = CassiniUVIS._load_data(filepath, label['^TIME_SERIES'], '>u2')

        CassiniUVIS._insert_subfields(obs, label, detector, product_type='TIME_SERIES',
                                      sampling=None, line_window=None, line_bin=None,
                                      band_window=None, band_bin=None, samples=samples,
                                      array=array)

        # Update the observation shape
        obs.shape = (samples,)

        return obs

    @staticmethod
    def _get_spectrum(filepath, tstart, label, data, *, fov, overrides):
        """The observation object given that it is a SPECTRUM.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            tstart (float): The start time of the observation in seconds TDB.
            label (dict): The PDS label as a dictionary.
            data (bool): True to include the data array.
            fov (FOV | None): The full FOV of the detector; None for the default.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Pixel: The observation.
        """

        # Determine the detector
        detector = label['PRODUCT_ID'][:3]
        assert detector in ('EUV', 'FUV')

        # Get array shape
        info = label['SPECTRUM']
        bands = info['ROWS']

        # Define the cadence (such as it is)
        assert info['COLUMNS'] == 1
        texp = label['INTEGRATION_DURATION']

        # Define the FOV
        resolution = label['SLIT_STATE']
        if fov is None:
            fov = CassiniUVIS._FOVS[(detector, resolution, 64)]

        line0 = info['UL_CORNER_SPATIAL']
        line1 = info['LR_CORNER_SPATIAL'] + 1
        if (line0,line1) != (0,64):
            fov = oops.fov.SliceFOV(fov, (0,line0), (1,line1-line0))

        line_bin = info['BIN_SPATIAL']
        if line_bin != 1:
            fov = oops.fov.SubsampledFOV(fov, (1,line_bin))

        # Define the observation
        cadence = oops.cadence.Metronome(tstart, texp, texp, 1)
        params = CassiniUVIS._default_params(label, cadence, fov, _FRAME_IDS[detector])
        obs = _build_observation(oops.observation.Pixel, ('b',), params, filepath,
                                 overrides)

        # Load the data array if necessary
        array = None
        if data:
            column = info['SPECTRUM']
            assert column['DATA_TYPE'] == 'MSB_UNSIGNED_INTEGER'
            assert column['BYTES'] == 2

            array = CassiniUVIS._load_data(filepath, label['^SPECTRUM'], '>u2')

        CassiniUVIS._insert_subfields(obs, label, detector, product_type='SPECTRUM',
                                      sampling=resolution, line_window=(line0,line1),
                                      line_bin=line_bin,
                                      band_window=(info['UL_CORNER_SPECTRAL'],
                                                   info['LR_CORNER_SPECTRAL']+1),
                                      band_bin=info['BIN_SPECTRAL'], samples=1,
                                      array=array)

        # Update the observation shape
        obs.shape = (bands,)

        return obs

    @staticmethod
    def _default_params(label, cadence, fov, frame_id):
        """The Observation constructor inputs before any overrides.

        Parameters:
            label (dict): The PDS label as a dictionary.
            cadence (Cadence): The cadence of the samples.
            fov (FOV): The field of view.
            frame_id (str): The ID of the instrument frame.

        Returns:
            dict: The inputs "cadence", "fov", "path", "frame", "target", "lightsource"
            and "calibrations", with the target derived from TARGET_NAME in `label` and
            the Sun as the light source.
        """

        target = _TARGET_NAME_REPAIRS.get(label['TARGET_NAME'], label['TARGET_NAME'])
        return {
            'cadence'     : cadence,
            'fov'         : fov,
            'path'        : 'CASSINI',
            'frame'       : frame_id,
            'target'      : target,
            'lightsource' : 'SUN',
            'calibrations': [],         # TBD
        }

    @staticmethod
    def _insert_subfields(obs, label, detector, *, product_type, sampling, line_window,
                          line_bin, band_window, band_bin, samples, array):
        """Insert the UVIS-specific subfields into an observation.

        Parameters:
            obs (Observation): The observation to modify.
            label (dict): The PDS label as a dictionary.
            detector (str): The detector name.
            product_type (str): "QUBE", "TIME_SERIES" or "SPECTRUM".
            sampling (str | None): The slit state; None to omit the `sampling` subfield.
            line_window (tuple[int, int] | None): The first line and one past the last.
            line_bin (int | None): The binning factor along the line axis.
            band_window (tuple[int, int] | None): The first band and one past the last.
            band_bin (int | None): The binning factor along the band axis.
            samples (int): The number of samples along the time axis.
            array (numpy.ndarray | None): The data array; None to omit the `data`
                subfield.
        """

        obs.insert_subfield('dict', label)
        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        if sampling is not None:
            obs.insert_subfield('sampling', sampling)
        obs.insert_subfield('product_type', product_type)
        obs.insert_subfield('label_target', label['TARGET_NAME'])

        obs.insert_subfield('line_window', line_window)
        obs.insert_subfield('line_bin', line_bin)

        obs.insert_subfield('band_window', band_window)
        obs.insert_subfield('band_bin', band_bin)

        obs.insert_subfield('samples', samples)

        if array is not None:
            obs.insert_subfield('data', array)

    @staticmethod
    def _load_data(filepath, body, dtype):
        """The contents of a UVIS data file as a NumPy array.

        If the data file named in the label does not exist, the same name in lower case
        is tried.

        Parameters:
            filepath (FCPath): Path to the label file.
            body (str): Basename of the data file, as named in the label.
            dtype (numpy.dtype): The data type of the values in the file.

        Returns:
            numpy.ndarray: The values read from the file.
        """

        data_filepath = FCPath(filepath).with_name(body)

        try:
            local_path = data_filepath.retrieve()
        except FileNotFoundError:
            data_filepath = data_filepath.with_name(body.lower())
            local_path = data_filepath.retrieve()

        return np.fromfile(local_path, sep='', dtype=dtype)

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
    def _initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                    gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the UVIS instrument.

        Fills in the FOV of each detector. Must be called first. After the first call,
        later calls to this function are ignored.

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

        # The instrument kernel keys each detector by its NAIF ID rather than its name
        ins = CassiniUVIS._INSTRUMENT_KERNEL['INS']

        # Construct a flat FOV for each detector
        fovs = {}
        for (key, (detector, resolution)) in _ABBREVS.items():

            # Get the FOV angles
            info = ins[_NAIF_IDS[key]]

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
        for frame_id in _FRAME_IDS.values():
            if not oops.Frame.frame_id_exists(frame_id):
                oops.frame.SpiceFrame(frame_id, frame_id=frame_id)

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
