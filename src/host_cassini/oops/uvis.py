##########################################################################################
# host_cassini/oops/uvis.py
##########################################################################################

import numbers
import re

import numpy as np
import julian

from filecache import FCPath
from pdsparser import Pds3Label

import oops
from oops.host import HostFormatError, HostIndexError
from ._cassini import (_Cassini, _cassini_obs_id_code, _CMATRIX_ROTATION,
                       _file_errors, _metadata_subfields, _pds3_options_with_defaults,
                       _read_label, _without_nulls)

__all__ = ['CassiniUVIS']


# The bands and lines of the EUV and FUV detectors, which are the full extent of every
# QUBE; an index row gives only the window within them
_DETECTOR_BANDS = 1024
_DETECTOR_LINES = 64

# The length of one HDAC integration in milliseconds. An index row gives the sampling
# interval of an HDAC time series only as DWELL_TIME, the number of integrations at each
# filament level. SAMPLING_PARAMETER_INTERVAL in the product labels is 125 ms times
# DWELL_TIME in 1,555 of 1,559 HDAC labels sampled across all 60 volumes (every 25th HDAC
# row); the other 4, in COUVIS_0031 and COUVIS_0032, give DWELL_TIME 72 and 125 ms (#33).
_HDAC_INTEGRATION_MS = 125

# The keys of the PDS3 objects of the three kinds of UVIS product, each a pointer to the
# data file too
_PRODUCT_OBJECTS = ('QUBE', 'TIME_SERIES', 'SPECTRUM')

# The correction to the text of a UVIS product label, as (pattern, replacement), which
# CassiniUVIS._correct_label_text applies before the parser reads the label: a comma after
# a number at the end of a line, which the strict parser refuses, in the 2 current labels
# FUV2013_069_22_35.LBL and HSP2013_069_22_34.LBL of COUVIS_0042:
#     RIGHT_ASCENSION           = 182.089,
# The comma is deleted.
_TRAILING_COMMA = (r'(?m)^([A-Z][A-Z0-9_]* *= *[-+]?[\d.]+),[ \t]*$', r'\1')

# DATA_SET_ID of a Cassini UVIS product: CO-<S, J or X>-UVIS-...
_UVIS_DATA_SET_ID = re.compile(r'CO-[A-Z]-UVIS-')

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

    #: The name under which this host registers with :class:`~oops.host.host_.Host` and
    #: that appears in its error messages.
    NAME = 'Cassini UVIS'

    # The order of the supplements is SETI/rms-data-projects#255 item 1.
    #
    #: The supplements of 54 of the 61 UVIS metadata indexes list the products in
    #: another order than the index, as much as 1,455 rows away, so each index row is
    #: paired with the supplement row for the same product.
    MERGE_SUPPLEMENT_BY_FILE = True

    #: The UVIS index has no VOLUME_ID column; its FILE_NAME already begins with the
    #: volume, as in "/COUVIS_0010/DATA/D2005_001/EUV2005_001_00_00.LBL".
    INDEX_FILE_KEY = ('FILE_NAME',)

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, enclose=False, pds3_options=None, select=None,
                  astrometry=False, target=None, lightsource=None, path=None, frame=None,
                  cadence=None, fov=None, calibrations=None, timeshift=None,
                  navigation=None, tracker=None, parallel=None):
        """One or more Observations based on the label of a Cassini UVIS file.

        The label is read, and unless `astrometry` is True, so is the data file that it
        names. The label is parsed with the "strict" method unless `pds3_options` gives
        another, after the correction of the known defect of the archive's labels.

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
                instrument, such as a :class:`~oops.frame.Cmatrix`, used as given.
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

        Returns:
            Observation | tuple[Observation, ...]: A :class:`~oops.observation.Pixel` for
            a SPECTRUM, a TIME_SERIES, or a one-line QUBE, or a
            :class:`~oops.observation.TimedImage` for a multi-line QUBE. A QUBE with
            multiple windows returns a tuple of observations, one per window, unless
            `enclose` is True. With `select`, the observation chosen by an index, or a
            tuple of the observations chosen by a slice or a tuple.

            Each observation has subfields `dict`, `label_dict`, `host`, `instrument`,
            `detector`, `product_type`, `line_window`, `line_bin`, `band_window`,
            `band_bin`, `samples`, `spice_kernels`, `spice_to_frame`, `spice_frame_name`,
            `spice_frame_id`, `ck_object_id` (-82000), `ck_sclk_id` (-82) and
            `spice_frame_is_frozen` (False); `sampling` unless it is a TIME_SERIES;
            `label_target` and `observation_id` when the label gives their items;
            `filepath`, `basename` and `image_url`, which name the data file;
            `label_url` and `label_abspath`, which name the label; and `image_abspath`
            and `data`, unless `astrometry` is True. `label_dict` is the parsed label, as
            its parser gives it, and `dict` is the same without the values that PDS3
            writes for no value ("UNK", "UNKNOWN", "N/A", "NULL", "NOT APPLICABLE",
            "---" or the empty string), each of which is left out, or None in a list.

        Raises:
            FileNotFoundError: If the label, or the data file of a label read without
                `astrometry`, does not exist.
            HostFormatError: If the label or the data file cannot be read or
                interpreted: if the label lacks an item that the observation needs, gives
                an invalid one or describes an unsupported option, or if the data file
                does not hold the number of values that the label gives. Subclass
                PdsHostError is raised if the label cannot be parsed or holds a value
                that cannot be converted. Each message names the file.
            OopsIndexError: If an index in `select` is out of range.
            OopsTypeError: If `select` has an invalid type.
            OopsValueError: If `select` is a string.
            WrongHostError: If the file is not the label of a Cassini UVIS product.
        """

        CassiniUVIS.initialize()  # define everything the first time through; use
                                  # defaults unless initialize() was called explicitly.

        pds3_options = _pds3_options_with_defaults(pds3_options)

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

        given = fileinfo.filepath if isinstance(fileinfo, Pds3Label) else fileinfo
        with _file_errors(given):
            fileinfo = _read_label(CassiniUVIS, fileinfo, astrometry=astrometry,
                                   pds3_options=pds3_options)
            label_path = FCPath(fileinfo.filepath)
            label = fileinfo.as_dict()
            key = next((key for key in _PRODUCT_OBJECTS if key in label), 'SPECTRUM')
            image_path = label_path.with_name(label['^' + key])
            subfields = _metadata_subfields(_without_nulls(label))
            subfields['label_dict'] = label
            return CassiniUVIS._make_observations(label, image_path=image_path,
                                                  label_path=label_path,
                                                  astrometry=astrometry,
                                                  enclose=enclose, overrides=overrides,
                                                  select=select, subfields=subfields)

    @staticmethod
    def from_index(filepath, *, enclose=False, supplement=oops.Host.SUPPLEMENT,
                   label=None, row_dicts=None, pds3_options=None, filter=None,
                   select=None, target=None, lightsource=None, path=None, frame=None,
                   cadence=None, fov=None, calibrations=None, timeshift=None,
                   navigation=None, tracker=None, parallel=None):
        """An iterator over Observations, one entry for each row of a data file index.

        It reads a metadata index of the PDS Ring-Moon Systems Node, such as
        metadata/COUVIS_0xxx/COUVIS_0001/COUVIS_0001_index.lbl, with its supplement,
        which a UVIS index needs. The call reads the labels of both tables; the rows are
        read as the iteration proceeds, 100 at a time, and each row's observations are
        built when the iteration reaches it. A caller that needs ``len()`` or indexing
        wraps the call in ``list()``.

        Each index row is paired with the supplement row for the same product, wherever
        it is in the supplement: the index's FILE_NAME, which begins with the volume,
        names the same file as the supplement's VOLUME_ID and FILE_SPECIFICATION_NAME,
        compared without regard to case or extension. No other column is compared.

        Each row's observation is the one that :meth:`from_file` gives for the product
        with `astrometry` True, built from the row's columns: the kind of product from
        DATA_OBJECT_TYPE; the window and binning of a QUBE or SPECTRUM from
        MINIMUM_BAND_NUMBER, MAXIMUM_BAND_NUMBER, BAND_BINNING_FACTOR,
        WINDOW_MINIMUM_LINE_NUMBER, WINDOW_MAXIMUM_LINE_NUMBER and LINE_BINNING_FACTOR;
        the number of samples of a QUBE or TIME_SERIES from LINE_SAMPLES; and the
        sampling interval of an HSP time series from INTEGRATION_DURATION, in
        milliseconds, or of an HDAC time series from DWELL_TIME, in integrations of 125
        milliseconds. A row gives one window, so for a QUBE with several windows it gives
        the window that encloses them all, binned by the smallest of their binnings,
        where :meth:`from_file` with `enclose` gives the same window unbinned.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            enclose (bool, optional): True to return a single observation for each row,
                regardless of how many windows are defined. If multiple windows are used,
                then the observation is defined by the enclosing limits in line and band,
                and the binning is assumed to be 1. If False and multiple windows are
                used, the function returns a tuple of observations rather than a single
                observation.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of the supplemental index file, in any directory. The default,
                `oops.Host.SUPPLEMENT`, reads the file beside `filepath` whose name has
                "_supplemental_index" in place of "_index". None raises an error, because
                the supplement is required.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, already merged with the supplement, one dictionary per row, if
                `filepath` has already been read. If provided, the tables are not read
                and `supplement` is ignored.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser, with
                which the label of the index, and with their method the label of the
                supplement, are parsed. The method is "strict" unless they give another.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included.
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
                instrument, such as a :class:`~oops.frame.Cmatrix`, used as given.
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

        Returns:
            collections.abc.Iterator[Observation | ObsArray]: The Observation or tuple of
            Observations for each filtered row of the index file, in index order. Each
            has the subfields of :meth:`from_file` with `astrometry` True, except
            `label_dict` and the file subfields: `filepath` is the row's VOLUME_ID and
            FILE_SPECIFICATION_NAME joined by "/", relative to the root of the volumes,
            and `basename` is the file name. `dict` is the merged row as the tables give
            it, with the "_mask" entry of each column.

        Raises:
            FileNotFoundError: If the label of `filepath` or of the supplement does not
                exist; when the iteration starts, if a table file does not exist.
            HostIndexError: When called, if `supplement` is None while `row_dicts` is
                not given, or if the label of the index or the supplement does not
                describe a table that can be read; during the iteration, as the reading
                of the rows finds that the supplement does not list the index's products.
            PdsHostError: If the PDS3 label of the index or the supplement cannot be
                parsed.
            WrongHostError: If `filepath` is not a Cassini UVIS index file.
        """

        CassiniUVIS.initialize()

        if supplement is None and row_dicts is None:
            raise HostIndexError(f'Cassini UVIS index needs its supplemental index: '
                                 f'{filepath}')

        # The index's FILE_NAME and the supplement's FILE_SPECIFICATION_NAME name the
        # same file, but the other columns that the two tables share write the same
        # value differently, -999 against -9.9e+99 for a fill, so no column is compared
        rows = CassiniUVIS._index_rows(filepath, supplement=supplement, label=label,
                                       row_dicts=row_dicts,
                                       pds3_options=_pds3_options_with_defaults(
                                           pds3_options),
                                       filter=filter, validate=())

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

        # Build each row's observations when the iteration reaches it
        def observations():
            for row_dict in rows:
                image_path = (str(row_dict['VOLUME_ID']) + '/'
                              + str(row_dict['FILE_SPECIFICATION_NAME']))
                yield CassiniUVIS._make_observations(
                    CassiniUVIS._label_from_index_row(row_dict), image_path=image_path,
                    label_path=None, astrometry=True, enclose=enclose,
                    overrides=overrides, select=select,
                    subfields=_metadata_subfields(row_dict))

        return observations()

    @staticmethod
    def _label_from_index_row(row_dict):
        """A UVIS label dictionary built from one row of an index merged with its
        supplemental index.

        A row names the kind of product in DATA_OBJECT_TYPE, where a product label has
        a QUBE, TIME_SERIES or SPECTRUM object, so the object is built from the row's
        columns.

        Parameters:
            row_dict (dict[str, Any]): The merged row. It is not modified.

        Returns:
            dict[str, Any]: A copy of `row_dict`, plus the object that DATA_OBJECT_TYPE
            names, with the keywords of that object that :meth:`_make_observations`
            reads, as :meth:`from_index` describes.
        """

        kind = row_dict['DATA_OBJECT_TYPE']
        label = dict(row_dict)

        if kind == 'TIME_SERIES':
            if row_dict['PRODUCT_ID'].startswith('HDAC'):
                interval = _HDAC_INTEGRATION_MS * int(row_dict['DWELL_TIME'])
            else:
                interval = float(row_dict['INTEGRATION_DURATION'])
            label['TIME_SERIES'] = {
                'ROWS'                       : int(row_dict['LINE_SAMPLES']),
                'COLUMNS'                    : 1,
                'SAMPLING_PARAMETER_UNIT'    : 'MILLISECOND',
                'SAMPLING_PARAMETER_INTERVAL': interval,
            }
            return label

        line0 = int(row_dict['WINDOW_MINIMUM_LINE_NUMBER'])
        line1 = int(row_dict['WINDOW_MAXIMUM_LINE_NUMBER'])
        line_bin = int(row_dict['LINE_BINNING_FACTOR'])
        band0 = int(row_dict['MINIMUM_BAND_NUMBER'])
        band1 = int(row_dict['MAXIMUM_BAND_NUMBER'])
        band_bin = int(row_dict['BAND_BINNING_FACTOR'])

        if kind == 'SPECTRUM':
            label['SPECTRUM'] = {
                'ROWS'              : (band1 - band0 + 1) // band_bin,
                'COLUMNS'           : 1,
                'UL_CORNER_SPATIAL' : line0,
                'LR_CORNER_SPATIAL' : line1,
                'BIN_SPATIAL'       : line_bin,
                'UL_CORNER_SPECTRAL': band0,
                'LR_CORNER_SPECTRAL': band1,
                'BIN_SPECTRAL'      : band_bin,
            }
            return label

        # A qube whose line window bins to a single line was stored with one line
        lines = 1 if line1 - line0 + 1 == line_bin else _DETECTOR_LINES
        label['QUBE'] = {
            'CORE_ITEMS'    : [_DETECTOR_BANDS, lines, int(row_dict['LINE_SAMPLES'])],
            'UL_CORNER_LINE': line0,
            'LR_CORNER_LINE': line1,
            'LINE_BIN'      : line_bin,
            'UL_CORNER_BAND': band0,
            'LR_CORNER_BAND': band1,
            'BAND_BIN'      : band_bin,
        }
        return label

    @staticmethod
    def _make_observations(label, *, image_path, label_path, astrometry, enclose,
                           overrides, select, subfields):
        """The observation(s) described by a UVIS label.

        Parameters:
            label (dict): The label content from a detached PDS3 label, or one built from
                an index row.
            image_path (str | FCPath): The data file, an FCPath for a file and the
                relative path that the index gives for a row.
            label_path (FCPath | None): The PDS3 label that was read; None for a row.
            astrometry (bool): True to omit the data arrays.
            enclose (bool): True to return a single observation for each row, regardless
                of how many windows are defined. If multiple windows are used, then the
                observation (and the optional data array) are defined by the enclosing
                limits in line and band, and the binning is assumed to be 1. If False and
                multiple windows are used, the function returns a tuple of observations
                rather than a single observation.
            overrides (dict): The standard override options of :meth:`from_file`.
            select (int | slice | tuple[int | slice, ...] | None): An index, a slice, or
                a tuple of indices and slices to apply to the returned observations,
                treating a single observation as a tuple of one; None for all.
            subfields (dict[str, Any]): The subfields that come from the metadata, which
                each observation receives.

        Returns:
            Observation | tuple[Observation, ...]: Observation(s) from the product after
            `select` has been applied.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
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
            result = CassiniUVIS._get_qube(image_path, label, enclose,
                                           astrometry=astrometry, overrides=overrides)
        elif 'TIME_SERIES' in label:
            result = CassiniUVIS._get_time_series(image_path, label,
                                                  astrometry=astrometry,
                                                  overrides=overrides)
        else:
            result = CassiniUVIS._get_spectrum(image_path, label, astrometry=astrometry,
                                               overrides=overrides)

        for obs in (result if isinstance(result, tuple) else (result,)):
            oops.Host._set_path_attributes(obs, image_path, label_path)
            obs.insert_subfields(subfields)

        if select is None:
            return result
        if not isinstance(result, tuple):
            result = (result,)
        return oops.Host._apply_select(result, select)

    @staticmethod
    def _get_qube(filepath, label, enclose, *, astrometry, overrides):
        """The observation object given that it is a QUBE.

        Parameters:
            filepath (str | FCPath): The data file, which is read for the array.
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

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
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

            array = CassiniUVIS._load_data(filepath, bands * lines * samples)

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
            filepath (str | FCPath): The data file, which is read for the array.
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

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
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
            if dline % line_bin != 0:
                raise HostFormatError(f'line window of {dline} lines is not a multiple '
                                      f'of LINE_BIN {line_bin} in {filepath}')
            fov = oops.fov.SubsampledFOV(fov, (1, line_bin))
            dline_binned = dline // line_bin
            shape = (dline_binned,) + shape[1:]

            if array is not None:
                array = array[:dline_binned]

        # Bin the bands
        if rebin and band_bin > 1:
            dband_binned = dband // band_bin
            shape = shape[:-1] + (dband_binned,)

            if array is not None:
                array = array[..., :dband_binned]

        # Define the Observation parameters
        params = CassiniUVIS._default_params(label, cadence, fov, frame_name)
        oops.Host._apply_overrides(params, **overrides)

        # Construct the Observation
        if lines == 1:
            obs = oops.observation.Pixel(('t', 'b'), **params)
        else:
            obs = oops.observation.TimedImage(('v', 'ut', 'b'), **params)

        if array is not None:
            obs.insert_subfield('data', array)

        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('sampling', resolution)
        obs.insert_subfield('product_type', 'QUBE')

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
            filepath (str | FCPath): The data file, which is read for the array.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Pixel: The observation.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
        """

        # Determine the detector
        product_id = label['PRODUCT_ID']
        if product_id.startswith('HSP'):
            detector = 'HSP'
        elif product_id.startswith('HDAC'):
            detector = 'HDAC'
        else:
            raise HostFormatError(f'Time series is neither HSP nor HDAC: {filepath}')

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
        oops.Host._apply_overrides(params, **overrides)

        # Construct the Observation
        obs = oops.observation.Pixel(('t',), **params)

        # Load the data array if necessary
        if not astrometry:
            column = info['PHOTOMETER_COUNTS']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in TIME_SERIES: '
                                      f'{filepath}')
            array = CassiniUVIS._load_data(filepath, samples)
            obs.insert_subfield('data', array)

        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('product_type', 'TIME_SERIES')

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
            filepath (str | FCPath): The data file, which is read for the array.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            Pixel: The observation.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
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
        oops.Host._apply_overrides(params, **overrides)
        obs = oops.observation.Pixel(('b',), **params)

        # Load the data array if necessary
        if not astrometry:
            column = info['SPECTRUM']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in SPECTRUM: {filepath}')
            array = CassiniUVIS._load_data(filepath, bands)
            obs.insert_subfield('data', array)

        obs.insert_subfield('host', 'Cassini')
        obs.insert_subfield('instrument', 'UVIS')
        obs.insert_subfield('detector', detector)
        obs.insert_subfield('product_type', 'SPECTRUM')
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
    def _load_data(filepath, count):
        """The contents of a UVIS data file as a NumPy array.

        Parameters:
            filepath (FCPath): The data file.
            count (int): The number of values that the label gives the file.

        Returns:
            numpy.ndarray: The values read from the file, as MSB 2-byte unsigned
            integers; the caller checks that the label describes the data that way.

        Raises:
            HostFormatError: If the file does not hold `count` values, so that the array
                would not be the one that the label describes.
        """

        array = np.fromfile(filepath.retrieve(), sep='', dtype='>u2')
        if array.size != count:
            raise HostFormatError(f'{filepath} holds {array.size} values but its label '
                                  f'gives {count}')
        return array

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 label of a UVIS product, with its known defect corrected.

        A comma after a number at the end of a line, as in two labels of the archive,
        which the "strict" parser refuses, is deleted. A label without this defect is
        returned unchanged.

        Parameters:
            text (str): The text of the label, with <LF> line terminators.
            filepath (FCPath): The path to the label.

        Returns:
            str: The corrected text.
        """

        return re.sub(*_TRAILING_COMMA, text)

    _OBS_CODE_CLEANUP = re.compile(r'(?:UR|)(.*?)(?:OCC|TI|V1|RDI|TK[ABC]|)$')

    @staticmethod
    def _star_from_obs_id(obs_id):
        """The StarSource from a Cassini Observation ID; None otherwise.

        If the star is identified by a different name in the OBSERVATION ID, add this as
        an `alt_name`. The product labels of the archive give OBSERVATION_ID as a
        number, such as 888, which names no star.
        """

        if not isinstance(obs_id, str):
            return None

        # Extract the code, strip "UR", "OCC", etc.
        obs_code = _cassini_obs_id_code(obs_id)
        obs_code = CassiniUVIS._OBS_CODE_CLEANUP.match(obs_code).group(1)

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

    @staticmethod
    def _detect_in_index(label):
        """Whether the given index label describes Cassini UVIS data.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an index file.

        Returns:
            bool | None: True if the label was read from a file whose path contains
            "COUVIS", in any case; otherwise None, meaning the host must be inferred from
            the index rows.
        """

        # The label itself records nothing that identifies the instrument, but the PDS
        # volume and RMS Node metadata names all start with COUVIS
        if label.filepath is not None and 'COUVIS' in label.filepath.as_posix().upper():
            return True
        return None

    @staticmethod
    def _detect_in_row(row_dict):
        """Whether the given row of an index file refers to Cassini UVIS data.

        Parameters:
            row_dict (dict): The content of one row of an index file.

        Returns:
            bool: True if the row's DATA_SET_ID names a Cassini UVIS data set, such as
            "CO-S-UVIS-2-CUBE-V1.2"; False otherwise.
        """

        return bool(_UVIS_DATA_SET_ID.match(row_dict.get('DATA_SET_ID', '')))

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
                leaving their handling to the caller. 'reconstructed' also loads the
                reconstructed C kernels of the Jupiter flyby and the cruise C kernels,
                which the reconstructed kernels take precedence over where both cover a
                time.
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
