##########################################################################################
# host_cassini/oops/uvis.py
##########################################################################################
"""The Cassini UVIS host, which makes observations from Ultraviolet Imaging Spectrograph
products.

The :doc:`user guide </user_guide/user_guide>` describes the products it reads, the
observations it returns, and the errors it raises.
"""

import numbers
import re

import numpy as np
import julian

from filecache import FCPath

import oops
from oops.host import HostFormatError, HostIndexError
from ._cassini import (_Cassini, _cassini_obs_id_code, _CMATRIX_ROTATION,
                       _common_subfields, _detect_index_by_path, _is_null, _known_path,
                       _label_dict, _pds3_options_with_defaults, _read_pds3_label,
                       _require_file_columns, _row_items, _without_nulls)

__all__ = ['CassiniUVIS']

_DEBUG = False      # True to assert that the data array must have null values outside the
                    # active windows
_ARRAY_NULL = 65535  # Incorrectly -1 in many labels

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

# DATA_SET_ID of a Cassini UVIS product: CO-<S, J or X>-UVIS-...
_UVIS_DATA_SET_ID = re.compile(r'CO-[A-Z]-UVIS-')

# TARGET_NAME in a UVIS label -> the target body; "NONE" if no body was targeted. A
# placeholder, such as UNK or N/A, also gives "NONE".
_TARGET_NAME_REPAIRS = {
    'AG'        : 'AEGAEON',
    'ATLAS:'    : 'ATLAS',
    'CALIB'     : 'NONE',
    'DA'        : 'DAPHNIS',
    'IPH'       : 'NONE',
    'ME'        : 'METHONE',
    'PL'        : 'PALLENE',
    'PO'        : 'POLYDEUCES',
    'SKY'       : 'NONE',
    'SOLAR WIND': 'NONE',
    'STAR'      : 'NONE',
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

    # Pairing by file works around the order until the supplements are corrected, and
    # stays right once they are: SETI/rms-data-projects#255 item 1; #30.
    #
    #: The supplements of 54 of the 61 UVIS metadata indexes in the holdings (COUVIS_0001
    #: to COUVIS_0060 and COUVIS_0999) list the products in another order than the index,
    #: as much as 1,455 rows away (survey of 2026-11-19), so each index row is paired with
    #: the supplement row for the same product.
    MERGE_SUPPLEMENT_BY_FILE = True

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, enclose=False, pds3_options=None, select=None,
                  astrometry=False, target=None, lightsource=None, path=None, frame=None,
                  cadence=None, fov=None, calibrations=None, timeshift=None,
                  navigation=None, tracker=None, parallel=None):
        """One or more Observations based on the label of a Cassini UVIS file.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label): The path to the PDS3 label
                of a UVIS data file, or the parsed label. A parsed label that came from no
                file has no file subfields, and a full read of it raises an error.
            enclose (bool, optional): True to return a single observation, regardless of
                how many windows are defined. If multiple windows are used, then the
                observation (and the optional data array) are defined by the enclosing
                limits in line and band, and the binning is assumed to be 1. If False and
                multiple windows are used, the function returns a tuple of observations
                rather than a single observation.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. A label
                that the requested parser, by default "fast", rejects or cannot convert
                is parsed again by each later one of "fast", "strict" and "loose".
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to the returned observations,
                treating a single observation as a tuple of one. For example, if the file
                defines two windows but `select=1`, only the second window's observation
                is returned. Its type is checked before the label is read.
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

        Returns:
            Observation | tuple[Observation, ...]: A :class:`~oops.observation.Pixel` for
            a SPECTRUM, a TIME_SERIES, or a one-line QUBE, or a
            :class:`~oops.observation.TimedImage` for a multi-line QUBE. A QUBE with
            multiple windows returns a tuple of observations, one per window, unless
            `enclose` is True. With `select`, the observation chosen by an index, or a
            tuple of the observations chosen by a slice or a tuple.

            Each observation has subfields `dict`, `label_dict`, `host`, `instrument`,
            `detector`, `product_type`, `line_window`, `line_bin`, `band_window`,
            `band_bin`, `samples` and `time_from_sclk` (always False); `sampling` unless
            it is a TIME_SERIES; `label_target`, `observation_id`,
            `spacecraft_clock_start_count` and `spacecraft_clock_stop_count` when the
            label gives their items; `filepath`, `basename` and `image_url`, which name
            the label, unless the label came from no file; and `abspath`, the local path
            of the data file that was read, and `data`, unless `astrometry` is True.
            `label_dict` is the label as parsed, with times as text and sets as lists,
            and `dict` is the same without the missing values ("UNK", "UNKNOWN", "N/A",
            "NULL", "---", and the empty string); both can be written as JSON.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid or describes
                an unsupported option, or if a full read needs the data file of a label
                that came from no file.
            OopsIndexError: If an index in `select` is out of range.
            OopsSpiceError: If no frame override is given and the C kernels lack the
                detector's pointing at the time at which an observation evaluates its
                frame (the midtime, after any cadence override, or the time that a
                `tracker` selects, moved by a time shift).
            OopsTypeError: If `select` has an invalid type.
            OopsValueError: If `select` is a string.
            OSError: If `filepath` cannot be read.
            PdsHostError: If no parser can read the PDS3 label; the message names the
                file and gives the error of each parser method tried.
            WrongHostError: If the file is not the label of a Cassini UVIS product.
        """

        # Argument errors are raised before anything is read (#20)
        oops.Host._check_select(select)

        CassiniUVIS.initialize()  # define everything the first time through; use
                                  # defaults unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        # Read the label
        fileinfo = _read_pds3_label(CassiniUVIS, fileinfo, astrometry=astrometry,
                                    pds3_options=pds3_options)
        filepath = _known_path(fileinfo.filepath)
        if filepath is None and not astrometry:
            raise HostFormatError('no file path is known for this Cassini UVIS PDS3 '
                                  'label, so its data file cannot be found')
        label = fileinfo.as_dict()
        label_dict = _label_dict(label)

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

        subfields = _common_subfields(_without_nulls(label_dict), label_dict=label_dict)
        return CassiniUVIS._make_observations(label, filepath, astrometry=astrometry,
                                              enclose=enclose, overrides=overrides,
                                              select=select, check=True,
                                              subfields=subfields)

    @staticmethod
    def from_index(filepath, *, enclose=False, supplement=oops.Host.SUPPLEMENT,
                   label=None, row_dicts=None, pds3_options=None, filter=None,
                   select=None, target=None, lightsource=None, path=None, frame=None,
                   cadence=None, fov=None, calibrations=None, timeshift=None,
                   navigation=None, tracker=None, parallel=None):
        """An iterator over Observations, one entry for each row of a data file index.

        It reads a Ring-Moon Systems Node metadata index, such as
        metadata/COUVIS_0xxx/COUVIS_0001/COUVIS_0001_index.lbl, and its supplement. When
        this method is called, it checks the arguments and reads the labels of the
        index and its supplement. The rows are read as the iteration proceeds, a few at a
        time, through ``oops.Host._index_rows``, and each row's observations are built
        when the iteration reaches that row, so an error in a row is raised at that row,
        after every earlier row has been yielded.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            enclose (bool, optional): True to return a single observation for each row,
                regardless of how many windows are defined. If multiple windows are used,
                then the observation (and the optional data array) are defined by the
                enclosing limits in line and band, and the binning is assumed to be 1. If
                False and multiple windows are used, the function returns a tuple of
                observations rather than a single observation.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of the supplemental index file, whose columns are merged into
                each row; a Cassini UVIS index needs its supplement. Its rows must list
                the same products as the index's, in any order: each index row is paired
                with the supplement row for the same product, the index's FILE_NAME
                matching the supplement's VOLUME_ID and FILE_SPECIFICATION_NAME. The
                supplement is read only as far ahead as the partner of the current index
                row. The default, `oops.Host.SUPPLEMENT`, reads the file in the
                directory of `filepath` whose name matches except that "_index" is
                replaced by "_supplemental_index" in upper or lower case. None or an
                empty string raises an error unless `row_dicts` is given.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, one dictionary per row, if `filepath` and `supplement` have
                already been read; each is filtered and built when the iteration reaches
                it. If provided, `supplement` and `label` are ignored.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. The
                options are checked when the call is made, unless `row_dicts` is given,
                when they are not used. The index's label is parsed with all of them, and
                the supplement's label with the method alone. Each label is parsed with
                one method, "fast" unless they give another, and no later method is
                tried.
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

        Returns:
            collections.abc.Iterator[Observation | ObsArray]: The Observation or tuple
            of Observations for each filtered row of the index file, in index order.

            Each row's observation is the one that :meth:`from_file` gives for the
            product with `astrometry` True, built from the row's columns: the kind of
            product from DATA_OBJECT_TYPE; the window and binning of a QUBE or SPECTRUM
            from MINIMUM_BAND_NUMBER, MAXIMUM_BAND_NUMBER, BAND_BINNING_FACTOR,
            WINDOW_MINIMUM_LINE_NUMBER, WINDOW_MAXIMUM_LINE_NUMBER and
            LINE_BINNING_FACTOR; the number of samples of a QUBE or TIME_SERIES from
            LINE_SAMPLES; and the sampling interval of an HSP time series from
            INTEGRATION_DURATION, in milliseconds, or of an HDAC time series from
            DWELL_TIME, in integrations of 125 milliseconds. A row gives one window, so
            for a QUBE with several windows it gives the window that encloses them all,
            binned by the smallest of their binnings, where :meth:`from_file` with
            `enclose` gives the same window unbinned.
            The `dict` subfield is the row in Python types, with the QUBE, TIME_SERIES
            or SPECTRUM object built from it, without the table's "_mask" entries and
            without its missing values: those the table's label declares null
            constants, such as -999; "UNK", "UNKNOWN", "N/A", "NULL", "---", and the
            empty string; and 1e32 and -1e32. `label_target`, `observation_id` and the
            clock counts are given as for :meth:`from_file`, and `time_from_sclk` is
            False. Where the C kernels cannot supply the pointing, the observation's
            frame raises :exc:`oops.OopsSpiceError` when it is used.

            `filepath` and `image_url` name the row's file, its FILE_SPECIFICATION_NAME,
            as a full path or URL within the volume named by VOLUME_ID, which is in the
            "volumes" tree beside a "metadata" tree holding the index, or otherwise in
            the index's own directory. No observation has `abspath`, since the file is
            not read.

        Raises:
            HostFormatError: If some aspect of an index row is not valid or describes an
                unsupported option, if it lacks a column that its observation needs, or
                if the sampling interval it gives for a time series or the
                BAND_BINNING_FACTOR it gives for a spectrum is not positive; raised when
                the iteration reaches that row.
            FileNotFoundError: If the label of the supplement does not exist; when the
                iteration starts, if a table file does not exist.
            HostIndexError: When called, if the label of the index or the supplement does
                not describe a table that can be read, or lacks a column that gives a
                row's product; or if `supplement` is None or an empty string while
                `row_dicts` is not given.
                During the iteration, at the row concerned: if an index row's product is
                not in the supplement, which is found once the whole supplement has been
                read; if the supplement gives a product a second time while its first
                row is still held; if a row of `row_dicts` has no supplemental columns
                merged in; if a row gives its VOLUME_ID or FILE_SPECIFICATION_NAME as a
                declared null constant or as text that the archive writes for no value;
                after every row that a table holds, if it holds fewer rows than its label
                says; or at the first of the 100 rows read with it, if a row is longer
                than the label's row length. After the last row, if the supplement lists
                a product that the index does not, or a table holds more rows than its
                label says. A value that cannot be read as its column's number gives a
                warning, and is missing from the row.
            OopsIndexError: If an index in `select` is out of range for a row; raised when
                the iteration reaches that row.
            OopsTypeError: If `select` has an invalid type; when the iteration reaches a
                row whose value cannot be compared with a range in `filter`.
            OopsValueError: If `select` is a string, or if `filter` uses a column that
                the index does not have; for `row_dicts` that is not a sequence, the
                `filter` error is raised when the iteration reads the first row. Also if
                `pds3_options` names a method the parser does not have or an option it
                does not take. The call checks the options when it is made, unless
                `row_dicts` is given.
            PdsHostError: If the PDS3 label of the index or the supplement cannot be
                parsed.
            OSError: If `filepath` or `supplement` cannot be read.
            WrongHostError: If `filepath` is not a Cassini UVIS index file.
        """

        oops.Host._check_select(select)

        # The supplement is required, so None or an empty path, either of which the index
        # reader takes to mean "read the index alone", is refused (#20)
        if row_dicts is None and (supplement is None or supplement == ''):
            raise HostIndexError(f'Cassini UVIS index needs its supplemental index: '
                                 f'{filepath}')

        # The labels are read now and the rows as the iteration reaches them. The index
        # gives each product as "/COUVIS_0001/DATA/.../X.LBL" and the supplement by its
        # volume and "DATA/.../X.LBL", in another order for most volumes until the
        # supplements are corrected (SETI/rms-data-projects#255 item 1; #30). The rows
        # are paired by these paths, compared without regard to case, which is the
        # whole check of a pair.
        rows = CassiniUVIS._index_rows(filepath, supplement=supplement, label=label,
                                       row_dicts=row_dicts, pds3_options=pds3_options,
                                       filter=filter, validate=(),
                                       file_key='FILE_NAME',
                                       supplement_key=('VOLUME_ID',
                                                       'FILE_SPECIFICATION_NAME'))
        volumes = oops.Host._volumes_directory(filepath)

        CassiniUVIS.initialize()

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
                # The file path and most of the observation's parameters come from the
                # supplement, which rows given by the caller may lack
                if 'FILE_SPECIFICATION_NAME' not in row_dict:
                    raise HostIndexError(f'Cassini UVIS needs a supplemental index: '
                                         f'{filepath}')

                # The table's declared null constants, -999 among them, and the text
                # that the archive writes for "no value" are missing
                items = _row_items(row_dict, masks=True)
                _require_file_columns(items, 'UVIS', filepath)
                fpath = oops.Host._index_file_path(volumes, items)  # path or URL (#29)
                row_label = CassiniUVIS._label_from_index_row(items, fpath)
                yield CassiniUVIS._make_observations(
                    row_label, fpath, astrometry=True, enclose=enclose,
                    overrides=overrides, select=select, check=False,
                    subfields=_common_subfields(row_label))

        return observations()

    @staticmethod
    def _label_from_index_row(row_dict, filepath):
        """A UVIS label dictionary built from one row of an index merged with its
        supplemental index.

        A row names the kind of product in DATA_OBJECT_TYPE, where a product label has
        a QUBE, TIME_SERIES or SPECTRUM object, so the object is built from the row's
        columns (#33).

        Parameters:
            row_dict (dict[str, Any]): The merged row. It is not modified.
            filepath (FCPath): The product's PDS3 label, for error messages.

        Returns:
            dict[str, Any]: A copy of `row_dict`, plus the object that DATA_OBJECT_TYPE
            names, with the keywords of that object that :meth:`_make_observations`
            reads, as :meth:`from_index` describes.

        Raises:
            HostFormatError: If the row lacks a column that its object needs, gives
                another kind of product, or gives a time series a sampling interval or a
                spectrum a BAND_BINNING_FACTOR that is not positive.
        """

        def column(name):
            if name not in row_dict:
                raise HostFormatError(f'Cassini UVIS index row has no {name}: {filepath}')
            return row_dict[name]

        kind = column('DATA_OBJECT_TYPE')
        label = dict(row_dict)

        if kind == 'TIME_SERIES':
            if column('PRODUCT_ID').startswith('HDAC'):
                interval = _HDAC_INTEGRATION_MS * int(column('DWELL_TIME'))
            else:
                interval = float(column('INTEGRATION_DURATION'))
            if interval <= 0:
                raise HostFormatError(f'Cassini UVIS index row gives a sampling interval '
                                      f'that is not positive: {filepath}')
            label['TIME_SERIES'] = {
                'ROWS'                       : int(column('LINE_SAMPLES')),
                'COLUMNS'                    : 1,
                'SAMPLING_PARAMETER_UNIT'    : 'MILLISECOND',
                'SAMPLING_PARAMETER_INTERVAL': interval,
            }
            return label

        if kind not in ('QUBE', 'SPECTRUM'):
            raise HostFormatError(f'unsupported Cassini UVIS DATA_OBJECT_TYPE {kind!r}: '
                                  f'{filepath}')

        line0 = int(column('WINDOW_MINIMUM_LINE_NUMBER'))
        line1 = int(column('WINDOW_MAXIMUM_LINE_NUMBER'))
        line_bin = int(column('LINE_BINNING_FACTOR'))
        band0 = int(column('MINIMUM_BAND_NUMBER'))
        band1 = int(column('MAXIMUM_BAND_NUMBER'))
        band_bin = int(column('BAND_BINNING_FACTOR'))

        if kind == 'SPECTRUM':
            if band_bin <= 0:
                raise HostFormatError(f'Cassini UVIS index row gives a '
                                      f'BAND_BINNING_FACTOR that is not positive: '
                                      f'{filepath}')
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
            'CORE_ITEMS'    : [_DETECTOR_BANDS, lines, int(column('LINE_SAMPLES'))],
            'UL_CORNER_LINE': line0,
            'LR_CORNER_LINE': line1,
            'LINE_BIN'      : line_bin,
            'UL_CORNER_BAND': band0,
            'LR_CORNER_BAND': band1,
            'BAND_BIN'      : band_bin,
        }
        return label

    @staticmethod
    def _make_observations(label, filepath, *, astrometry, enclose, overrides, select,
                           check, subfields):
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
            select (int | slice | tuple[int | slice, ...] | None): An index, a slice, or
                a tuple of indices and slices to apply to the returned observations,
                treating a single observation as a tuple of one; None for all.
            check (bool): True to raise OopsSpiceError now if the C kernels cannot give
                the pointing at the time at which an observation evaluates its frame;
                False to give such an observation a frame that raises it when it is
                used.
            subfields (dict[str, Any]): The subfields that come from the metadata, which
                each observation receives.

        Returns:
            Observation | tuple[Observation, ...]: Observation(s) from `filepath` after
            `select` has been applied.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
        """

        # Load any needed SPICE kernels, for a cadence override's time too
        tstart = CassiniUVIS._tdb(label, 'START_TIME', filepath)
        tstop = CassiniUVIS._tdb(label, 'STOP_TIME', filepath)
        if _Cassini.load_observation_kernels(tstart, tstop, overrides):
            CassiniUVIS._define_frames()

        # Figure out the PDS object class and get the observation(s)
        if 'QUBE' in label:
            result = CassiniUVIS._get_qube(filepath, label, enclose,
                                           astrometry=astrometry, overrides=overrides,
                                           check=check)
        elif 'TIME_SERIES' in label:
            result = CassiniUVIS._get_time_series(filepath, label, astrometry=astrometry,
                                                  overrides=overrides, check=check)
        else:
            result = CassiniUVIS._get_spectrum(filepath, label, astrometry=astrometry,
                                               overrides=overrides, check=check)

        for obs in (result if isinstance(result, tuple) else (result,)):
            obs.insert_subfields(subfields)

        if select is None:
            return result
        if not isinstance(result, tuple):
            result = (result,)
        return oops.Host._apply_select(result, select)

    @staticmethod
    def _get_qube(filepath, label, enclose, *, astrometry, overrides, check):
        """The observation object given that it is a QUBE.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            enclose (bool): True to combine multiple windows into a single observation
                defined by their enclosing limits in line and band; False to return one
                observation per window.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError if the C kernels cannot give the
                pointing; False to give the observation a frame that raises it when used.

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
            (array, abspath) = (None, None)
        else:
            if (info['CORE_ITEM_TYPE'] != 'MSB_UNSIGNED_INTEGER'
                    or info['CORE_ITEM_BYTES'] != 2):
                raise HostFormatError(f'unexpected QUBE data type in {filepath}')
            if info['SUFFIX_ITEMS'] != [0, 0, 0]:
                raise HostFormatError(f'unexpected QUBE SUFFIX_ITEMS in {filepath}')

            (array, abspath) = CassiniUVIS._load_data(filepath, label, 'QUBE')

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
                                             abspath=abspath, overrides=overrides,
                                             check=check)

        # Multiple windows combined into one enclosure
        if enclose:
            return CassiniUVIS._get_one_qube(filepath, label, array, line0=min(line0),
                                             line1=max(line1)+1, line_bin=min(line_bin),
                                             band0=min(band0), band1=max(band1)+1,
                                             band_bin=min(band_bin), rebin=False,
                                             abspath=abspath, overrides=overrides,
                                             check=check)

        # Separate windows
        return tuple(CassiniUVIS._get_one_qube(filepath, label, array, line0=line0[w],
                                               line1=line1[w]+1, line_bin=line_bin[w],
                                               band0=band0[w], band1=band1[w]+1,
                                               band_bin=band_bin[w], rebin=True,
                                               abspath=abspath, overrides=overrides,
                                               check=check)
                     for w in range(len(line0)))

    @staticmethod
    def _get_one_qube(filepath, label, array, *, line0, line1, line_bin, band0,
                      band1, band_bin, rebin, abspath, overrides, check):
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
            abspath (pathlib.Path | None): The local path of the data file if this call
                retrieved it to read `array`; otherwise None.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError if the C kernels cannot give the
                pointing; False to give the observation a frame that raises it when used.

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
        tstart = CassiniUVIS._tdb(label, 'START_TIME', filepath)
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
        params = CassiniUVIS._params(label, cadence, fov, frame_name, filepath,
                                     abspath=abspath, overrides=overrides, check=check)

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
    def _get_time_series(filepath, label, *, astrometry, overrides, check):
        """The observation object given that it is a TIME_SERIES.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError if the C kernels cannot give the
                pointing; False to give the observation a frame that raises it when used.

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
        tstart = CassiniUVIS._tdb(label, 'START_TIME', filepath)
        texp = info['SAMPLING_PARAMETER_INTERVAL'] * 0.001
        cadence = oops.cadence.Metronome(tstart, texp, texp, samples)

        # Load the data array if necessary
        if astrometry:
            (array, abspath) = (None, None)
        else:
            column = info['PHOTOMETER_COUNTS']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in TIME_SERIES: '
                                      f'{filepath}')
            (array, abspath) = CassiniUVIS._load_data(filepath, label, 'TIME_SERIES')

        # Define the Observation parameters
        fov = CassiniUVIS._FOVS[(detector, '', 1)]
        params = CassiniUVIS._params(label, cadence, fov, frame_name, filepath,
                                     abspath=abspath, overrides=overrides, check=check)

        # Construct the Observation
        obs = oops.observation.Pixel(('t',), **params)
        if array is not None:
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
    def _get_spectrum(filepath, label, *, astrometry, overrides, check):
        """The observation object given that it is a SPECTRUM.

        Parameters:
            filepath (FCPath): The full path to the PDS label.
            label (dict): The PDS label as a dictionary.
            astrometry (bool): False to include the data array.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError if the C kernels cannot give the
                pointing; False to give the observation a frame that raises it when used.

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
        tstart = CassiniUVIS._tdb(label, 'START_TIME', filepath)
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

        # Load the data array if necessary
        if astrometry:
            (array, abspath) = (None, None)
        else:
            column = info['SPECTRUM']
            if column['DATA_TYPE'] != 'MSB_UNSIGNED_INTEGER' or column['BYTES'] != 2:
                raise HostFormatError(f'non-standard data type in SPECTRUM: {filepath}')
            (array, abspath) = CassiniUVIS._load_data(filepath, label, 'SPECTRUM')

        # Define the observation
        params = CassiniUVIS._params(label, cadence, fov, frame_name, filepath,
                                     abspath=abspath, overrides=overrides, check=check)
        obs = oops.observation.Pixel(('b',), **params)
        if array is not None:
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
    def _params(label, cadence, fov, frame_name, filepath, *, abspath=None, overrides,
                check):
        """The Observation constructor inputs, with the overrides applied.

        Parameters:
            label (dict): The PDS label as a dictionary.
            cadence (Cadence): The cadence of the samples.
            fov (FOV): The field of view.
            frame_name (str): The SPICE name of the instrument frame.
            filepath (FCPath | None): The path to the PDS label, or None if it is not
                known.
            abspath (pathlib.Path, optional): The local path of the data file, given
                only when this call retrieved the file to read its array.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError if the C kernels cannot give the
                pointing at the time at which the observation evaluates its frame; False
                to give the observation a frame that raises it when it is used.

        Returns:
            dict: The inputs "cadence", "fov", "path", "frame", "target", "lightsource"
            and "calibrations", with the target derived from TARGET_NAME in `label` and
            the Sun as the light source, before the overrides; and the file subfields.

        Raises:
            OopsSpiceError: If `check` is True, no frame override is given, and the C
                kernels cannot give the pointing at the time at which the observation
                evaluates its frame.
        """

        target = label.get('TARGET_NAME')
        target = 'NONE' if _is_null(target) else _TARGET_NAME_REPAIRS.get(target, target)
        lightsource = CassiniUVIS._star_from_obs_id(label.get('OBSERVATION_ID')) or 'SUN'

        params = {
            'target'      : target,
            'lightsource' : lightsource,
            'path'        : 'CASSINI',
            'frame'       : frame_name,
            'cadence'     : cadence,
            'fov'         : fov,
            'calibrations': [],
        }

        # A frame that the C kernels cannot supply raises OopsSpiceError, never a cspyce
        # error: now from from_file, and from from_index when the frame is used. It is
        # evaluated as the observation will evaluate it once the overrides are applied.
        if overrides['frame'] is None:
            params['frame'] = oops.Host._check_spice_frame(
                frame_name, cadence, overrides,
                description=f'no Cassini UVIS {frame_name} pointing', filepath=filepath,
                defer=not check)

        # Parsed content from no file has no file subfields, and `abspath` is given only
        # by a read that retrieved the data file
        oops.Host._apply_overrides(params, filepath, abspath=abspath, **overrides)
        return params

    @staticmethod
    def _tdb(label, key, filepath):
        """A time of a UVIS label, in seconds TDB.

        Parameters:
            label (dict): The PDS label as a dictionary.
            key (str): "START_TIME" or "STOP_TIME".
            filepath (FCPath | None): The path to the PDS label, named in the error.

        Returns:
            float: The time.

        Raises:
            HostFormatError: If the label lacks the time or gives an invalid one.
        """

        if key not in label:
            raise HostFormatError(f'{key} is missing from {filepath}')
        try:
            return julian.tdb_from_iso(label[key])
        except (TypeError, ValueError) as err:
            raise HostFormatError(f'invalid {key} {label[key]!r} in {filepath}') from err

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
            tuple[numpy.ndarray, pathlib.Path]: The values read from the file, as MSB
            2-byte unsigned integers, which the caller checks that the label describes;
            and the local path that ``FCPath.retrieve()`` gave for the file.
        """

        basename = label['^' + key]
        data_filepath = FCPath(filepath).with_name(basename)

        try:
            local_path = data_filepath.retrieve()
        except FileNotFoundError:
            data_filepath = data_filepath.with_name(basename.lower())
            local_path = data_filepath.retrieve()

        return (np.fromfile(local_path, sep='', dtype='>u2'), local_path)

    _OBS_CODE_CLEANUP = re.compile(r'(?:UR|)(.*?)(?:OCC|TI|V1|RDI|TK[ABC]|)$')

    @staticmethod
    def _star_from_obs_id(obs_id):
        """The StarSource from a Cassini Observation ID; None otherwise.

        If the star is identified by a different name in the OBSERVATION ID, add this as
        an `alt_name`.
        """

        # The product labels in the archive give OBSERVATION_ID as a number, which names
        # no star (#36)
        if not isinstance(obs_id, str):
            return None

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

    @staticmethod
    def _detect_in_index(label):
        """Whether the given index label describes Cassini UVIS data.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an index file.

        Returns:
            bool | None: True if the label was read from a file whose name starts with
            "COUVIS", in any case; False if the name starts with "COISS" or "COVIMS", or
            if the label gives an INSTRUMENT_ID that is not "UVIS"; otherwise None,
            meaning the host must be inferred from the index rows.
        """

        # The UVIS metadata index labels record nothing that identifies the instrument,
        # but the RMS Node metadata index names all start with COUVIS
        status = _detect_index_by_path(label, 'COUVIS')
        if (status is None
                and str(label.get('INSTRUMENT_ID', 'UVIS')).strip() != 'UVIS'):
            return False
        return status

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
                reconstructed C kernels of the Jupiter flyby.
            planets (int | Sequence[int], optional): The number of a planet, 1 to 9, or a
                sequence of such numbers, whose systems
                :meth:`~oops.Body.define_solar_system` defines. None or 0 means all
                nine.
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
