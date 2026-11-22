##########################################################################################
# host_cassini/oops/vims.py
#
# Known shortcomings:
#
# For SAMPLING_MODE_ID == "UNDER" (aka Nyquist sampling), the FOV boundary will be
# slightly off because an FOV object cannot handle overlapping, double-sized pixels.
# The proper boundary is 1/64th larger along the line direction to account for the larger
# pixel.
##########################################################################################
"""The Cassini VIMS host, which makes a pair of observations from each Visual and
Infrared Mapping Spectrometer cube.

The :doc:`user guide </user_guide/user_guide>` describes the products it reads, the
observations it returns, and the errors it raises.
"""

import re

import numpy as np
from numpy.lib.stride_tricks import as_strided

import julian

from filecache import FCPath
from pdsparser import Pds3Label

import oops
from oops.host import HostFormatError, HostIndexError, WrongHostError
from ._cassini import (_Cassini, _cassini_obs_id_code, _CMATRIX_ROTATION,
                       _common_subfields, _detect_index_by_path, _fix_cassini_ring_target,
                       _is_null, _known_path, _label_dict, _pds3_options_with_defaults,
                       _read_pds3_label, _require_file_columns, _row_items,
                       _tdb_from_sclk, _without_nulls)

__all__ = ['CassiniVIMS']

# Timing correction as of 1/9/15 -- MRS
# EXTRA_INTERSAMPLE_DELAY =  0.000363     # observed empirically in V1555349441
# TIME_CORRECTION         = -0.337505

_TIME_FACTOR = 1.01725

# From Matt Hedman's IDL program navims.pro
#
# vims_params=fltarr(6)
# vims_params(0)=0.495e-3 ;pixel size of IR channels (rad) ;070910
# vims_params(1)=0.506e-3 ; pixel size of VIS channels (rad)
# vims_params(2)=-1.7 ; VIS offset x (in pixels)
# vims_params(3)=+1.5 ; VIS offset z (in pixels)
# vims_params(4)=1.98; HI-RES IR
# vims_params(5)=2.99; HI-RES VIS
#
# xi=(xo-1+i-31.5)*vims_params[0]
# zi=(zo-1+j-31.5)*vims_params[0]
# xv=(xo-1+i-31.5+vims_params[2])*vims_params[1]
# zv=(zo-1+j-31.5+vims_params[3])*vims_params[1]
#
# if hires(0) eq 1 then begin
#     hrf=vims_params(4)
#     xi=(xo-1+sq(1)/2./hrf+i/hrf-31.5)*vims_params[0]
# end
#
# if hires(1) eq 1 then begin
#     hrf=vims_params(5)
#     xv=(xo-1+sq(1)/hrf+i/hrf-31.5 +vims_params[2])*vims_params[1]
#     zv=(zo-1-0+sq(3)/hrf+j/hrf-31.5 +vims_params[3])*vims_params[1]
# end
#
# aimpointi=[xi,zi]
# aimpointv=[xv,zv]

_IR_NORMAL_PIXEL  = 0.495e-3
_VIS_NORMAL_PIXEL = 0.506e-3

_IR_HIRES_FACTOR  = 1.98
_VIS_HIRES_FACTOR = 2.99

_IR_NORMAL_SCALE  = oops.Pair((_IR_NORMAL_PIXEL,  _IR_NORMAL_PIXEL))
_VIS_NORMAL_SCALE = oops.Pair((_VIS_NORMAL_PIXEL, _VIS_NORMAL_PIXEL))

_IR_HIRES_SCALE  = oops.Pair((_IR_NORMAL_PIXEL/_IR_HIRES_FACTOR, _IR_NORMAL_PIXEL))
_VIS_HIRES_SCALE = oops.Pair((_VIS_NORMAL_PIXEL/_VIS_HIRES_FACTOR,
                              _VIS_NORMAL_PIXEL/_VIS_HIRES_FACTOR))

_IR_OVER_VIS = _IR_NORMAL_PIXEL / _VIS_NORMAL_PIXEL

_IR_FULL_FOV  = oops.fov.FlatFOV(_IR_NORMAL_SCALE,  oops.Pair((64,64)))
_VIS_FULL_FOV = oops.fov.FlatFOV(_VIS_NORMAL_SCALE, oops.Pair((64,64)))

# TARGET_NAME in a VIMS label -> the target body; "NONE" if no body was targeted. A
# placeholder, such as UNK or N/A, also gives "NONE".
_TARGET_NAME_REPAIRS = {
    'DARK SKY': 'NONE',
    'DUST'    : 'NONE',
    'OTHER'   : 'NONE',
    'SKY'     : 'NONE',
}

# The keywords of the VIMS labels in which -999 is a fill value: the exposure of a
# channel that is off (100 of 400 labels sampled on 2026-11-19), and the temperatures, in
# kelvins, data rate and compression ratio of v1294644598_1 of COVIMS_0001. None of them
# can be -999.
_FILL_KEYWORDS = frozenset({'DETECTOR_TEMPERATURE', 'EXPOSURE_DURATION',
                            'INSTRUMENT_DATA_RATE', 'INSTRUMENT_TEMPERATURE',
                            'INST_CMPRS_RATIO', 'OPTICS_TEMPERATURE'})

# These are star targets and need to be defined as LightSources
_TARGET_STARS = {'FOMALHAUT', 'SPICA'}

# These are repairs to star names embedded in the OBSERVATION_ID
# A second value is an alternative name.
_OBS_CODE_REPAIRS = {
    'ALCENA'    : 'alf01 Cen',
    'ALPBO'     : 'alf Boo',
    'CWSTAR'    : 'CW Leo',
    'GAMCRUETA' : 'gam Cru',
}

# Names for the `select` option of from_file() -> index in the returned (vis, ir) pair
_CHANNEL_NAMES = {'VIS': 0, 'IR': 1}

# The index does not record a cube's band count, which varies with the mission phase
# rather than with any index column. Observations from an index use this value, which
# only matters for reading data, never for geometry.
_INDEX_BANDS = 352

# A POINT or OCCULTATION row records SWATH_WIDTH = SWATH_LENGTH = 1, which is not the
# cube's shape, and the row's timing does not constrain the shape either: tour cubes are
# 64 x 64 even when STOP_TIME implies fewer lines. Nearly every such cube fills the
# 64-line buffer: every cruise POINT cube is 1 sample wide and 97% are 64 lines, and at
# least 97% of OCCULTATION cubes are 64 x 64.
_CORE_ITEMS_FROM_MODE = {'POINT'      : (1, _INDEX_BANDS, 64),
                         'OCCULTATION': (64, _INDEX_BANDS, 64)}

# A comment that opens on the line of a value and closes on the next line, which no PDS3
# parser accepts, because a PDS3 comment ends on the line where it starts. 119 of 570
# sampled labels have one, such as v1481842120_1_048.lbl of COVIMS_0005:
#     SNAPSHOT_MODE_FLAG               = OFF/* IR high resolution,
#                                          IR low resolution, Visible (degree K) */
# The two lines of the comment are joined. The labels are in the holdings, so the repair
# stays; CassiniVIMS._correct_label_text applies it.
_TWO_LINE_COMMENT = (r'(/\*(?:[^*\n]|\*(?!/))*)\n([^\n]*?\*/)', r'\1 \2')

# The SPICE frames of the VIS channel, the IR channel, and the IR channel's solar port
_VIS_FRAME_NAME    = 'CASSINI_VIMS_V'
_IR_FRAME_NAME     = 'CASSINI_VIMS_IR'
_IR_SOL_FRAME_NAME = 'CASSINI_VIMS_IR_SOL'
_FRAME_IDS = {_VIS_FRAME_NAME: -82370, _IR_FRAME_NAME: -82371, _IR_SOL_FRAME_NAME: -82372}

##########################################################################################
# FMT files have fixed values
##########################################################################################

_CORE_DESCRIPTION_FMT = Pds3Label("""\
  CORE_ITEM_BYTES                = 2
  CORE_ITEM_TYPE                 = SUN_INTEGER
  CORE_BASE                      = 0.0
  CORE_MULTIPLIER                = 1.0
  CORE_VALID_MINIMUM             = -4095
  CORE_NULL                      = -8192
  CORE_LOW_REPR_SATURATION       = -32767
  CORE_LOW_INSTR_SATURATION      = -32766
  CORE_HIGH_REPR_SATURATION      = -32764
  CORE_HIGH_INSTR_SATURATION     = -32765
  CORE_MINIMUM_DN                = -122
  CORE_NAME                      = "RAW DATA NUMBER"
  CORE_UNIT                      = DIMENSIONLESS
""", method='fast').as_dict()

_SUFFIX_DESCRIPTION_FMT = Pds3Label("""\
  GROUP                          = SAMPLE_SUFFIX
    SUFFIX_NAME                  = BACKGROUND
    SUFFIX_UNIT                  = DIMENSIONLESS
    SUFFIX_ITEM_BYTES            = 4
    SUFFIX_ITEM_TYPE             = SUN_INTEGER
    SUFFIX_BASE                  = 0.0
    SUFFIX_MULTIPLIER            = 1.0
    SUFFIX_VALID_MINIMUM         = 0
    SUFFIX_NULL                  = -8192
    SUFFIX_LOW_REPR_SAT          = -32767
    SUFFIX_LOW_INSTR_SAT         = -32766
    SUFFIX_HIGH_REPR_SAT         = -32764
    SUFFIX_HIGH_INSTR_SAT        = -32765
  END_GROUP                      = SAMPLE_SUFFIX

  GROUP                          = BAND_SUFFIX
    SUFFIX_NAME                  = (X_SCAN_DRIVE_CURRENT,
                                    Z_SCAN_DRIVE_CURRENT,
                                    X_SCAN_MIRROR_POSITION,
                                    Z_SCAN_MIRROR_POSITION)
    SUFFIX_UNIT                  = (DIMENSIONLESS,DIMENSIONLESS,
                                    DIMENSIONLESS,DIMENSIONLESS)
    SUFFIX_ITEM_TYPE             = (SUN_INTEGER,SUN_INTEGER,
                                    SUN_INTEGER,SUN_INTEGER)
    SUFFIX_ITEM_BYTES            = (4,4,4,4)
    SUFFIX_BASE                  = (0.0,0.0,0.0,0.0)
    SUFFIX_MULTIPLIER            = (1.0,1.0,1.0,1.0)
    SUFFIX_VALID_MINIMUM         = (0,0,0,0)
    SUFFIX_NULL                  = (-8192,-8192,-8192,-8192)
    SUFFIX_LOW_REPR_SAT          = (-32767,-32767,-32767,-32767)
    SUFFIX_LOW_INSTR_SAT         = (-32766,-32766,-32766,-32766)
    SUFFIX_HIGH_INSTR_SAT        = (-32765,-32765,-32765,-32765)
    SUFFIX_HIGH_REPR_SAT         = (-32764,-32764,-32764,-32764)
  END_GROUP                      = BAND_SUFFIX
""", method='fast').as_dict()

_BAND_BIN_CENTER_FMT = Pds3Label("""\
  GROUP                          = BAND_BIN
    BAND_BIN_CENTER = (0.35,0.36,0.37,0.37,0.38,0.39,0.40,0.40,0.41,0.42,
      0.42,0.43,0.44,0.45,0.45,0.46,0.47,0.48,0.49,0.49,0.50,0.51,0.51,0.52,
      0.53,0.53,0.54,0.55,0.56,0.56,0.57,0.58,0.59,0.59,0.60,0.61,0.62,0.62,
      0.63,0.64,0.64,0.65,0.66,0.67,0.67,0.68,0.69,0.70,0.70,0.71,0.72,0.72,
      0.73,0.74,0.75,0.75,0.76,0.77,0.78,0.78,0.79,0.80,0.81,0.81,0.82,0.83,
      0.83,0.84,0.85,0.86,0.86,0.87,0.88,0.89,0.89,0.90,0.91,0.92,0.92,0.93,
      0.94,0.94,0.95,0.96,0.97,0.97,0.98,0.99,1.00,1.00,1.01,1.02,1.02,1.03,
      1.04,1.05,0.863,0.879,0.896,0.912,0.928,0.945,0.961,0.977,0.994,1.010,
      1.026,1.043,1.060,1.077,1.093,1.109,1.125,1.142,1.159,1.175,1.191,1.207,
      1.224,1.240,1.257,1.273,1.290,1.306,1.322,1.338,1.355,1.372,1.388,1.404,
      1.421,1.437,1.453,1.470,1.487,1.503,1.519,1.535,1.552,1.569,1.585,1.597,
      1.620,1.637,1.651,1.667,1.684,1.700,1.717,1.733,1.749,1.766,1.783,1.799,
      1.815,1.831,1.848,1.864,1.882,1.898,1.914,1.930,1.947,1.964,1.980,1.997,
      2.013,2.029,2.046,2.063,2.079,2.095,2.112,2.128,2.145,2.162,2.178,2.194,
      2.211,2.228,2.245,2.261,2.277,2.294,2.311,2.328,2.345,2.363,2.380,2.397,
      2.413,2.430,2.446,2.462,2.479,2.495,2.512,2.528,2.544,2.559,2.577,2.593,
      2.610,2.625,2.642,2.656,2.676,2.691,2.707,2.728,2.743,2.758,2.776,2.794,
      2.811,2.827,2.845,2.861,2.877,2.894,2.910,2.926,2.942,2.958,2.972,2.996,
      3.009,3.025,3.043,3.059,3.075,3.092,3.107,3.125,3.142,3.158,3.175,3.192,
      3.209,3.227,3.243,3.261,3.278,3.294,3.311,3.328,3.345,3.361,3.377,3.394,
      3.410,3.427,3.444,3.460,3.476,3.493,3.508,3.525,3.542,3.558,3.575,3.591,
      3.609,3.626,3.644,3.660,3.678,3.695,3.712,3.729,3.746,3.763,3.779,3.796,
      3.812,3.830,3.846,3.857,3.877,3.894,3.910,3.926,3.943,3.959,3.975,3.992,
      4.008,4.024,4.042,4.058,4.076,4.092,4.110,4.127,4.144,4.161,4.178,4.193,
      4.206,4.219,4.237,4.255,4.273,4.292,4.310,4.328,4.346,4.361,4.378,4.393,
      4.410,4.427,4.443,4.461,4.477,4.495,4.511,4.529,4.547,4.563,4.581,4.598,
      4.615,4.631,4.649,4.665,4.682,4.698,4.715,4.732,4.749,4.765,4.782,4.798,
      4.815,4.831,4.848,4.864,4.881,4.898,4.915,4.932,4.949,4.967,4.984,5.001,
      5.017,5.036,5.052,5.069,5.086,5.102)
    BAND_BIN_UNIT                = MICROMETER
    BAND_BIN_ORIGINAL_BAND = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
      19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
      43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,
      67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
      91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,
      111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,
      129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,
      147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,
      165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,
      183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,
      201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,
      219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,
      237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,
      255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,
      273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,
      291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,
      309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,
      327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,
      345,346,347,348,349,350,351)
  END_GROUP                      = BAND_BIN
""", method='fast').as_dict()

# (detector, sampling) -> undersampling factors (u,v) of a VIMS pixel
_SHRINKAGE = {
    ('IR',  'NORMAL'): (1,1),
    ('IR',  'HI-RES'): (2,1),
    ('IR',  'UNDER' ): (2,1),
    ('VIS', 'NORMAL'): (1,1),
    ('VIS', 'HI-RES'): (3,3),
}


class CassiniVIMS(oops.Host):
    """The Cassini VIMS host: Observation constructors for VIS and IR cubes."""

    #: The name under which this host registers with :class:`~oops.host.host_.Host` and
    #: that appears in its error messages.
    NAME = 'Cassini VIMS'

    # Pairing by file works around the order until the supplements are corrected, and
    # stays right once they are: SETI/rms-data-projects#257 item 1; #34.
    #
    #: The supplements of 25 of the 95 VIMS metadata indexes in the holdings (COVIMS_0001
    #: to COVIMS_0094 and COVIMS_0999) list the cubes in another order than the index, as
    #: much as 28,990 rows away, in COVIMS_0003 and the cumulative COVIMS_0999 (survey of
    #: 2026-11-19), so each index row is paired with the supplement row for the same cube.
    MERGE_SUPPLEMENT_BY_FILE = True

    _INSTRUMENT_KERNEL = None
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, pds3_options=None, select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, cadence=None,
                  fov=None, calibrations=None, timeshift=None, navigation=None,
                  tracker=None, parallel=None):
        """A pair of Observations based on a Cassini VIMS cube or its PDS3 label.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label): The path to a VIMS cube
                in ISIS format, the path to its detached PDS3 label, or the parsed label.
                A parsed label that came from no file has no file subfields, and a full
                read of it raises an error.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. A
                comment that opens on a value's line and closes on the next, which no
                parser accepts, has its two lines joined first. A label that the
                requested parser, by default "fast", rejects or cannot convert is parsed
                again by each later one of "fast", "strict" and "loose".
            select (int | slice | str | tuple[int | slice | str, ...], optional): An
                index, slice, or tuple of indices and slices to apply to the returned
                (vis, ir) pair. A channel can also be chosen by name, "VIS" or "IR"
                (without regard to case). For example, `select=1` or `select='IR'` returns
                only the IR observation. Its type and names are checked before the file
                is read.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observations.
            target (str | oops.Body, optional): Override for the target Body of both
                observations, which is otherwise derived from TARGET_NAME in the label.
                Use "NONE" for inertial pointing, indicating that no Solar System body
                was tracked.
            lightsource (oops.LightSource, optional): Override for the LightSource of both
                observations, which is otherwise the Sun.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of both channels.
            cadence (oops.Cadence, optional): Override for the Cadence of the observation.
            fov (oops.FOV, optional): Override for the default FOV of the observing
                instrument. Note that this should only be used if either the IR or VIS
                channel has been explicitly chosen using `select`.
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
            tuple[Observation | None, Observation | None] | Observation | None: Without
            `select`, the pair (vis, ir), in which a channel that is off is None. With
            `select`, the observation chosen by an index or name, or a tuple of the
            observations chosen by a slice or a tuple.

            Each observation has subfields `dict`, `host`, `instrument`, `detector`,
            `sampling` and `time_from_sclk`, True only if the cube's time backplanes
            gave its times; `label_dict` for a detached PDS3 label or `header_dict` for
            the ISIS label of a cube, the label as parsed, with times as text and sets
            as lists; `label_target`, `observation_id`, `spacecraft_clock_start_count`
            and `spacecraft_clock_stop_count` when the label gives their items;
            `filepath`, `basename` and `image_url`, which name the file given, unless
            the label came from no file; and `abspath`, the local path of the cube that
            was read, and `data`, unless `astrometry` is True. `dict` is the label in the
            standard form, with the cube's object under both "QUBE" and "SPECTRAL_QUBE",
            without the missing values: "UNK", "UNKNOWN", "N/A", "NULL", "---", and the
            empty string, and -999 in the exposures, temperatures, data rate, and
            compression ratio. All three dictionaries can be written as JSON.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid or describes
                an unsupported option, or if a full read needs the cube of a label that
                came from no file.
            OopsIndexError: If an index in `select` is out of range.
            OopsSpiceError: If no frame override is given and the C kernels lack a
                channel's pointing at the time at which its observation evaluates its
                frame (the midtime, after any cadence override, or the time that a
                `tracker` selects, moved by a time shift), or if the spacecraft clock
                kernel cannot convert the cube's time backplanes.
            OopsTypeError: If `select` has an invalid type.
            OopsValueError: If a string in `select` is not a channel name, or if
                `pds3_options` holds an option that the PDS3 label parser does not accept
                while the method parses the label of a cube or a PDS3 label file; the
                options are not checked for a Pds3Label passed in.
            PdsHostError: If no parser can read the PDS3 or ISIS label; the message
                names the file and gives the error of each parser method tried.
            OSError: If the file cannot be read.
            WrongHostError: If the file is not a Cassini VIMS file.
        """

        # Argument errors are raised before anything is read (#20)
        oops.Host._check_select(select, names=_CHANNEL_NAMES)

        CassiniVIMS.initialize()  # define everything the first time through; use
                                  # defaults unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        # Read the label
        fileinfo = _read_pds3_label(CassiniVIMS, fileinfo, astrometry=astrometry,
                                    pds3_options=pds3_options)
        filepath = _known_path(fileinfo.filepath)
        if filepath is None and not astrometry:
            raise HostFormatError('no file path is known for this Cassini VIMS label, so '
                                  'its cube cannot be found')

        # Define the overrides
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

        # The label as read is `header_dict` for the ISIS label of a cube, which holds a
        # QUBE object, and `label_dict` for a detached PDS3 label
        raw = fileinfo.as_dict()
        read_dict = _label_dict(raw)
        label = CassiniVIMS._standard_label(raw)
        subfields = _common_subfields(_without_nulls(_label_dict(label), _FILL_KEYWORDS))
        subfields['header_dict' if 'QUBE' in read_dict else 'label_dict'] = read_dict

        # Define and select the object(s)
        pair = CassiniVIMS._make_observations(label, filepath, astrometry=astrometry,
                                              overrides=overrides, check=True,
                                              subfields=subfields)
        return oops.Host._apply_select(pair, select, _CHANNEL_NAMES)

    @staticmethod
    def from_index(filepath, *, supplement=oops.Host.SUPPLEMENT, label=None,
                   row_dicts=None, pds3_options=None, filter=None, select=None,
                   target=None, lightsource=None, path=None, frame=None, cadence=None,
                   fov=None, calibrations=None, timeshift=None, navigation=None,
                   tracker=None, parallel=None):
        """An iterator over Observations, one entry for each row of a data file index.

        It reads a Ring-Moon Systems Node metadata index, such as
        metadata/COVIMS_0xxx/COVIMS_0001/COVIMS_0001_index.lbl, and its supplement. When
        this method is called, it checks the arguments and reads the labels of the index
        and its supplement. The rows are read as the iteration proceeds, a few at a time,
        through ``oops.Host._index_rows``, and each row's observations are built when the
        iteration reaches that row, so an error in a row is raised at that row, after
        every earlier row has been yielded.

        With `row_dicts`, the given rows take the place of the tables, and the call reads
        only those rows. The errors below that come from reading a label or a table are
        then never raised: the labels need not exist, the index need not be Cassini VIMS,
        and `pds3_options` is not checked. What is still checked is `select`,
        `filter` against the first row, and each row when the iteration reaches it.

        The observations are for astrometry only and have no `data`.

        A supplemental index that records them would supply the cube's actual shape
        (CORE_SAMPLES, CORE_BANDS, CORE_LINES), the label's exposures
        (IR_EXPOSURE_DURATION, VIS_EXPOSURE_DURATION) and, for a cube with time
        backplanes, the spacecraft clock count of the first sample
        (BACKPLANE_SCLK_START_COUNT), which then replaces START_TIME as each
        observation's start. No supplement in the archive has these columns yet.

        Without those columns, the shape and the exposures are the index's best guess: a
        row records the swath but not the cube's band count, so each label's CORE_ITEMS
        lists a nominal 352 bands. A row does not record the pixel count of a POINT or
        OCCULTATION cube either; nearly every such cube fills the full 64-line buffer,
        so its observations assume 1 x 64 (POINT) or 64 x 64 (OCCULTATION) pixels with
        nominal timing, as :meth:`from_file` assumes with `astrometry` for the same
        cube. A channel whose recorded exposure is missing, zero or negative is treated
        as off, because only the cube's label holds its real exposure.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                data file index.
            supplement (str | pathlib.Path | FCPath | object, optional): The path to the
                PDS label of the supplemental index file, whose columns are merged into
                each row; a Cassini VIMS index needs its supplement. Its rows must list
                the same cubes as the index's, in any order: each index row is paired
                with the supplement row for the same cube, the index's VOLUME_ID,
                PATH_NAME and FILE_NAME matching the supplement's VOLUME_ID and
                FILE_SPECIFICATION_NAME without regard to the file name's extension. The
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
                it. If provided, they are used in place of the tables and their labels,
                and `supplement`, `label` and `pds3_options` are ignored.
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
            select (int | slice | str | tuple[int | slice | str, ...]): An index, slice,
                or tuple of indices and slices to apply to the returned array of
                Observation objects. For example, if the file contains two Observations
                but `select=1`, only the second Observation will be returned.
                Alternatively, specify one or more host-specific names of the data arrays.
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
                instrument. Note that this should only be used if either the IR or VIS
                channel has been explicitly chosen using `select`.
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
            collections.abc.Iterator[tuple[Observation | None, Observation | None]
            | Observation | None]: One entry per filtered row of the index, in index
            order: the (vis, ir) pair, in which a channel that is off is None, or the
            observation or tuple of observations chosen by `select`.

            The `dict` subfield is the row in Python types, with a "SPECTRAL_QUBE" whose
            CORE_ITEMS are the shape used, without the table's "_mask" entries and
            without its missing values: those the table's label declares null
            constants, such as -999; "UNK", "UNKNOWN", "N/A", "NULL", "---", and the
            empty string; and 1e32 and -1e32. `label_target`, `observation_id` and the
            clock counts are given as for :meth:`from_file`, and `time_from_sclk` is True
            when BACKPLANE_SCLK_START_COUNT gives the start. Where the C kernels cannot
            supply the pointing, an observation's frame raises :exc:`oops.OopsSpiceError`
            when it is used.

            `filepath` and `image_url` name the row's file, its FILE_SPECIFICATION_NAME,
            as a full path or URL within the volume named by VOLUME_ID, which is in the
            "volumes" tree beside a "metadata" tree holding the index, or otherwise in
            the index's own directory. No observation has `abspath`, since the file is
            not read.

        Raises:
            HostFormatError: If some aspect of an index row is not valid or describes an
                unsupported option; raised when the iteration reaches that row.
            FileNotFoundError: If the label of the supplement does not exist; when the
                iteration starts, if a table file does not exist.
            HostIndexError: When called, if the label of the index or the supplement does
                not describe a table that can be read, or lacks a column that gives a
                row's cube; or if `supplement` is None or an empty string while
                `row_dicts` is not given.
                During the iteration, at the row concerned: if an index row's cube is not
                in the supplement, which is found once the whole supplement has been
                read; if the supplement gives a cube a second time while its first row is
                still held; if a row of `row_dicts` has no supplemental columns merged
                in; if a row gives its VOLUME_ID or FILE_SPECIFICATION_NAME as a declared
                null constant or as text that the archive writes for no value; after
                every row that a table holds, if it holds fewer rows than its label says;
                or at the first of the 100 rows read with it, if a row is longer than the
                label's row length. After the last row, if the supplement lists a cube
                that the index does not, or a table holds more rows than its label says.
                A value that cannot be read as its column's number gives a warning, and
                is missing from the row.
            OopsIndexError: If an index in `select` is out of range for a row; raised when
                the iteration reaches that row.
            OopsSpiceError: If the spacecraft clock kernel cannot convert a row's
                BACKPLANE_SCLK_START_COUNT; raised when the iteration reaches that row.
            OopsTypeError: If `select` has an invalid type; when the iteration reaches a
                row whose value cannot be compared with a range in `filter`.
            OopsValueError: If a string in `select` is not a channel name, or if
                `filter` uses a column that the index does not have; for `row_dicts` that
                is not a sequence, the `filter` error is raised when the iteration reads
                the first row. Also if `pds3_options` names a method the parser does not
                have or an option it does not take. The call checks the options when it
                is made, unless `row_dicts` is given.
            PdsHostError: If the PDS3 label of the index or the supplement cannot be
                parsed.
            OSError: If `filepath` or `supplement` cannot be read.
            WrongHostError: If `filepath` is not a Cassini VIMS index file.
        """

        oops.Host._check_select(select, names=_CHANNEL_NAMES)

        # The supplement is required, so None or an empty path, either of which the index
        # reader takes to mean "read the index alone", is refused (#20)
        if row_dicts is None and (supplement is None or supplement == ''):
            raise HostIndexError(f'Cassini VIMS index needs its supplemental index: '
                                 f'{filepath}')

        # The labels are read now and the rows as the iteration reaches them. The index
        # gives each cube by PATH_NAME and FILE_NAME, its data file, as
        # "/data/.../v1294638283_1.qub", and the supplement by its label,
        # "data/.../v1294638283_1.lbl", in another order for some volumes until the
        # supplements are corrected (SETI/rms-data-projects#257 item 1; #34). The rows
        # are paired by these paths, compared without regard to case or extension,
        # which is the whole check of a pair. Given rows are used as they are:
        # _index_rows() returns them before it reads either label, so the label checks
        # are not made.
        rows = CassiniVIMS._index_rows(filepath, supplement=supplement, label=label,
                                       row_dicts=row_dicts, pds3_options=pds3_options,
                                       filter=filter, validate=(),
                                       file_key=('VOLUME_ID', 'PATH_NAME', 'FILE_NAME'),
                                       supplement_key=('VOLUME_ID',
                                                       'FILE_SPECIFICATION_NAME'))
        volumes = oops.Host._volumes_directory(filepath)

        CassiniVIMS.initialize()

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
                    raise HostIndexError(f'Cassini VIMS needs a supplemental index: '
                                         f'{filepath}')

                # The table's declared null constants, -999 among them, and the text
                # that the archive writes for "no value" are missing
                items = _row_items(row_dict, masks=True)
                _require_file_columns(items, 'VIMS', filepath)
                fpath = oops.Host._index_file_path(volumes, items)  # path or URL (#29)
                row_label = CassiniVIMS._label_from_index_row(items, fpath)
                meta = dict(items, SPECTRAL_QUBE={
                    'CORE_ITEMS': list(row_label['SPECTRAL_QUBE']['CORE_ITEMS'])})
                subfields = _common_subfields(
                    meta, time_from_sclk=('BACKPLANE_SCLK_START_COUNT' in items))
                pair = CassiniVIMS._make_observations(row_label, fpath, astrometry=True,
                                                      overrides=overrides, check=False,
                                                      subfields=subfields)
                yield oops.Host._apply_select(pair, select, _CHANNEL_NAMES)

        return observations()

    @staticmethod
    def _standard_label(label):
        """A VIMS label dictionary in a standard form, whether from ISIS or PDS3.

        Parameters:
            label (dict[str, Any]): The dictionary from an ISIS header or a detached PDS3
                label. It is modified in place.

        Returns:
            dict[str, Any]: The label with standardized keywords. The cube's description
            is under both "QUBE" and "SPECTRAL_QUBE" and the non-structural keywords are
            at the top level.
        """

        # Convert ISIS .qub info to a standard PDS3 label dictionary
        if 'QUBE' in label:

            # PDS3 labels use "SPECTRAL_QUBE" in place of "QUBE"
            label['SPECTRAL_QUBE'] = label['QUBE']

            # Copy the non-structural items to the top-level dictionary
            for (key, value) in label['QUBE'].items():
                if (key in {'AXES', 'AXIS_NAME'}
                        or key[:5] == 'CORE_'
                        or key[:7] in {'SUFFIX_', 'SAMPLE_', 'BAND_SU'}):
                    continue
                label[key] = value

            # Rename "PACKING" to "PACKING_FLAG"
            label['PACKING_FLAG'] = label['PACKING']
            return label

        # Otherwise, convert PDS3 label info to a standard ISIS dictionary, inserting the
        # info from the .FMT files
        label['SPECTRAL_QUBE'] = {**label['SPECTRAL_QUBE'], **_CORE_DESCRIPTION_FMT,
                                  **_SUFFIX_DESCRIPTION_FMT}
        label.update(_BAND_BIN_CENTER_FMT)

        # ISIS labels use "QUBE" in place of "SPECTRAL_QUBE"
        label['QUBE'] = label['SPECTRAL_QUBE']
        return label

    @staticmethod
    def _label_from_index_row(row_dict, filepath):
        """A VIMS label dictionary in the standard form of :meth:`_standard_label`, built
        from one row of an index merged with its supplemental index.

        Parameters:
            row_dict (dict[str, Any]): The merged row, in Python types and without missing
                values. It is not modified.
            filepath (FCPath): The cube's file, named in an error.

        Returns:
            dict[str, Any]: A copy of `row_dict`, plus a "SPECTRAL_QUBE" (also "QUBE")
            whose CORE_ITEMS are the row's CORE_SAMPLES, CORE_BANDS and CORE_LINES when
            the supplemental index records them, and otherwise (SWATH_WIDTH, 352,
            SWATH_LENGTH), or the nominal full buffer in POINT and OCCULTATION modes.
            The (IR, VIS) pairs POWER_STATE_FLAG, EXPOSURE_DURATION and SAMPLING_MODE_ID
            are built from the index's separate IR_ and VIS_ columns, with the
            EXPOSURE_DURATION columns of the supplemental index preferred when present;
            a channel whose exposure is missing, zero or negative is marked off. A
            BACKPLANE_SCLK_START_COUNT replaces START_TIME.

        Raises:
            OopsSpiceError: If the spacecraft clock kernel cannot convert
                BACKPLANE_SCLK_START_COUNT.
        """

        label = dict(row_dict)

        # A supplemental index that records the cube's shape is definitive. Otherwise,
        # except in POINT and OCCULTATION modes, the cube's samples and lines are the
        # swath's width and length
        mode = row_dict['INSTRUMENT_MODE_ID']
        if 'CORE_SAMPLES' in row_dict:
            core_items = (row_dict['CORE_SAMPLES'], row_dict['CORE_BANDS'],
                          row_dict['CORE_LINES'])
        elif mode in _CORE_ITEMS_FROM_MODE:
            core_items = _CORE_ITEMS_FROM_MODE[mode]
        else:
            core_items = (row_dict['SWATH_WIDTH'], _INDEX_BANDS, row_dict['SWATH_LENGTH'])
        label['SPECTRAL_QUBE'] = {'CORE_ITEMS': core_items}
        label['QUBE'] = label['SPECTRAL_QUBE']

        # The IR_EXPOSURE and VIS_EXPOSURE columns sometimes hold a -999 placeholder for
        # a cube whose label records the exposure; the EXPOSURE_DURATION columns of a
        # supplemental index that has them repeat the label. Only the cube's label holds
        # an exposure that the index does not give, and without one the channel's
        # cadence cannot be built, so the channel is treated as off.
        if 'IR_EXPOSURE_DURATION' in row_dict or 'VIS_EXPOSURE_DURATION' in row_dict:
            names = ('IR_EXPOSURE_DURATION', 'VIS_EXPOSURE_DURATION')
        else:
            names = ('IR_EXPOSURE', 'VIS_EXPOSURE')
        exposures = tuple(row_dict.get(name, 0.) for name in names)
        flags = (row_dict.get('IR_POWER_STATE_FLAG', 'OFF'),
                 row_dict.get('VIS_POWER_STATE_FLAG', 'OFF'))
        label['EXPOSURE_DURATION'] = exposures
        label['POWER_STATE_FLAG'] = tuple('OFF' if texp <= 0. else flag
                                          for (flag, texp) in zip(flags, exposures,
                                                                  strict=True))
        label['SAMPLING_MODE_ID'] = (row_dict.get('IR_SAMPLING_MODE_ID', ''),
                                     row_dict.get('VIS_SAMPLING_MODE_ID', ''))

        # A cube with time backplanes records its first spacecraft clock count in a
        # supplemental index that has this column; it is more accurate than START_TIME
        if 'BACKPLANE_SCLK_START_COUNT' in row_dict:
            tdb = _tdb_from_sclk(row_dict['BACKPLANE_SCLK_START_COUNT'], filepath)
            label['START_TIME'] = julian.iso_from_tai(julian.tai_from_tdb(tdb), digits=3)
        return label

    @staticmethod
    def _make_observations(label, filepath, *, astrometry, overrides, check, subfields):
        """The VIS and IR observations described by a standardized VIMS label.

        Parameters:
            label (dict): The label, as returned by :meth:`_standard_label` or
                :meth:`_label_from_index_row`.
            filepath (FCPath | None): The path to the cube or to its detached PDS3 label,
                or None if it is not known.
            astrometry (bool): True to omit the data arrays.
            overrides (dict): The standard override options of :meth:`from_file`.
            check (bool): True to raise OopsSpiceError now if the C kernels cannot give
                a channel's pointing at the time at which its observation evaluates its
                frame; False to give such an observation a frame that raises it when it
                is used.
            subfields (dict[str, Any]): The subfields that come from the metadata, which
                each observation receives; `time_from_sclk` becomes True if the times
                come from the cube's time backplanes.

        Returns:
            tuple[Observation | None, Observation | None]: Observation, if any, for VIS
            and IR.

        Raises:
            HostFormatError: If some aspect of the label or file is not valid.
            OopsSpiceError: If `check` is True, no frame override is given, and the C
                kernels cannot give a channel's pointing at the time at which its
                observation evaluates its frame; or if the spacecraft clock kernel
                cannot convert the cube's time backplanes.
        """

        qube_dict = label['SPECTRAL_QUBE']

        # Load any needed SPICE kernels, for a cadence override's time too
        tstart = julian.tdb_from_iso(label['START_TIME'])
        tstop = julian.tdb_from_iso(label['STOP_TIME'])
        using_cks = _Cassini.load_observation_kernels(tstart, tstop, overrides)
        if using_cks:
            CassiniVIMS._define_frames()

        # Check power state of each channel: [0] is IR; [1] is VIS
        ir_is_off  = label['POWER_STATE_FLAG'][0] == 'OFF'
        vis_is_off = label['POWER_STATE_FLAG'][1] == 'OFF'

        ##################################################################################
        # Load the data arrays
        ##################################################################################

        (samples, bands, lines) = qube_dict['CORE_ITEMS']
        if bands not in (352, 256, 96):
            raise HostFormatError(f'unrecognized band axis size {bands} in {filepath}')

        if astrometry:  # always True for from_index()
            vis_data = None
            ir_data = None
            times = None
            abspath = None
        else:
            filepath = FCPath(filepath)
            if filepath.suffix.lower() == '.lbl':
                data_file = filepath.parent / label['^QUBE']
                header_recs = label['^QUBE_offset'] - 1
            else:
                data_file = filepath
                header_recs = label['^QUBE'] - 1

            (array, times) = CassiniVIMS._load_data_and_times(data_file, header_recs,
                                                              label)

            # Only a cube retrieved in this call has a local path that is known to
            # exist, which becomes `abspath`. The read above retrieved it, so this
            # retrieval finds it in place.
            abspath = data_file.retrieve()
            if array.shape != (lines, samples, bands):
                raise HostFormatError(f'invalid array shape {array.shape} in {filepath}')

            if bands == 352:
                vis_data = array[:,:,:96]   # index order is [line, sample, band]
                ir_data  = array[:,:,96:]
            elif bands == 256:              # only happens in a few early cubes
                vis_data = None
                ir_data = array
            else:
                vis_data = array
                ir_data = None

        ##################################################################################
        # Define the FOVs
        ##################################################################################

        swath_width = label['SWATH_WIDTH']
        swath_length = label['SWATH_LENGTH']

        frame_size = swath_width * swath_length
        frames = (samples * lines) // frame_size
        if samples * lines != frames * frame_size:
            raise HostFormatError(f'mismatched array shape and size in {filepath}')

        # Replace multiple one-line frames by one frame, multiple lines
        if frames > 1 and frame_size != 1:
            if swath_length != 1:
                raise HostFormatError(f'invalid SWATH_LENGTH {swath_length} '
                                      f'in {filepath}')
            swath_length = frames
            lines = frames
            frame_size = swath_width * swath_length

        uv_shape = (swath_width, swath_length)

        x_offset = label['X_OFFSET']
        z_offset = label['Z_OFFSET']
        uv_los = (33. - x_offset, 33. - z_offset)

        vis_sampling = label['SAMPLING_MODE_ID'][1].strip()  # handle ' NORMAL'
        ir_sampling  = label['SAMPLING_MODE_ID'][0].strip()
        sampling = (vis_sampling, ir_sampling)

        # VIS FOV
        if vis_sampling == 'HI-RES':
            uv_los_hires = (_VIS_HIRES_FACTOR * uv_los[0] - uv_shape[0],
                            _VIS_HIRES_FACTOR * uv_los[1] - uv_shape[1])
            vis_fov = oops.fov.FlatFOV(_VIS_HIRES_SCALE, uv_shape, uv_los=uv_los_hires)

        elif uv_shape == (64,64):
            vis_fov = _VIS_FULL_FOV

        else:
            vis_fov = oops.fov.FlatFOV(_VIS_NORMAL_SCALE, uv_shape,
                                       uv_los=(_IR_OVER_VIS * uv_los[0],
                                               _IR_OVER_VIS * uv_los[1]))

        # IR FOV
        if label['INSTRUMENT_MODE_ID'] == 'OCCULTATION':
            if ir_sampling == 'NORMAL':
                ir_fov = oops.fov.FlatFOV(_IR_NORMAL_SCALE, uv_shape, uv_los=uv_los)
            else:
                ir_fov = oops.fov.FlatFOV(_IR_HIRES_SCALE, uv_shape, uv_los=uv_los)

        elif ir_sampling in ('HI-RES','UNDER'):
            ir_fov = oops.fov.FlatFOV(_IR_HIRES_SCALE, uv_shape,
                                      uv_los=(_IR_HIRES_FACTOR * uv_los[0]
                                              - uv_shape[0]/2., uv_los[1]))

        elif uv_shape == (64,64):
            ir_fov = _IR_FULL_FOV

        else:
            ir_fov = oops.fov.FlatFOV(_IR_NORMAL_SCALE, uv_shape, uv_los=uv_los)

        # Nyquist sampling
        # TBD: VIMS IR sampling mode UNDER is untested!!
        # In UNDER sampling the IR detector is twice the size of the sample spacing, so
        # the pixels overlap. An FOV cannot represent overlapping pixels, so the boundary
        # is slightly off, as noted at the top of this file. GapFOV models the opposite
        # case, pixels smaller than the sample spacing, so it does not apply here.
        #
        # Note that this assignment also overrides every branch of the "IR FOV" logic
        # above.
        ir_fov = oops.fov.FlatFOV(_IR_NORMAL_SCALE, uv_shape, uv_los=uv_los)

        ##################################################################################
        # Define the cadences
        ##################################################################################

        # Define cadences based on header parameters
        ir_texp  = label['EXPOSURE_DURATION'][0] * 0.001 * _TIME_FACTOR
        vis_texp = label['EXPOSURE_DURATION'][1] * 0.001 * _TIME_FACTOR

        interline_delay = label['INTERLINE_DELAY_DURATION'] * 0.001 * _TIME_FACTOR

        # Adjust the timing of one line, multiple frames
        if frames > 1 and frame_size != 1:
            if 'INTERFRAME_DELAY_DURATION' not in label:
                raise HostFormatError(f'INTERFRAME_DELAY_DURATION is missing from '
                                      f'{filepath}')
            interline_delay = label['INTERFRAME_DELAY_DURATION'] * 0.001 * _TIME_FACTOR

        length_stride = max(ir_texp * swath_width, vis_texp) + interline_delay

        backplane_cadence = None

        # Define a cadence based on the time backplane, if it is present
        if times is None:
            pass

        elif label['OVERWRITTEN_CHANNEL_FLAG'] == 'ON':
            times = times.ravel()
            if times[0] >= times[1]:
                raise HostFormatError(f'invalid embedded times in {filepath}')
            if not vis_is_off:
                raise HostFormatError(f'OCCULTATION mode with VIS channel ON in '
                                      f'{filepath}')
            backplane_cadence = oops.cadence.Sequence(times, ir_texp)

        else:
            # Otherwise the embedded times just provide a better tstart. For a packed
            # cube, this approximates the per-frame times by the uniform header cadence.
            times = times.ravel()
            tstart = times[0]

        # The header cadence of each active channel. An unused channel's exposure can be
        # zero or a -999 placeholder, so no cadence is built for a channel that is off.
        if not vis_is_off:
            vis_header_cadence = oops.cadence.Metronome(tstart, length_stride,
                                                        vis_texp, swath_length)
        if not ir_is_off:
            ir_fast_cadence = oops.cadence.Metronome(tstart, ir_texp, ir_texp,
                                                     swath_width)

        # At this point...
        #   vis_header_cadence  defined if the VIS channel is on, always 1-D.
        #   ir_fast_cadence     defined if the IR channel is on, always 1-D.
        #   backplane_cadence   defined if timing was recorded, always 1-D.

        ##################################################################################
        # Define the coordinate frames
        ##################################################################################

        ir_frame_name = _IR_FRAME_NAME
        if (label.get('TARGET_NAME') == 'SUN'
                or '_SOL' in str(label.get('OBSERVATION_ID', ''))):
            ir_frame_name = _IR_SOL_FRAME_NAME
        frame_names = (_VIS_FRAME_NAME, ir_frame_name)

        ##################################################################################
        # Define the Observation objects
        ##################################################################################

        vis_info = None
        ir_info  = None

        # POINT/OCCULTATION case
        if swath_width == 1 and swath_length == 1:

            # The IR raster's timing, which the VIS channel shares; not built when both
            # channels are off, because the IR exposure can then be a placeholder
            if backplane_cadence is None and not (ir_is_off and vis_is_off):
                fast_stride = ir_texp   # + EXTRA_INTERSAMPLE_DELAY
                fastcad = oops.cadence.Metronome(tstart, fast_stride, ir_texp, samples)

                slow_stride = ir_texp * samples + interline_delay
                slowcad = oops.cadence.Metronome(tstart, slow_stride, slow_stride, lines)

            # A cruise POINT cube usually keeps the VIS channel on: its pixels share the
            # IR raster's timing with the VIS exposure. A cube with embedded timing never
            # has VIS on, so the IR raster's cadences are always defined here.
            if not vis_is_off:
                viscad = oops.cadence.Metronome(tstart, fast_stride, vis_texp, samples)
                viscad = oops.cadence.DualCadence(slowcad, viscad)
                vis_cadence = oops.cadence.ReshapedCadence(viscad, (samples*lines,))

                if vis_data is not None:
                    vis_data = vis_data.reshape((frames, 96))

                vis_info = (oops.observation.Pixel, ('t','b'), vis_cadence, vis_fov,
                            _VIS_FRAME_NAME)

            if not ir_is_off:
                if backplane_cadence is None:
                    fullcad = oops.cadence.DualCadence(slowcad, fastcad)
                    ir_cadence = oops.cadence.ReshapedCadence(fullcad, (samples*lines,))
                else:
                    ir_cadence = backplane_cadence

                if ir_data is not None:
                    ir_data = ir_data.reshape((frames, 256))

                ir_info = (oops.observation.Pixel, ('t','b'), ir_cadence, ir_fov,
                           ir_frame_name)

        # Single LINE case
        elif swath_length == 1:
            if not vis_is_off:
                if vis_data is not None:
                    vis_data = vis_data.reshape((samples, 96))

                vis_info = (oops.observation.Slit1D, ('u','b'),
                            oops.cadence.SnapCadence(tstart, vis_texp),
                            vis_fov, _VIS_FRAME_NAME)

            if not ir_is_off:
                if ir_data is not None:
                    ir_data = ir_data.reshape((samples, 256))

                if backplane_cadence is not None:
                    ir_fast_cadence = backplane_cadence

                ir_info = (oops.observation.RasterSlit1D, ('ut','b'), ir_fast_cadence,
                           ir_fov, ir_frame_name)

        # Single 2-D IMAGE case
        elif samples == swath_width and lines == swath_length:
            if not vis_is_off:
                vis_info = (oops.observation.TimedImage, ('vt','u','b'),
                            vis_header_cadence, vis_fov, _VIS_FRAME_NAME)

            if not ir_is_off:
                if backplane_cadence is None:
                    # The IR lines follow the same stride as the VIS cadence, whether or
                    # not the VIS channel is on
                    line_cadence = oops.cadence.Metronome(tstart, length_stride,
                                                          length_stride, swath_length)
                    ir_cadence = oops.cadence.DualCadence(line_cadence, ir_fast_cadence)
                else:
                    ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                                              (lines,samples))

                ir_info = (oops.observation.TimedImage, ('vslow','ufast','b'),
                           ir_cadence, ir_fov, ir_frame_name)

        else:
            raise HostFormatError(f'unsupported VIMS format in file {filepath}')

        ##################################################################################
        # Define Observation attributes
        ##################################################################################

        # Determine target and light source; with no target named, nothing is tracked
        label_target = label.get('TARGET_NAME')
        label_target = ('NONE' if _is_null(label_target)
                        else _TARGET_NAME_REPAIRS.get(label_target, label_target))
        if label_target in _TARGET_STARS:
            label_lightsource = oops.lightsource.star_lookup(label_target)
            label_target = 'NONE'
        else:
            obs_id = label.get('OBSERVATION_ID', '')
            label_lightsource = CassiniVIMS._star_from_obs_id(obs_id) or 'SUN'
            label_target = _fix_cassini_ring_target(label_target, label, tstart)

        # The time came from a clock count if the cube's time backplanes gave it
        subfields = dict(subfields)
        subfields['time_from_sclk'] = subfields['time_from_sclk'] or times is not None

        # Create the parameter dictionaries and apply the overrides
        obs_pair = [None, None]
        detectors = ('VIS', 'IR')
        arrays = (vis_data, ir_data)
        for k, info in enumerate((vis_info, ir_info)):
            if info is not None:
                (obs_subclass, axes, cadence, fov, frame) = info
                # A frame that the C kernels cannot supply raises OopsSpiceError, never
                # a cspyce error: now from from_file, and from from_index when the frame
                # is used. It is evaluated as the observation will evaluate it once the
                # overrides are applied.
                if using_cks:
                    frame = oops.Host._check_spice_frame(
                        frame, cadence, overrides,
                        description=f'no Cassini VIMS {frame} pointing',
                        filepath=filepath, defer=not check)

                params = {
                    'target'      : label_target,
                    'lightsource' : label_lightsource,
                    'path'        : 'CASSINI',
                    'frame'       : frame,
                    'cadence'     : cadence,
                    'fov'         : fov,
                    'calibrations': [],
                }

                # Apply standard overrides. Parsed content from no file has no file
                # subfields, and `abspath` is given only by a read that retrieved the
                # cube.
                oops.Host._apply_overrides(params, filepath, abspath=abspath,
                                           **overrides)

                # Construct
                obs = obs_subclass(axes, **params)
                if arrays[k] is not None:
                    obs.insert_subfield('data', arrays[k])

                obs.insert_subfields(subfields)
                obs.insert_subfield('host', 'Cassini')
                obs.insert_subfield('instrument', 'VIMS')
                obs.insert_subfield('detector', detectors[k])
                obs.insert_subfield('sampling', sampling[k])

                kernels = _Cassini.used_kernels(obs.time, 'vims', ck=using_cks,
                                                return_all_planets=False)
                obs.insert_subfield('spice_kernels', kernels)
                obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
                obs.insert_subfield('spice_frame_name', frame_names[k])
                obs.insert_subfield('spice_frame_id', _FRAME_IDS[frame_names[k]])
                obs_pair[k] = obs

        return tuple(obs_pair)

    @staticmethod
    def _load_data_and_times(data_file, header_recs, label):
        """Load the data array from the file.

        If time backplanes are present, also return an array of times in seconds TDB as
        derived from these backplanes.

        Parameters:
            data_file (FCPath): The path to the cube.
            header_recs (int): The number of records that precede the cube's data.
            label (dict): The label, as returned by :meth:`_standard_label`.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray | None]: (data, times), where:

            * `data`: The data in axis order (line, sample, band).
            * `times`: The time sampling array in (line, sample) axis order, or None if
              no time backplane is found in the file.

        Raises:
            HostFormatError: If the byte order or data type of the core or the
                suffix is not recognized or if a line of core samples does not fill a
                whole number of suffix items.
        """

        # This procedure is absurdly complicated, but it has been carefully debugged
        # against real cubes; change it only with real cubes at hand.

        qube_dict = label['SPECTRAL_QUBE']

        # Extract key parameters from the file header
        core_items   = qube_dict['CORE_ITEMS']
        core_samples = core_items[0]
        core_bands   = core_items[1]
        core_lines   = core_items[2]
        core_item_bytes = qube_dict.get('CORE_ITEM_BYTES', 2)
        core_item_type  = qube_dict.get('CORE_ITEM_TYPE', 'SUN_INTEGER')

        sample_suffix_items = qube_dict['SUFFIX_ITEMS'][0]
        band_suffix_items   = qube_dict['SUFFIX_ITEMS'][1]

        suffix_item_bytes = 4

        if sample_suffix_items:
            suffix_item_type = qube_dict.get('SAMPLE_SUFFIX_ITEM_TYPE', 'SUN_INTEGER')
        else:
            suffix_item_type = qube_dict.get('BAND_SUFFIX_ITEM_TYPE', 'SUN_INTEGER')
        if isinstance(suffix_item_type, (list, tuple)):
            suffix_item_type = suffix_item_type[0]  # all back/sideplanes are same type

        record_bytes = label['RECORD_BYTES']
        header_bytes = record_bytes * header_recs

        # Make sure we have byte-aligned values
        if (core_samples * core_item_bytes) % suffix_item_bytes != 0:
            raise HostFormatError(f'misaligned array elements in {data_file}')

        # Determine the dtype and strides for the core item array
        band_stride = (core_samples * core_item_bytes +
                       sample_suffix_items * suffix_item_bytes)

        core_items_in_line = core_samples * core_bands
        suffix_items_in_line = ((core_samples + sample_suffix_items) *
                                (core_bands   + band_suffix_items)
                                - core_items_in_line)
        line_stride = (core_items_in_line * core_item_bytes +
                       suffix_items_in_line * suffix_item_bytes)

        # Locate the cube data in the file in units of core_item_bytes
        offset = header_bytes // core_item_bytes
        size = line_stride * core_lines

        # Determine the dtype for the file core
        if 'SUN_' in core_item_type or 'MSB_' in core_item_type:
            core_dtype = '>'
        elif 'PC_' in core_item_type or 'LSB_' in core_item_type:
            core_dtype = '<'
        else:
            raise HostFormatError(f'unrecognized CORE_ITEM_TYPE {core_item_type} in '
                                  f'{data_file}')

        if 'UNSIGNED' in core_item_type:
            core_dtype += 'u'
            native_dtype = 'int'
        elif 'INTEGER' in core_item_type:
            core_dtype += 'i'
            native_dtype = 'int'
        elif 'REAL' in core_item_type:
            core_dtype += 'f'
            native_dtype = 'float'
        else:
            raise HostFormatError(f'unrecognized CORE_ITEM_TYPE {core_item_type} in '
                                  f'{data_file}')

        core_dtype += str(core_item_bytes)

        # Read the file as core dtypes
        local_path = data_file.retrieve()
        array = np.fromfile(local_path, dtype=core_dtype)

        # Slice away the core lines, leaving off the line suffix
        array = array[offset:offset+size]

        # Create a data array using new strides in (line, sample, band) order
        data = as_strided(array, strides=(line_stride, core_item_bytes, band_stride),
                          shape=(core_lines, core_samples, core_bands))

        # Convert core to a native 3-D array
        data = data.astype(native_dtype)

        # If there are no time backplanes, we're done
        band_suffix_name = qube_dict['BAND_SUFFIX_NAME']
        if 'SLICE_TIME_SECONDS' not in band_suffix_name:
            return (data, None)

        # Determine the dtype for the file suffix
        if 'SUN_' in suffix_item_type or 'MSB_' in suffix_item_type:
            suffix_item_dtype = '>'
        elif 'PC_' in suffix_item_type or 'LSB_' in suffix_item_type:
            suffix_item_dtype = '<'
        else:
            raise HostFormatError(f'unrecognized suffix item type {suffix_item_type} in '
                                  f'{data_file}')

        if 'UNSIGNED' in suffix_item_type:
            suffix_item_dtype += 'u'
        elif 'INTEGER' in suffix_item_type:
            suffix_item_dtype += 'i'
        elif 'REAL'    in suffix_item_type:
            suffix_item_dtype += 'f'
        else:
            raise HostFormatError(f'unrecognized suffix item type {suffix_item_type} in '
                                  f'{data_file}')

        suffix_item_dtype += str(suffix_item_bytes)

        # The offset array skips over the first (bands,samples) to begin at the memory
        # location of the first band suffix backplane
        suffix_offset = core_bands * (core_samples * core_item_bytes
                                      + sample_suffix_items * suffix_item_bytes)
        offset_array = np.frombuffer(array.data, offset=suffix_offset,
                                     dtype=suffix_item_dtype)

        # Extract the band suffix array using new strides in (backplane, line, sample)
        # order
        backplane_stride = suffix_item_bytes * (core_samples + sample_suffix_items)
        backplane = as_strided(offset_array,
                               strides=(backplane_stride, line_stride, suffix_item_bytes),
                               shape=(band_suffix_items, core_lines, core_samples))

        # Convert to spacecraft clock
        seconds = backplane[band_suffix_name.index('SLICE_TIME_SECONDS')]
        ticks = backplane[band_suffix_name.index('SLICE_TIME_TICKS')]
        sclock = seconds + ticks/15959.

        # Convert to TDB
        mask = (seconds == -8192)       # Sometimes all are -8192 except first
        if np.any(mask):
            if not np.all(mask.ravel()[1:]):
                raise HostFormatError(f'invalid embedded times in {data_file}')
            times = np.empty(sclock.shape)
            times[...] = CassiniVIMS._tdb_from_clock_seconds(sclock[0,0], data_file)

        else:
            sclock_min = sclock.min()
            tdb_min = CassiniVIMS._tdb_from_clock_seconds(sclock_min, data_file)
            sclock_max = sclock.max()
            tdb_max = CassiniVIMS._tdb_from_clock_seconds(sclock_max, data_file)
            times = tdb_min + (sclock - sclock_min) * ((tdb_max - tdb_min) /
                                                       (sclock_max - sclock_min))

        return (data, times)

    @staticmethod
    def _tdb_from_clock_seconds(sclock, data_file):
        """The time in seconds TDB of a spacecraft clock reading in decimal seconds.

        The Cassini clock count that the clock kernel converts counts the whole seconds
        and then 1/256 s ticks after the point, so "1640775127.128" is half a second
        after "1640775127.000". A time backplane gives seconds and a decimal fraction
        instead, so the whole seconds are converted as a count and the fraction is added
        in seconds.

        Parameters:
            sclock (float): The clock reading in seconds, with a decimal fraction.
            data_file (FCPath): The cube, named in an error.

        Returns:
            float: The time in seconds TDB.

        Raises:
            OopsSpiceError: If the spacecraft clock kernel cannot convert the count.
        """

        whole = int(np.floor(sclock))
        return _tdb_from_sclk(f'{whole}.000', data_file) + (float(sclock) - whole)

    @staticmethod
    def meshgrid_and_times(obs, *, oversample=6, extend=1.5):
        """A Meshgrid and time array oversampling a VIMS field of view.

        The meshgrid oversamples and extends the dimensions of the field of view of a
        VIMS observation.

        Parameters:
            obs (Observation): The VIMS observation object for which to generate a
                meshgrid and a time array.
            oversample (float, optional): The factor by which to oversample the field of
                view, in units of the full-resolution VIMS pixel size.
            extend (float, optional): Pixels by which to extend the field of view, in
                units of the oversampled pixel.

        Returns:
            tuple[Meshgrid, Scalar]: (meshgrid, time), where:

            * `meshgrid`: The oversampled :class:`~oops.Meshgrid`.
            * `time`: The time in seconds TDB at each point of the meshgrid.

        Raises:
            WrongHostError: If `obs` is not a VIMS observation.
        """

        if obs.instrument != 'VIMS':
            raise WrongHostError(f'not a VIMS observation: {obs.instrument}')

        (ushrink,vshrink) = _SHRINKAGE[(obs.detector, obs.sampling)]

        oversample = float(oversample)
        undersample = (ushrink, vshrink)

        ustep = ushrink / oversample
        vstep = vshrink / oversample

        origin = (-extend * ustep, -extend * vstep)

        limit = (obs.fov.uv_shape.vals[0] + extend * ustep,
                 obs.fov.uv_shape.vals[1] + extend * vstep)

        meshgrid = oops.Meshgrid.for_fov(obs.fov, origin, undersample, oversample, limit,
                                         swap=True)

        time = obs.uvt(obs.fov.nearest_uv(meshgrid.uv).swapxy())[1]

        return (meshgrid, time)

    _OBS_CODE_CLEANUP = re.compile(r'(?:UR|)(.*?)(?:CAL|OCCA?B?)$')

    @staticmethod
    def _star_from_obs_id(obs_id):
        """The StarSource from a Cassini Observation ID; None otherwise.

        If the star is identified by a different name in the OBSERVATION ID, add this as
        an `alt_name`.
        """

        # Extract the code, strip "UR", "OCC", etc.
        obs_code = _cassini_obs_id_code(obs_id)
        match = CassiniVIMS._OBS_CODE_CLEANUP.match(obs_code)
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
        """True if the given parsed PDS3 label describes data from Cassini VIMS.

        An ISIS cube's label holds the identifying keywords inside its QUBE object rather
        than at the top level, so both places are checked.

        Parameters:
            label (Pds3Label | dict): A parsed PDS3 label.

        Returns:
            bool: True if `label` describes Cassini VIMS data.
        """

        for dict_ in (label, label.get('QUBE') or {}):
            if (dict_.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                    and dict_.get('INSTRUMENT_ID', '').startswith('VIMS')):
                return True
        return False

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 label of a VIMS cube or index, with its known defects
        repaired.

        ``oops.Host`` calls this before any parser reads a label of this host, the labels
        of the index tables included. A comment that opens on the line of a value and
        closes on the next line, as in 119 of 570 sampled cube labels, is joined into one
        line. A label without this defect is returned unchanged.

        Parameters:
            text (str): The text of the label, with <LF> line terminators.
            filepath (FCPath): The path to the label.

        Returns:
            str: The repaired text.
        """

        return re.sub(*_TWO_LINE_COMMENT, text)

    @staticmethod
    def _detect_in_index(label):
        """Whether the given index label describes Cassini VIMS data.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an index file.

        Returns:
            bool | None: True if the label's INSTRUMENT_HOST_ID is "CO" and its
            INSTRUMENT_ID is "VIMS", or if the label was read from a file whose name
            starts with "COVIMS", in any case; False if the name starts with "COISS" or
            "COUVIS"; otherwise None, meaning the host must be inferred from the index
            rows.
        """

        # The metadata index labels give these
        if (label.get('INSTRUMENT_HOST_ID', '') == 'CO'
                and label.get('INSTRUMENT_ID', '') == 'VIMS'):
            return True
        return _detect_index_by_path(label, 'COVIMS')

    @staticmethod
    def _detect_in_row(row_dict):
        """Whether the given row of an index file refers to Cassini VIMS data.

        Parameters:
            row_dict (dict): The content of one row of an index file.

        Returns:
            bool: True if the row's VOLUME_ID names a Cassini VIMS volume, such as
            "COVIMS_0040"; False otherwise.
        """

        return row_dict.get('VOLUME_ID', '').startswith('COVIMS_')

    ######################################################################################
    # Initialization
    ######################################################################################

    @staticmethod
    def initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                   gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the VIMS instrument.

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
                nine. Saturn, 6, must be among them.
            asof (str, optional): A date in ISO format. Each kernel is chosen from
                those released no later than the start (00:00) of that date, the leap
                seconds kernel included, and a query of which every kernel was released
                after the start of the date silently uses the kernel it would choose
                without a date. A set of SP or C kernels keeps only those released by
                the start of the date, so the later part of the mission can lack
                coverage. The C kernels never fall back; unless `ck` is 'none',
                ValueError is raised instead. Two loads ignore the date: the gap-filling
                C kernels, and the VIMS instrument kernel. None to choose the newest
                kernels.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.

        Raises:
            KeyError: If `planets` leaves out Saturn.
            RuntimeError: If `asof` is a date from 2003-11-05 through 2004-04-26,
                because the frames kernel of Saturn's small moons chosen for those dates
                has a malformed line that SPICE rejects.
            ValueError: If `ck` is not 'none' and none of the C kernels of the chosen
                set was released by the start of `asof`, as for any date up to and
                including 2003-11-04.
        """

        # Quick exit after first call
        if CassiniVIMS._initialized:
            return

        _Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                            mst_pck=mst_pck, irregulars=irregulars)
        _Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        CassiniVIMS._INSTRUMENT_KERNEL = _Cassini.spice_instrument_kernel('VIMS')[0]
        CassiniVIMS._initialized = True

    @staticmethod
    def _define_frames():
        """Register the SPICE frames of the VIS and IR channels and the IR solar port.

        Built lazily (and only once) so that observations using a custom frame never
        construct or depend on the SPICE frames.
        """

        # Each frame is tested against the Frame registry, because registering a second
        # frame under an existing ID renames it rather than failing.
        for frame_id in (_VIS_FRAME_NAME, _IR_FRAME_NAME, _IR_SOL_FRAME_NAME):
            if not oops.Frame.frame_id_exists(frame_id):
                oops.frame.SpiceFrame(frame_id, frame_id=frame_id)

    @staticmethod
    def _reset():
        """Reset the internal Cassini VIMS parameters.

        Can be useful for debugging.
        """

        CassiniVIMS._INSTRUMENT_KERNEL = None
        CassiniVIMS._initialized = False
        _Cassini.reset()


CassiniVIMS._register()

##########################################################################################
