##########################################################################################
# host_cassini/oops/vims.py
#
# Known shortcomings:
#
# For SAMPLING_MODE_ID == "UNDER" (aka Nyquist sampling), the FOV boundary will be
# be slightly off because an FOV object cannot handle overlapping, double-sized pixels.
# The proper boundary is 1/64th larger along the line direction to account for the larger
# pixel.
##########################################################################################

import numpy as np
from numpy.lib import stride_tricks

import cspyce
import julian

from filecache import FCPath
from pdsparser import Pds3Label

import oops
from ._cassini import _apply_select, _build_observation, _Cassini

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

# TARGET_NAME in a VIMS label -> the target body; "NONE" if no body was targeted
_TARGET_NAME_REPAIRS = {
    'DARK SKY': 'NONE',
    'OTHER'   : 'NONE',
    'SKY'     : 'NONE',
    'UNK'     : 'NONE',
}

# These are star targets and need to be defined as LightSources
_TARGET_STARS = {'FOMALHAUT', 'SPICA'}

# Names for the `select` option of from_file() -> index in the returned (vis, ir) pair
_CHANNELS = {'VIS': 0, 'IR': 1}

# The SPICE frames of the VIS channel, the IR channel, and the IR channel's solar port
_VIS_FRAME_ID    = 'CASSINI_VIMS_V'
_IR_FRAME_ID     = 'CASSINI_VIMS_IR'
_IR_SOL_FRAME_ID = 'CASSINI_VIMS_IR_SOL'

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

    NAME = 'Cassini VIMS'

    _INSTRUMENT_KERNEL = None
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, pds3_method='fast', select=None, astrometry=False,
                  target=None, lightsource=None, path=None, frame=None, fov=None,
                  calibrations=None, timeshift=None, navigation=None, parallel=None,
                  tracker=None, **kwargs):
        """A pair of Observations based on a Cassini VIMS cube or its PDS3 label.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label): The path to a VIMS cube
                in ISIS format, the path to its detached PDS3 label, or the parsed label.
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

            select (int | slice | str | tuple[int | slice | str, ...], optional): An
                index, a slice, or a tuple of indices and slices to apply to the returned
                (vis, ir) pair. A channel can also be chosen by name, "VIS" or "IR"
                (without regard to case). For example, `select=1` or `select='IR'`
                returns only the IR observation.
            astrometry (bool, optional): True to specify that the returned Observations
                will only be used for timing and astrometry. In this case, no data arrays
                are read and returned in the Observations.
            target (str | oops.Body, optional): Override for the target Body of both
                observations, which is otherwise derived from TARGET_NAME in the label.
                Use "NONE" for inertial pointing, indicating that no Solar System body
                was tracked.
            lightsource (oops.Lightsource, optional): Override for the Lightsource of both
                observations, which is otherwise the Sun.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of both channels.
            fov (oops.FOV, optional): Not supported, because the VIS and IR channels have
                different FOVs; must be None.
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
            tuple[Observation | None, Observation | None] | Observation | None: Without
            `select`, the pair (vis, ir), where:

            * `vis`: The VIS observation, or None if the VIS channel was inactive.
            * `ir`: The IR observation, or None if the IR channel was inactive.

            With `select`, the observation chosen by an index or name, or a tuple of the
            observations chosen by a slice or a tuple.

            Each observation has subfields `dict`, `host`, `instrument`, `detector`,
            `sampling`, `label_target`, `filepath` and `basename`, and `data` unless
            `astrometry` is True.

        Raises:
            ValueError: If `fov` is given, or if `select` names neither channel.
            IndexError: If an index in `select` is out of range.
            TypeError: If `select` has an invalid type.
            OopsValueError: If the cube has an unsupported shape.
        """

        # Check for unsupported options
        if fov is not None:
            raise ValueError('disallowed Cassini VIMS.from_file() option fov')

        CassiniVIMS._initialize()  # define everything the first time through; use
                                   # defaults unless _initialize() was called explicitly.

        fileinfo = CassiniVIMS._read_fileinfo(fileinfo, formats='P',
                                              astrometry=astrometry,
                                              pds3_method=pds3_method)
        filepath = FCPath(fileinfo.filepath)
        (label, data_file, header_recs) = CassiniVIMS._standard_label(fileinfo.dict,
                                                                      filepath)

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

        pair = CassiniVIMS._make_observations(label, filepath, data_file, header_recs,
                                              astrometry=astrometry, overrides=overrides)
        return _apply_select(pair, select, _CHANNELS)

    @staticmethod
    def _standard_label(dict_, filepath):
        """A VIMS label dictionary in a standard form, whether from ISIS or PDS3.

        Parameters:
            dict_ (dict): The parsed label of an ISIS cube or of a detached PDS3 label.
            filepath (FCPath): The path to the cube or to its detached PDS3 label.

        Returns:
            tuple[dict, FCPath, int]: (label, data_file, header_recs), where:

            * `label`: A copy of `dict_` that has the cube's description in both
              "QUBE" and "SPECTRAL_QUBE" and its non-structural keywords at the top
              level, so that ISIS and PDS3 labels can be read the same way.
            * `data_file`: The path to the cube.
            * `header_recs`: The number of records that precede the cube's data.
        """

        label = dict(dict_)

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

            return (label, filepath, label['^QUBE'] - 1)

        # Otherwise, convert PDS3 label info to a standard ISIS dictionary, inserting the
        # info from the .FMT files
        label['SPECTRAL_QUBE'] = {**label['SPECTRAL_QUBE'], **_CORE_DESCRIPTION_FMT,
                                  **_SUFFIX_DESCRIPTION_FMT}
        label.update(_BAND_BIN_CENTER_FMT)

        # ISIS labels use "QUBE" in place of "SPECTRAL_QUBE"
        label['QUBE'] = label['SPECTRAL_QUBE']

        return (label, filepath.with_name(label['^QUBE']), label['^QUBE_offset'] - 1)

    @staticmethod
    def _make_observations(label, filepath, data_file, header_recs, *, astrometry,
                           overrides):
        """The VIS and IR observations described by a standardized VIMS label.

        Parameters:
            label (dict): The label, as returned by :meth:`_standard_label`.
            filepath (FCPath): The path to the cube or to its detached PDS3 label.
            data_file (FCPath): The path to the cube.
            header_recs (int): The number of records that precede the cube's data.
            astrometry (bool): True to omit the data arrays.
            overrides (dict): The standard override options of :meth:`from_file`.

        Returns:
            tuple[Observation | None, Observation | None]: (vis, ir), as described for
            :meth:`from_file`.

        Raises:
            OopsValueError: If the cube has an unsupported shape.
        """

        qube_dict = label['SPECTRAL_QUBE']

        # Load any needed SPICE kernels
        tstart = julian.tdb_from_iso(label['START_TIME'])
        _Cassini.load_spks(tstart, tstart + 3600.)
        if overrides['frame'] is None:
            _Cassini.load_cks(tstart, tstart + 3600.)
            CassiniVIMS._define_frames()

        # Check power state of each channel: [0] is IR; [1] is VIS
        ir_is_off  = label['POWER_STATE_FLAG'][0] == 'OFF'
        vis_is_off = label['POWER_STATE_FLAG'][1] == 'OFF'

        ##################################################################################
        # Load the data arrays
        ##################################################################################

        (samples, bands, lines) = qube_dict['CORE_ITEMS']
        assert bands in (352, 256, 96)

        vis_data = None
        ir_data = None
        times = None

        if not astrometry:
            (array, times) = CassiniVIMS._load_data_and_times(data_file, header_recs,
                                                              label)
            assert array.shape == (lines, samples, bands), 'incorrect array shape'

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
        assert samples * lines == frames * frame_size

        # Replace multiple one-line frames by one frame, multiple lines
        if frames > 1 and frame_size != 1:
            assert swath_length == 1
            swath_length = frames
            lines = frames
            frame_size = swath_width * swath_length

        uv_shape = (swath_width, swath_length)

        x_offset = label['X_OFFSET']
        z_offset = label['Z_OFFSET']
        uv_los = (33. - x_offset, 33. - z_offset)

        vis_sampling = label['SAMPLING_MODE_ID'][1].strip()  # handle ' NORMAL'
        ir_sampling  = label['SAMPLING_MODE_ID'][0].strip()

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
        ### TBD: VIMS IR sampling mode UNDER is untested!!
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
        vis_texp_nonzero = max(vis_texp, 1.e-8) # avoids divide-by-zero in cadences

        interframe_delay = label['INTERFRAME_DELAY_DURATION'] * 0.001 * _TIME_FACTOR
        interline_delay  = label['INTERLINE_DELAY_DURATION']  * 0.001 * _TIME_FACTOR

        # Adjust the timing of one line, multiple frames
        if frames > 1 and frame_size != 1:
            interline_delay = interframe_delay

        length_stride = max(ir_texp * swath_width, vis_texp) + interline_delay

        backplane_cadence = None

        # Define a cadence based on the time backplane, if it is present
        if times is None:
            pass

        elif label['OVERWRITTEN_CHANNEL_FLAG'] == 'ON':
            times = times.ravel()
            assert times[0] < times[1]
            assert vis_is_off
            backplane_cadence = oops.cadence.Sequence(times, ir_texp)

        elif label['PACKING_FLAG'] == 'ON':
            times = times.ravel()
            assert times[0] == times[1]
            tstart = times[0]
            frame_cadence = oops.cadence.Sequence(times[::frame_size], texp=0.)

        else:       # No packing plus no embedded timing just means a better tstart
            times = times.ravel()
            assert times[0] == times[1]
            tstart = times[0]

        vis_header_cadence = oops.cadence.Metronome(tstart, length_stride,
                                                    vis_texp_nonzero, swath_length)
        ir_fast_cadence = oops.cadence.Metronome(tstart, ir_texp, ir_texp, swath_width)

        # At this point...
        #   vis_header_cadence  always defined, always 1-D.
        #   ir_fast_cadence     always defined, always 1-D.
        #   backplane_cadence   defined if timing was recorded, always 1-D.

        ##################################################################################
        # Define the coordinate frames
        ##################################################################################

        vis_frame_id = _VIS_FRAME_ID
        ir_frame_id  = _IR_FRAME_ID

        if (label['TARGET_NAME'] == 'SUN' or '_SOL' in label['OBSERVATION_ID']):
            ir_frame_id = _IR_SOL_FRAME_ID

        ##################################################################################
        # Construct the Observation objects
        ##################################################################################

        target_name = label['TARGET_NAME']
        label_target = _TARGET_NAME_REPAIRS.get(target_name, target_name)

        def build(obs_class, axes, cadence, fov, frame_id):
            params = {
                'cadence'     : cadence,
                'fov'         : fov,
                'path'        : 'CASSINI',
                'frame'       : frame_id,
                'target'      : label_target,
                'lightsource' : 'SUN',
                'calibrations': [],         # TBD
            }
            return _build_observation(obs_class, axes, params, filepath, overrides)

        vis_obs = None
        ir_obs  = None

        # POINT/OCCULTATION case
        if swath_width == 1 and swath_length == 1:
            assert vis_is_off
            if backplane_cadence is None:
                fast_stride = ir_texp   # + EXTRA_INTERSAMPLE_DELAY
                fastcad = oops.cadence.Metronome(tstart, fast_stride, ir_texp, samples)

                slow_stride = ir_texp * samples + interline_delay
                slowcad = oops.cadence.Metronome(tstart, slow_stride, slow_stride, lines)
                fullcad = oops.cadence.DualCadence(slowcad, fastcad)
                ir_cadence = oops.cadence.ReshapedCadence(fullcad, (samples*lines,))

            else:
                ir_cadence = backplane_cadence

            if ir_data is not None:
                ir_data = ir_data.reshape((frames, 256))

            ir_obs = build(oops.observation.Pixel, ('t','b'), ir_cadence, ir_fov,
                           ir_frame_id)

        # Single LINE case
        elif swath_length == 1:
            if not vis_is_off:
                if vis_data is not None:
                    vis_data = vis_data.reshape((samples, 96))

                vis_obs = build(oops.observation.Slit1D, ('u','b'),
                                oops.cadence.SnapCadence(tstart, vis_texp_nonzero),
                                vis_fov, vis_frame_id)

            if not ir_is_off:
                if ir_data is not None:
                    ir_data = ir_data.reshape((samples, 256))

                if backplane_cadence is not None:
                    ir_fast_cadence = backplane_cadence

                ir_obs = build(oops.observation.RasterSlit1D, ('ut','b'), ir_fast_cadence,
                               ir_fov, ir_frame_id)

        # Single 2-D IMAGE case
        elif samples == swath_width and lines == swath_length:
            if not vis_is_off:
                vis_obs = build(oops.observation.TimedImage, ('vt','u','b'),
                                vis_header_cadence, vis_fov, vis_frame_id)

            if not ir_is_off:
                if backplane_cadence is None:
                    ir_cadence = oops.cadence.DualCadence(vis_header_cadence,
                                                          ir_fast_cadence)
                else:
                    ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                                              (lines,samples))

                ir_obs = build(oops.observation.TimedImage, ('vslow','ufast','b'),
                               ir_cadence, ir_fov, ir_frame_id)

        # Multiple LINE case
        elif swath_length == 1 and swath_length == lines:
            if not vis_is_off:
                vis_obs = build(oops.observation.TimedImage, ('vt','u','b'),
                                frame_cadence, vis_fov, vis_frame_id)

            if not ir_is_off:
                if backplane_cadence is None:
                    ir_cadence = oops.cadence.DualCadence(frame_cadence, ir_fast_cadence)
                else:
                    ir_cadence = oops.cadence.ReshapedCadence(backplane_cadence,
                                                              (lines,samples))

                ir_obs = build(oops.observation.TimedImage, ('vslow','ufast','b'),
                               ir_cadence, ir_fov, ir_frame_id)

        else:
            raise oops.OopsValueError(f'unsupported VIMS format in file {filepath}')

        # Insert the subfields and data arrays
        for (obs, detector, sampling, data) in ((vis_obs, 'VIS', vis_sampling, vis_data),
                                                (ir_obs,  'IR',  ir_sampling,  ir_data)):
            if obs is None:
                continue

            obs.insert_subfield('dict', label)
            obs.insert_subfield('host', 'Cassini')
            obs.insert_subfield('instrument', 'VIMS')
            obs.insert_subfield('detector', detector)
            obs.insert_subfield('sampling', sampling)
            obs.insert_subfield('label_target', label['TARGET_NAME'])
            if data is not None:
                obs.insert_subfield('data', data)

        return (vis_obs, ir_obs)

    @staticmethod
    def _load_data_and_times(data_file, header_recs, label):
        """Load the data array from the file.

        If time backplanes are present, also return an array of times in seconds TDB as
        derived from these backplanes.

        This procedure is absurdly complicated but it has been rather carefully debugged.
        --MRS 7/4/12.

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
            OopsTypeError: If the byte order or data type of the core or the suffix is
                not recognized.
        """

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
        assert (core_samples * core_item_bytes) % suffix_item_bytes == 0, \
            'misaligned items'

        ##################################################################################

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
        elif 'PC_' in core_item_type or  'LSB_' in core_item_type:
            core_dtype = '<'
        else:
            raise oops.OopsTypeError('Unrecognized byte order: ' + core_item_type)

        if 'UNSIGNED' in core_item_type:
            core_dtype += 'u'
            native_dtype = 'int'
        elif 'INTEGER' in core_item_type:
            core_dtype += 'i'
            native_dtype = 'int'
        elif 'REAL'    in core_item_type:
            core_dtype += 'f'
            native_dtype = 'float'
        else:
            raise oops.OopsTypeError('Unrecognized core data type: ' + core_item_type)

        core_dtype += str(core_item_bytes)

        # Read the file as core dtypes
        local_path = data_file.retrieve()
        array = np.fromfile(local_path, dtype=core_dtype)

        # Slice away the core lines, leaving off the line suffix
        array = array[offset:offset+size]

        # Create a data array using new strides in (line, sample, band) order
        data = stride_tricks.as_strided(array,
                                        strides = (line_stride, core_item_bytes,
                                                   band_stride),
                                        shape   = (core_lines,  core_samples,
                                                   core_bands))

        # Convert core to a native 3-D array
        data = data.astype(native_dtype)

        # If there are no time backplanes, we're done
        band_suffix_name = qube_dict['BAND_SUFFIX_NAME']
        if 'SLICE_TIME_SECONDS' not in band_suffix_name:
            return (data, None)

        ##################################################################################

        # Determine the dtype for the file core
        if 'SUN_' in core_item_type or 'MSB_' in suffix_item_type:
            suffix_item_dtype = '>'
        elif 'PC_' in core_item_type or  'LSB_' in suffix_item_type:
            suffix_item_dtype = '<'
        else:
            raise oops.OopsTypeError('Unrecognized byte order: ' + suffix_item_type)

        if 'UNSIGNED' in suffix_item_type:
            suffix_item_dtype += 'u'
        elif 'INTEGER' in suffix_item_type:
            suffix_item_dtype += 'i'
        elif 'REAL'    in suffix_item_type:
            suffix_item_dtype += 'f'
        else:
            raise oops.OopsTypeError('Unrecognized suffix data type: ' + suffix_item_type)

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
        backplane = stride_tricks.as_strided(offset_array,
                                             strides = (backplane_stride, line_stride,
                                                        suffix_item_bytes),
                                             shape   = (band_suffix_items, core_lines,
                                                        core_samples))

        # Convert to spacecraft clock
        seconds = backplane[band_suffix_name.index('SLICE_TIME_SECONDS')]
        ticks = backplane[band_suffix_name.index('SLICE_TIME_TICKS')]
        sclock = seconds + ticks/15959.

        # Convert to TDB
        mask = (seconds == -8192)       # Sometimes all are -8192 except first
        if np.any(mask):
            assert np.all(mask.ravel()[1:])
            formatted = f'{sclock[0,0]:16.3f}'
            times = np.empty(sclock.shape)
            times[...] = cspyce.scs2e(-82, formatted)

        else:
            sclock_min = sclock.min()
            formatted = f'{sclock_min:16.3f}'
            tdb_min = cspyce.scs2e(-82, formatted) + (sclock_min - float(formatted))

            sclock_max = sclock.max()
            formatted = f'{sclock_max:16.3f}'
            tdb_max = cspyce.scs2e(-82, formatted) + (sclock_max - float(formatted))

            times = tdb_min + (sclock - sclock_min) * ((tdb_max - tdb_min) /
                                                       (sclock_max - sclock_min))

        return (data, times)

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
            ValueError: If `obs` is not a VIMS observation.
        """

        if obs.instrument != 'VIMS':
            raise ValueError(f'not a VIMS observation: {obs.instrument}')

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

    ######################################################################################
    # Initialization
    ######################################################################################

    @staticmethod
    def _initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                    gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the VIMS instrument.

        Must be called first. After the first call, later calls to this function are
        ignored.

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
        for frame_id in (_VIS_FRAME_ID, _IR_FRAME_ID, _IR_SOL_FRAME_ID):
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
