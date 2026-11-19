##########################################################################################
# host_cassini/oops/iss.py
##########################################################################################

import datetime
import re

import numpy as np
import cspyce
import julian

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarError, VicarImage, VicarLabel

import oops
from ._cassini import (_Cassini, _CMATRIX_ROTATION, _drop_unfetched_abspath,
                       _fix_cassini_ring_target, _index_file_path,
                       _pds3_options_with_defaults)

__all__ = ['CassiniISS']

# Create a master version of the NAC and WAC distortion models from
#   Owen Jr., W.M., 2003. Cassini ISS Geometric Calibration of April 2003.
#   JPL IOM 312.E-2003.
# These polynomials convert from X,Y (radians) to U,V (pixels).

_NAC_F = 2002.703    # mm
_NAC_E2 = 8.28e-6    # / mm2
_NAC_E5 = 5.45e-6    # / mm
_NAC_E6 = -19.67e-6  # / mm
_NAC_KX = 83.33333   # samples/mm
_NAC_KY = 83.3428    # lines/mm

_NAC_COEFF = np.zeros((4,4,2))
_NAC_COEFF[1,0,0] = _NAC_KX         * _NAC_F
_NAC_COEFF[3,0,0] = _NAC_KX*_NAC_E2 * _NAC_F**3
_NAC_COEFF[1,2,0] = _NAC_KX*_NAC_E2 * _NAC_F**3
_NAC_COEFF[1,1,0] = _NAC_KX*_NAC_E5 * _NAC_F**2
_NAC_COEFF[2,0,0] = _NAC_KX*_NAC_E6 * _NAC_F**2

_NAC_COEFF[0,1,1] = _NAC_KY         * _NAC_F
_NAC_COEFF[2,1,1] = _NAC_KY*_NAC_E2 * _NAC_F**3
_NAC_COEFF[0,3,1] = _NAC_KY*_NAC_E2 * _NAC_F**3
_NAC_COEFF[0,2,1] = _NAC_KY*_NAC_E5 * _NAC_F**2
_NAC_COEFF[1,1,1] = _NAC_KY*_NAC_E6 * _NAC_F**2

_WAC_F = 200.7761    # mm
_WAC_E2 = 60.89e-6   # / mm2
_WAC_E5 = 4.93e-6    # / mm
_WAC_E6 = -72.28e-6  # / mm
_WAC_KX = 83.33333   # samples/mm
_WAC_KY = 83.34114   # lines/mm

_WAC_COEFF = np.zeros((4,4,2))
_WAC_COEFF[1,0,0] = _WAC_KX         * _WAC_F
_WAC_COEFF[3,0,0] = _WAC_KX*_WAC_E2 * _WAC_F**3
_WAC_COEFF[1,2,0] = _WAC_KX*_WAC_E2 * _WAC_F**3
_WAC_COEFF[1,1,0] = _WAC_KX*_WAC_E5 * _WAC_F**2
_WAC_COEFF[2,0,0] = _WAC_KX*_WAC_E6 * _WAC_F**2

_WAC_COEFF[0,1,1] = _WAC_KY         * _WAC_F
_WAC_COEFF[2,1,1] = _WAC_KY*_WAC_E2 * _WAC_F**3
_WAC_COEFF[0,3,1] = _WAC_KY*_WAC_E2 * _WAC_F**3
_WAC_COEFF[0,2,1] = _WAC_KY*_WAC_E5 * _WAC_F**2
_WAC_COEFF[1,1,1] = _WAC_KY*_WAC_E6 * _WAC_F**2

# For testing the numerical inverse fitting
#    _WAC_COEFF = np.zeros((4,4,2))
#    _WAC_COEFF[0,1,0] = 1.5
#    _WAC_COEFF[1,0,0] = 0.5
#    _WAC_COEFF[0,1,1] = 0.75
#    _WAC_COEFF[1,0,1] = 2.0

_DISTORTION_COEFF_XY_TO_UV = {'NAC': _NAC_COEFF, 'WAC': _WAC_COEFF}

# Create a master version of the inverse distortion model.
# These coefficients were computed by numerically solving the above polynomials. These
# polynomials convert from U,V (pixels) to X,Y (radians). Maximum errors from applying the
# original distortion model and then inverting:
# NAC
#   X DIFF MIN MAX -7.01489382641e-10 1.15568299657e-10
#   Y DIFF MIN MAX -7.7440587623e-10 8.81658628916e-10
#   U DIFF MIN MAX -0.00138077186011 1.94695478513e-05
#   V DIFF MIN MAX -0.000474712833352 0.000563339044731
# WAC
#   X DIFF MIN MAX -3.29122915029e-07 5.42305755497e-08
#   Y DIFF MIN MAX -3.92109405705e-07 3.98905653939e-07
#   U DIFF MIN MAX -0.0535697887647 0.000916852346108
#   V DIFF MIN MAX -0.0182888189072 0.0187174796013

_NAC_INV_COEFF = np.zeros((4,4,2))
_NAC_INV_COEFF[:,:,0] = [
    [ -1.14799845e-10,  7.80494024e-14,  1.73312704e-15, -5.95242349e-19],
    [  5.99190197e-06, -3.91823615e-13, -7.12054264e-15,  0.00000000e+00],
    [  1.42455664e-12, -2.15242793e-18,  0.00000000e+00,  0.00000000e+00],
    [ -7.11199158e-15,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
_NAC_INV_COEFF[:,:,1] = [
    [ -4.86573092e-12,  5.99122009e-06, -3.91358768e-13, -7.12774967e-15],
    [  1.03832114e-13,  1.41728538e-12, -1.55778300e-18,  0.00000000e+00],
    [  2.03725690e-16, -7.11687723e-15,  0.00000000e+00,  0.00000000e+00],
    [  1.67086926e-21,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

_WAC_INV_COEFF = np.zeros((4,4,2))
_WAC_INV_COEFF[:,:,0] = [
    [ -5.42748411e-08,  5.06916433e-12,  8.16888725e-13, -3.85258837e-17],
    [  5.97679707e-05, -3.53472340e-12, -5.13400740e-13,  0.00000000e+00],
    [  5.66784001e-11, -1.27038656e-16,  0.00000000e+00,  0.00000000e+00],
    [ -5.09498280e-13,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
_WAC_INV_COEFF[:,:,1] = [
    [ -3.31749330e-10,  5.97618831e-05, -3.50726141e-12, -5.17011679e-13],
    [  6.70738287e-12,  5.34696724e-11, -8.85892191e-17,  0.00000000e+00],
    [  1.35389219e-14, -5.11900149e-13,  0.00000000e+00,  0.00000000e+00],
    [  7.39668979e-19,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

# For testing the numerical inverse fitting
#    _WAC_INV_COEFF[:,:,0] = [
#        [  1.14377216e-09,  5.71428571e-01, -5.17435487e-16,  1.39385624e-17],
#        [ -2.85714286e-01, -3.18326095e-14, -1.04402502e-17,  0.00000000e+00],
#        [ -6.02467295e-14, -1.36440914e-16,  0.00000000e+00,  0.00000000e+00],
#        [ -2.77233884e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
#    _WAC_INV_COEFF[:,:,1] = [
#        [ -2.86046684e-09, -1.90476190e-01,  1.72236899e-14, -3.66611741e-18],
#        [  7.61904762e-01,  4.33089979e-14,  1.25388484e-16,  0.00000000e+00],
#        [  7.96957986e-14,  1.53755115e-16,  0.00000000e+00,  0.00000000e+00],
#        [  4.82114213e-17,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

_DISTORTION_COEFF_UV_TO_XY = {'NAC': _NAC_INV_COEFF, 'WAC': _WAC_INV_COEFF}

# Target name as found in PDS label -> correct name
_TARGET_NAME_REPAIRS = {
    'ERRIAPO'  : 'ERRIAPUS',
    'K07S4'    : 'AEGAEON',
    'S8_2004'  : 'FORNJOT',
    'S12_2004' : 'S/2004_S_12',
    'S13_2004' : 'S/2004_S_13',
    'S14_2004' : 'HATI',
    'S18_2004' : 'BESTLA',
    'SKADI'    : 'SKATHI',
    'SUTTUNG'  : 'SUTTUNGR',
    'THRYM'    : 'THRYMR',
    'DARK SKY' : 'NONE',
    'SKY'      : 'NONE',
}

# These are star targets and need to be defined as LightSources
_TARGET_STARS = {'FOMALHAUT', 'SPICA'}

# The VICAR headers of the early cruise images, from 1999 to July 2000 (the first 1250
# images of volume COISS_1001), predate the PDS3 keywords. Their keywords are renamed to
# the PDS3 keywords that the archive's labels give the same values: VICAR name -> PDS3
# name.
_EARLY_VICAR_RENAMES = {
    'BIAS'                 : 'BIAS_STRIP_MEAN',
    'CALIB_LAMP_STATE_FLAG': 'CALIBRATION_LAMP_STATE_FLAG',
    'COMPRESSION_RATIO'    : 'INST_CMPRS_RATIO',
    'CONVERSION_TYPE'      : 'DATA_CONVERSION_TYPE',
    'ENCODING_TYPE'        : 'INST_CMPRS_TYPE',
    'EXTENDED_PIXEL_VALUE' : 'DARK_STRIP_MEAN',
    'SENSOR_HEAD_ELEC_TEMP': 'SENSOR_HEAD_ELEC_TEMPERATURE',
    'SHUTTER_MODE_ID'      : 'SHUTTER_STATE_ID',  # "ENABLED" or "DISABLED"
    'SHUTTER_EVENT_TYPE'   : 'SHUTTER_MODE_ID',   # "NACONLY", "WACONLY" or "BOTSIM"
}

# INSTRUMENT_ID -> INSTRUMENT_NAME, as the labels of the early cruise images spell it
_EARLY_VICAR_INSTRUMENT_NAMES = {
    'ISSNA': 'IMAGING SCIENCE SUBSYSTEM NARROW ANGLE',
    'ISSWA': 'IMAGING SCIENCE SUBSYSTEM WIDE ANGLE',
}

# The PDS3 null constants, compared without regard to case or surrounding blanks. A value
# that is one of these states nothing: the item is missing from that source.
_PDS3_NULLS = {'UNK', 'N/A', 'NULL'}

# The number that both sources write for a reading that was not made, which counts as
# missing like a null constant: the back optics temperature of a camera with no back
# sensor, the COMPRESSION_RATIO of the early cruise headers, and the temperatures of
# products whose headers were filled with predictions, such as the DETECTOR_TEMPERATURE
# of -999 degrees C, below absolute zero, in N1640662528_3_CALIB.LBL of
# calibrated/COISS_2059, whose DESCRIPTION says "Predicted values used to populate
# header" (#10).
_FILL_VALUE = -999.

# The fill values of the index tables, besides the PDS3 null constants and -999. The index
# tables in the metadata tree declare these as null constants, and the image labels give
# their items as UNK or N/A; the tables in the volumes' index directories write the same
# values but declare no null constants. The masks that the table reader builds from the
# declared constants are not used, because those labels also declare 0 the null constant
# of BIAS_STRIP_MEAN and DARK_STRIP_MEAN, which the image labels give as the value
# (38,006 rows). 1e32 and -1e32 are missing in every column (#28).
_INDEX_FILL_BOUND = 1.e32
_INDEX_FILL_VALUES = {
    'COMMAND_SEQUENCE_NUMBER'     : 65534,
    'ELECTRONICS_BIAS'            : 65534,
    'EXPECTED_PACKETS'            : 65534,
    'ORDER_NUMBER'                : 65534,
    'PARALLEL_CLOCK_VOLTAGE_INDEX': 65534,
    'RECEIVED_PACKETS'            : 65534,
    'SEQUENCE_NUMBER'             : 65534,
    'VALID_MAXIMUM'               : 65534,
    'INST_CMPRS_PARAM'            : -2147483648,
    'MISSING_LINES'               : -2147483648,
    # Written only for images whose rows have no target geometry (PHASE_ANGLE -1e32):
    # 8,464 rows each, and 10,558 for CENTER_LONGITUDE
    'EMISSION_ANGLE'              : 0.,
    'INCIDENCE_ANGLE'             : 0.,
    'CENTER_LATITUDE'             : 0.,
    'CENTER_LONGITUDE'            : 0.,
}

# The clock counts of the index rows, which the tables write with their trailing zeros
# dropped, such as "1473199010.12" for the label's "1473199010.120" (#27)
_INDEX_CLOCK_COUNTS = ('SPACECRAFT_CLOCK_START_COUNT', 'SPACECRAFT_CLOCK_STOP_COUNT')
_CLOCK_COUNT = re.compile(r'(\d+)(?:\.(\d{0,3}))?')

# The last of the early cruise images, the first 1250 images of COISS_1001, from
# W1294561143_1 (1999-01-09) to N1343357427_1 (2000-07-27), whose VICAR headers predate
# the PDS3 keywords. Their labels give START_TIME equal to STOP_TIME, and a start count
# that is the stop count less the exposure in milliseconds counted as seconds; the
# headers give no start count. The next image, N1347928997_3 (2000-09-18), gives both
# counts correctly (#10, #27).
_LAST_EARLY_CRUISE_IMAGE = 1343357427

# The NAIF ID of the Cassini spacecraft clock
_CASSINI_SCLK = -82

# Items without which no Snapshot can be built.
_REQUIRED_ITEMS = ('START_TIME', 'EXPOSURE_DURATION', 'INSTRUMENT_MODE_ID',
                   'INSTRUMENT_NAME')

# The leading number of electrons per DN in GAIN_MODE_ID -> gain_mode
_GAIN_MODES = (('215', 0), ('95', 1), ('29', 2), ('12', 3))

# Items copied unchanged into a subfield when present: item -> subfield
_OPTIONAL_SUBFIELDS = (
    ('TARGET_NAME'                 , 'label_target'),
    ('OBSERVATION_ID'              , 'observation_id'),
    ('SPACECRAFT_CLOCK_START_COUNT', 'spacecraft_clock_start_count'),
    ('SPACECRAFT_CLOCK_STOP_COUNT' , 'spacecraft_clock_stop_count'),
)

# The two sources of an image's metadata, the detached PDS3 label and the VICAR header of
# the image file, as the source of an item where they disagree.
_LABEL = 'label'
_HEADER = 'header'

# The source of each item on which the label and the header disagree in the archive. The
# source is fixed for the item, the same for every image, and chosen for being right
# where the two differ; the issue of label differences, #10, gives the counts and the
# images on which the chosen source is wrong. An item not listed takes the label, the
# archive's curated description of the product. A value that is missing from the chosen
# source, as a null constant or the fill value, is taken from the other.
_RESOLUTION_RULES = {
    # Label: every differing label value is a float32 printed to 6 decimal places, and the
    # header's is that float32 rounded to 6 significant digits (#10)
    'BIAS_STRIP_MEAN'             : _LABEL,
    'DARK_STRIP_MEAN'             : _LABEL,
    'DETECTOR_TEMPERATURE'        : _LABEL,
    'FILTER_TEMPERATURE'          : _LABEL,
    'INST_CMPRS_RATIO'            : _LABEL,
    'SENSOR_HEAD_ELEC_TEMPERATURE': _LABEL,
    # Label: the label is the float32 of the header's value, 182.783997 for 182.784 (#10)
    'INSTRUMENT_DATA_RATE'        : _LABEL,
    # Label: float32 values printed to 6 decimal places; the header's 6 significant digits
    # are better only for values below 0.1, in a handful of products, by at most 5e-7
    # (#10)
    'EXPECTED_MAXIMUM'            : _LABEL,
    'INST_CMPRS_RATE'             : _LABEL,
    # Label: the front and back temperatures as PDS3 defines the item. The cruise headers
    # give the front temperature alone, which their labels round to 5 digits (#10)
    'OPTICS_TEMPERATURE'          : _LABEL,
    # Label: the cruise headers carry an older data set name; the COISS_1001 catalog and
    # volume description give the label's (#10)
    'DATA_SET_ID'                 : _LABEL,
    # Label: the cruise headers say "CASSINI"; the catalog says "CASSINI-HUYGENS" (#10)
    'MISSION_NAME'                : _LABEL,
    # Label: the instrument catalog's spelling of the camera's name; either spelling gives
    # the camera (#10)
    'INSTRUMENT_NAME'             : _LABEL,
    # Label: a calibrated header keeps the raw product's creation time, while the label
    # gives the calibrated file's own (#10)
    'PRODUCT_CREATION_TIME'       : _LABEL,
    # Label: a calibrated header keeps the raw product's description, while the label
    # describes the calibrated product; the raw one stays in header_dict (#10)
    'DESCRIPTION'                 : _LABEL,
    # Label: the two differ by a millisecond, and nothing shows which is right (#10)
    'EARTH_RECEIVED_START_TIME'   : _LABEL,
    'EARTH_RECEIVED_STOP_TIME'    : _LABEL,
    # Header: every early cruise label says "SCIENCE CRUISE", which the mission catalog
    # dates 2002 to 2004, while each header names the phase of its 1999-2000 date in that
    # catalog; elsewhere the two differ only in the underscores the headers write (#10)
    'MISSION_PHASE_NAME'          : _HEADER,
    # Header: the labels replace ";" and "," with spaces and "@" with a line break (#10)
    'METHOD_DESC'                 : _HEADER,
}

# A time as the VICAR headers write it, ending in a "Z" that the PDS3 labels leave out
_HEADER_TIME = re.compile(r'\d{4}-(?:\d{3}|\d{2}-\d{2})T\d{2}:\d{2}:\d{2}(?:\.\d*)?Z')

# The order in which the PDS3 label parsers are tried, from the one requested onward
_PARSER_METHODS = ('fast', 'strict', 'loose')

# Repairs to the text of a PDS3 label, as (pattern, replacement):
#
# A quoted value that starts on the line after its keyword, as in N1569827183_4.LBL of
# COISS_2037, which the fast parser rejects or misreads, is moved up to that line.
_NEXT_LINE_VALUE = (r'(?m)^([A-Z][A-Z0-9_]* =)[ \t]*\r?\n[ \t]+"', r'\1 "')

# The DESCRIPTION value of the calibrated labels of calibrated/COISS_1xxx_v1, and of some
# of calibrated/COISS_2xxx_v1, opens a quote that it never closes, so no parser accepts
# them. The quote is closed at the end of the line before the next keyword (#11).
_UNTERMINATED_DESCRIPTION = (r'(?m)^(DESCRIPTION = "[^"]*?)(\r?\n)(?=[A-Z][A-Z0-9_]* = )',
                             r'\1"\2')

# The calibrated DESCRIPTION of some calibrated labels, as in N1640662528_3_CALIB.LBL of
# calibrated/COISS_2059, ends with a quote on a line of its own, and the raw label's
# description follows as a second quoted string on the next line, so no parser accepts
# them. The two strings are joined into one (#11).
_SPLIT_DESCRIPTION = (r'\r?\n"(\r?\n[ \t]+)"', r'\1')

_LABEL_REPAIRS = (_NEXT_LINE_VALUE, _UNTERMINATED_DESCRIPTION, _SPLIT_DESCRIPTION)

# A keyword name that a parsed PDS3 label can hold, including the suffixed keys that the
# parser adds, such as "START_TIME_fmt", and its "objects" list
_PDS3_NAME = re.compile(r'\^?[A-Za-z][A-Za-z0-9_:]*')


class CassiniISS(oops.Host):
    """The Cassini ISS host: Snapshot constructors for NAC and WAC images."""

    #: The name under which this host registers with :class:`~oops.host.host_.Host` and
    #: that appears in its error messages.
    NAME = 'Cassini ISS'

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, fast_distortion=True, return_all_planets=False,
                  pds3_options=None, astrometry=False, target=None, lightsource=None,
                  path=None, frame=None, cadence=None, fov=None, calibrations=None,
                  timeshift=None, navigation=None, tracker=None, parallel=None, **kwargs):
        """A Snapshot object based on a given Cassini ISS image file.

        What is read depends on the input. Given the VICAR image file, only that file is
        read: its header, and its array unless `astrometry` is True. Given the detached
        PDS3 label, the label is read and, unless `astrometry` is True, so is the whole
        image file that its ^IMAGE pointer names; with `astrometry`, the image file need
        not exist. The array is always the one that the image file's own VICAR header
        locates.

        The metadata comes from whichever of the label and the header were read. An item
        that a source gives as a PDS3 null value ("UNK", "N/A" or "NULL", in any case) or
        as the fill value -999 counts as missing from it, and an item missing from every
        source read is left out: its subfield is absent, never None. Where the label and
        the header give different values for an item, the item's fixed source supplies
        it, the same source for every image: the header for MISSION_PHASE_NAME, which
        the early cruise labels give as "SCIENCE CRUISE" while each header names the
        phase of the image's date, and for METHOD_DESC, whose semicolons, commas and "@"
        the labels replace; the label for every other item. A value missing from the
        fixed source is taken from the other. The header's values are first written in
        the label's forms: a time without its final "Z", IMAGE_NUMBER as text,
        IMAGE_OBSERVATION_TYPE as a list, a single OPTICS_TEMPERATURE as the pair of it
        and None, and MISSION_PHASE_NAME with spaces for underscores. Times, field of
        view, pointing, target and light source are computed from the merged metadata.

        If the observation's frame comes from SPICE, the C kernels must give the camera's
        pointing at its midtime, after any cadence override. Nothing is checked when a
        `frame` override is given.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The path to the VICAR image file or to its detached PDS3 label, or the
                parsed content of either as a Pds3Label, VicarLabel, or VicarImage, each
                read as the file it came from.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. A label
                that the requested parser, by default "fast", rejects, cannot convert, or
                misreads is parsed again by each later one of "fast", "strict" and
                "loose", and a label with a known archive defect is repaired in memory.
                `repairs` may be a list of (pattern, replacement) pairs or a single pair.
            astrometry (bool, optional):
                True to specify that the returned Observations will only be used for
                timing and astrometry. In this case, no data arrays are read and returned
                in the Observation.
            target (str | oops.Body, optional): Override for the default target Body of
                the observation. Use "NONE" for inertial pointing, indicating that no
                Solar System body was tracked.
            lightsource (oops.LightSource, optional): Override for the default LightSource
                of the observation.
            path (str | oops.Path, optional): Override for the Path of the observer.
            frame (str | oops.Frame, optional): Override for the Frame of the observing
                instrument, such as a :class:`~oops.frame.Cmatrix`. With it, the C
                kernels are neither loaded nor checked.
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
            **kwargs: Additional keyword arguments; they are accepted and ignored. This
                includes `select`.

        Returns:
            Snapshot: The observation, with these subfields inserted:

            * `host`, `instrument`, `detector`, `sampling`, `texp`, `spice_kernels`,
              `spice_to_frame`, `spice_frame_name`, `spice_frame_id`, `filepath`,
              `basename` and `image_url`, always.
            * `abspath`, unless `fileinfo` is a remote PDS3 label read with `astrometry`,
              whose image file is never retrieved.
            * `data`, unless `astrometry` is True.
            * `filter`, `filter1` and `filter2`; `gain_mode`; `label_target`;
              `observation_id`; and `spacecraft_clock_start_count` and
              `spacecraft_clock_stop_count`, each when the metadata gives its item.
              The clock counts are text, as the label spells them, such as
              "1459509269.122". For an early cruise image, one of the first 1250 of
              COISS_1001, whose label gives a start count that is the stop count less the
              exposure in milliseconds counted as seconds, and whose header gives none,
              the stop count is IMAGE_NUMBER with zero ticks, and the start count is
              computed from it and EXPOSURE_DURATION with the Cassini spacecraft clock
              kernel; `dict` holds the same counts.
            * `dict`: the merged metadata, keyed by the PDS3 names. It holds every item
              of each source read, except those whose value is missing, and in a list a
              missing element is None. The VICAR headers of the early cruise images,
              which use older names, are given the names and spellings of their labels,
              and their FILTER1_NAME and FILTER2_NAME appear only within FILTER_NAME.
              Every other keyword keeps its own name. The keys that the label parser
              derives from a keyword, such as "START_TIME_fmt", and its END are left
              out. An item has the same value and type from either input wherever the
              two sources agree: times are text, as the label writes them, and sets are
              lists.
            * `label_dict`: every keyword of the PDS3 label as its parser gives it, with
              times as the label's text and sets as lists; only the upper-case keywords,
              without END or the keys that the parser adds, such as "START_TIME_fmt",
              "^IMAGE_offset" and "objects"; when the label was read.
            * `header_dict`: every parameter of the VICAR header, keyed by its name, or,
              for one that occurs more than once, by its name and occurrence counted
              from zero, such as "TASK[1]"; when the header was read.

            `dict`, `label_dict` and `header_dict` can be written as JSON. `filepath`,
            `basename`, `abspath` and `image_url` name the image file, also when
            `fileinfo` is its PDS3 label.

        Raises:
            FileNotFoundError: If the file, or the image file of a PDS3 label read without
                `astrometry`, does not exist in the case spelled or in upper or lower
                case.
            HostFormatError: If START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID or
                INSTRUMENT_NAME is missing from every source read, or START_TIME or
                INSTRUMENT_MODE_ID is invalid; the message names the item and the image
                file.
            OopsSpiceError: If no frame override is given and the C kernels hold no
                pointing for the camera at the midtime of the observation. The message
                reads "no Cassini ISS NAC pointing at <time> for <image file>: <code>",
                ending with the SPICE error code, such as "SPICE(CKINSUFFDATA)"; the
                `spice_code` and `filepath` attributes hold the code and the image file.
            OSError: If the file cannot be read.
            PdsHostError: If the PDS3 label cannot be parsed, or its ^IMAGE pointer names
                no file; the message names the label.
            VicarHostError: If the VICAR header cannot be parsed, or the image file ends
                before its array; the message names the image file.
            WrongHostError: If the file, or the parsed content given, is not of a Cassini
                ISS image.
        """

        CassiniISS.initialize()  # define everything the first time through; use defaults
                                 # unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        fileinfo = CassiniISS._parse_fileinfo(fileinfo, astrometry=astrometry,
                                              pds3_options=pds3_options)

        label = None
        header = None
        data = None
        fetched = True
        if isinstance(fileinfo, Pds3Label):
            label = fileinfo
            if astrometry:
                # Only the label is read; the image file need not exist
                filepath = CassiniISS._image_path(label)
                fetched = filepath.is_local()
            else:
                # The image file is a VICAR file, and its own header, not the label's
                # ^IMAGE pointer, locates the array. The pointer is one record off in
                # every calibrated file, where it leaves out the binary header record,
                # and in the raw files of COISS_1006 whose VICAR label fills one record,
                # where it points one record past the first line (#6).
                (filepath, header) = CassiniISS._read_image_of_label(label)
                data = header.data_2d
        else:
            header = fileinfo
            if not astrometry:
                if not isinstance(header, VicarImage):
                    # A VicarImage built from a VicarLabel has no array; re-read the file
                    header = CassiniISS._read_vicar_image(FCPath(header.filepath))
                data = header.data_2d
            filepath = FCPath(fileinfo.filepath)

        label_dict = None if label is None else CassiniISS._label_dict(label)
        header_dict = None if header is None else CassiniISS._header_dict(header)
        dict_ = CassiniISS._early_cruise_counts(
            CassiniISS._merged_dict(label_dict, header_dict))

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

        snapshot = CassiniISS._make_snapshot(dict_, filepath=filepath, data=data,
                                             fast_distortion=fast_distortion,
                                             return_all_planets=return_all_planets,
                                             overrides=overrides, check_pointing=True)
        if label_dict is not None:
            snapshot.insert_subfield('label_dict', label_dict)
        if header_dict is not None:
            snapshot.insert_subfield('header_dict', header_dict)
        if not fetched:
            # The image file of a remote label was never retrieved, so the local cache
            # path that names it holds nothing (#13)
            snapshot.delete_subfield('abspath')
        return snapshot

    @staticmethod
    def from_index(filepath, *, supplement=oops.Host.SUPPLEMENT, label=None,
                   row_dicts=None, pds3_options=None, filter=None, fast_distortion=True,
                   return_all_planets=False, select=None, target=None, lightsource=None,
                   path=None, frame=None, cadence=None, fov=None, calibrations=None,
                   timeshift=None, navigation=None, tracker=None, parallel=None,
                   **kwargs):
        """A list of Snapshot objects, one for each row of a Cassini ISS index file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to a Cassini ISS index
                file or its PDS label.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of a supplemental index file, whose columns are merged
                into each row. Its rows must name the same files in the same order as the
                index; where both tables give a column, the index value is kept. The
                default, `oops.Host.SUPPLEMENT`, reads the file in the directory of
                `filepath` whose name matches except that "_index" is replaced by
                "_supplemental_index" in upper or lower case, if there is one; the
                archive has none for ISS. None reads no supplement.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (list[dict[str, Any]], optional): A list of dictionaries, one per
                row of the index. Include if `filepath` has already been read. If
                provided, `supplement` and `label` are ignored.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included in the returned list.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            select (None): Must be None. Each row describes a single observation, so
                there is nothing to select from; any other value raises an error.
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
                instrument. Note that this should only be used if you know in advance that
                only WAC or only NAC image observations will be returned.
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
            **kwargs: Any other keyword argument raises an error.

        Returns:
            list[Snapshot]: One observation per filtered row of the index, each with
            subfields `dict`, `host`, `instrument`, `detector`, `sampling`, `texp`,
            `spice_kernels`, `filepath`, `basename`, `image_url`, `spice_to_frame`,
            `spice_frame_name` and `spice_frame_id` inserted; `abspath` when the image
            file is local or has already been fetched; and `filter`, `filter1`,
            `filter2`, `gain_mode`, `label_target`, `observation_id`,
            `spacecraft_clock_start_count` and `spacecraft_clock_stop_count` where the
            row gives their columns. `filepath` and `image_url` are the full path or URL
            of the row's image file: its FILE_SPECIFICATION_NAME within its volume,
            whose root is the directory above an "index" directory holding the index,
            the volume named by VOLUME_ID in the "volumes" tree beside a "metadata" tree
            holding it, or otherwise the index's own directory.

            `dict` is the row in Python types (a list for an array column), with its
            text stripped of padding and without the table's "_mask" entries. The
            masks are not applied, because the metadata indexes declare real values as
            null constants (a BIAS_STRIP_MEAN or DARK_STRIP_MEAN of 0, which the image
            labels give). Instead these are missing: a PDS3 null constant, such as UNK
            or NULL; the fill values -999, 1e32 and -1e32; 65534 and -2147483648 in the
            columns whose labels give UNK or N/A for them; and an EMISSION_ANGLE,
            INCIDENCE_ANGLE, CENTER_LATITUDE or CENTER_LONGITUDE of 0, written for an
            image with no target geometry. A
            missing element of an array column is None. The clock counts are spelled
            as the labels spell them, with three digits after the point, where the
            tables drop trailing zeros. Those of an early cruise image are computed as
            :meth:`from_file` describes. The pointing is not checked, and every
            filtered row gives an observation.

        Raises:
            FileNotFoundError: If `supplement` names a file that does not exist.
            HostFormatError: If a row lacks START_TIME, EXPOSURE_DURATION,
                INSTRUMENT_MODE_ID or INSTRUMENT_NAME, or gives an invalid START_TIME or
                INSTRUMENT_MODE_ID.
            HostIndexError: If the supplement does not have the same rows as the index.
            OopsTypeError: If a keyword argument is not one of those listed.
            OopsValueError: If `select` is not None.
            PdsHostError: If the PDS3 label cannot be parsed.
            OSError: If `filepath` cannot be read.
            WrongHostError: If `filepath` is not a Cassini ISS index file.
        """

        # A mistaken keyword, such as "supplementary", must not be dropped silently (#20)
        if kwargs:
            raise oops.OopsTypeError(f'unexpected keyword argument for '
                                     f'CassiniISS.from_index: {next(iter(kwargs))!r}')
        if select is not None:
            raise oops.OopsValueError(f'each row of a Cassini ISS index is one '
                                      f'observation; select must be None, not '
                                      f'{select!r}')

        CassiniISS.initialize()

        # Read and filter the index table. ISS needs no supplement, so the default looks
        # for one without requiring it, while one that is named must exist.
        if row_dicts is None:
            require = supplement is not oops.Host.SUPPLEMENT
            row_dicts = CassiniISS._read_index_rows(filepath, supplement=supplement,
                                                    require_supplement=require,
                                                    pds3_options=pds3_options,
                                                    label=label)
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

        # Create the list of Snapshot objects
        snapshots = []
        for row_dict in row_dicts:
            fpath = _index_file_path(filepath, row_dict)
            items = CassiniISS._early_cruise_counts(CassiniISS._row_items(row_dict))
            obs = CassiniISS._make_snapshot(items, filepath=fpath, data=None,
                                            fast_distortion=fast_distortion,
                                            return_all_planets=return_all_planets,
                                            overrides=overrides, check_pointing=False)
            snapshots.append(_drop_unfetched_abspath(obs, fpath))

        return snapshots

    @staticmethod
    def _make_snapshot(dict_, filepath, data, *, fast_distortion, return_all_planets,
                       overrides, check_pointing):
        """A Snapshot built from the metadata of an ISS image or one row of an ISS index.

        Loads the SPK kernels covering the exposure. Unless `overrides` supplies a frame,
        it also loads the CK kernels and registers the camera frames.

        Parameters:
            dict_ (dict[str, Any]): The metadata, from which missing values have been
                removed. It must contain START_TIME, EXPOSURE_DURATION,
                INSTRUMENT_MODE_ID and INSTRUMENT_NAME; each other item that it lacks
                leaves out the subfields that would be taken from it.
            filepath (str | FCPath): The path to the image file; used for the `filepath`
                and `basename` subfields, and named in errors.
            data (np.ndarray | None): The image array, inserted as the `data` subfield;
                None to insert no `data`.
            fast_distortion (bool | None): True to use a pre-inverted polynomial; False to
                use a dynamically solved polynomial; None to use a FlatFOV.
            return_all_planets (bool): Report the kernels for all planets, not just
                Jupiter or Saturn, in the `spice_kernels` subfield.
            overrides (dict[str, Any]): The values of the standard overrides, `target`,
                `lightsource`, `path`, `frame`, `cadence`, `fov`, `calibrations`,
                `timeshift`, `navigation`, `tracker` and `parallel`, keyed by name, each
                None to keep the default.
            check_pointing (bool): True to require that the C kernels give the camera's
                pointing at the midtime of the observation, when no frame override is
                given.

        Returns:
            Snapshot: The observation, with the subfields listed in :meth:`from_file`.

        Raises:
            HostFormatError: If a required item is missing or invalid.
            OopsSpiceError: If `check_pointing` is True, no frame override is given, and
                the C kernels hold no pointing for the camera at the midtime of the
                observation.
        """

        CassiniISS._require_items(dict_, filepath)
        try:
            tstart = julian.tdb_from_iso(dict_['START_TIME'])
        except (TypeError, ValueError) as err:
            raise oops.host.HostFormatError(f'invalid START_TIME '
                                            f'{dict_["START_TIME"]!r} in {filepath}'
                                            ) from err
        texp   = dict_['EXPOSURE_DURATION'] / 1000.
        mode   = dict_['INSTRUMENT_MODE_ID']        # "FULL", "SUM2", or "SUM4"
        camera = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'
        if (camera, mode, fast_distortion) not in CassiniISS._FOVS:
            raise oops.host.HostFormatError(f'invalid INSTRUMENT_MODE_ID {mode!r} in '
                                            f'{filepath}')

        # Determine target and light source; with no target named, nothing is tracked
        label_target = dict_.get('TARGET_NAME', 'NONE')
        label_target = _TARGET_NAME_REPAIRS.get(label_target, label_target)
        if label_target in _TARGET_STARS:
            label_lightsource = oops.lightsource.star_lookup(label_target)
            label_target = 'NONE'
        else:
            label_target = _fix_cassini_ring_target(label_target, dict_, tstart)
            label_lightsource = 'SUN'

        # Make sure the SPICE kernels are loaded; construct the frame
        _Cassini.load_spks(tstart, tstart + texp)
        using_cks = overrides['frame'] is None
        if using_cks:
            _Cassini.load_cks(tstart, tstart + texp)
            CassiniISS._define_camera_frames()

        # Define the Snapshot parameters
        params = {
            'target'      : label_target,
            'lightsource' : label_lightsource,
            'path'        : 'CASSINI',
            'frame'       : 'CASSINI_ISS_' + camera,
            'cadence'     : oops.cadence.SnapCadence(tstart, texp),
            'fov'         : CassiniISS._FOVS[camera, mode, fast_distortion],
            'calibrations': [],         # TBD
        }

        # Apply standard overrides
        oops.Host._apply_overrides(params, filepath, **overrides)

        obs = oops.observation.Snapshot(('v','u'), **params)
        if using_cks and check_pointing:
            # Report missing pointing now, with its SPICE code, not when the frame is
            # first used; at the midtime of the observation, after any cadence override
            # (#15)
            CassiniISS._check_pointing(camera, obs.midtime, filepath)
        if data is not None:
            obs.insert_subfield('data', data)

        # Extra attributes
        obs.insert_subfields(CassiniISS._get_subfields(dict_))

        # With a custom frame, pointing never came from a CK, so any CK that happens to be
        # furnished (e.g. the gapfill CKs loaded unconditionally by _Cassini.initialize())
        # is unrelated to this observation and must not be reported as used.
        kernels = _Cassini.used_kernels(obs.time, 'iss',
                                        return_all_planets=return_all_planets,
                                        ck=using_cks)
        obs.insert_subfield('spice_kernels', kernels)
        obs.insert_subfield('spice_to_frame', _CMATRIX_ROTATION)
        obs.insert_subfield('spice_frame_name', 'CASSINI_ISS_' + camera)
        obs.insert_subfield('spice_frame_id', -82360 if camera == 'NAC' else -82361)

        return obs

    @staticmethod
    def _require_items(dict_, filepath):
        """Raise HostFormatError unless the metadata gives every required item.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without missing values.
            filepath (str | FCPath): The image file, named in the error.

        Raises:
            HostFormatError: If START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID or
                INSTRUMENT_NAME is missing; the message names the item and `filepath`.
        """

        for item in _REQUIRED_ITEMS:
            if item not in dict_:
                raise oops.host.HostFormatError(f'{item} is missing from {filepath}')

    @staticmethod
    def _check_pointing(camera, time, filepath):
        """Raise OopsSpiceError unless the C kernels give the camera's pointing at a time.

        Parameters:
            camera (str): "NAC" or "WAC".
            time (float): The time in seconds TDB.
            filepath (str | FCPath): The image file, named in the error.

        Raises:
            OopsSpiceError: If SPICE cannot evaluate the camera frame at `time`. It is
                built by ``OopsSpiceError.from_spice_error``, so its message
                reads "no Cassini ISS NAC pointing at <time> for <filepath>: <code>",
                with the SPICE error code, such as "SPICE(CKINSUFFDATA)", when SPICE gives
                one, and its cause is the exception that SPICE raised. An exception of
                any class other than those that cspyce raises propagates unchanged.
        """

        # The SPICE frame, which the C kernels define; the camera frame is fixed to it
        frame = oops.Frame.as_frame('CASSINI_ISS_' + camera + '_FLIPPED')
        try:
            frame.transform_at_time(time)
        except (LookupError, OSError, RuntimeError, ValueError) as err:
            raise oops.OopsSpiceError.from_spice_error(
                err, description=f'no Cassini ISS {camera} pointing', time=time,
                filepath=filepath) from err

    @staticmethod
    def _get_subfields(dict_):
        """The subfields taken from the metadata of an image.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without missing values.

        Returns:
            dict[str, Any]: The subfields `dict`, `host`, `instrument`, `detector`,
            `texp` and `sampling`, and each of `filter`, `filter1`, `filter2`,
            `gain_mode`, `label_target`, `observation_id`,
            `spacecraft_clock_start_count` and `spacecraft_clock_stop_count` for which
            `dict_` gives the items. The filter subfields need both filter names; a
            FILTER_NAME with a None element gives none of them.
        """

        params = {'dict': dict_}

        # Merge filter names, ignoring "CL1" and "CL2"; a FILTER_NAME with a missing
        # element names no filter pair
        filter1 = filter2 = None
        if 'FILTER_NAME' in dict_:
            filter1, filter2 = dict_['FILTER_NAME']
        elif 'FILTER1_NAME' in dict_ and 'FILTER2_NAME' in dict_:
            filter1 = dict_['FILTER1_NAME']
            filter2 = dict_['FILTER2_NAME']

        if filter1 is not None and filter2 is not None:
            if filter1[:2] == 'CL':
                if filter2[:2] == 'CL':
                    filter_ = 'CLEAR'
                else:
                    filter_ = filter2
            else:
                if filter2[:2] == 'CL':
                    filter_ = filter1
                else:
                    filter_ = '+'.join(sorted((filter1, filter2)))

            params['filter'] = filter_
            params['filter1'] = filter1
            params['filter2'] = filter2

        # gain_mode, from the leading number of electrons per DN
        gain = dict_.get('GAIN_MODE_ID', '')
        for (prefix, gain_mode) in _GAIN_MODES:
            if gain.startswith(prefix):
                params['gain_mode'] = gain_mode
                break

        # Other parameters
        params['host'] = 'Cassini'
        params['instrument'] = 'ISS'
        params['detector'] = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'
        params['texp'] = dict_['EXPOSURE_DURATION'] / 1000.
        params['sampling'] = dict_['INSTRUMENT_MODE_ID']  # "FULL", "SUM2", or "SUM4"
        for (item, name) in _OPTIONAL_SUBFIELDS:
            if item in dict_:
                params[name] = dict_[item]

        return params

    @staticmethod
    def _image_path(label):
        """The path to the image file that a detached PDS3 label describes.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an ISS image.

        Returns:
            FCPath: The file named by the label's ^IMAGE pointer, in the label's
            directory.

        Raises:
            PdsHostError: If the label's ^IMAGE pointer names no file.
        """

        pointer = label.get('^IMAGE', None)
        if not isinstance(pointer, str):
            raise oops.host.PdsHostError('^IMAGE names no image file in '
                                         f'{label.filepath}')
        return FCPath(label.filepath).parent / pointer

    @staticmethod
    def _read_image_of_label(label):
        """The image file that a detached PDS3 label describes, read in full.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an ISS image.

        Returns:
            tuple[FCPath, VicarImage]: The path to the image file and its content. The
            file is the one named by the label's ^IMAGE pointer, in the label's directory;
            if no file of that name exists, the same name in upper or lower case is used
            if that exists.

        Raises:
            FileNotFoundError: If no image file of any of these names exists.
            PdsHostError: If the label's ^IMAGE pointer names no file.
            VicarHostError: If the image file's VICAR header cannot be parsed.
        """

        # A copy of the archive may have renamed its files to the other case (#14)
        filepath = CassiniISS._image_path(label)
        alternatives = dict.fromkeys((filepath.name.upper(), filepath.name.lower()))
        alternatives.pop(filepath.name, None)
        try:
            return (filepath, CassiniISS._read_vicar_image(filepath))
        except FileNotFoundError as err:
            error = err

        for name in alternatives:
            path = filepath.with_name(name)
            try:
                return (path, CassiniISS._read_vicar_image(path))
            except FileNotFoundError:
                continue

        raise error

    @staticmethod
    def _read_vicar_image(filepath):
        """The content of a VICAR image file.

        Parameters:
            filepath (FCPath): The path to the image file.

        Returns:
            VicarImage: The image array and header.

        Raises:
            FileNotFoundError: If the file does not exist.
            VicarHostError: If the VICAR header cannot be parsed, or the file ends before
                the array that the header describes.
        """

        # Raised as the host's own error (#12); a file cut short makes the array's
        # reshape fail with a ValueError (#18)
        try:
            return VicarImage(filepath, strict=False)
        except (VicarError, ValueError) as err:
            raise oops.host.VicarHostError(f'{err} in {filepath}') from err

    @staticmethod
    def _parse_fileinfo(fileinfo, *, astrometry, pds3_options):
        """The parsed PDS3 label or VICAR header of an ISS image.

        A PDS3 label is parsed with the method in `pds3_options`, and if that parser
        rejects the label, fails to convert one of its values, or (for the fast parser)
        reads a keyword that is not a PDS3 name, by each later method in the order
        "fast", "strict", "loose"; the fast parser is followed by a second fast attempt
        with each quoted value that starts on the line after its keyword moved up to that
        line. A label that every one of these rejects is parsed once more with that
        repair, with the unterminated DESCRIPTION value of some calibrated labels closed,
        and with the split DESCRIPTION value of others joined, by the requested method
        and then by the last one. Any `repairs` in `pds3_options`, a list of (pattern,
        replacement) pairs or a single pair, are applied in every attempt. A method that
        is not one of the three is tried alone, and then with the repairs.

        Parsed content is not parsed again, but must describe a Cassini ISS image.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The path to the PDS3 label or the VICAR image file, or its parsed content.
            astrometry (bool): True to read only the header of a VICAR file.
            pds3_options (dict[str, Any]): The options for the PDS3 label parser,
                including its method.

        Returns:
            Pds3Label | VicarLabel | VicarImage: The parsed content.

        Raises:
            OSError: If the file cannot be read.
            PdsHostError: If the PDS3 label cannot be parsed; the message names the
                label.
            VicarHostError: If the VICAR header cannot be parsed, or the image file ends
                before its array; the message names the file.
            WrongHostError: If the file or the parsed content is not of a Cassini ISS
                image.
        """

        # Parsed content is checked as its file would be (#19)
        if isinstance(fileinfo, Pds3Label):
            if not CassiniISS._detect_in_pds3(fileinfo):
                raise oops.Host._no_host_error([CassiniISS], 'PDS3 label',
                                               fileinfo.filepath)
            return fileinfo
        if isinstance(fileinfo, (VicarLabel, VicarImage)):
            if not CassiniISS._detect_in_vicar(fileinfo):
                raise oops.Host._no_host_error([CassiniISS], 'VICAR file',
                                               fileinfo.filepath)
            return fileinfo

        if FCPath(fileinfo).suffix.lower() != '.lbl':
            # A VICAR image file, which the PDS3 parsers play no part in reading. A file
            # cut short makes the array's reshape fail with a plain ValueError (#18).
            try:
                return CassiniISS._read_fileinfo(fileinfo, formats='PV',
                                                 astrometry=astrometry,
                                                 pds3_options=pds3_options)
            except oops.host.WrongHostError:
                raise
            except ValueError as err:   # including VicarHostError, which names no file
                raise oops.host.VicarHostError(f'{err} in {fileinfo}') from err

        method = pds3_options['method']
        if method in _PARSER_METHODS:
            methods = _PARSER_METHODS[_PARSER_METHODS.index(method):]
        else:
            methods = (method,)
        attempts = []
        for method in methods:
            attempts.append((method, ()))
            if method == 'fast':
                attempts.append((method, (_NEXT_LINE_VALUE,)))
        attempts += [(method, _LABEL_REPAIRS)
                     for method in dict.fromkeys((methods[0], methods[-1]))]

        given_repairs = pds3_options.get('repairs', [])
        if (isinstance(given_repairs, tuple) and len(given_repairs) == 2
                and all(isinstance(item, str) for item in given_repairs)):
            given_repairs = [given_repairs]     # one (pattern, replacement) pair

        misread = None
        error = None
        for (method, repairs) in attempts:
            options = pds3_options | {'method': method,
                                      'repairs': list(given_repairs) + list(repairs)}
            try:
                parsed = CassiniISS._read_fileinfo(fileinfo, formats='PV',
                                                   astrometry=astrometry,
                                                   pds3_options=options)
            except (oops.host.PdsHostError, oops.OopsValueError) as err:
                # An OopsValueError is a value that this parser could not convert, such
                # as hour 25 in a time, which a later parser may still read (#17)
                error = err
                continue

            # The fast parser misreads a quoted value that starts on the next line and
            # holds "=", such as METHOD_DESC in N1579539166_1.LBL of COISS_2041, making
            # a keyword of its text (#11)
            if (method == 'fast' and isinstance(parsed, Pds3Label)
                    and not all(_PDS3_NAME.fullmatch(key) for key in parsed.as_dict())):
                misread = misread or parsed
                continue
            return parsed

        if misread is not None:
            return misread
        raise oops.host.PdsHostError(f'{error} in {fileinfo}') from error

    @staticmethod
    def _label_dict(label):
        """The content of a parsed PDS3 label, as the `label_dict` subfield gives it.

        This is every keyword of the label's ``as_dict()``, in label order, with each
        date or time as the text that the label gives it (the parser's "_fmt" form, such
        as "2005-016T13:01:59.864") and each set as the list of its members in label
        order, so that the dictionary can be written as JSON. END, any other key whose
        value is None, and the lower-case keys that the parser adds (see :meth:`_items`)
        are left out.

        Parameters:
            label (Pds3Label): The parsed PDS3 label.

        Returns:
            dict[str, Any]: The label's content.
        """

        # Consumers write the metadata out as JSON, which takes no datetime or set (#16);
        # the derived keys are dropped only after the times and sets are read from them
        return CassiniISS._items(CassiniISS._json_form(label.as_dict()))

    @staticmethod
    def _json_form(dict_):
        """A copy of a parsed PDS3 dictionary with its times as text and sets as lists.

        Parameters:
            dict_ (dict[str, Any]): A dictionary from ``Pds3Label.as_dict()``, or one of
                the OBJECT or GROUP dictionaries inside it.

        Returns:
            dict[str, Any]: The dictionary, with each date, time, or date-time value
            replaced by the text of its "_fmt" key, each set replaced by the list of its
            "_list" key, and each dictionary inside it converted the same way.
        """

        result = {}
        for (key, value) in dict_.items():
            if isinstance(value, dict):
                value = CassiniISS._json_form(value)
            elif isinstance(value, (datetime.date, datetime.time)):  # incl. datetime
                value = dict_[key + '_fmt']
            elif isinstance(value, (set, frozenset)):
                value = dict_[key + '_list']
            result[key] = value
        return result

    @staticmethod
    def _header_dict(header):
        """The content of a VICAR header, as the `header_dict` subfield gives it.

        Every parameter appears. One that occurs once is keyed by its name; each
        occurrence of one that repeats, such as TASK in the processing history, is keyed
        by its name and its occurrence counted from zero, as in "TASK[0]" and "TASK[1]".

        Parameters:
            header (VicarLabel | VicarImage): The parsed VICAR header.

        Returns:
            dict[str, Any]: The header's parameters, in header order.
        """

        # as_dict() keys each occurrence of a repeated parameter by a (name, n) tuple,
        # which JSON cannot take as a key; as_dict(unique=False) keeps only the first
        # (#16)
        return {(f'{key[0]}[{key[1]}]' if isinstance(key, tuple) else key): value
                for (key, value) in header.as_dict(unique=True).items()}

    @staticmethod
    def _merged_dict(label_dict, header_dict):
        """The metadata of an image, merged from its PDS3 label and its VICAR header.

        The keys that the label parser derives from a keyword (see :meth:`_items`) are
        left out of the label's keywords. The header's keywords are given their PDS3
        names and forms (see :meth:`_pds3_dict_from_vicar`). Missing values are removed
        from both (see :meth:`_without_nulls`). An item that only one source gives takes
        that source's value. An item that both give takes the value of its source in
        `_RESOLUTION_RULES`, the label for every item not listed there. The label's
        keywords come first, in label order, then the header's other keywords, in header
        order.

        Parameters:
            label_dict (dict[str, Any] | None): The `label_dict` of the PDS3 label; None
                if it was not read.
            header_dict (dict[str, Any] | None): The `header_dict` of the VICAR header;
                None if it was not read.

        Returns:
            dict[str, Any]: The merged metadata.
        """

        label = {}
        if label_dict is not None:
            label = CassiniISS._without_nulls(CassiniISS._items(label_dict))
        header = {}
        if header_dict is not None:
            header = CassiniISS._without_nulls(
                CassiniISS._pds3_dict_from_vicar(header_dict))

        merged = {}
        for (key, value) in label.items():
            if key in header and _RESOLUTION_RULES.get(key, _LABEL) == _HEADER:
                value = header[key]
            merged[key] = value
        for (key, value) in header.items():
            if key not in label:
                merged[key] = value
        return merged

    @staticmethod
    def _items(dict_):
        """A copy of a parsed PDS3 dictionary without the keys the parser adds.

        The PDS3 keywords are upper case. The keys that the parser adds are not: those it
        derives from a keyword, named by the keyword plus a lower-case suffix, such as
        "START_TIME_fmt", "DETECTOR_TEMPERATURE_unit" or "^IMAGE_offset", and its
        "objects" and "groups" lists. They state nothing of their own, and the VICAR
        header has none.

        Parameters:
            dict_ (dict[str, Any]): The content of a PDS3 label, or an OBJECT or GROUP
                dictionary inside it.

        Returns:
            dict[str, Any]: The upper-case keywords of `dict_`, without any whose value
            is None, such as END. Each dictionary inside is treated the same way.
        """

        result = {}
        for (key, value) in dict_.items():
            if key != key.upper() or value is None:
                continue
            result[key] = CassiniISS._items(value) if isinstance(value, dict) else value
        return result

    @staticmethod
    def _row_items(row_dict):
        """The metadata of one row of an index table, in Python types.

        Parameters:
            row_dict (dict[str, Any]): The row, as the index reader gives it: NumPy values
                and, for each column, a "_mask" entry.

        Returns:
            dict[str, Any]: The row's columns, each value converted to a Python value or,
            for an array column, a list, and text stripped of its padding. A fill value
            of the index tables (see `_INDEX_FILL_VALUES`) and every missing value in the
            sense of :meth:`_without_nulls` is removed; such an element of an array is
            None. The clock counts are given three digits after the point. The "_mask"
            entries are left out and the masks are not applied.
        """

        # The masks are not applied, because the metadata index labels declare real
        # values as null constants (see _INDEX_FILL_VALUES)
        items = {}
        for (key, value) in row_dict.items():
            if key.endswith('_mask') and key[:-5] in row_dict:
                continue
            value = np.asarray(value).tolist()
            if isinstance(value, list):
                value = [CassiniISS._index_value(key, item) for item in value]
            else:
                value = CassiniISS._index_value(key, value)
            items[key] = value

        # The tables drop a count's trailing zeros, which also changes the number of
        # clock ticks it reads as: "1473199010.12" for the label's "1473199010.120"
        for key in _INDEX_CLOCK_COUNTS:
            match = (_CLOCK_COUNT.fullmatch(items[key]) if isinstance(items.get(key), str)
                     else None)
            if match:
                items[key] = f'{match[1]}.{(match[2] or "").ljust(3, "0")}'

        return CassiniISS._without_nulls(items)

    @staticmethod
    def _index_value(key, value):
        """A value of an index row, or None if it is one of the tables' fill values.

        Parameters:
            key (str): The column name.
            value (Any): The value, or one element of an array column's value, as a
                Python value.

        Returns:
            Any: None for 1e32 or -1e32, or for the fill value of the column in
            `_INDEX_FILL_VALUES`; text stripped of its padding; any other value
            unchanged.
        """

        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and (abs(value) == _INDEX_FILL_BOUND
                                                or value == _INDEX_FILL_VALUES.get(key)):
            return None
        return value

    @staticmethod
    def _early_cruise_counts(dict_):
        """The metadata of an image with the clock counts of an early cruise image fixed.

        The labels of the early cruise images give a start count that is the stop count
        less the exposure in milliseconds counted as seconds, and their headers give no
        start count. The ISS index tables of COISS_1001 in the volumes' index directories
        give the image number as the start count, and as the stop count the image number
        less the exposure in centiseconds counted as seconds. For an image whose
        IMAGE_NUMBER is no later than that of the last early cruise image, the stop count
        is its IMAGE_NUMBER, the whole seconds of the clock at the shutter's close, with
        zero ticks, as the labels give it; the start count is the clock count that the
        Cassini spacecraft clock kernel gives for the time EXPOSURE_DURATION milliseconds
        earlier. Without an EXPOSURE_DURATION, no start count is given.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without missing values; modified in
                place.

        Returns:
            dict[str, Any]: `dict_`.
        """

        number = dict_.get('IMAGE_NUMBER')
        if (not isinstance(number, str) or not number.isdigit()
                or int(number) > _LAST_EARLY_CRUISE_IMAGE):
            return dict_

        # The clock kernel names the partition, "1/1294562649.000"; the labels do not
        stop = number + '.000'
        dict_['SPACECRAFT_CLOCK_STOP_COUNT'] = stop
        if 'EXPOSURE_DURATION' not in dict_:
            dict_.pop('SPACECRAFT_CLOCK_START_COUNT', None)
            return dict_

        tdb = cspyce.scs2e(_CASSINI_SCLK, stop) - dict_['EXPOSURE_DURATION'] / 1000.
        start = cspyce.sce2s(_CASSINI_SCLK, tdb)
        dict_['SPACECRAFT_CLOCK_START_COUNT'] = start.rpartition('/')[2]
        return dict_

    @staticmethod
    def _without_nulls(dict_):
        """A copy of a dictionary without its missing values.

        Parameters:
            dict_ (dict[str, Any]): The keywords of a label.

        Returns:
            dict[str, Any]: The keywords whose values are not missing in the sense of
            :meth:`_is_null`. In a list some of whose elements are missing, each of them
            is None: every list in an ISS label is a sequence whose positions carry
            meaning, such as the front and back temperatures of OPTICS_TEMPERATURE or the
            two filters of FILTER_NAME. Each dictionary inside is treated the same way.
        """

        result = {}
        for (key, value) in dict_.items():
            if CassiniISS._is_null(value):
                continue
            if isinstance(value, dict):
                value = CassiniISS._without_nulls(value)
            elif isinstance(value, list):
                value = [None if CassiniISS._is_null(item) else item for item in value]
            result[key] = value
        return result

    @staticmethod
    def _is_null(value):
        """True if a value states nothing.

        Parameters:
            value (Any): The value of a keyword.

        Returns:
            bool: True if `value` is None; a PDS3 null constant, "UNK", "N/A" or "NULL",
            in any case and with any surrounding blanks; the fill value -999; or a
            non-empty list, tuple or set of nothing but these.
        """

        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().upper() in _PDS3_NULLS
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return value == _FILL_VALUE
        if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 0:
            return all(CassiniISS._is_null(item) for item in value)
        return False

    @staticmethod
    def _pds3_dict_from_vicar(dict_):
        """The keywords of an ISS VICAR header under their PDS3 label names and forms.

        Every header's values are written in the forms of the PDS3 labels:

        * A time loses the "Z" that the headers write at its end.
        * IMAGE_NUMBER is text, as the labels quote it.
        * IMAGE_OBSERVATION_TYPE is a list, as the labels' set is given.
        * OPTICS_TEMPERATURE given as one number, the front temperature, is the pair of it
          and a missing (None) back temperature.
        * MISSION_PHASE_NAME has spaces where the headers write underscores.

        Most VICAR headers already use the keywords of the PDS3 label. Those of the early
        cruise images, which carry no INSTRUMENT_HOST_NAME, use older names. For these,
        each keyword is given the name and spelling that the image's PDS3 label gives the
        same value:

        * INSTRUMENT_HOST_NAME is "CASSINI ORBITER", and INSTRUMENT_NAME is the name of
          the camera that INSTRUMENT_ID identifies.
        * FILTER_NAME is the list of FILTER1_NAME and FILTER2_NAME, which are then left
          out.
        * GAIN_MODE_ID is spelled in full, e.g. "12 ELECTRONS PER DN" for "12 e/DN".
        * START_TIME and STOP_TIME are IMAGE_TIME.
        * SPACECRAFT_CLOCK_STOP_COUNT is IMAGE_NUMBER followed by ".000".
        * The VICAR SHUTTER_MODE_ID is SHUTTER_STATE_ID, and SHUTTER_EVENT_TYPE is
          SHUTTER_MODE_ID.
        * BIAS is BIAS_STRIP_MEAN, CALIB_LAMP_STATE_FLAG is CALIBRATION_LAMP_STATE_FLAG,
          COMPRESSION_RATIO is INST_CMPRS_RATIO, CONVERSION_TYPE is DATA_CONVERSION_TYPE,
          ENCODING_TYPE is INST_CMPRS_TYPE, EXTENDED_PIXEL_VALUE is DARK_STRIP_MEAN, and
          SENSOR_HEAD_ELEC_TEMP is SENSOR_HEAD_ELEC_TEMPERATURE.

        Each of these is made only when the header gives the keywords it comes from, and
        SPACECRAFT_CLOCK_STOP_COUNT only when IMAGE_NUMBER is not missing. Every other
        keyword keeps its VICAR name and value.

        Parameters:
            dict_ (dict[str, Any]): The keywords of the VICAR header.

        Returns:
            dict[str, Any]: A new dictionary of its keywords under their PDS3 names.
        """

        pds3 = {key: CassiniISS._pds3_form(key, value) for (key, value) in dict_.items()}
        if 'INSTRUMENT_HOST_NAME' in pds3:
            return pds3

        pds3 = {_EARLY_VICAR_RENAMES.get(key, key): value
                for (key, value) in pds3.items()}
        pds3['INSTRUMENT_HOST_NAME'] = 'CASSINI ORBITER'
        instrument_id = pds3.get('INSTRUMENT_ID')
        if instrument_id in _EARLY_VICAR_INSTRUMENT_NAMES:
            pds3['INSTRUMENT_NAME'] = _EARLY_VICAR_INSTRUMENT_NAMES[instrument_id]
        if 'FILTER1_NAME' in pds3 and 'FILTER2_NAME' in pds3:
            # FILTER_NAME holds the whole of each, so each appears under that name alone
            pds3['FILTER_NAME'] = [pds3.pop('FILTER1_NAME'), pds3.pop('FILTER2_NAME')]
        if isinstance(pds3.get('GAIN_MODE_ID'), str):
            pds3['GAIN_MODE_ID'] = pds3['GAIN_MODE_ID'].replace(' e/DN',
                                                                ' ELECTRONS PER DN')
        if 'IMAGE_TIME' in pds3:
            pds3['START_TIME'] = pds3['IMAGE_TIME']
            pds3['STOP_TIME'] = pds3['IMAGE_TIME']

        # The image number is the whole seconds of the clock count at the shutter's
        # close, which the labels of these images give with zero ticks
        if 'IMAGE_NUMBER' in pds3 and not CassiniISS._is_null(pds3['IMAGE_NUMBER']):
            pds3['SPACECRAFT_CLOCK_STOP_COUNT'] = f'{pds3["IMAGE_NUMBER"]}.000'
        return pds3

    @staticmethod
    def _pds3_form(key, value):
        """A VICAR header value written in the form that the PDS3 labels give its item.

        Parameters:
            key (str): The keyword.
            value (Any): The header's value.

        Returns:
            Any: The value as described in :meth:`_pds3_dict_from_vicar`; any other value
            unchanged.
        """

        # The label's forms, so that an item has one value and type from either input
        # (#16)
        if isinstance(value, str) and _HEADER_TIME.fullmatch(value):
            return value[:-1]
        if key == 'IMAGE_NUMBER' and isinstance(value, int):
            return str(value)
        if key == 'IMAGE_OBSERVATION_TYPE' and isinstance(value, str):
            return [value]
        if key == 'OPTICS_TEMPERATURE' and isinstance(value, float):
            return [value, None]
        if key == 'MISSION_PHASE_NAME' and isinstance(value, str):
            return value.replace('_', ' ')
        return value

    ######################################################################################
    # Detectors
    ######################################################################################

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes data from this host/instrument.
        """

        return (label.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and label.get('INSTRUMENT_ID', '').startswith('ISS'))

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes data from this host/instrument."""

        # Most VICAR headers use the PDS3 names; the early cruise headers have no
        # INSTRUMENT_HOST_NAME but give MISSION_NAME = "CASSINI"
        return (CassiniISS._detect_in_pds3(label)
                or (label.get('MISSION_NAME', '').startswith('CASSINI')
                    and label.get('INSTRUMENT_ID', '').startswith('ISS')))

    @staticmethod
    def _detect_in_index(label):
        """Whether the given index label describes Cassini ISS data.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an index file.

        Returns:
            bool | None: True if the label was read from a file whose path contains
            "COISS", in any case; otherwise None, meaning the host must be inferred from
            the index rows.
        """

        # The label itself records nothing that identifies the instrument, but the PDS
        # volume and RMS Node metadata names all start with COISS
        if label.filepath is not None and 'COISS' in label.filepath.as_posix().upper():
            return True
        return None

    @staticmethod
    def _detect_in_row(row_dict):
        """True if the given row of an index file label refers to this host/instrument."""

        return (row_dict.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and row_dict.get('INSTRUMENT_NAME', '').startswith('IMAGING SCIENCE'))

    ######################################################################################
    # Initialization
    ######################################################################################

    @staticmethod
    def initialize(*, ck='reconstructed', planets=None, asof=None, spk='reconstructed',
                   gapfill=True, mst_pck=True, irregulars=True):
        """Initialize key information about the ISS instrument.

        Optional. Call it before the first :meth:`~from_file` or :meth:`~from_index` to
        override the kernel defaults. Once it has run, explicitly or implicitly, later
        calls are ignored.

        Parameters:
            ck (str, optional): The set of C kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no C kernels automatically,
                leaving their handling to the caller. 'reconstructed' also loads the
                reconstructed C kernels of the Jupiter flyby.
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
            HostError: If the instrument kernel gives a camera's field of view
                in units other than degrees.
        """

        # Quick exit after first call
        if CassiniISS._initialized:
            return

        _Cassini.initialize(ck=ck, planets=planets, asof=asof, spk=spk, gapfill=gapfill,
                            mst_pck=mst_pck, irregulars=irregulars)
        _Cassini.load_instruments(asof=asof)

        # Load the instrument kernel
        CassiniISS._INSTRUMENT_KERNEL = _Cassini.spice_instrument_kernel('ISS')[0]

        # Construct a Polynomial FOV for each camera
        fovs = {}
        for detector in ['NAC', 'WAC']:
            info = CassiniISS._INSTRUMENT_KERNEL['INS']['CASSINI_ISS_' + detector]

            # Full field of view
            lines = info['PIXEL_LINES']
            samples = info['PIXEL_SAMPLES']

            xfov = info['FOV_REF_ANGLE']
            yfov = info['FOV_CROSS_ANGLE']
            units = info['FOV_ANGLE_UNITS']
            if units != 'DEGREES':
                raise oops.host.HostError(f'unsupported FOV_ANGLE_UNITS {units!r} for '
                                          f'CASSINI_ISS_{detector} in the ISS kernel')

            uscale = np.arctan(np.tan(xfov * oops.RPD) / (samples/2.))
            vscale = np.arctan(np.tan(yfov * oops.RPD) / (lines/2.))

            # Display directions: [u,v] = [right,down]
            full_fov = oops.fov.PolynomialFOV(
                                (samples,lines),
                                coefft_uv_from_xy=_DISTORTION_COEFF_XY_TO_UV[detector],
                                coefft_xy_from_uv=None)
            full_fov_fast = oops.fov.PolynomialFOV(
                                (samples,lines),
                                coefft_uv_from_xy=_DISTORTION_COEFF_XY_TO_UV[detector],
                                coefft_xy_from_uv=_DISTORTION_COEFF_UV_TO_XY[detector])
            full_fov_none = oops.fov.FlatFOV((uscale,vscale), (samples,lines))

            # Load the dictionary, include the subsampling modes
            fovs[detector, 'FULL', False] = full_fov
            fovs[detector, 'SUM2', False] = oops.fov.SubsampledFOV(full_fov, 2)
            fovs[detector, 'SUM4', False] = oops.fov.SubsampledFOV(full_fov, 4)
            fovs[detector, 'FULL', True] = full_fov_fast
            fovs[detector, 'SUM2', True] = oops.fov.SubsampledFOV(full_fov_fast, 2)
            fovs[detector, 'SUM4', True] = oops.fov.SubsampledFOV(full_fov_fast, 4)
            fovs[detector, 'FULL', None] = full_fov_none
            fovs[detector, 'SUM2', None] = oops.fov.SubsampledFOV(full_fov_none, 2)
            fovs[detector, 'SUM4', None] = oops.fov.SubsampledFOV(full_fov_none, 4)
            fovs[detector, 'FULL'] = full_fov_none
            fovs[detector, 'SUM2'] = oops.fov.SubsampledFOV(full_fov_none, 2)
            fovs[detector, 'SUM4'] = oops.fov.SubsampledFOV(full_fov_none, 4)

        CassiniISS._FOVS = fovs
        CassiniISS._initialized = True

    @staticmethod
    def _define_camera_frames():
        """Register the SPICE-derived CASSINI_ISS_NAC and CASSINI_ISS_WAC frames.

        :meth:`initialize` must have been called first. Built lazily (and only once) so
        that observations using a custom C-matrix never construct or depend on the SPICE
        camera frames.
        """

        # Each camera is tested individually against the Frame registry rather than
        # against a flag on this class. Registering a second frame under an ID that
        # already exists does not fail; it is renamed CASSINI_ISS__NAC_2 and never found
        # again, so the test has to ask the registry what it actually holds.
        for camera in ('NAC', 'WAC'):
            frame_id = 'CASSINI_ISS_' + camera
            if oops.Frame.frame_id_exists(frame_id):
                continue

            # The SpiceFrame takes the "flipped" ID, leaving the camera's own ID for the
            # Cmatrix that rotates it into the oops-frame convention
            spice_frame = oops.frame.SpiceFrame(frame_id, frame_id=frame_id + '_FLIPPED')
            oops.frame.Cmatrix(_CMATRIX_ROTATION, spice_frame, frame_id=frame_id)

    @staticmethod
    def _reset():
        """Reset the internal Cassini ISS parameters.

        Can be useful for debugging.
        """

        CassiniISS._INSTRUMENT_KERNEL = None
        CassiniISS._FOVS = {}
        CassiniISS._initialized = False
        _Cassini.reset()


CassiniISS._register()

##########################################################################################
