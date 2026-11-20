##########################################################################################
# host_cassini/oops/iss.py
##########################################################################################
"""The Cassini ISS host, which makes a Snapshot from each narrow or wide angle image.

The :doc:`user guide </user_guide/user_guide>` describes the products it reads, how it
merges the two labels of each product, and the errors it raises.
"""

import re

import numpy as np
import julian

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarError, VicarImage, VicarLabel

import oops
from ._cassini import (_Cassini, _CMATRIX_ROTATION, _detect_index_by_path,
                       _FILL_VALUE, _fix_cassini_ring_target, _is_null, _items,
                       _known_path, _label_dict, _pds3_options_with_defaults, _row_items,
                       _sclk_from_tdb, _tdb_from_sclk, _without_nulls)

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

# The keywords of the ISS labels and VICAR headers in which -999 is a fill value: the
# back optics temperature of a camera with no back sensor (54 of 400 labels sampled on
# 2026-11-19, and their headers), the COMPRESSION_RATIO (INST_CMPRS_RATIO) of the early
# cruise headers, and the temperatures and rates of products whose headers were filled
# with predictions, such as the DETECTOR_TEMPERATURE of -999 degrees C, below absolute
# zero, in N1640662528_3_CALIB.LBL of calibrated/COISS_2059, whose DESCRIPTION says
# "Predicted values used to populate header" (#10). None of them can be -999.
_FILL_KEYWORDS = frozenset({'DETECTOR_TEMPERATURE', 'FILTER_TEMPERATURE',
                            'INSTRUMENT_DATA_RATE', 'INST_CMPRS_RATE', 'INST_CMPRS_RATIO',
                            'OPTICS_TEMPERATURE', 'SENSOR_HEAD_ELEC_TEMPERATURE'})

# How the null constants that an index label declares for a column are read, decided
# once per table from its label (see CassiniISS._index_fill_rules). Besides these, the
# text that the archive writes for "no value", 1e32 and -1e32 are missing in every
# column (see _cassini._row_items), and so is -999 in the columns of _FILL_KEYWORDS,
# where it cannot be a real value whatever the label declares. Survey of the 127 ISS
# metadata index labels, 2026-11-19: 125 declare a null constant for 65 of their columns,
# and two, COISS_2115 and COISS_2116, declare none at all. The 125 declare -999 for every
# column of _FILL_KEYWORDS except INST_CMPRS_RATIO, for which they declare -1e32 while
# 128 rows give -999; until the index is corrected, that -999 is missing through
# _FILL_KEYWORDS, which a corrected table, writing its declared fill, does not need.
# SETI/rms-data-projects#253 item 16.
#
# The keys of a PDS3 column that declare a special constant, each of which the table
# reader treats as missing, as rms-pdstable reads them
_SPECIAL_CONSTANT_KEYS = ('INVALID_CONSTANT', 'MISSING_CONSTANT', 'UNKNOWN_CONSTANT',
                          'NOT_APPLICABLE_CONSTANT', 'NULL_CONSTANT', 'INVALID',
                          'MISSING', 'UNKNOWN', 'NOT_APPLICABLE', 'NULL')

# Until the index is corrected, a declared null constant of 0 is not applied: the 125
# labels declare 0 for BIAS_STRIP_MEAN and DARK_STRIP_MEAN, which the image labels give
# as the value (38,006 rows), and for EMISSION_ANGLE and INCIDENCE_ANGLE, where 0 is a
# possible angle (see _ZERO_WITHOUT_GEOMETRY). A column whose label declares 0 is read
# without its mask; one that declares only other constants, as a corrected label does,
# is read with it. SETI/rms-data-projects#253 item 5; #28.
_DECLARED_ZERO = 0

# Until the index is corrected, the fill values that the 125 labels declare are also
# missing in a column for which a label declares no null constant, as COISS_2115 and
# COISS_2116 declare none for any column; the image labels give these items as UNK or
# N/A. A column whose label declares a constant follows that declaration instead.
# SETI/rms-data-projects#253 item 15; #28.
_UNDECLARED_FILL_VALUES = {
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
}

# The angles that the index tables give as 0 in a row that has no target geometry, whose
# PHASE_ANGLE is -1e32: 8,464 rows each, and 10,558 for CENTER_LONGITUDE, in COISS_1999
# and COISS_2999; no row with target geometry gives any of them as 0 (survey of every ISS
# metadata index, 2026-11-19). Until the index is corrected, a 0 is missing in a row
# that has no PHASE_ANGLE; a corrected table gives the fill value there, and a real 0
# in a row with target geometry is kept. SETI/rms-data-projects#253 item 5.
_ZERO_WITHOUT_GEOMETRY = ('EMISSION_ANGLE', 'INCIDENCE_ANGLE', 'CENTER_LATITUDE',
                          'CENTER_LONGITUDE')

# The clock counts of the index rows, which the tables write with their trailing zeros
# dropped, such as "1473199010.12" for the label's "1473199010.120". Until the index is
# corrected, each count is given three digits after the point; a count that already has
# three, as a corrected table writes it, is unchanged. SETI/rms-data-projects#253 item 1;
# #27.
_INDEX_CLOCK_COUNTS = ('SPACECRAFT_CLOCK_START_COUNT', 'SPACECRAFT_CLOCK_STOP_COUNT')
_CLOCK_COUNT = re.compile(r'(\d+)(?:\.(\d{0,3}))?')

# The last of the early cruise images, the first 1250 images of COISS_1001, from
# W1294561143_1 (1999-01-09) to N1343357427_1 (2000-07-27), whose VICAR headers predate
# the PDS3 keywords. Their labels give START_TIME equal to STOP_TIME, and a start count
# that is the stop count less the exposure in milliseconds counted as seconds; the
# headers give no start count. The next image, N1347928997_3 (2000-09-18), gives both
# counts correctly (#10, #27).
_LAST_EARLY_CRUISE_IMAGE = 1343357427

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

# Repairs to the text of a PDS3 label, as (pattern, replacement), which
# CassiniISS._correct_label_text applies before any parser reads the label. The labels
# are in the holdings, so the repairs stay. Each pattern matches only text that no PDS3
# parser reads as the label means it, so a label without the defect is parsed from its
# file unchanged.
#
# A quoted value that starts on the line after its keyword, as in N1569827183_4.LBL of
# COISS_2037, which the fast parser rejects, or, if the value holds "=", misreads (#11).
# It is moved up to the keyword's line, where every parser reads it.
_NEXT_LINE_VALUE = (r'(?m)^([A-Z][A-Z0-9_]* =)[ \t]*\n[ \t]+"', r'\1 "')

# The DESCRIPTION value of the calibrated labels of calibrated/COISS_1xxx_v1, and of some
# of calibrated/COISS_2xxx_v1, opens a quote that it never closes, so no parser accepts
# them. The quote is closed at the end of the line before the next keyword (#11).
_UNTERMINATED_DESCRIPTION = (r'(?m)^(DESCRIPTION = "[^"]*?)(\n)(?=[A-Z][A-Z0-9_]* = )',
                             r'\1"\2')

# The calibrated DESCRIPTION of some calibrated labels, as in N1640662528_3_CALIB.LBL of
# calibrated/COISS_2059, ends with a quote on a line of its own, and the raw label's
# description follows as a second quoted string on the next line, so no parser accepts
# them. The two strings are joined into one (#11).
_SPLIT_DESCRIPTION = (r'\n"(\n[ \t]+)"', r'\1')

_LABEL_REPAIRS = (_NEXT_LINE_VALUE, _UNTERMINATED_DESCRIPTION, _SPLIT_DESCRIPTION)

# A keyword name that a parsed PDS3 label can hold, including the suffixed keys that the
# parser adds, such as "START_TIME_fmt", and its "objects" list
_PDS3_NAME = re.compile(r'\^?[A-Za-z][A-Za-z0-9_:]*')


class CassiniISS(oops.Host):
    """The Cassini ISS host: Snapshot constructors for NAC and WAC images."""

    #: The name under which this host registers with :class:`~oops.host.host_.Host` and
    #: that appears in its error messages.
    NAME = 'Cassini ISS'

    #: None of the 127 ISS metadata indexes in the holdings, the 10 of COISS_1xxx and
    #: the 117 of COISS_2xxx, has a supplemental index (survey of 2026-11-19), so the
    #: rows of a supplement that the caller gives are paired with the index rows by
    #: position.
    MERGE_SUPPLEMENT_BY_FILE = False

    #: A metadata index, named in the message that refuses a volume's own index.
    METADATA_INDEX_EXAMPLE = 'COISS_2001_index.lbl'

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, fast_distortion=True, return_all_planets=False,
                  pds3_options=None, select=None, astrometry=False, target=None,
                  lightsource=None, path=None, frame=None, cadence=None, fov=None,
                  calibrations=None, timeshift=None, navigation=None, tracker=None,
                  parallel=None, **kwargs):
        """A Snapshot object based on a given Cassini ISS image file.

        What is read depends on the input. Given the VICAR image file, only that file is
        read: its header, and its array unless `astrometry` is True. Given the detached
        PDS3 label, the label is read and, unless `astrometry` is True, so is the whole
        image file that its ^IMAGE pointer names; with `astrometry`, the image file need
        not exist. The array is always the one that the image file's own VICAR header
        locates.

        The metadata comes from whichever of the label and the header were read. An item
        that a source gives as text that the archive writes for no value ("UNK",
        "UNKNOWN", "N/A", "NULL", "---" or the empty string, in any case) or as the fill
        value -999 counts as missing from it, and an item missing from every
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
        pointing at the time at which the observation reads them: its midtime, after any
        cadence override, or the time that a `tracker` selects, moved by a time shift
        whose letters include "T". An "F" shift moves the camera frame, which is fixed to
        the SPICE frame, and not the time at which the C kernels are read. Nothing is
        checked when a `frame` override is given.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The path to the VICAR image file or to its detached PDS3 label, or the
                parsed content of either as a Pds3Label, VicarLabel, or VicarImage, each
                read as the file it came from. Parsed content that came from no file has
                no file subfields, and a full read of it that needs the image file
                raises an error.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. A label
                with a known archive defect is repaired in memory first. A label that the
                requested parser, by default "fast", rejects, cannot convert, or misreads
                is parsed again by each later one of "fast", "strict" and "loose".
                `repairs` may be a list of (pattern, replacement) pairs or a single pair,
                applied after the host's own repairs.
            select (None, optional): Must be None. An ISS image is a single observation,
                so there is nothing to select from; any other value raises an error.
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
            **kwargs: Any other keyword argument raises an error.

        Returns:
            Snapshot: The observation, with these subfields inserted:

            * `host`, `instrument`, `detector`, `sampling`, `texp`, `time_from_sclk`
              (always False: the time never comes from a clock count), `spice_kernels`,
              `spice_to_frame`, `spice_frame_name` and `spice_frame_id`, always.
            * `filepath`, `basename` and `image_url`, unless `fileinfo` is parsed content
              that came from no file.
            * `abspath`, when the image file is local or already in the local cache. A
              remote PDS3 label read with `astrometry` does not retrieve its image
              file, so the image has an `abspath` only if an earlier read cached it.
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
            FileNotFoundError: If the file does not exist, or if the image file of a PDS3
                label read without `astrometry` exists neither in the case that the
                label spells nor in upper or lower case.
            HostFormatError: If START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID or
                INSTRUMENT_NAME is missing from every source read, or START_TIME or
                INSTRUMENT_MODE_ID is invalid; the message names the item and the image
                file. Also if a full read needs the image file of parsed content that
                came from no file.
            OopsSpiceError: If no frame override is given and the C kernels hold no
                pointing for the camera at the midtime of the observation. The message
                reads "no Cassini ISS NAC pointing at <time> for <image file>: <code>",
                ending with the SPICE error code, such as "SPICE(CKINSUFFDATA)"; the
                `spice_code` and `filepath` attributes hold the code and the image file.
                Also if the spacecraft clock kernel cannot convert the clock counts of an
                early cruise image.
            OopsTypeError: If a keyword argument is not one of those listed.
            OopsValueError: If `select` is not None.
            OSError: If the file cannot be read.
            PdsHostError: If the PDS3 label cannot be parsed, or its ^IMAGE pointer names
                no file; the message names the label and, for a label that cannot be
                parsed, gives the error of each parser method tried.
            VicarHostError: If the VICAR header cannot be parsed, or the image file ends
                before its array; the message names the image file.
            WrongHostError: If the file, or the parsed content given, is not of a Cassini
                ISS image.
        """

        # A mistaken keyword, such as "astrometery", must not be dropped silently, and
        # an argument error is raised before anything is read, as in from_index (#20)
        CassiniISS._check_keywords('from_file', kwargs)
        oops.Host._check_select(select, unit='Cassini ISS image')

        CassiniISS.initialize()  # define everything the first time through; use defaults
                                 # unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        fileinfo = CassiniISS._parse_fileinfo(fileinfo, astrometry=astrometry,
                                              pds3_options=pds3_options)

        label = None
        header = None
        data = None
        if isinstance(fileinfo, Pds3Label):
            label = fileinfo
            if astrometry:
                # Only the label is read; the image file need not exist
                filepath = (None if _known_path(label.filepath) is None
                            else CassiniISS._image_path(label))
            else:
                # The image file is a VICAR file, and its own header, not the label's
                # ^IMAGE pointer, locates the array. The pointer is one record off in
                # every calibrated file, where it leaves out the binary header record,
                # and in the raw files of COISS_1006 whose VICAR label fills one record,
                # where it points one record past the first line (#6).
                (filepath, header, image) = CassiniISS._read_image_of_label(label)
                data = image.data_2d
        else:
            filepath = _known_path(fileinfo.filepath)
            header = fileinfo
            if isinstance(fileinfo, VicarImage) and filepath is not None:
                # rms-vicar writes the reading machine into HOST and BHOST when it reads
                # the array, so the header is read again as the file writes it
                header = CassiniISS._read_vicar_label(filepath)
            if not astrometry:
                if isinstance(fileinfo, VicarImage):
                    image = fileinfo
                elif filepath is None:
                    raise oops.host.HostFormatError('no file path is known for this '
                                                    'Cassini ISS VICAR header, so its '
                                                    'image array cannot be read')
                else:
                    # A VicarLabel holds no array; the file is read in full
                    image = CassiniISS._read_vicar_image(filepath)
                data = image.data_2d

        label_dict = None if label is None else _label_dict(label.as_dict())
        header_dict = None if header is None else CassiniISS._header_dict(header)
        dict_ = CassiniISS._merged_dict(label_dict, header_dict)

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
        return snapshot

    @staticmethod
    def from_index(filepath, *, supplement=oops.Host.SUPPLEMENT, label=None,
                   row_dicts=None, pds3_options=None, filter=None, fast_distortion=True,
                   return_all_planets=False, select=None, target=None, lightsource=None,
                   path=None, frame=None, cadence=None, fov=None, calibrations=None,
                   timeshift=None, navigation=None, tracker=None, parallel=None,
                   **kwargs):
        """An iterator over Snapshots, one for each row of a Cassini ISS index file.

        When this method is called, it checks the arguments and reads the labels of the
        index and of any supplement. The rows are read as the iteration proceeds, a few
        at a time, through ``oops.Host._index_rows``, and each row's Snapshot is built
        when the iteration reaches that row, so an error in a row is raised at that row,
        after every earlier row has been yielded.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to a Cassini ISS index
                file or its PDS label.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of a supplemental index file, whose columns are merged
                into each row. Its rows must list the same files as the index's, in the
                same order, by VOLUME_ID and FILE_SPECIFICATION_NAME where both tables
                have those columns. Where both tables give a column, the index value is
                kept. The default, `oops.Host.SUPPLEMENT`, reads the file in the
                directory of `filepath` whose name matches except that "_index" is
                replaced by "_supplemental_index" in upper or lower case, if there is
                one; the archive has none for ISS. None reads the index alone.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, one dictionary per row, if `filepath` has already been read;
                each is filtered and built when the iteration reaches it. If provided,
                `supplement` is ignored, and the null constants that the index declares
                are read from `label` or, if it is not given, from the label of
                `filepath`, as for rows read from the table.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            filter (dict[str, Any], optional): A dictionary of index column names and
                their values or a set or tuple of values. If provided, only rows of the
                index in which each named column equals the value, falls in the set of
                values, or falls in a range defined by a tuple of two values, will be
                included.
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
            collections.abc.Iterator[Snapshot]: One observation per filtered row of the
            index, in index order, each with subfields `dict`, `host`, `instrument`,
            `detector`, `sampling`, `texp`, `spice_kernels`, `filepath`, `basename`,
            `image_url`, `spice_to_frame`, `spice_frame_name` and `spice_frame_id`
            inserted; `abspath` when the image file is local or has already been fetched;
            and `filter`, `filter1`, `filter2`, `gain_mode`, `label_target`,
            `observation_id`, `spacecraft_clock_start_count` and
            `spacecraft_clock_stop_count` where the row gives their columns; and
            `time_from_sclk`, always False. `filepath` and `image_url` are the full path
            or URL of the row's image file: its FILE_SPECIFICATION_NAME within the
            volume named by VOLUME_ID, which is in the "volumes" tree beside a "metadata"
            tree holding the index, or otherwise in the index's own directory.

            `dict` is the row in Python types (a list for an array column), with its
            text stripped of padding and without the table's "_mask" entries. These are
            missing: a value that the index label declares a null constant of its
            column, unless that constant is 0, which the metadata index labels declare
            for real values (a BIAS_STRIP_MEAN or DARK_STRIP_MEAN of 0, which the image
            labels give) and for possible angles; text that PDS3 writes for no value,
            such as UNK or NULL; the fill values 1e32 and -1e32; -999 in the
            temperature, data rate and compression columns, where it cannot be a real
            value; 65534 and -2147483648, the null constants that the other labels
            declare, in the columns listed for them where the index label does not
            declare a null constant; and an EMISSION_ANGLE, INCIDENCE_ANGLE,
            CENTER_LATITUDE or CENTER_LONGITUDE of 0 in a row that has no target
            geometry (no PHASE_ANGLE), where the tables write 0 for a missing angle. A
            missing element of an array column is None. The clock counts are spelled as
            the labels spell them, with three digits after the point, where the tables
            drop trailing zeros. Those of an early cruise image are computed as
            :meth:`from_file` describes.

            The pointing is not checked, and every filtered row gives an observation.
            Where the C kernels cannot supply the camera's pointing at the time at
            which an observation evaluates its frame, that frame raises
            :exc:`oops.OopsSpiceError` when it is used.

        Raises:
            FileNotFoundError: If the label of `filepath` does not exist and `label` is
                not given, also for rows given in `row_dicts`; if `supplement` gives a
                file whose label does not exist; when the iteration starts, if a table
                file does not exist.
            HostFormatError: If a row lacks START_TIME, EXPOSURE_DURATION,
                INSTRUMENT_MODE_ID or INSTRUMENT_NAME, or gives an invalid START_TIME or
                INSTRUMENT_MODE_ID; raised when the iteration reaches that row.
            HostIndexError: When called, if `filepath` is a volume's own index, in its
                "index" directory, rather than a metadata index, or if the label of the
                index or the supplement does not describe a table that can be read.
                During the iteration: at the row concerned, if the supplement does not
                list the same files as the index in the same order; after every row that
                a table holds, if it holds fewer rows than its label says; at the first
                of the ten rows read with it, if a row is longer than the label's row
                length; and after the last row, if a table holds more rows than its
                label says. A value that cannot be read as its column's number gives a
                warning, and the row is read with it as text, or missing where the
                column's declared null constants apply.
            OopsSpiceError: If the spacecraft clock kernel cannot convert the clock counts
                of an early cruise image; raised when the iteration reaches that row.
            OopsTypeError: If a keyword argument is not one of those listed; when the
                iteration reaches a row whose value cannot be compared with a range in
                `filter`.
            OopsValueError: If `select` is not None, or if `filter` uses a column that the
                index does not have; for `row_dicts` that is not a sequence, the latter
                is raised when the iteration reads the first row.
            PdsHostError: If the PDS3 label of the index or the supplement cannot be
                parsed.
            OSError: If `filepath` cannot be read.
            WrongHostError: If `filepath` is not a Cassini ISS index file.
        """

        # A mistaken keyword, such as "supplementary", must not be dropped silently (#20)
        CassiniISS._check_keywords('from_index', kwargs)
        oops.Host._check_select(select, unit='row of a Cassini ISS index')

        # The labels are read now and the rows as the iteration reaches them. The index
        # label, parsed once, also tells how its declared null constants are read, for
        # rows given in row_dicts too, which oops.Host.from_index passes without the
        # label. The archive has no ISS supplement, so the default looks for one without
        # requiring it, while one that is given must exist.
        label = CassiniISS._read_index_label(filepath, label=label,
                                             pds3_options=pds3_options)
        (masked, fills) = CassiniISS._index_fill_rules(label)
        rows = CassiniISS._index_rows(filepath, supplement=supplement,
                                      require_supplement=(supplement is not
                                                          oops.Host.SUPPLEMENT),
                                      label=label, row_dicts=row_dicts,
                                      pds3_options=pds3_options, filter=filter)
        volumes = oops.Host._volumes_directory(filepath)

        CassiniISS.initialize()

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

        # Build each row's Snapshot when the iteration reaches it
        def snapshots():
            for row_dict in rows:
                fpath = oops.Host._index_file_path(volumes, row_dict)
                obs = CassiniISS._make_snapshot(CassiniISS._row_items(row_dict, masked,
                                                                      fills),
                                                filepath=fpath, data=None,
                                                fast_distortion=fast_distortion,
                                                return_all_planets=return_all_planets,
                                                overrides=overrides,
                                                check_pointing=False)
                yield obs

        return snapshots()

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
        CassiniISS._early_cruise_counts(dict_, filepath)
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

        # The camera frame, a fixed Cmatrix on the SPICE frame that the C kernels define,
        # is evaluated as the observation will evaluate it once the cadence, time-shift
        # and tracker overrides are applied. With check_pointing, missing pointing raises
        # OopsSpiceError now, with its SPICE code (#15); otherwise, as for an index row,
        # the frame raises it when it is used, never a cspyce error.
        if using_cks:
            checked = oops.Host._check_spice_frame(
                'CASSINI_ISS_' + camera, params['cadence'], overrides,
                description=f'no Cassini ISS {camera} pointing', filepath=filepath,
                defer=not check_pointing)
            if isinstance(checked, oops.frame.SpiceErrorFrame):
                params['frame'] = checked

        # Apply standard overrides. Parsed content from no file has no file subfields,
        # and `abspath` is given only when the file is on the local disk.
        oops.Host._apply_overrides(params, filepath, **overrides)

        obs = oops.observation.Snapshot(('v','u'), **params)
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
        # element does not give the filter pair
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

        # Other parameters; the time is never taken from a clock count
        params['host'] = 'Cassini'
        params['instrument'] = 'ISS'
        params['time_from_sclk'] = False
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
            label (Pds3Label): The parsed PDS3 label of an ISS image, read from a file.

        Returns:
            FCPath: The file named by the label's ^IMAGE pointer, in the label's
            directory.

        Raises:
            PdsHostError: If the label's ^IMAGE pointer does not give a file name.
        """

        pointer = label.get('^IMAGE', None)
        if not isinstance(pointer, str):
            raise oops.host.PdsHostError('^IMAGE does not give the name of the image '
                                         f'file in {label.filepath}')
        return FCPath(label.filepath).parent / pointer

    @staticmethod
    def _read_image_of_label(label):
        """The image file that a detached PDS3 label describes, read in full.

        Parameters:
            label (Pds3Label): The parsed PDS3 label of an ISS image.

        Returns:
            tuple[FCPath, VicarLabel, VicarImage]: The path to the image file, its VICAR
            header as the file writes it, and its content. The file is the one named by
            the label's ^IMAGE pointer, in the label's directory; if no file of that name
            exists, the same name in upper or lower case is used if that exists.

        Raises:
            FileNotFoundError: If no image file of any of these names exists.
            HostFormatError: If the label was not read from a file, so that no image file
                can be found.
            PdsHostError: If the label's ^IMAGE pointer does not give a file name.
            VicarHostError: If the image file's VICAR header cannot be parsed.
        """

        if _known_path(label.filepath) is None:
            raise oops.host.HostFormatError('no file path is known for this Cassini ISS '
                                            'PDS3 label, so its image file cannot be '
                                            'found')

        # A copy of the archive may have renamed its files to the other case (#14)
        filepath = CassiniISS._image_path(label)
        alternatives = dict.fromkeys((filepath.name.upper(), filepath.name.lower()))
        alternatives.pop(filepath.name, None)
        error = None
        for path in [filepath] + [filepath.with_name(name) for name in alternatives]:
            try:
                image = CassiniISS._read_vicar_image(path)
            except FileNotFoundError as err:
                error = error or err        # the error for the name the label gives
                continue
            return (path, CassiniISS._read_vicar_label(path), image)

        raise error

    @staticmethod
    def _read_vicar_image(filepath):
        """The content of a VICAR image file.

        Parameters:
            filepath (FCPath): The path to the image file.

        Returns:
            VicarImage: The image array and header. rms-vicar writes the reading machine
            into the header's HOST and BHOST; :meth:`_read_vicar_label` gives the header
            as the file writes it.

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
    def _read_vicar_label(filepath):
        """The VICAR header of an image file, as the file writes it.

        Parameters:
            filepath (FCPath): The path to the image file.

        Returns:
            VicarLabel: The header.

        Raises:
            FileNotFoundError: If the file does not exist.
            VicarHostError: If the VICAR header cannot be parsed.
        """

        try:
            return VicarLabel(filepath, strict=False)
        except (VicarError, ValueError) as err:
            raise oops.host.VicarHostError(f'{err} in {filepath}') from err

    @staticmethod
    def _parse_fileinfo(fileinfo, *, astrometry, pds3_options):
        """The parsed PDS3 label or VICAR header of an ISS image.

        A PDS3 label is read by ``oops.Host._read_fileinfo`` with its parser fallback:
        its text is first corrected by :meth:`_correct_label_text`, then parsed with the
        method in `pds3_options` and, if that method rejects it or cannot convert one of
        its values, with each later method in the order "fast", "strict", "loose". A
        label that the fast parser reads with a keyword that is not a PDS3 name, which is
        how it misreads a quoted value holding "=" that starts on the line after its
        keyword, is parsed again from the strict method on; if no later method reads it,
        the fast parser's result is kept. Any `repairs` in `pds3_options` are applied
        after the correction.

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
            except ValueError as err:   # VicarHostError too, whose message lacks the file
                raise oops.host.VicarHostError(f'{err} in {fileinfo}') from err

        # The correction, the parser fallback, and a value that one method cannot
        # convert, such as hour 25 in a time (#17), are handled by oops
        parsed = CassiniISS._read_fileinfo(fileinfo, formats='PV', astrometry=astrometry,
                                           pds3_options=pds3_options, pds3_fallback=True)

        # The fast parser misreads a quoted value that starts on the next line and holds
        # "=", such as METHOD_DESC in N1579539166_1.LBL of COISS_2041, making a keyword of
        # its text (#11). The correction moves such a value up to its keyword's line; a
        # label that the fast parser still misreads is read by the later methods.
        if (pds3_options['method'] == 'fast'
                and not all(_PDS3_NAME.fullmatch(key) for key in parsed.as_dict())):
            try:
                return CassiniISS._read_fileinfo(fileinfo, formats='P',
                                                 astrometry=astrometry,
                                                 pds3_options=pds3_options
                                                 | {'method': 'strict'},
                                                 pds3_fallback=True)
            except oops.host.PdsHostError:
                pass
        return parsed

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 label of an ISS image, with its known defects repaired.

        ``oops.Host`` calls this before any parser reads a label of this host, the labels
        of the index tables included. Three defects of the archive's labels are repaired
        (#11): a quoted value that starts on the line after its keyword is moved up to
        that line; the unterminated DESCRIPTION of some calibrated labels is closed at the
        end of the line before the next keyword; and the DESCRIPTION of others, closed on
        a line of its own and followed by a second quoted string, is joined into one
        string. A label without these defects is returned unchanged.

        Parameters:
            text (str): The text of the label, with <LF> line terminators.
            filepath (FCPath): The path to the label.

        Returns:
            str: The repaired text.
        """

        for (pattern, replacement) in _LABEL_REPAIRS:
            text = re.sub(pattern, replacement, text)
        return text

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

        The keys that the label parser derives from a keyword (see `_items` in
        _cassini.py) are left out of the label's keywords. The header's keywords are
        given their PDS3 names and forms (see :meth:`_pds3_dict_from_vicar`). Missing
        values are removed from both (see `_without_nulls` in _cassini.py), -999 among
        them in the keywords of `_FILL_KEYWORDS`. An item that only one source gives
        takes that source's value. An item that both give takes the value of its source
        in `_RESOLUTION_RULES`, the label for every item not listed there. The label's
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
            label = _without_nulls(_items(label_dict), _FILL_KEYWORDS)
        header = {}
        if header_dict is not None:
            header = _without_nulls(CassiniISS._pds3_dict_from_vicar(header_dict),
                                    _FILL_KEYWORDS)

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
    def _index_fill_rules(label):
        """How the missing values of an ISS index table are told, from its label.

        Parameters:
            label (Pds3Label | dict[str, Any]): The parsed label of the index.

        Returns:
            tuple[frozenset[str], dict[str, Any]]: The columns whose masks apply, those
            for which the label declares a null constant other than 0 (see
            `_DECLARED_ZERO`); and the fill value of each column, beyond its mask: -999
            in each column of `_FILL_KEYWORDS`, and the value of `_UNDECLARED_FILL_VALUES`
            in each column listed there for which the label does not declare a null
            constant.
        """

        declared = {}
        content = label.as_dict() if isinstance(label, Pds3Label) else label
        for table in content.values():
            if not isinstance(table, dict):
                continue
            for column in table.values():
                if isinstance(column, dict) and column.get('OBJECT') == 'COLUMN':
                    declared[column.get('NAME')] = [column[key] for key in
                                                    _SPECIAL_CONSTANT_KEYS
                                                    if key in column]

        masked = frozenset(name for (name, values) in declared.items()
                           if values and _DECLARED_ZERO not in values)
        fills = dict.fromkeys(_FILL_KEYWORDS, _FILL_VALUE)
        fills |= {name: value for (name, value) in _UNDECLARED_FILL_VALUES.items()
                  if not declared.get(name)}
        return (masked, fills)

    @staticmethod
    def _row_items(row_dict, masked, fills):
        """The metadata of one row of an ISS index table, in Python types.

        Parameters:
            row_dict (dict[str, Any]): The row, as the index reader gives it: NumPy values
                and, for each column, a "_mask" entry.
            masked (collections.abc.Container[str]): The columns whose masks apply, from
                :meth:`_index_fill_rules`.
            fills (dict[str, Any]): The fill value of each column, beyond its mask, from
                :meth:`_index_fill_rules`.

        Returns:
            dict[str, Any]: The row's columns as `_row_items` in _cassini.py gives them,
            with the masks of the columns in `masked` and the values of `fills` missing,
            and an angle of `_ZERO_WITHOUT_GEOMETRY` that is 0 missing in a row without
            PHASE_ANGLE. The clock counts are given three digits after the point.
        """

        items = _row_items(row_dict, masks=masked, fill_values=fills)
        if 'PHASE_ANGLE' not in items:
            for key in _ZERO_WITHOUT_GEOMETRY:
                if items.get(key) == 0:
                    del items[key]

        # The tables drop a count's trailing zeros, which also changes the number of
        # clock ticks it reads as: "1473199010.12" for the label's "1473199010.120"
        for key in _INDEX_CLOCK_COUNTS:
            match = (_CLOCK_COUNT.fullmatch(items[key]) if isinstance(items.get(key), str)
                     else None)
            if match:
                items[key] = f'{match[1]}.{(match[2] or "").ljust(3, "0")}'

        return items

    @staticmethod
    def _early_cruise_counts(dict_, filepath):
        """The metadata of an image with the clock counts of an early cruise image fixed.

        The labels of the early cruise images give a start count that is the stop count
        less the exposure in milliseconds counted as seconds, and their headers give no
        start count. The ISS index tables of COISS_1001 in the volumes' index directories
        give the image number as the start count, and as the stop count the image number
        less the exposure in tenths of a second counted as seconds. For an image whose
        IMAGE_NUMBER is no later than that of the last early cruise image, the stop count
        is its IMAGE_NUMBER, the whole seconds of the clock at the shutter's close, with
        zero ticks, as the labels give it; the start count is the clock count that the
        Cassini spacecraft clock kernel gives for the time EXPOSURE_DURATION milliseconds
        earlier.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without missing values, including
                EXPOSURE_DURATION; modified in place.
            filepath (FCPath | None): The image file, named in an error.

        Returns:
            dict[str, Any]: `dict_`.

        Raises:
            OopsSpiceError: If the spacecraft clock kernel cannot convert the counts.
        """

        number = dict_.get('IMAGE_NUMBER')
        if (not isinstance(number, str) or not number.isdigit()
                or int(number) > _LAST_EARLY_CRUISE_IMAGE):
            return dict_

        # The clock kernel names the partition, "1/1294562649.000"; the labels do not
        stop = number + '.000'
        dict_['SPACECRAFT_CLOCK_STOP_COUNT'] = stop
        tdb = _tdb_from_sclk(stop, filepath) - dict_['EXPOSURE_DURATION'] / 1000.
        dict_['SPACECRAFT_CLOCK_START_COUNT'] = _sclk_from_tdb(tdb, filepath)
        return dict_

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
        if 'IMAGE_NUMBER' in pds3 and not _is_null(pds3['IMAGE_NUMBER']):
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
            "COISS", in any case; False if the path contains "COUVIS" or "COVIMS", or if
            the label gives an INSTRUMENT_ID that does not start with "ISS"; otherwise
            None, meaning the host must be inferred from the index rows.
        """

        # The ISS index labels record nothing that identifies the instrument, but the PDS
        # volume and RMS Node metadata names all start with COISS. A UVIS or VIMS index
        # is turned down from its path or label, so the dispatcher can tell from the label
        # alone.
        status = _detect_index_by_path(label, 'COISS')
        if (status is None
                and not str(label.get('INSTRUMENT_ID', 'ISS')).strip().startswith('ISS')):
            return False
        return status

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
