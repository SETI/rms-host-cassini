##########################################################################################
# host_cassini/oops/iss.py
##########################################################################################

import datetime
import re

import numpy as np
import julian

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarImage, VicarLabel

import oops
from ._cassini import (_Cassini, _CMATRIX_ROTATION, _POINTING_SUBFIELDS, _file_errors,
                       _fix_cassini_ring_target, _pds3_options_with_defaults,
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
    'UNK'      : 'NONE',
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

# The image number of the last of the early cruise images, the first 1250 images of
# COISS_1001, from W1294561143_1 (1999-01-09) to N1343357427_1 (2000-07-27). Their labels
# give a start count that is the stop count less the exposure in milliseconds counted as
# seconds, and their headers give no start count. The next image, N1347928997_3
# (2000-09-18), gives both counts correctly.
_LAST_EARLY_CRUISE_IMAGE = 1343357427

# The clock counts of the index rows, which the tables write with their trailing zeros
# dropped, such as "1473199010.12" for the label's "1473199010.120"; 1,930 counts of the
# ISS indexes are short, and each matches its label once padded on the right
# (SETI/rms-data-projects#253 item 1). A count with three digits after the point is
# unchanged.
_INDEX_CLOCK_COUNTS = ('SPACECRAFT_CLOCK_START_COUNT', 'SPACECRAFT_CLOCK_STOP_COUNT')
_SHORT_CLOCK_COUNT = re.compile(r'(\d+)(?:\.(\d{0,2}))?')

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
# images on which the chosen source is wrong. The evidence covers 3,399 products. An item
# not listed takes the label, the archive's curated description of the product. A value
# that is missing from the chosen source, as a null value or the fill value, is taken
# from the other.
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

# A time as the VICAR headers write it, most ending in a "Z" that the PDS3 labels leave
# out; the early cruise headers write none.
# The day number 000, which no calendar has, is left out too: the one header that writes
# it, that of N1444745852_1 in COISS_1008, keeps it as text, as its corrected label does.
_HEADER_TIME = re.compile(r'\d{4}-(?:(?!000)\d{3}|\d{2}-\d{2})'
                          r'T\d{2}:\d{2}:\d{2}(?:\.\d*)?Z?')

# The corrections to the text of an ISS product label, as (pattern, replacement), which
# CassiniISS._correct_label_text applies before the parser reads the label. Each is for
# a defect that the strict parser refuses in the current archive, and changes only the
# text that has the defect. The count of each is from a scan of every current label.
_LABEL_CORRECTIONS = (
    # A pointer that names a file in another directory, which PDS3 does not allow, in
    # 232,524 raw COISS_2xxx labels, such as N1601512976_1.LBL of COISS_2049:
    #     ^DESCRIPTION = "../../label/vicar2.txt"
    # The file name is kept, which PDS3 looks for in the volume's label directory too.
    (r'(\^DESCRIPTION = ")\.\./\.\./label/', r'\1'),
    # A blank in place of the leading zero of a day, in 110,185 calibrated COISS_2xxx
    # labels, such as N1454725799_1_CALIB.LBL of COISS_2001:
    #     PRODUCT_CREATION_TIME = 2019-06- 9T09:58:43
    (r'(PRODUCT_CREATION_TIME = \d{4}-\d\d-) (\d)', r'\g<1>0\2'),
    # An unquoted N/A, in 133 raw and 133 calibrated COISS_2xxx labels, such as
    # N1463528225_2.LBL of COISS_2002:
    #     MISSING_LINES = N/A
    (r'(MISSING_LINES = )N/A', r'\1"N/A"'),
    # The second line of the raw label's two-line DESCRIPTION, left after the closing
    # quote of the calibrated DESCRIPTION, in 54 calibrated labels (50 COISS_2xxx and 4
    # COISS_1xxx), such as N1431946322_3_CALIB.LBL of COISS_1008:
    #       CALIBRATION_STAGE = '10'
    #     "
    #           header. Finalized using N1431946322_2.IMG."
    # The orphan line is deleted.
    (r'(\n"\n)[ \t]+[^"\n]*"[ \t]*\n', r'\1'),
    # The raw label's description as a second quoted string after the closing quote of
    # the calibrated DESCRIPTION, in 168 calibrated COISS_2xxx labels, such as
    # N1640662528_3_CALIB.LBL of COISS_2059:
    #       CALIBRATION_STAGE = '10'
    #     "
    #         "Extended header missing. Predicted values used to populate
    #         header.DSN gap truncation. Finalized using N1640662528_2.IMG."
    # The two strings are joined into one.
    (r'\n"(\n[ \t]+)"', r'\1'),
    # The day number 000, which no calendar has, in the raw and calibrated labels of
    # N1444745852_1 in COISS_1008:
    #     EARTH_RECEIVED_START_TIME = 0000-000T00:00:00.000
    # The value is quoted, so that it is read as text.
    (r'(EARTH_RECEIVED_START_TIME = )(\d{4}-000T[\d:.]+)', r'\1"\2"'),
)


class CassiniISS(oops.Host):
    """The Cassini ISS host: Snapshot constructors for NAC and WAC images."""

    #: The name under which this host registers with :class:`~oops.host.host_.Host` and
    #: that appears in its error messages.
    NAME = 'Cassini ISS'

    #: No ISS metadata index has a supplement, so the rows of a supplement that the
    #: caller gives are paired with the index rows by position.
    MERGE_SUPPLEMENT_BY_FILE = False

    _INSTRUMENT_KERNEL = None
    _FOVS = {}
    _initialized = False

    @staticmethod
    def from_file(fileinfo, *, fast_distortion=True, return_all_planets=False,
                  pds3_options=None, select=None, astrometry=False, target=None,
                  lightsource=None, path=None, frame=None, cadence=None, fov=None,
                  calibrations=None, timeshift=None, navigation=None, tracker=None,
                  parallel=None):
        """A Snapshot object based on a given Cassini ISS image file or its PDS3 label.

        What is read depends on the input. Given the VICAR image file, only that file is
        read: its header, and its array unless `astrometry` is True. Given the detached
        PDS3 label, the label is read and, unless `astrometry` is True, so is the image
        file that its ^IMAGE pointer names; with `astrometry`, the image file need not
        exist. The array is always the one that the image file's own VICAR header
        locates, which for many labels is not where the label's ^IMAGE pointer puts it.
        The label is parsed with the "strict" method unless `pds3_options` gives another,
        after the corrections of the known defects of the archive's labels.

        The metadata comes from whichever of the label and the header were read. An item
        that a source gives as text that PDS3 writes for no value ("UNK", "UNKNOWN",
        "N/A", "NULL", "NOT APPLICABLE", "---" or the empty string, in any case) counts as
        missing from it, as does -999 in a temperature, data rate or compression item.
        Where the label and the header both give an item, one fixed source supplies it,
        the same for every image: the header for MISSION_PHASE_NAME, which the early
        cruise labels give as "SCIENCE CRUISE" while each header names the phase of the
        image's date, and for METHOD_DESC, whose semicolons, commas and "@" the labels
        replace; the label for every other item. A value missing from the fixed source is
        taken from the other. An item missing from every source read is left out.

        The header's values are first written in the label's forms and types: a time is
        a :class:`datetime.datetime`, IMAGE_NUMBER is text, IMAGE_OBSERVATION_TYPE is a
        set, a single OPTICS_TEMPERATURE is the pair of it and None, and
        MISSION_PHASE_NAME has spaces for underscores. The VICAR headers of the early
        cruise images, the first 1250 of COISS_1001, which use older keywords, are given
        the names and spellings of their labels: INSTRUMENT_HOST_NAME "CASSINI ORBITER";
        INSTRUMENT_NAME from INSTRUMENT_ID; FILTER_NAME, the pair of FILTER1_NAME and
        FILTER2_NAME; GAIN_MODE_ID spelled in full, such as "12 ELECTRONS PER DN";
        START_TIME and STOP_TIME, both IMAGE_TIME; and SHUTTER_STATE_ID, SHUTTER_MODE_ID,
        BIAS_STRIP_MEAN, CALIBRATION_LAMP_STATE_FLAG, INST_CMPRS_RATIO,
        DATA_CONVERSION_TYPE, INST_CMPRS_TYPE, DARK_STRIP_MEAN and
        SENSOR_HEAD_ELEC_TEMPERATURE, from SHUTTER_MODE_ID, SHUTTER_EVENT_TYPE, BIAS,
        CALIB_LAMP_STATE_FLAG, COMPRESSION_RATIO, CONVERSION_TYPE, ENCODING_TYPE,
        EXTENDED_PIXEL_VALUE and SENSOR_HEAD_ELEC_TEMP.

        The clock counts of an early cruise image are computed: the stop count is
        IMAGE_NUMBER followed by ".000", and the start count is the same with the shutter
        disabled and otherwise the count that the Cassini spacecraft clock kernel gives
        for the time EXPOSURE_DURATION earlier.

        With the shutter disabled (SHUTTER_STATE_ID "DISABLED"), the exposure starts at
        START_TIME; otherwise it ends at IMAGE_TIME and starts EXPOSURE_DURATION earlier.

        The pointing is not checked. Where the C kernels lack it, using the frame raises
        the error that SPICE reports.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The path to the VICAR image file or to its detached PDS3 label, or the
                parsed content of either, read as the file it came from.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to the observation, treated as
                a tuple of one; None to return the observation itself.
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
                instrument, such as a :class:`~oops.frame.Cmatrix`, used as given. With
                it, the C kernels are not loaded.
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
            Snapshot | tuple[Snapshot, ...]: The observation, or the observation or tuple
            that `select` chooses, with these subfields:

            * `host`, `instrument`, `detector`, `sampling`, `texp`, `spice_kernels`,
              `spice_to_frame`, `spice_frame_name`, `spice_frame_id`, `ck_object_id`
              (-82000), `ck_sclk_id` (-82) and `spice_frame_is_frozen` (False), always.
            * `filepath` and `basename`, which name the image file also when `fileinfo`
              is its label, and `image_url`, its absolute path or URL.
            * `image_abspath`, the local path of the image file, when it was read: unless
              `fileinfo` is the label and `astrometry` is True.
            * `label_url` and `label_abspath`, the absolute path or URL and the local path
              of the label, when it was read.
            * `data`, unless `astrometry` is True.
            * `filter`, `filter1` and `filter2`; `gain_mode`; `label_target`;
              `observation_id`; and `spacecraft_clock_start_count` and
              `spacecraft_clock_stop_count`, each when the metadata gives its items.
            * `dict`: the merged metadata, keyed by the PDS3 names, as described above.
              It holds every item of each source read whose value is not missing, the
              keys that the label parser adds included, such as "START_TIME_fmt"; in a
              list, a missing element is None.
            * `label_dict`: the parsed PDS3 label, as its parser gives it, when the label
              was read.
            * `header_dict`: the VICAR header, as its parser gives it, the first value of
              a keyword that repeats, when the image file was read.

        Raises:
            FileNotFoundError: If the file, or the image file that a label read without
                `astrometry` names, does not exist.
            HostFormatError: If the label or the image file cannot be read or
                interpreted: if it lacks an item that the observation needs or gives an
                invalid one, if the image file ends before its array, or if the array is
                not the size that the label gives. Subclass PdsHostError is raised if the
                label cannot be parsed, and VicarHostError if the VICAR content of the
                image file cannot be read, as for an image of more than one band. Each
                message names the file.
            WrongHostError: If the file, or the parsed content given, is not of a Cassini
                ISS image.
        """

        CassiniISS.initialize()  # define everything the first time through; use defaults
                                 # unless initialize() was called explicitly.

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

        given = (fileinfo.filepath
                 if isinstance(fileinfo, (Pds3Label, VicarLabel, VicarImage))
                 else fileinfo)
        with _file_errors(given):
            (label, header, data, image_path) = CassiniISS._read_files(
                fileinfo, astrometry=astrometry, pds3_options=pds3_options)
            label_dict = None if label is None else label.as_dict()
            header_dict = None if header is None else header.as_dict(unique=False)
            dict_ = CassiniISS._merged_dict(label_dict, header_dict)
            CassiniISS._early_cruise_counts(dict_)
            label_path = None if label is None else FCPath(label.filepath)
            snapshot = CassiniISS._make_snapshot(dict_, image_path=image_path,
                                                 label_path=label_path, data=data,
                                                 fast_distortion=fast_distortion,
                                                 return_all_planets=return_all_planets,
                                                 overrides=overrides)

        if label_dict is not None:
            snapshot.insert_subfield('label_dict', label_dict)
        if header_dict is not None:
            snapshot.insert_subfield('header_dict', header_dict)
        return CassiniISS._select(snapshot, select)

    @staticmethod
    def from_index(filepath, *, supplement=oops.Host.SUPPLEMENT, label=None,
                   row_dicts=None, pds3_options=None, filter=None, fast_distortion=True,
                   return_all_planets=False, select=None, target=None, lightsource=None,
                   path=None, frame=None, cadence=None, fov=None, calibrations=None,
                   timeshift=None, navigation=None, tracker=None, parallel=None):
        """An iterator over Snapshots, one for each row of a Cassini ISS index file.

        It reads a metadata index of the PDS Ring-Moon Systems Node, such as
        metadata/COISS_2xxx/COISS_2001/COISS_2001_index.lbl. The call reads the labels of
        the index and of any supplement; the rows are read as the iteration proceeds, 100
        at a time, and each row's Snapshot is built when the iteration reaches it. A
        caller that needs ``len()`` or indexing wraps the call in ``list()``.

        Each row gives the times, field of view, target and light source as a label
        does; the exposure timing is that of :meth:`from_file`. The C kernels and SP
        kernels of each row's exposure are loaded when the iteration reaches the row. The
        pointing is not checked; where the C kernels lack it, using the frame raises the
        error that SPICE reports.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to the PDS label of a
                Cassini ISS index file.
            supplement (str | pathlib.Path | FCPath | object | None, optional): The path
                to the PDS label of a supplemental index file, in any directory, whose
                columns are merged into each row; its rows must list the same files as
                the index in the same order, and every other column that both tables
                give must agree. The default, `oops.Host.SUPPLEMENT`, reads the file
                beside `filepath` whose name has "_supplemental_index" in place of
                "_index", if it exists; no ISS index has one. None reads the index alone.
            label (Pds3Label, optional): The PDS label of `filepath` if it has already
                been read.
            row_dicts (collections.abc.Iterable[dict[str, Any]], optional): The rows of
                the index, one dictionary per row, if `filepath` has already been read. If
                provided, the tables are not read and `supplement` is ignored.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser, with
                which the label of the index, and with their method the label of the
                supplement, are parsed. The method is "strict" unless they give another.
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
            select (int | slice | tuple[int | slice, ...], optional): An index, a slice,
                or a tuple of indices and slices to apply to each row's observation,
                treated as a tuple of one; None for the observation itself.
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

        Returns:
            collections.abc.Iterator[Snapshot | tuple[Snapshot, ...]]: One observation, or
            what `select` chooses, for every filtered row of the index, in index order.
            Each has the subfields of :meth:`from_file` that the row gives, except the
            file subfields: `filepath` is the row's VOLUME_ID and
            FILE_SPECIFICATION_NAME joined by "/", relative to the root of the volumes,
            and `basename` is the file name. `dict` is the row as the table gives it,
            with the "_mask" entry of each column, except that a clock count with fewer
            than three digits after the point is padded on the right with zeros, as the
            labels give it, such as "1473199010.120" for "1473199010.12", and one with
            no point is followed by ".000".

        Raises:
            FileNotFoundError: If the label of `filepath` does not exist, or if
                `supplement` names a file that does not exist; when the iteration starts,
                if a table file does not exist.
            HostIndexError: When called, if the label of the index or the supplement does
                not describe a table that can be read; during the iteration, as the
                reading of the rows finds that the supplement does not match the index.
            PdsHostError: If the PDS3 label of the index or the supplement cannot be
                parsed.
            WrongHostError: If `filepath` is not a Cassini ISS index file.
        """

        CassiniISS.initialize()

        rows = CassiniISS._index_rows(filepath, supplement=supplement,
                                      require_supplement=(supplement is not
                                                          oops.Host.SUPPLEMENT),
                                      label=label, row_dicts=row_dicts,
                                      pds3_options=_pds3_options_with_defaults(
                                          pds3_options),
                                      filter=filter)

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
                row_dict = CassiniISS._padded_clock_counts(row_dict)
                image_path = (str(row_dict['VOLUME_ID']) + '/'
                              + str(row_dict['FILE_SPECIFICATION_NAME']))
                obs = CassiniISS._make_snapshot(row_dict, image_path=image_path,
                                                label_path=None, data=None,
                                                fast_distortion=fast_distortion,
                                                return_all_planets=return_all_planets,
                                                overrides=overrides)
                yield CassiniISS._select(obs, select)

        return snapshots()

    @staticmethod
    def _make_snapshot(dict_, *, image_path, label_path, data, fast_distortion,
                       return_all_planets, overrides):
        """A Snapshot built from the metadata of an ISS image or one row of an ISS index.

        Loads the SPK kernels covering the exposure. Unless `overrides` supplies a frame,
        it also loads the CK kernels and registers the camera frames.

        Parameters:
            dict_ (dict[str, Any]): The metadata or the index row. It must contain
                EXPOSURE_DURATION, INSTRUMENT_MODE_ID and INSTRUMENT_NAME; START_TIME if
                SHUTTER_STATE_ID is "DISABLED", and otherwise IMAGE_TIME. Each other item
                that it lacks leaves out the subfields that would be taken from it.
            image_path (str | FCPath): The image file, an FCPath for a file and the
                relative path that the index gives for a row; passed to
                ``oops.Host._set_path_attributes``.
            label_path (FCPath | None): The PDS3 label that was read; None if none was.
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

        Returns:
            Snapshot: The observation, with the subfields listed in :meth:`from_file`,
            except `label_dict` and `header_dict`.
        """

        # With the shutter disabled, START_TIME, STOP_TIME and IMAGE_TIME all mark the
        # start of the exposure window, as the index labels define them; otherwise
        # IMAGE_TIME is the close of the shutter. In 427 shutter-enabled COISS_1001
        # frames, such as W1294561143_1, START_TIME equals IMAGE_TIME (#3).
        texp = dict_['EXPOSURE_DURATION'] / 1000.
        if dict_.get('SHUTTER_STATE_ID') == 'DISABLED':
            tstart = julian.tdb_from_iso(dict_['START_TIME'])
        else:
            tstart = julian.tdb_from_iso(dict_['IMAGE_TIME']) - texp
        mode   = dict_['INSTRUMENT_MODE_ID']        # "FULL", "SUM2", or "SUM4"
        camera = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'
        fov    = CassiniISS._FOVS[camera, mode, fast_distortion]

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
            'fov'         : fov,
            'calibrations': [],         # TBD
        }

        # Apply standard overrides
        oops.Host._apply_overrides(params, **overrides)

        obs = oops.observation.Snapshot(('v','u'), **params)
        if data is not None:
            obs.insert_subfield('data', data)
        oops.Host._set_path_attributes(obs, image_path, label_path)

        # Extra attributes
        obs.insert_subfields(CassiniISS._get_subfields(dict_))
        obs.insert_subfields(_POINTING_SUBFIELDS)

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
    def _select(obs, select):
        """The observation that `select` chooses.

        Parameters:
            obs (Snapshot): The observation.
            select (int | slice | tuple[int | slice, ...] | None): The selection, applied
                to the observation as a tuple of one; None for the observation itself.

        Returns:
            Snapshot | tuple[Snapshot, ...]: The chosen observation or tuple.
        """

        if select is None:
            return obs
        return oops.Host._apply_select((obs,), select)

    @staticmethod
    def _get_subfields(dict_):
        """The subfields taken from the metadata of an image.

        Parameters:
            dict_ (dict[str, Any]): The metadata or the index row.

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
    def _read_files(fileinfo, *, astrometry, pds3_options):
        """The label, header and array of an ISS image, as the input determines.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The path to the image file or its PDS3 label, or the parsed content of
                either.
            astrometry (bool): True to read no array, and given a label, no image file.
            pds3_options (dict[str, Any]): The options for the PDS3 label parser.

        Returns:
            tuple[Pds3Label | None, VicarLabel | VicarImage | None, np.ndarray | None,
            FCPath]: The PDS3 label, if one was read; the VICAR header, if the image file
            was read; the array, if it was read; and the path to the image file.

        Raises:
            HostFormatError: If the array is not the size that the label gives.
            WrongHostError: If the content is not of a Cassini ISS image.
        """

        # A VICAR header, given as content or read here with the loose VICAR reader,
        # which reads the 33-character keyword UNEVEN_BIT_WEIGHT_CORRECTION_FLAG of
        # calibrated COISS_1001 N1294562651_1_CALIB.IMG, refused by the strict one
        if isinstance(fileinfo, (VicarLabel, VicarImage)) or (
                not isinstance(fileinfo, Pds3Label)
                and FCPath(fileinfo).suffix.lower() != '.lbl'):
            if isinstance(fileinfo, (VicarLabel, VicarImage)):
                header = fileinfo
            elif astrometry:
                header = VicarLabel(fileinfo, strict=False)
            else:
                header = VicarImage(fileinfo, strict=False)
            image_path = FCPath(header.filepath)
            if not CassiniISS._detect_in_vicar(header):
                raise oops.Host._no_host_error([CassiniISS], 'VICAR file', image_path)
            if astrometry:
                return (None, header, None, image_path)
            if not isinstance(header, VicarImage):
                header = VicarImage(image_path, strict=False)
            return (None, header, header.data_2d, image_path)

        label = CassiniISS._read_fileinfo(fileinfo, formats='P', astrometry=astrometry,
                                          pds3_options=pds3_options)
        if not CassiniISS._detect_in_pds3(label):
            raise oops.Host._no_host_error([CassiniISS], 'PDS3 label', label.filepath)
        image_path = FCPath(label.filepath).parent / label['^IMAGE']
        if astrometry:
            return (label, None, None, image_path)

        # The array is read through the VICAR file, never at the label's ^IMAGE offset,
        # which is one record short in every calibrated label, such as
        # N1777325846_1_CALIB.LBL of calibrated/COISS_2090, and in 437 of the 8,024 raw
        # labels of COISS_1006, such as N1361112165_1.LBL (#6)
        image = VicarImage(image_path, strict=False)
        data = image.data_2d
        shape = (label['IMAGE']['LINES'], label['IMAGE']['LINE_SAMPLES'])
        if data.shape != shape:
            raise oops.host.HostFormatError(f'{image_path}: the array is '
                                            f'{data.shape[0]} x {data.shape[1]} but the '
                                            f'label gives {shape[0]} x {shape[1]}')
        return (label, image, data, image_path)

    @classmethod
    def _correct_label_text(cls, text, filepath):
        """The text of a PDS3 label of an ISS image, with its known defects corrected.

        Six defects of the archive's labels, each of which the "strict" parser refuses,
        are corrected: a ^DESCRIPTION pointer to "../../label/vicar2.txt", which keeps
        the file name alone; a PRODUCT_CREATION_TIME with a blank for the leading zero of
        its day, which gets the zero; MISSING_LINES = N/A, which is quoted; the orphan
        second line of a raw description after the closing quote of a calibrated
        DESCRIPTION, which is deleted; a second quoted string after the closing quote of
        a calibrated DESCRIPTION, which is joined to it; and an EARTH_RECEIVED_START_TIME
        with the day number 000, which is quoted. A label without these defects is
        returned unchanged.

        Parameters:
            text (str): The text of the label, with <LF> line terminators.
            filepath (FCPath): The path to the label.

        Returns:
            str: The corrected text.
        """

        for (pattern, replacement) in _LABEL_CORRECTIONS:
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _merged_dict(label_dict, header_dict):
        """The metadata of an image, merged from its PDS3 label and its VICAR header.

        The header's keywords are given their PDS3 names and forms (see
        :meth:`_pds3_dict_from_vicar`). Missing values are removed from both, -999 among
        them in the keywords of `_FILL_KEYWORDS`. An item that only one source gives
        takes that source's value. An item that both give takes the value of its source
        in `_RESOLUTION_RULES`, the label for every item not listed there. The label's
        keywords come first, in label order, then the header's other keywords, in header
        order.

        Parameters:
            label_dict (dict[str, Any] | None): The parsed PDS3 label; None if it was not
                read.
            header_dict (dict[str, Any] | None): The parsed VICAR header; None if it was
                not read.

        Returns:
            dict[str, Any]: The merged metadata.
        """

        label = {}
        if label_dict is not None:
            label = _without_nulls(label_dict, _FILL_KEYWORDS)
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
    def _padded_clock_counts(row_dict):
        """An index row with its short clock counts padded, as the labels give them.

        Parameters:
            row_dict (dict[str, Any]): The row, as the index reader gives it. It is not
                modified.

        Returns:
            dict[str, Any]: A copy of the row in which each clock count with fewer than
            three digits after the point is padded on the right with zeros, and one with
            no point is followed by ".000".
        """

        row_dict = dict(row_dict)
        for key in _INDEX_CLOCK_COUNTS:
            match = _SHORT_CLOCK_COUNT.fullmatch(str(row_dict.get(key, '')))
            if match:
                row_dict[key] = f'{match[1]}.{(match[2] or "").ljust(3, "0")}'
        return row_dict

    @staticmethod
    def _early_cruise_counts(dict_):
        """Compute the clock counts of an early cruise image in its metadata.

        For an image whose IMAGE_NUMBER is no later than that of the last early cruise
        image, the stop count is its IMAGE_NUMBER, the whole seconds of the clock at the
        shutter's close, followed by ".000", as the labels give it. With the shutter
        disabled, the start count is the stop count; otherwise it is the count that the
        Cassini spacecraft clock kernel gives for the time EXPOSURE_DURATION
        milliseconds earlier. Any other image is left as it is.

        Parameters:
            dict_ (dict[str, Any]): The metadata of an image, from its label or header;
                modified in place.
        """

        number = dict_.get('IMAGE_NUMBER')
        if number is None or int(number) > _LAST_EARLY_CRUISE_IMAGE:
            return

        stop = number + '.000'
        dict_['SPACECRAFT_CLOCK_STOP_COUNT'] = stop
        if dict_.get('SHUTTER_STATE_ID') == 'DISABLED':
            dict_['SPACECRAFT_CLOCK_START_COUNT'] = stop
        else:
            tdb = _tdb_from_sclk(stop) - dict_['EXPOSURE_DURATION'] / 1000.
            dict_['SPACECRAFT_CLOCK_START_COUNT'] = _sclk_from_tdb(tdb)

    @staticmethod
    def _pds3_dict_from_vicar(dict_):
        """The keywords of an ISS VICAR header under their PDS3 label names and forms.

        Every header's values are written in the forms and types of the PDS3 labels (see
        :meth:`_pds3_form`). The VICAR headers of the early cruise images, which carry no
        INSTRUMENT_HOST_NAME, use older names; for these, each keyword is given the name
        and spelling that the image's PDS3 label gives the same value, as
        :meth:`from_file` describes. Every other keyword keeps its VICAR name and value.

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
        return pds3

    @staticmethod
    def _pds3_form(key, value):
        """A VICAR header value written in the form and type that the PDS3 labels give
        its item.

        Parameters:
            key (str): The keyword.
            value (Any): The header's value.

        Returns:
            Any: A time, with or without the final "Z" of most headers, as a
            :class:`datetime.datetime`, as the label parser gives it; IMAGE_NUMBER as
            text; IMAGE_OBSERVATION_TYPE as a set; a single OPTICS_TEMPERATURE, the front
            temperature, as the pair of it and None; MISSION_PHASE_NAME with spaces for
            underscores; any other value unchanged.
        """

        if isinstance(value, str) and _HEADER_TIME.fullmatch(value):
            return _datetime(value.rstrip('Z'))
        if key == 'IMAGE_NUMBER' and isinstance(value, int):
            return str(value)
        if key == 'IMAGE_OBSERVATION_TYPE':
            return {value} if isinstance(value, str) else set(value)
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


def _datetime(text):
    """A time of a VICAR header as the PDS3 label parser gives a time of a label.

    Parameters:
        text (str): The time in ISO format, without the final "Z" of most headers, such
            as "2004-037T02:07:06.418".

    Returns:
        datetime.datetime: The time, to the microsecond, rounded as the parser rounds.
    """

    (day, sec) = julian.day_sec_from_iso(text)
    (hour, minute, second) = julian.hms_from_sec(sec)
    whole = int(second)
    return datetime.datetime(*julian.ymd_from_day(day), hour, minute, whole,
                             int((second - whole) * 1000000 + 0.499999))

##########################################################################################
