##########################################################################################
# host_cassini/oops/iss.py
##########################################################################################

import math
import re

import numpy as np
import julian

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarError, VicarImage

import oops
from ._cassini import (_Cassini, _CMATRIX_ROTATION, _fix_cassini_ring_target,
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

# The PDS3 null constants. A value that is one of these, or a list or set of nothing but
# these, states nothing: the item is missing from that source.
_PDS3_NULLS = {'UNK', 'N/A', 'NULL'}

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
# the image file, as the winner of an item they disagree on.
_LABEL = 'label'
_HEADER = 'header'

# The items on which the label and the header disagree in the archive, and which source
# wins each, from a comparison of 2,493 raw and calibrated products (#10). The label, the
# archive's curated description of the product, wins any item not listed, and is listed
# where it wins for a reason of its own. Before this table applies, a PDS3 null value
# loses to a value, and a number rounded to fewer digits loses to the precise one; that
# settles BIAS_STRIP_MEAN, DARK_STRIP_MEAN, DETECTOR_TEMPERATURE, FILTER_TEMPERATURE,
# INSTRUMENT_DATA_RATE, INST_CMPRS_RATIO and SENSOR_HEAD_ELEC_TEMPERATURE, which the
# header writes to 6 significant digits (524 to 1,869 products each), and
# EXPECTED_MAXIMUM and INST_CMPRS_RATE, where the label is the rounded one in 21
# products.
_RESOLUTION_RULES = {
    # 301 of 1,880: the cruise headers carry an older data set name
    'DATA_SET_ID'              : _LABEL,
    # 906 of 1,605, all calibrated: the header keeps the raw image's description, the
    # label describes the calibration; the raw one stays in header_dict
    'DESCRIPTION'              : _LABEL,
    # 34 and 44 of 1,880: the two differ by 1 or 2 milliseconds
    'EARTH_RECEIVED_START_TIME': _LABEL,
    'EARTH_RECEIVED_STOP_TIME' : _LABEL,
    # 887 of 1,880: later labels write " - " in the name; either spelling gives the camera
    'INSTRUMENT_NAME'          : _LABEL,
    # 34 of 1,605: the labels replace ";" and "," with spaces and "@" with a line break
    'METHOD_DESC'              : _HEADER,
    # 888 of 2,493: the cruise headers say "CASSINI"
    'MISSION_NAME'             : _LABEL,
    # 694 of 2,493: the headers write underscores; the early cruise headers name a finer
    # phase than the labels' "SCIENCE CRUISE"
    'MISSION_PHASE_NAME'       : _LABEL,
    # 888 of 2,493: the cruise headers give the front temperature alone, which the labels
    # give rounded to 5 digits beside an unknown back temperature
    'OPTICS_TEMPERATURE'       : _LABEL,
    # 1,133 of 1,880, all calibrated: the header keeps the raw product's creation time
    'PRODUCT_CREATION_TIME'    : _LABEL,
}

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

_LABEL_REPAIRS = (_NEXT_LINE_VALUE, _UNTERMINATED_DESCRIPTION)

# A keyword name that a parsed PDS3 label can hold, including the suffixed keys that the
# parser adds, such as "START_TIME_fmt", and its "objects" list
_PDS3_NAME = re.compile(r'\^?[A-Za-z][A-Za-z0-9_:]*')

# The short code at the start of a SPICE error message, such as "SPICE(CKINSUFFDATA)"
_SPICE_ERROR_CODE = re.compile(r'SPICE\([A-Z0-9_]+\)')


class _NoPointingError(oops.host.HostError):
    """The C kernels hold no pointing for the camera at the time of an image."""


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
        that a source gives as a PDS3 null value ("UNK", "N/A" or "NULL") counts as
        missing from it, and an item missing from every source read is left out: its
        subfield is absent, never None. Where the label and the header give different
        values for an item, a rule for that item chooses one; the label wins unless the
        header's value is the more precise number or the item's rule names the header.
        Times, field of view, pointing, target and light source are computed from the
        merged metadata.

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
                values to be passed to the :class:`~pdsparser.Pds3Label` parser. With the
                default method, "fast", a label that the fast parser rejects or misreads
                is parsed again by the "strict" and then the "loose" parser, and a label
                with a known archive defect is repaired in memory first.
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
              "1459509269.122".
            * `dict`: the merged metadata, keyed by the PDS3 names. It holds every item
              of each source read, under its own name, except those whose value is a
              PDS3 null value. The VICAR headers of the early cruise images, which use
              older names, are given the names and spellings of their labels.
            * `label_dict`: the PDS3 label as its parser gives it, when the label was
              read.
            * `header_dict`: the VICAR header as ``VicarLabel.as_dict(unique=False)``
              gives it, when the header was read.

            `filepath`, `basename`, `abspath` and `image_url` name the image file, also
            when `fileinfo` is its PDS3 label.

        Raises:
            FileNotFoundError: If the file, or the image file of a PDS3 label read without
                `astrometry`, does not exist in the case spelled or in upper or lower
                case.
            HostError: If no frame override is given and the C kernels hold no pointing
                for the camera at the midtime of the exposure. The message ends with the
                SPICE error code, such as "SPICE(CKINSUFFDATA)", when SPICE gives one.
            HostFormatError: If START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID or
                INSTRUMENT_NAME is missing from every source read, or START_TIME or
                INSTRUMENT_MODE_ID is invalid; the message names the item and the file.
            OSError: If the file cannot be read.
            PdsHostError: If the PDS3 label cannot be parsed, or its ^IMAGE pointer names
                no file.
            VicarHostError: If the VICAR header cannot be parsed.
            WrongHostError: If the file is not a Cassini ISS file.
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

        label_dict = None if label is None else label.as_dict()
        header_dict = None if header is None else header.as_dict(unique=False)
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

        source = filepath if label is None else FCPath(label.filepath)
        snapshot = CassiniISS._make_snapshot(dict_, filepath=filepath, data=data,
                                             fast_distortion=fast_distortion,
                                             return_all_planets=return_all_planets,
                                             overrides=overrides, source=source,
                                             check_pointing=True)
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
    def from_index(filepath, *, row_dicts=None, pds3_options=None, filter=None,
                   fast_distortion=True, return_all_planets=False, target=None,
                   lightsource=None, path=None, frame=None, cadence=None, fov=None,
                   calibrations=None, timeshift=None, navigation=None, tracker=None,
                   parallel=None, **kwargs):
        """A list of Snapshot objects, one for each row of a Cassini ISS index file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to a Cassini ISS index
                file or its PDS label.
            row_dicts (list[dict[str, Any]], optional): A list of dictionaries, one per
                row of the index. Include if `filepath` has already been read.
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
            **kwargs: Additional keyword arguments; they are accepted and ignored. This
                includes `select` and `supplement`.

        Returns:
            list[Snapshot]: One observation per filtered row of the index, each with
            subfields `dict`, `host`, `instrument`, `detector`, `sampling`, `texp`,
            `spice_kernels`, `filepath`, `basename`, `spice_to_frame`, `spice_frame_name`
            and `spice_frame_id` inserted, and `filter`, `filter1`, `filter2`,
            `gain_mode`, `label_target`, `observation_id`,
            `spacecraft_clock_start_count` and `spacecraft_clock_stop_count` where the
            row gives their columns. `dict` is the row without its PDS3 null values;
            a column that holds one is treated as missing.

        Raises:
            HostFormatError: If a row lacks START_TIME, EXPOSURE_DURATION,
                INSTRUMENT_MODE_ID or INSTRUMENT_NAME, or gives an invalid START_TIME or
                INSTRUMENT_MODE_ID.
            PdsHostError: If the PDS3 label cannot be parsed.
            OSError: If `filepath` cannot be read.
            WrongHostError: If `filepath` is not a Cassini ISS index file.
        """

        CassiniISS.initialize()

        # Read and filter the index table
        if row_dicts is None:
            row_dicts = CassiniISS._read_index_rows(filepath, pds3_options=pds3_options)
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
            fpath = row_dict['VOLUME_ID'] + '/' + row_dict['FILE_SPECIFICATION_NAME']
            obs = CassiniISS._make_snapshot(CassiniISS._without_nulls(row_dict),
                                            filepath=fpath, data=None,
                                            fast_distortion=fast_distortion,
                                            return_all_planets=return_all_planets,
                                            overrides=overrides, source=fpath,
                                            check_pointing=False)
            snapshots.append(obs)

        return snapshots

    @staticmethod
    def _make_snapshot(dict_, filepath, data, *, fast_distortion, return_all_planets,
                       overrides, source, check_pointing):
        """A Snapshot built from the metadata of an ISS image or one row of an ISS index.

        Loads the SPK kernels covering the exposure. Unless `overrides` supplies a frame,
        it also loads the CK kernels and registers the camera frames.

        Parameters:
            dict_ (dict[str, Any]): The metadata, from which PDS3 null values have been
                removed. It must contain START_TIME, EXPOSURE_DURATION,
                INSTRUMENT_MODE_ID and INSTRUMENT_NAME; each other item that it lacks
                leaves out the subfields that would be taken from it.
            filepath (str | FCPath): The path to the image file; used for the `filepath`
                and `basename` subfields.
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
            source (str | FCPath): The file the metadata came from, named in errors.
            check_pointing (bool): True to require that the C kernels give the camera's
                pointing at the midtime of the exposure, when no frame override is given.

        Returns:
            Snapshot: The observation, with the subfields listed in :meth:`from_file`.

        Raises:
            HostFormatError: If a required item is missing or invalid.
            HostError: If `check_pointing` is True and the C kernels hold no pointing for
                the camera at the midtime of the exposure. The message ends with the
                SPICE error code, such as "SPICE(CKINSUFFDATA)", when SPICE gives one.
        """

        CassiniISS._require_items(dict_, source)
        try:
            tstart = julian.tdb_from_iso(dict_['START_TIME'])
        except (TypeError, ValueError) as err:
            raise oops.host.HostFormatError(f'invalid START_TIME '
                                            f'{dict_["START_TIME"]!r} in {source}'
                                            ) from err
        texp   = dict_['EXPOSURE_DURATION'] / 1000.
        mode   = dict_['INSTRUMENT_MODE_ID']        # "FULL", "SUM2", or "SUM4"
        camera = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'
        if (camera, mode, fast_distortion) not in CassiniISS._FOVS:
            raise oops.host.HostFormatError(f'invalid INSTRUMENT_MODE_ID {mode!r} in '
                                            f'{source}')

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
            if check_pointing:
                # Report missing pointing now, with its SPICE code, not when the frame
                # is first used (#15)
                CassiniISS._check_pointing(camera, tstart + texp/2., source)

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
    def _require_items(dict_, source):
        """Raise HostFormatError unless the metadata gives every required item.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without PDS3 null values.
            source (str | FCPath): The file the metadata came from, named in the error.

        Raises:
            HostFormatError: If START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID or
                INSTRUMENT_NAME is missing; the message names the item and `source`.
        """

        for item in _REQUIRED_ITEMS:
            if item not in dict_:
                raise oops.host.HostFormatError(f'{item} is missing from {source}')

    @staticmethod
    def _check_pointing(camera, time, source):
        """Raise HostError unless the C kernels give the camera's pointing at a time.

        Parameters:
            camera (str): "NAC" or "WAC".
            time (float): The time in seconds TDB.
            source (str | FCPath): The file of the image, named in the error.

        Raises:
            HostError: If the camera frame cannot be evaluated at `time`. Its cause is the
                exception that SPICE raised, and its message ends with the SPICE error
                code, such as "SPICE(CKINSUFFDATA)", when that exception gives one.
        """

        # The SPICE frame, which the C kernels define; the camera frame is fixed to it
        frame = oops.Frame.as_frame('CASSINI_ISS_' + camera + '_FLIPPED')
        try:
            frame.transform_at_time(time)
        except (LookupError, OSError, RuntimeError, ValueError) as err:
            code = _SPICE_ERROR_CODE.search(str(err))
            suffix = f': {code[0]}' if code else ''
            iso = julian.iso_from_tai(julian.tai_from_tdb(time), digits=3)
            raise _NoPointingError(f'no Cassini ISS {camera} pointing at {iso} for '
                                   f'{source}{suffix}') from err

    @staticmethod
    def _get_subfields(dict_):
        """The subfields taken from the metadata of an image.

        Parameters:
            dict_ (dict[str, Any]): The metadata, without PDS3 null values.

        Returns:
            dict[str, Any]: The subfields `dict`, `host`, `instrument`, `detector`,
            `texp` and `sampling`, and each of `filter`, `filter1`, `filter2`,
            `gain_mode`, `label_target`, `observation_id`,
            `spacecraft_clock_start_count` and `spacecraft_clock_stop_count` for which
            `dict_` gives the items.
        """

        params = {'dict': dict_}

        # Merge filter names, ignoring "CL1" and "CL2"
        if 'FILTER_NAME' in dict_:
            filter1, filter2 = dict_['FILTER_NAME']
        elif 'FILTER1_NAME' in dict_ and 'FILTER2_NAME' in dict_:
            filter1 = dict_['FILTER1_NAME']
            filter2 = dict_['FILTER2_NAME']
        else:
            filter1 = None

        if filter1 is not None:
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
            VicarHostError: If the VICAR header cannot be parsed.
        """

        try:
            return VicarImage(filepath, strict=False)
        except VicarError as err:      # raised as the host's own error (#12)
            raise oops.host.VicarHostError(f'{err} in {filepath}') from err

    @staticmethod
    def _parse_fileinfo(fileinfo, *, astrometry, pds3_options):
        """The parsed PDS3 label or VICAR header of an ISS image.

        A PDS3 label is parsed with the method in `pds3_options`. If that is "fast", the
        default, and the fast parser rejects the label or reads a keyword that is not a
        PDS3 name, the label is parsed again: by the fast parser with each quoted value
        that starts on the line after its keyword moved up to that line, then by the
        "strict" parser, then by the "loose" one. A label that every method rejects is
        parsed once more with that repair and with the unterminated DESCRIPTION value of
        some calibrated labels closed, by the fast parser and then by the loose one, or
        with any other method by that method alone.

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
            PdsHostError: If the PDS3 label cannot be parsed.
            VicarHostError: If the VICAR header cannot be parsed.
            WrongHostError: If the file is not a Cassini ISS image.
        """

        method = pds3_options['method']
        if method == 'fast':
            attempts = [('fast', ()), ('fast', (_NEXT_LINE_VALUE,)), ('strict', ()),
                        ('loose', ()), ('fast', _LABEL_REPAIRS),
                        ('loose', _LABEL_REPAIRS)]
        else:
            attempts = [(method, ()), (method, _LABEL_REPAIRS)]

        misread = None
        error = None
        for (method, repairs) in attempts:
            repairs = list(pds3_options.get('repairs', [])) + list(repairs)
            options = pds3_options | {'method': method, 'repairs': repairs}
            try:
                parsed = CassiniISS._read_fileinfo(fileinfo, formats='PV',
                                                   astrometry=astrometry,
                                                   pds3_options=options)
            except oops.host.PdsHostError as err:
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
        raise error

    @staticmethod
    def _merged_dict(label_dict, header_dict):
        """The metadata of an image, merged from its PDS3 label and its VICAR header.

        The header's keywords are first given their PDS3 names (see
        :meth:`_pds3_dict_from_vicar`), and PDS3 null values are removed from both. An
        item that only one source gives takes that source's value. An item that both
        give takes the value that :meth:`_resolve` chooses. The label's keywords come
        first, in label order, then the header's other keywords, in header order.

        Parameters:
            label_dict (dict[str, Any] | None): The keywords of the PDS3 label; None if it
                was not read.
            header_dict (dict[str, Any] | None): The keywords of the VICAR header; None if
                it was not read.

        Returns:
            dict[str, Any]: The merged metadata.
        """

        label = {} if label_dict is None else CassiniISS._without_nulls(label_dict)
        header = {}
        if header_dict is not None:
            header = CassiniISS._pds3_dict_from_vicar(dict(header_dict))
            header = CassiniISS._without_nulls(header)

        merged = {}
        for (key, value) in label.items():
            merged[key] = (CassiniISS._resolve(key, value, header[key]) if key in header
                           else value)
        for (key, value) in header.items():
            if key not in label:
                merged[key] = value
        return merged

    @staticmethod
    def _resolve(key, label_value, header_value):
        """The value of an item that the PDS3 label and the VICAR header both give.

        A number, or list of numbers, that the other source gives to more digits loses
        to it. Otherwise the source named for the item in `_RESOLUTION_RULES` wins, and
        the label wins every item not named there, including those whose two values are
        the same value written differently (a time as text or as a datetime, a number as
        text or as a number).

        Parameters:
            key (str): The PDS3 name of the item.
            label_value (Any): The label's value, not a PDS3 null value.
            header_value (Any): The header's value, not a PDS3 null value.

        Returns:
            Any: The chosen value.
        """

        if CassiniISS._is_rounding_of(label_value, header_value):
            return header_value
        if CassiniISS._is_rounding_of(header_value, label_value):
            return label_value
        if _RESOLUTION_RULES.get(key, _LABEL) == _HEADER:
            return header_value
        return label_value

    @staticmethod
    def _is_rounding_of(short, precise):
        """True if a number, or each number of a list, is another rounded to fewer digits.

        Parameters:
            short (Any): The value that might be rounded.
            precise (Any): The value that might be the precise one.

        Returns:
            bool: True if `short` and `precise` are unequal real numbers and `short` is
            `precise` rounded to the number of significant digits that `short` is written
            with; or if they are unequal lists of the same length, each element of which
            either equals its counterpart or is its rounding in this sense.
        """

        if isinstance(short, list) and isinstance(precise, list):
            return (len(short) == len(precise) and short != precise
                    and all(a == b or CassiniISS._is_rounding_of(a, b)
                            for (a, b) in zip(short, precise, strict=True)))

        if not (isinstance(short, float) and isinstance(precise, float)):
            return False
        if short == precise:
            return False

        # Significant digits of `short`: its zeros after the last nonzero decimal digit
        # are not counted, and those before the decimal point are. The last of them sets
        # the decimal place to which `precise` would have been rounded.
        if short == 0.:
            return False
        mantissa = repr(short).lower().partition('e')[0].lstrip('-')
        (whole, _, fraction) = mantissa.partition('.')
        digits = len((whole + fraction.rstrip('0')).lstrip('0'))
        decimals = max(digits, 1) - 1 - math.floor(math.log10(abs(short)))

        # Half a unit in the last digit, widened slightly so that a value which ends in 5
        # and is rounded either way, as the headers do, still counts
        return abs(precise - short) <= 0.5 * 10.**(-decimals) * (1. + 1.e-9)

    @staticmethod
    def _without_nulls(dict_):
        """A copy of a dictionary without its PDS3 null values.

        Parameters:
            dict_ (dict[str, Any]): The keywords of a label.

        Returns:
            dict[str, Any]: The keywords whose values are not "UNK", "N/A" or "NULL", nor
            a list or set of nothing but these.
        """

        return {key: value for (key, value) in dict_.items()
                if not CassiniISS._is_null(value)}

    @staticmethod
    def _is_null(value):
        """True if a value is a PDS3 null value, or a list or set of nothing but these.

        Parameters:
            value (Any): The value of a keyword.

        Returns:
            bool: True if `value` is "UNK", "N/A" or "NULL", or a non-empty list, tuple or
            set of nothing but these.
        """

        if isinstance(value, str):
            return value.strip() in _PDS3_NULLS
        if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 0:
            return all(CassiniISS._is_null(item) for item in value)
        return False

    @staticmethod
    def _pds3_dict_from_vicar(dict_):
        """The keywords of an ISS VICAR header under their PDS3 label names and meanings.

        Most VICAR headers already use the keywords of the PDS3 label. Those of the early
        cruise images, which carry no INSTRUMENT_HOST_NAME, use older names. For these,
        each keyword is given the name and spelling that the image's PDS3 label gives the
        same value:

        * INSTRUMENT_HOST_NAME is "CASSINI ORBITER", and INSTRUMENT_NAME is the name of
          the camera that INSTRUMENT_ID identifies.
        * FILTER_NAME is the list of FILTER1_NAME and FILTER2_NAME.
        * GAIN_MODE_ID is spelled in full, e.g. "12 ELECTRONS PER DN" for "12 e/DN".
        * START_TIME and STOP_TIME are IMAGE_TIME.
        * SPACECRAFT_CLOCK_STOP_COUNT is IMAGE_NUMBER followed by ".000".

        Each of these is made only when the header gives the keywords it comes from, and
        SPACECRAFT_CLOCK_STOP_COUNT only when IMAGE_NUMBER is not a PDS3 null value.
        * The VICAR SHUTTER_MODE_ID is SHUTTER_STATE_ID, and SHUTTER_EVENT_TYPE is
          SHUTTER_MODE_ID.
        * BIAS is BIAS_STRIP_MEAN, CALIB_LAMP_STATE_FLAG is CALIBRATION_LAMP_STATE_FLAG,
          COMPRESSION_RATIO is INST_CMPRS_RATIO, CONVERSION_TYPE is DATA_CONVERSION_TYPE,
          ENCODING_TYPE is INST_CMPRS_TYPE, EXTENDED_PIXEL_VALUE is DARK_STRIP_MEAN, and
          SENSOR_HEAD_ELEC_TEMP is SENSOR_HEAD_ELEC_TEMPERATURE.

        Every other keyword keeps its VICAR name and value.

        Parameters:
            dict_ (dict[str, Any]): The keywords of the VICAR header.

        Returns:
            dict[str, Any]: `dict_` itself if it already holds INSTRUMENT_HOST_NAME;
            otherwise a new dictionary of its keywords under their PDS3 names.
        """

        if 'INSTRUMENT_HOST_NAME' in dict_:
            return dict_

        pds3 = {_EARLY_VICAR_RENAMES.get(key, key): value
                for (key, value) in dict_.items()}
        pds3['INSTRUMENT_HOST_NAME'] = 'CASSINI ORBITER'
        instrument_id = dict_.get('INSTRUMENT_ID')
        if instrument_id in _EARLY_VICAR_INSTRUMENT_NAMES:
            pds3['INSTRUMENT_NAME'] = _EARLY_VICAR_INSTRUMENT_NAMES[instrument_id]
        if 'FILTER1_NAME' in dict_ and 'FILTER2_NAME' in dict_:
            pds3['FILTER_NAME'] = [dict_['FILTER1_NAME'], dict_['FILTER2_NAME']]
        if isinstance(dict_.get('GAIN_MODE_ID'), str):
            pds3['GAIN_MODE_ID'] = dict_['GAIN_MODE_ID'].replace(' e/DN',
                                                                 ' ELECTRONS PER DN')
        if 'IMAGE_TIME' in dict_:
            pds3['START_TIME'] = dict_['IMAGE_TIME']
            pds3['STOP_TIME'] = dict_['IMAGE_TIME']

        # The image number is the whole seconds of the clock count at the shutter's
        # close, which the labels of these images give with zero ticks
        if 'IMAGE_NUMBER' in dict_ and not CassiniISS._is_null(dict_['IMAGE_NUMBER']):
            pds3['SPACECRAFT_CLOCK_STOP_COUNT'] = f'{dict_["IMAGE_NUMBER"]}.000'
        return pds3

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
