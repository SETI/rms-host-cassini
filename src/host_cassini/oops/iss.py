##########################################################################################
# host_cassini/oops/iss.py
##########################################################################################

import numpy as np
import julian

from filecache import FCPath
from pdsparser import Pds3Label
from vicar     import VicarImage

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
    'UNK'      : 'NONE',
}

# These are star targets and need to be defined as LightSources
_TARGET_STARS = {'FOMALHAUT', 'SPICA'}


class CassiniISS(oops.Host):
    """The Cassini ISS host: Snapshot constructors for NAC and WAC images."""

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

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | Pds3Label | VicarLabel | VicarImage):
                The file path or its parsed content as a Pds3Label, VicarLabel, or
                VicarImage.
            fast_distortion (bool | None, optional): True to use a pre-inverted
                polynomial; False to use a dynamically solved polynomial; None to use a
                :class:`~oops.fov.FlatFOV`.
            return_all_planets (bool, optional): Include kernels for all planets, not just
                Jupiter or Saturn.
            pds3_options (dict[str, Any], optional): Any keyword arguments and their
                values to be passed to the :class:`~pdsparser.Pds3Label` parser.
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
            Snapshot: The observation, with subfields `dict`, `host`, `instrument`,
            `detector`, `sampling`, `filter`, `filter1`, `filter2`, `gain_mode`, `texp`,
            `label_target`, `observation_id`, `spice_kernels`, `filepath`, `basename`,
            `spice_to_frame`, `spice_frame_name`, `spice_frame_id`, `abspath` and
            `image_url` inserted, and `data` unless `astrometry` is True.

        Raises:
            PdsHostError: If the PDS3 label cannot be parsed.
            VicarHostError: If the VICAR header cannot be parsed.
            OSError: If the file cannot be read.
            WrongHostError: If the file is not a Cassini ISS file.
        """

        CassiniISS.initialize()  # define everything the first time through; use defaults
                                 # unless initialize() was called explicitly.

        # Default Pds3Label options are method="fast", sources=False
        pds3_options = _pds3_options_with_defaults(pds3_options)

        fileinfo = CassiniISS._read_fileinfo(fileinfo, formats='PV',
                                             astrometry=astrometry,
                                             pds3_options=pds3_options)

        data = None
        if isinstance(fileinfo, Pds3Label):
            if not astrometry:
                data = oops.host.read_pds3_image_array(fileinfo)
            dict_ = fileinfo.as_dict()
        else:
            if not astrometry:
                if not isinstance(fileinfo, VicarImage):
                    # A VicarImage built from a VicarLabel has no array; re-read the file
                    fileinfo = VicarImage(fileinfo.filepath, strict=False)
                data = fileinfo.data_2d
            dict_ = fileinfo.as_dict(unique=False)

        filepath = FCPath(fileinfo.filepath)

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
                                             overrides=overrides)
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
            subfields `dict`, `host`, `instrument`, `detector`, `sampling`, `filter`,
            `filter1`, `filter2`, `gain_mode`, `texp`, `label_target`, `observation_id`,
            `spice_kernels`, `filepath`, `basename`, `spice_to_frame`, `spice_frame_name`
            and `spice_frame_id` inserted.

        Raises:
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
            obs = CassiniISS._make_snapshot(row_dict, filepath=fpath, data=None,
                                            fast_distortion=fast_distortion,
                                            return_all_planets=return_all_planets,
                                            overrides=overrides)
            snapshots.append(obs)

        return snapshots

    @staticmethod
    def _make_snapshot(dict_, filepath, data, *, fast_distortion, return_all_planets,
                       overrides):
        """A Snapshot built from the label of an ISS image or one row of an ISS index.

        Loads the SPK kernels covering the exposure. Unless `overrides` supplies a frame,
        it also loads the CK kernels and registers the camera frames.

        Parameters:
            dict_ (dict[str, Any]): The PDS3 label, VICAR label or index row. It must
                contain START_TIME, EXPOSURE_DURATION, INSTRUMENT_MODE_ID,
                INSTRUMENT_NAME, TARGET_NAME, OBSERVATION_ID and GAIN_MODE_ID, and either
                FILTER_NAME or both FILTER1_NAME and FILTER2_NAME.
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

        Returns:
            Snapshot: The observation, with the subfields listed in :meth:`from_file`.
        """

        # Note that VICAR and PDS3 dictionaries use the same names.
        tstart = julian.tdb_from_iso(dict_['START_TIME'])
        texp   = dict_['EXPOSURE_DURATION'] / 1000.
        mode   = dict_['INSTRUMENT_MODE_ID']        # "FULL", "SUM2", or "SUM4"
        camera = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'

        # Determine target and light source
        label_target = _TARGET_NAME_REPAIRS.get(dict_['TARGET_NAME'],
                                                dict_['TARGET_NAME'])
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
    def _get_subfields(dict_):
        """Define attributes of the Observation.

        Parameters:
            dict_ (dict): The parameter dictionary.

        Returns:
            params (dict): The selected attributes and their values.
        """

        params = {'dict': dict_}

        # Merge filter names, ignoring "CL1" and "CL2"
        if 'FILTER_NAME' in dict_:
            filter1, filter2 = dict_['FILTER_NAME']
        else:
            filter1 = dict_['FILTER1_NAME']
            filter2 = dict_['FILTER2_NAME']

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

        # gain_mode
        gain_mode = None
        if dict_['GAIN_MODE_ID'][:3] == '215':
            gain_mode = 0
        elif dict_['GAIN_MODE_ID'][:2] == '95':
            gain_mode = 1
        elif dict_['GAIN_MODE_ID'][:2] == '29':
            gain_mode = 2
        elif dict_['GAIN_MODE_ID'][:2] == '12':
            gain_mode = 3

        params['gain_mode'] = gain_mode

        # Other parameters
        params['host'] = 'Cassini'
        params['instrument'] = 'ISS'
        params['detector'] = 'WAC' if 'WIDE' in dict_['INSTRUMENT_NAME'] else 'NAC'
        params['texp'] = dict_['EXPOSURE_DURATION'] / 1000.
        params['sampling'] = dict_['INSTRUMENT_MODE_ID']  # "FULL", "SUM2", or "SUM4"
        params['label_target'] = dict_['TARGET_NAME']
        params['observation_id'] = dict_['OBSERVATION_ID']

        return params

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

        # PDS3 and VICAR use the same names
        return CassiniISS._detect_in_pds3(label)

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
