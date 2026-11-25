##########################################################################################
# tests/oops/test_iss.py
##########################################################################################
"""Tests for the Cassini ISS host, host_cassini.oops.iss."""

import datetime
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, cast

import cspyce
import julian
import numpy as np
import oops
from filecache import FCPath
from pdsparser import Pds3Label
import pytest
import vicar
from conftest import (approx_time, NAC_HALF_FOV_DEG, PIXELS, WAC_HALF_FOV_DEG, FakeSpice,
                      FakeSpiceDB, ik_dict, pds3_lines, tdb_in_month, write_table)

import host_cassini
import host_cassini.oops
import host_cassini.oops.iss as iss
from host_cassini.oops._cassini import _Cassini
from host_cassini.oops.iss      import CassiniISS

START_TIME = '2005-01-01T00:00:00.000'
IMAGE_TIME = '2005-01-01T00:00:01.000'
NAC_NAME = 'IMAGING SCIENCE SUBSYSTEM NARROW ANGLE'
WAC_NAME = 'IMAGING SCIENCE SUBSYSTEM WIDE ANGLE'
INDEX_PATH = 'COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG'


def label_dict(**changes: Any) -> dict[str, Any]:
    """The label keywords that the ISS host reads, as for a NAC image of Saturn."""

    label = {
        'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
        'INSTRUMENT_ID'       : 'ISSNA',
        'INSTRUMENT_NAME'     : NAC_NAME,
        'START_TIME'          : START_TIME,
        'IMAGE_TIME'          : IMAGE_TIME,
        'EXPOSURE_DURATION'   : 1000.0,
        'INSTRUMENT_MODE_ID'  : 'FULL',
        'FILTER_NAME'         : ['CL1', 'GRN'],
        'GAIN_MODE_ID'        : '12 ELECTRONS PER DN',
        'TARGET_NAME'         : 'SATURN',
        'OBSERVATION_ID'      : 'ISS_000SA_TEST001_PRIME',
    }
    label.update(changes)
    return label


OVERRIDE_NAMES = ('target', 'lightsource', 'path', 'frame', 'cadence', 'fov',
                  'calibrations', 'timeshift', 'navigation', 'tracker', 'parallel')


def make_snapshot(dict_: dict[str, Any], *, data: Any = None,
                  fast_distortion: bool | None = True,
                  **overrides: Any) -> oops.observation.Snapshot:
    """A Snapshot from `_make_snapshot`, with every override not given set to None."""

    all_overrides = {name: overrides.get(name) for name in OVERRIDE_NAMES}
    obs: oops.observation.Snapshot = CassiniISS._make_snapshot(
        dict_, image_path=INDEX_PATH, label_path=None, data=data,
        fast_distortion=fast_distortion, return_all_planets=False,
        overrides=all_overrides)
    return obs


def snapshot(**changes: Any) -> oops.observation.Snapshot:
    """A Snapshot made from `label_dict(**changes)` with default options."""

    return make_snapshot(label_dict(**changes))

##########################################################################################
# Distortion polynomials
##########################################################################################


def _evaluate(coefft: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Evaluate a (4,4,2) polynomial at (a,b), returning an array of shape (..., 2)."""

    result = np.zeros(a.shape + (2,))
    for i in range(4):
        for j in range(4):
            result += coefft[i, j] * (a**i * b**j)[..., np.newaxis]
    return result


@pytest.mark.parametrize(('camera', 'tolerance'), [('NAC', 0.0014), ('WAC', 0.054)])
def test_inverse_distortion_undoes_distortion(camera: str, tolerance: float) -> None:
    # Tolerances in pixels are the maximum errors stated beside the inverse coefficients.
    (u, v) = np.meshgrid(np.linspace(-512., 512., 33), np.linspace(-512., 512., 33))
    xy = _evaluate(iss._DISTORTION_COEFF_UV_TO_XY[camera], u, v)
    uv = _evaluate(iss._DISTORTION_COEFF_XY_TO_UV[camera], xy[..., 0], xy[..., 1])
    assert np.abs(uv[..., 0] - u).max() <= tolerance
    assert np.abs(uv[..., 1] - v).max() <= tolerance


@pytest.mark.parametrize(('camera', 'focal_length'), [('NAC', 2002.703),
                                                      ('WAC', 200.7761)])
def test_distortion_scale_is_focal_length_times_pixel_density(camera: str,
                                                              focal_length: float
                                                              ) -> None:
    assert iss._DISTORTION_COEFF_XY_TO_UV[camera][1, 0, 0] == pytest.approx(
        83.33333 * focal_length)


def test_cmatrix_rotation_spins_180_degrees_about_boresight() -> None:
    assert iss._CMATRIX_ROTATION.vals.tolist() == [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]

##########################################################################################
# Registration and detection
##########################################################################################


def test_iss_is_registered_with_oops() -> None:
    assert oops.Host._LOOKUP['Cassini ISS'] is CassiniISS


@pytest.mark.parametrize('module', [host_cassini, host_cassini.oops],
                         ids=['host_cassini', 'oops'])
def test_package_exports(module: Any) -> None:
    assert module.__all__ == ['CassiniISS', 'CassiniUVIS', 'CassiniVIMS']


def test_module_exports() -> None:
    assert iss.__all__ == ['CassiniISS']


@pytest.mark.parametrize('module', [host_cassini, host_cassini.oops],
                         ids=['host_cassini', 'oops'])
def test_package_exposes_host(module: Any) -> None:
    assert module.CassiniISS is iss.CassiniISS


# Prints whether oops and host_cassini.oops.iss are loaded after importing host_cassini,
# and again after its CassiniISS is used.
_LAZY_IMPORT_SCRIPT = """
import sys
names = ('oops', 'host_cassini.oops.iss')
import host_cassini
print(*(name in sys.modules for name in names))
from host_cassini import CassiniISS
print(*(name in sys.modules for name in names))
print(sys.modules['oops'].Host._LOOKUP['Cassini ISS'] is CassiniISS)
"""


def test_host_is_imported_lazily() -> None:
    src = pathlib.Path(__file__).parents[2] / 'src'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(src), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-c', _LAZY_IMPORT_SCRIPT], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == ['False False', 'True True', 'True']


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError,
                       match="module 'host_cassini' has no attribute 'Nope'"):
        _ = host_cassini.Nope


@pytest.mark.parametrize(('label', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'ISSNA'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'ISSWA'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'VIMS'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_ID': 'ISSNA'}, False),
    ({'INSTRUMENT_ID': 'ISSNA'}, False),
    ({}, False),
], ids=['nac', 'wac', 'other-instrument', 'other-host', 'no-host', 'empty'])
def test_detection(label: dict[str, str], expected: bool) -> None:
    assert CassiniISS._detect_in_pds3(label) is expected
    assert CassiniISS._detect_in_vicar(label) is expected


_INDEX_LABEL = ['PDS_VERSION_ID = PDS3', 'OBJECT = IMAGE_INDEX_TABLE',
                'END_OBJECT = IMAGE_INDEX_TABLE', 'END', '']


@pytest.mark.parametrize(('relpath', 'expected'), [
    ('COISS_2001/index/index.lbl', True),
    ('COISS_2999_index.lbl', True),
    ('index/index.lbl', None),
    ('coiss_2001/index/index.lbl', True),
], ids=['coiss-directory', 'coiss-file', 'no-coiss', 'lowercase'])
def test_detection_in_index(tmp_path: pathlib.Path, relpath: str,
                            expected: bool | None) -> None:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\r\n'.join(_INDEX_LABEL))
    assert CassiniISS._detect_in_index(Pds3Label(path)) is expected


def test_detection_in_index_without_a_path() -> None:
    assert CassiniISS._detect_in_index(Pds3Label(_INDEX_LABEL[:-1])) is None


@pytest.mark.parametrize(('row', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_NAME': NAC_NAME}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_NAME': WAC_NAME}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
      'INSTRUMENT_NAME': 'VISUAL AND INFRARED MAPPING SPECTROMETER'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_NAME': NAC_NAME}, False),
    ({}, False),
], ids=['nac', 'wac', 'other-instrument', 'other-host', 'empty'])
def test_detection_in_row(row: dict[str, str], expected: bool) -> None:
    assert CassiniISS._detect_in_row(row) is expected

##########################################################################################
# initialize()
##########################################################################################


def test_initialize_builds_every_fov(initialized_iss: FakeSpiceDB) -> None:
    expected: set[tuple[Any, ...]] = {
        (camera, mode, fast) for camera in ('NAC', 'WAC')
        for mode in ('FULL', 'SUM2', 'SUM4') for fast in (True, False, None)}
    expected |= {(camera, mode) for camera in ('NAC', 'WAC')
                 for mode in ('FULL', 'SUM2', 'SUM4')}
    assert set(CassiniISS._FOVS) == expected


@pytest.mark.parametrize('fast', [True, False, None])
@pytest.mark.parametrize(('mode', 'size'), [('FULL', 1024), ('SUM2', 512), ('SUM4', 256)])
@pytest.mark.parametrize('camera', ['NAC', 'WAC'])
def test_fov_shapes(initialized_iss: FakeSpiceDB, camera: str, mode: str, size: int,
                    fast: bool | None) -> None:
    assert tuple(CassiniISS._FOVS[camera, mode, fast].uv_shape.vals) == (size, size)


@pytest.mark.parametrize(('camera', 'half_fov'), [('NAC', NAC_HALF_FOV_DEG),
                                                  ('WAC', WAC_HALF_FOV_DEG)])
def test_flat_fov_scale(initialized_iss: FakeSpiceDB, camera: str,
                        half_fov: float) -> None:
    scale = np.arctan(np.tan(np.radians(half_fov)) / (PIXELS / 2.))
    uv_scale = CassiniISS._FOVS[camera, 'FULL', None].uv_scale.vals
    assert uv_scale == pytest.approx([scale, scale])


@pytest.mark.parametrize('mode', ['FULL', 'SUM2', 'SUM4'])
def test_two_part_fov_keys_are_flat(initialized_iss: FakeSpiceDB, mode: str) -> None:
    flat = CassiniISS._FOVS['NAC', mode, None]
    assert type(CassiniISS._FOVS['NAC', mode]) is type(flat)
    uv_scale = CassiniISS._FOVS['NAC', mode].uv_scale.vals
    assert uv_scale == pytest.approx(flat.uv_scale.vals)


@pytest.mark.parametrize(('camera', 'tolerance'), [('NAC', 0.0014), ('WAC', 0.054)])
def test_fast_and_solved_distortion_agree(initialized_iss: FakeSpiceDB, camera: str,
                                          tolerance: float) -> None:
    # Tolerances in pixels are the maximum U,V errors stated beside the inverse
    # coefficients, converted to radians at the camera's pixel scale.
    uv = oops.Pair([[0., 0.], [512., 512.], [1024., 1024.], [0., 1024.], [300., 700.]])
    fast = CassiniISS._FOVS[camera, 'FULL', True].xy_from_uv(uv).vals
    solved = CassiniISS._FOVS[camera, 'FULL', False].xy_from_uv(uv).vals
    pixel_scale = iss._DISTORTION_COEFF_UV_TO_XY[camera][1, 0, 0]
    assert np.abs(fast - solved).max() <= tolerance * pixel_scale


def test_fov_center_is_on_axis(initialized_iss: FakeSpiceDB) -> None:
    xy = CassiniISS._FOVS['NAC', 'FULL', True].xy_from_uv(oops.Pair((512., 512.))).vals
    assert xy == pytest.approx([0., 0.], abs=1.e-9)


def test_initialize_reads_instrument_kernel(initialized_iss: FakeSpiceDB) -> None:
    assert CassiniISS._INSTRUMENT_KERNEL is initialized_iss.kernel_dict


def test_initialize_rejects_fov_units_other_than_degrees(
        fake_spicedb: FakeSpiceDB) -> None:
    fake_spicedb.kernel_dict = ik_dict(units='RADIANS')
    with pytest.raises(oops.host.HostError,
                       match="FOV_ANGLE_UNITS 'RADIANS' for CASSINI_ISS_NAC"):
        CassiniISS.initialize()


def test_initialize_loads_default_instruments(initialized_iss: FakeSpiceDB) -> None:
    assert _Cassini.loaded_instruments == ['ISS', 'VIMS', 'CIRS', 'UVIS']


def test_initialize_passes_options_to_mission(fake_spicedb: FakeSpiceDB,
                                              fake_spice: FakeSpice) -> None:
    CassiniISS.initialize(ck='predicted', spk='predicted', planets=[6],
                          asof='2010-02-03', mst_pck=False, irregulars=False)
    assert fake_spicedb.named('select_spk')[0][1]['name'] == 'CAS-SPK-PREDICTED'
    assert fake_spicedb.named('select_ck')[0][1]['name'] == 'CAS-CK-PREDICTED'
    assert fake_spice.solar_system[0][1] == {'asof': '2010-02-03', 'planets': [6],
                                             'mst_pck': False, 'irregulars': False}


def test_initialize_ignores_later_calls(initialized_iss: FakeSpiceDB) -> None:
    fovs = CassiniISS._FOVS
    count = len(initialized_iss.calls)
    CassiniISS.initialize(spk='predicted')
    assert len(initialized_iss.calls) == count
    assert CassiniISS._FOVS is fovs


def test_reset_clears_state(initialized_iss: FakeSpiceDB) -> None:
    CassiniISS._reset()
    assert not CassiniISS._initialized
    assert CassiniISS._FOVS == {}
    assert CassiniISS._INSTRUMENT_KERNEL is None
    assert not _Cassini.initialized

##########################################################################################
# _define_camera_frames()
##########################################################################################


def test_define_camera_frames_builds_both_cameras(fake_spice: FakeSpice) -> None:
    CassiniISS._define_camera_frames()
    assert fake_spice.spice_frames == ['CASSINI_ISS_NAC', 'CASSINI_ISS_WAC']


@pytest.mark.parametrize('camera', ['NAC', 'WAC'])
def test_camera_frame_is_flipped_spice_frame(camera: str) -> None:
    # The fake SpiceFrame is J2000 itself, so the camera frame relative to J2000 is the
    # rotation alone.
    CassiniISS._define_camera_frames()
    frame = oops.Frame.as_frame('CASSINI_ISS_' + camera).wrt(oops.Frame.J2000)
    matrix = np.asarray(frame.transform_at_time(tdb_in_month(100.)).matrix.vals)
    assert matrix.tolist() == iss._CMATRIX_ROTATION.vals.tolist()


def test_camera_flipped_frame_is_registered() -> None:
    CassiniISS._define_camera_frames()
    assert oops.Frame.frame_id_exists('CASSINI_ISS_NAC_FLIPPED')


def test_define_camera_frames_only_once(fake_spice: FakeSpice) -> None:
    CassiniISS._define_camera_frames()
    CassiniISS._define_camera_frames()
    assert fake_spice.spice_frames == ['CASSINI_ISS_NAC', 'CASSINI_ISS_WAC']


def test_define_camera_frames_skips_existing_camera(fake_spice: FakeSpice) -> None:
    oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                       frame_id='CASSINI_ISS_NAC')
    CassiniISS._define_camera_frames()
    assert fake_spice.spice_frames == ['CASSINI_ISS_WAC']

##########################################################################################
# _make_snapshot()
##########################################################################################


uses_iss = pytest.mark.usefixtures('initialized_iss')


@uses_iss
def test_snapshot_ends_at_image_time() -> None:
    tstop = julian.tdb_from_iso(IMAGE_TIME)
    assert snapshot(START_TIME=IMAGE_TIME).time == approx_time((tstop - 1., tstop))


@uses_iss
def test_snapshot_with_shutter_disabled_starts_at_start_time() -> None:
    tstart = julian.tdb_from_iso(IMAGE_TIME)
    obs = snapshot(START_TIME=IMAGE_TIME, SHUTTER_STATE_ID='DISABLED')
    assert obs.time == approx_time((tstart, tstart + 1.))


@uses_iss
def test_snapshot_zero_exposure() -> None:
    tstop = julian.tdb_from_iso(IMAGE_TIME)
    obs = snapshot(EXPOSURE_DURATION=0.)
    assert obs.time == approx_time((tstop, tstop))


@uses_iss
@pytest.mark.parametrize(('name', 'camera'), [(NAC_NAME, 'NAC'), (WAC_NAME, 'WAC')])
def test_snapshot_camera(name: str, camera: str) -> None:
    obs = snapshot(INSTRUMENT_NAME=name)
    assert obs.detector == camera
    assert obs.frame == oops.Frame.as_wayframe('CASSINI_ISS_' + camera)
    assert obs.spice_frame_name == 'CASSINI_ISS_' + camera


@uses_iss
@pytest.mark.parametrize(('name', 'frame_id'), [(NAC_NAME, -82360), (WAC_NAME, -82361)])
def test_snapshot_spice_frame_id(name: str, frame_id: int) -> None:
    assert snapshot(INSTRUMENT_NAME=name).spice_frame_id == frame_id


@uses_iss
def test_snapshot_pointing_subfields() -> None:
    obs = snapshot()
    assert obs.ck_object_id == -82000
    assert obs.ck_sclk_id == -82
    assert obs.spice_frame_is_frozen is False


@uses_iss
def test_ck_object_clock_matches_ck_sclk_id() -> None:
    # cspyce.ckmeta derives the clock of a CK object without any kernel
    obs = snapshot()
    assert cspyce.ckmeta(obs.ck_object_id, 'SCLK') == obs.ck_sclk_id


@uses_iss
@pytest.mark.parametrize('fast', [True, False, None])
@pytest.mark.parametrize('mode', ['FULL', 'SUM2', 'SUM4'])
def test_snapshot_fov(mode: str, fast: bool | None) -> None:
    obs = make_snapshot(label_dict(INSTRUMENT_MODE_ID=mode), fast_distortion=fast)
    assert obs.fov is CassiniISS._FOVS['NAC', mode, fast]
    assert obs.sampling == mode


@uses_iss
def test_snapshot_path_is_cassini() -> None:
    assert snapshot().path == oops.Path.as_waypoint('CASSINI')


@uses_iss
@pytest.mark.parametrize(('filters', 'expected'), [
    (['CL1', 'CL2'], 'CLEAR'),
    (['CL1', 'GRN'], 'GRN'),
    (['RED', 'CL2'], 'RED'),
    (['RED', 'GRN'], 'GRN+RED'),
    (['IR2', 'IR1'], 'IR1+IR2'),
])
def test_snapshot_filter(filters: list[str], expected: str) -> None:
    obs = snapshot(FILTER_NAME=filters)
    assert obs.filter == expected
    assert obs.filter1 == filters[0]
    assert obs.filter2 == filters[1]


@uses_iss
def test_snapshot_filter_from_separate_keywords() -> None:
    dict_ = label_dict(FILTER1_NAME='RED', FILTER2_NAME='CL2')
    del dict_['FILTER_NAME']
    assert make_snapshot(dict_).filter == 'RED'


@uses_iss
def test_snapshot_without_filters() -> None:
    dict_ = label_dict()
    del dict_['FILTER_NAME']
    assert not hasattr(make_snapshot(dict_), 'filter')


@uses_iss
def test_snapshot_filter_with_a_missing_element() -> None:
    assert not hasattr(snapshot(FILTER_NAME=['CL1', None]), 'filter')


@uses_iss
@pytest.mark.parametrize(('gain', 'expected'), [
    ('215 ELECTRONS PER DN', 0),
    ('95 ELECTRONS PER DN', 1),
    ('29 ELECTRONS PER DN', 2),
    ('12 ELECTRONS PER DN', 3),
])
def test_snapshot_gain_mode(gain: str, expected: int) -> None:
    assert snapshot(GAIN_MODE_ID=gain).gain_mode == expected


@uses_iss
def test_snapshot_unknown_gain_mode_is_absent() -> None:
    assert not hasattr(snapshot(GAIN_MODE_ID='UNKNOWN'), 'gain_mode')


@uses_iss
@pytest.mark.parametrize(('label_target', 'target'), [
    ('SATURN', 'SATURN'),
    ('ERRIAPO', 'ERRIAPUS'),
    ('K07S4', 'AEGAEON'),
    ('S12_2004', 'S/2004_S_12'),
    ('DARK SKY', 'NONE'),
    ('UNK', 'NONE'),
])
def test_snapshot_target(label_target: str, target: str) -> None:
    obs = snapshot(TARGET_NAME=label_target)
    assert obs.target == target
    assert obs.label_target == label_target
    assert obs.lightsource == 'SUN'


@uses_iss
def test_snapshot_without_target() -> None:
    dict_ = label_dict()
    del dict_['TARGET_NAME']
    obs = make_snapshot(dict_)
    assert obs.target == 'NONE'
    assert not hasattr(obs, 'label_target')


@uses_iss
def test_snapshot_ring_target() -> None:
    obs = snapshot(OBSERVATION_ID='ISS_053RI_PHOTOMDRK002_PRIME')
    assert obs.target == 'SATURN_RING_PLANE'
    assert obs.observation_id == 'ISS_053RI_PHOTOMDRK002_PRIME'


@uses_iss
@pytest.mark.parametrize('star', ['FOMALHAUT', 'SPICA'])
def test_snapshot_star_target(star: str) -> None:
    obs = snapshot(TARGET_NAME=star)
    assert obs.target == 'NONE'
    assert obs.lightsource is oops.lightsource.star_lookup(star)


@uses_iss
def test_snapshot_subfields() -> None:
    dict_ = label_dict(SPACECRAFT_CLOCK_START_COUNT='1.000',
                       SPACECRAFT_CLOCK_STOP_COUNT='2.000')
    obs = make_snapshot(dict_)
    assert obs.host == 'Cassini'
    assert obs.instrument == 'ISS'
    assert obs.texp == 1.
    assert obs.dict is dict_
    assert obs.spacecraft_clock_start_count == '1.000'
    assert obs.spacecraft_clock_stop_count == '2.000'
    assert obs.spice_to_frame is iss._CMATRIX_ROTATION
    assert obs.calibrations == []


@uses_iss
def test_snapshot_index_row_path_attributes() -> None:
    obs = snapshot()
    assert obs.filepath == INDEX_PATH
    assert obs.basename == 'N1484506648_1.IMG'
    assert not hasattr(obs, 'image_url')


@uses_iss
def test_snapshot_data() -> None:
    data = np.zeros((PIXELS, PIXELS), dtype='uint8')
    obs = make_snapshot(label_dict(), data=data)
    assert obs.data is data


@uses_iss
def test_snapshot_without_data() -> None:
    assert not hasattr(snapshot(), 'data')


@uses_iss
def test_snapshot_reports_used_kernels() -> None:
    assert snapshot().spice_kernels == ['cas_v43.tf', 'cas_iss_v10.ti']


@uses_iss
def test_snapshot_queries_kernels_for_its_time(initialized_iss: FakeSpiceDB) -> None:
    obs = snapshot()
    assert initialized_iss.named('used_basenames')[0][1]['time'] == obs.time
    assert initialized_iss.named('used_basenames')[0][1]['inst'] == 'iss'
    assert initialized_iss.named('used_basenames')[0][1]['types'] is None


@uses_iss
def test_snapshot_loads_kernels_for_its_month() -> None:
    month = int((julian.tdb_from_iso(START_TIME) - _Cassini.TDB0) // _Cassini.DTDB)
    snapshot()
    assert _Cassini.SPK_LOADED[month]
    assert _Cassini.CK_LOADED[month]


@uses_iss
def test_snapshot_overrides() -> None:
    target = 'TITAN'
    calibrations = ['calibration']
    obs = make_snapshot(label_dict(), target=target, calibrations=calibrations)
    assert obs.target == target
    assert obs.calibrations is calibrations


@uses_iss
def test_snapshot_cadence_override() -> None:
    cadence = oops.cadence.SnapCadence(100., 2.)
    obs = make_snapshot(label_dict(), cadence=cadence)
    assert obs.cadence is cadence
    assert obs.time == (100., 102.)


@pytest.fixture
def custom_frame(initialized_iss: FakeSpiceDB) -> oops.Frame:
    return oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                              frame_id='TEST_CMATRIX')


def test_custom_frame_is_used(custom_frame: oops.Frame) -> None:
    obs = make_snapshot(label_dict(), frame=custom_frame)
    assert obs.frame == custom_frame.wayframe


def test_custom_frame_loads_no_cks(custom_frame: oops.Frame) -> None:
    make_snapshot(label_dict(), frame=custom_frame)
    assert not _Cassini.CK_LOADED.any()
    assert _Cassini.SPK_LOADED.any()


def test_custom_frame_defines_no_camera_frames(custom_frame: oops.Frame) -> None:
    make_snapshot(label_dict(), frame=custom_frame)
    assert not oops.Frame.frame_id_exists('CASSINI_ISS_NAC')


def test_custom_frame_reports_no_cks(custom_frame: oops.Frame,
                                     initialized_iss: FakeSpiceDB) -> None:
    make_snapshot(label_dict(), frame=custom_frame)
    assert 'CK' not in initialized_iss.named('used_basenames')[0][1]['types']

##########################################################################################
# from_file()
##########################################################################################


def write_vicar_image(directory: pathlib.Path, *, lines: int = 256, bands: int = 1,
                      binary_header: bool = False, keywords: dict[str, Any] | None = None,
                      name: str = 'N0000000001_1.IMG', **changes: Any) -> pathlib.Path:
    """Write a VICAR image with an ISS header; return its path.

    The image is an 8-bit ramp. `binary_header` adds a one-record binary header after the
    VICAR label, as the archive's calibrated files have. The header holds `keywords`, or
    by default `label_dict(INSTRUMENT_MODE_ID='SUM4', **changes)`; a keyword whose value
    is None is left out.
    """

    image = np.arange(bands * lines * lines, dtype='uint32') % 256
    image = image.reshape((bands, lines, lines) if bands > 1 else (lines, lines))
    header = bytes(lines) if binary_header else None
    vic = vicar.VicarImage(array=image.astype('uint8'), binheader=header)
    if keywords is None:
        keywords = label_dict(**({'INSTRUMENT_MODE_ID': 'SUM4'} | changes))
    for (key, value) in keywords.items():
        if value is not None:
            vic[key] = value
    path = directory / name
    vic.write_file(path)
    return path


def write_pds3_image(directory: pathlib.Path, *, lines: int = 256, label_lines: int = 0,
                     binary_header: bool = False, pointer_error: int = 0,
                     header: dict[str, Any] | None = None,
                     extra_lines: list[str] | None = None,
                     **changes: Any) -> pathlib.Path:
    """Write a VICAR image and a detached PDS3 label for it; return the label's path.

    Both labels hold `label_dict(INSTRUMENT_MODE_ID='SUM4', **changes)`, except that the
    VICAR header takes the keywords in `header` too, leaving out those whose value is
    None. The label's ^IMAGE pointer is `pointer_error` records away from the array, as
    the archive's calibrated labels are, and its IMAGE object gives `label_lines` lines
    and samples, by default those of the array. `extra_lines` are added to the label text
    as they are, after the keywords.
    """

    keywords = label_dict(**({'INSTRUMENT_MODE_ID': 'SUM4'} | changes))
    image_path = write_vicar_image(directory, lines=lines, binary_header=binary_header,
                                   keywords=keywords | (header or {}))
    vic = vicar.VicarLabel(image_path)
    recsize = cast(int, vic['RECSIZE'])
    record = (cast(int, vic['LBLSIZE']) // recsize + cast(int, vic['NLB']) + 1
              + pointer_error)
    label_lines = label_lines or lines

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {recsize}', f'FILE_RECORDS = {lines + record}',
            f'^IMAGE = ("{image_path.name}", {record})']
    text += pds3_lines({key: value for (key, value) in keywords.items()
                        if value is not None})
    text += extra_lines or []
    text += ['OBJECT = IMAGE', f'  LINES = {label_lines}',
             f'  LINE_SAMPLES = {label_lines}', '  SAMPLE_BITS = 8',
             '  SAMPLE_TYPE = MSB_UNSIGNED_INTEGER', 'END_OBJECT = IMAGE', 'END', '']
    path = directory / 'N0000000001_1.LBL'
    path.write_text('\r\n'.join(text), newline='')
    return path


uses_iss_file = pytest.mark.usefixtures('initialized_iss')


@uses_iss_file
def test_from_file_pds3(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path))
    assert obs.data.shape == (256, 256)
    assert obs.data[1, 2] == 258 % 256
    assert obs.fov is CassiniISS._FOVS['NAC', 'SUM4', True]


@uses_iss_file
def test_from_file_pds3_reads_the_array_the_vicar_header_places(
        tmp_path: pathlib.Path) -> None:
    # A calibrated label's ^IMAGE pointer is one record short of the array
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, binary_header=True,
                                                pointer_error=-1))
    assert obs.data[1, 2] == 258 % 256


@uses_iss_file
def test_from_file_pds3_path_attributes(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    obs = CassiniISS.from_file(path)
    image = tmp_path / 'N0000000001_1.IMG'
    assert obs.filepath == FCPath(image)
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.image_url == FCPath(image).absolute().as_posix()
    assert obs.image_abspath == image.resolve()
    assert obs.label_url == FCPath(path).absolute().as_posix()
    assert obs.label_abspath == path.resolve()
    assert not hasattr(obs, 'abspath')


@uses_iss_file
def test_from_file_pds3_astrometry(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    (tmp_path / 'N0000000001_1.IMG').unlink()
    obs = CassiniISS.from_file(path, astrometry=True)
    assert not hasattr(obs, 'data')
    assert not hasattr(obs, 'image_abspath')
    assert not hasattr(obs, 'header_dict')
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.label_abspath == path.resolve()


@uses_iss_file
def test_from_file_pds3_gives_both_raw_sources(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    obs = CassiniISS.from_file(path)
    assert obs.label_dict == Pds3Label(path, method='strict', sources=False).as_dict()
    assert obs.header_dict == vicar.VicarLabel(tmp_path / 'N0000000001_1.IMG').as_dict(
        unique=False)


@uses_iss_file
def test_from_file_pds3_array_must_match_the_label(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, label_lines=128)
    image = tmp_path / 'N0000000001_1.IMG'
    with pytest.raises(oops.host.HostFormatError,
                       match=re.escape(f'{FCPath(image)}: the array is 256 x 256 but '
                                       'the label gives 128 x 128')):
        CassiniISS.from_file(path)


@uses_iss_file
def test_from_file_vicar(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path, INSTRUMENT_NAME=WAC_NAME)
    obs = CassiniISS.from_file(path)
    assert obs.data.shape == (256, 256)
    assert obs.data[1, 2] == 258 % 256
    assert obs.detector == 'WAC'
    assert obs.image_abspath == path.resolve()
    assert not hasattr(obs, 'label_dict')
    assert not hasattr(obs, 'label_url')


@uses_iss_file
def test_from_file_vicar_astrometry(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_vicar_image(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('the VICAR image array was read')

    monkeypatch.setattr(vicar.VicarImage, '__init__', fail)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert not hasattr(obs, 'data')
    assert obs.image_abspath == path.resolve()
    assert obs.filter == 'GRN'


@uses_iss_file
def test_from_file_vicar_image_given(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    vic = cast(vicar.VicarImage, vicar.VicarImage.from_file(path, strict=False))
    vic.filepath = path
    obs = CassiniISS.from_file(vic)
    assert obs.data[1, 2] == 258 % 256


@uses_iss_file
def test_from_file_vicar_label_given_reads_data(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(vicar.VicarLabel(write_vicar_image(tmp_path)))
    assert obs.data[1, 2] == 258 % 256


@uses_iss_file
def test_from_file_vicar_label_given_astrometry(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(vicar.VicarLabel(write_vicar_image(tmp_path)),
                               astrometry=True)
    assert not hasattr(obs, 'data')


@uses_iss_file
def test_from_file_pds3_label_given(tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_image(tmp_path), method='strict')
    obs = CassiniISS.from_file(label, astrometry=True)
    assert obs.filter == 'GRN'


@uses_iss_file
def test_from_file_vicar_header_is_read_loosely(tmp_path: pathlib.Path) -> None:
    # The strict VICAR reader refuses a keyword longer than 32 characters, which the
    # writer refuses too, so a 32-character one is lengthened in the file
    path = write_vicar_image(tmp_path, UNEVEN_BIT_WEIGHT_CORRECTION_FLX=1)
    path.write_bytes(path.read_bytes().replace(b'CORRECTION_FLX=1  ',
                                               b'CORRECTION_FLAG=1 '))
    obs = CassiniISS.from_file(path)
    assert obs.header_dict['UNEVEN_BIT_WEIGHT_CORRECTION_FLAG'] == 1


@uses_iss_file
def test_from_file_select(tmp_path: pathlib.Path) -> None:
    result = CassiniISS.from_file(write_vicar_image(tmp_path), select=slice(None))
    assert [obs.instrument for obs in result] == ['ISS']


@uses_iss_file
def test_from_file_select_index(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path), select=0)
    assert obs.instrument == 'ISS'


@uses_iss_file
def test_from_file_rejects_unknown_keywords(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'enclose'"):
        CassiniISS.from_file(write_vicar_image(tmp_path), enclose=True)


def test_from_file_initializes_iss(tmp_path: pathlib.Path) -> None:
    CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert CassiniISS._initialized


@uses_iss_file
@pytest.mark.parametrize('kind', ['pds3', 'vicar'])
def test_host_from_file_dispatches_to_iss(tmp_path: pathlib.Path, kind: str) -> None:
    path = (write_pds3_image(tmp_path) if kind == 'pds3'
            else write_vicar_image(tmp_path))
    obs = oops.Host.from_file(path, astrometry=True)
    assert obs.instrument == 'ISS'


@uses_iss_file
def test_from_file_rejects_another_hosts_label(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, INSTRUMENT_HOST_NAME='GALILEO ORBITER')
    with pytest.raises(oops.host.WrongHostError, match='not a Cassini ISS PDS3 label'):
        CassiniISS.from_file(path, astrometry=True)


@uses_iss_file
def test_from_file_rejects_another_hosts_image(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path, INSTRUMENT_HOST_NAME='GALILEO ORBITER')
    with pytest.raises(oops.host.WrongHostError, match='not a Cassini ISS VICAR file'):
        CassiniISS.from_file(path, astrometry=True)

##########################################################################################
# Failures to read or interpret a file
##########################################################################################


@uses_iss_file
def test_from_file_truncated_image(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(oops.host.HostFormatError,
                       match=re.escape(f'{path}: ValueError: cannot reshape')):
        CassiniISS.from_file(path)


@uses_iss_file
def test_from_file_image_of_several_bands(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path, bands=2)
    with pytest.raises(oops.host.VicarHostError,
                       match=re.escape(f'{path}: VICAR array shape (2, 256, 256) is not '
                                       '2-D')):
        CassiniISS.from_file(path)


@uses_iss_file
def test_from_file_label_without_a_needed_item(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, EXPOSURE_DURATION=None)
    with pytest.raises(oops.host.HostFormatError,
                       match=re.escape(f"{path}: KeyError: 'EXPOSURE_DURATION'")):
        CassiniISS.from_file(path, astrometry=True)


@uses_iss_file
def test_from_file_label_that_cannot_be_parsed(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, extra_lines=['NOT A STATEMENT'])
    with pytest.raises(oops.host.PdsHostError,
                       match=re.escape(f'cannot parse the PDS3 label {FCPath(path)}')):
        CassiniISS.from_file(path, astrometry=True)


@uses_iss_file
def test_from_file_label_value_that_cannot_be_converted(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, extra_lines=['TEST_TIME = 2005-02-30T00:00:00'])
    with pytest.raises(oops.host.PdsHostError,
                       match=re.escape(f'cannot parse the PDS3 label {path}: day is out '
                                       'of range for month')):
        CassiniISS.from_file(path, astrometry=True)


@uses_iss_file
def test_from_file_missing_image_file(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    (tmp_path / 'N0000000001_1.IMG').unlink()
    with pytest.raises(FileNotFoundError):
        CassiniISS.from_file(path)


@uses_iss_file
def test_from_file_passes_a_spice_error(tmp_path: pathlib.Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    def scs2e(code: int, count: str) -> float:
        raise KeyError('SPICE(KERNELVARNOTFOUND) -- scs2e -- no clock kernel')

    monkeypatch.setattr(cspyce, 'scs2e', scs2e)
    path = write_vicar_image(tmp_path, IMAGE_NUMBER='1294562651')
    with pytest.raises(KeyError, match=re.escape('SPICE(KERNELVARNOTFOUND)')):
        CassiniISS.from_file(path)

##########################################################################################
# Metadata: dict, label_dict and header_dict
##########################################################################################


@uses_iss_file
def test_from_file_vicar_gives_its_header(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    obs = CassiniISS.from_file(path)
    assert obs.header_dict == vicar.VicarLabel(path).as_dict(unique=False)


@uses_iss_file
def test_from_file_label_dict_keeps_parser_keys(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert obs.label_dict['START_TIME_fmt'] == '2005-01-01T00:00:00.000'


@uses_iss_file
def test_from_file_dict_keeps_parser_keys(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert obs.dict['START_TIME_fmt'] == '2005-01-01T00:00:00.000'


@uses_iss_file
def test_from_file_null_text_is_absent(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, TARGET_DESC='N/A'),
                               astrometry=True)
    assert 'TARGET_DESC' not in obs.dict
    assert obs.label_dict['TARGET_DESC'] == 'N/A'


@uses_iss_file
def test_from_file_fill_value_is_absent(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, DETECTOR_TEMPERATURE=-999.),
                               astrometry=True)
    assert 'DETECTOR_TEMPERATURE' not in obs.dict


@uses_iss_file
def test_from_file_fill_element_is_none(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path,
                                                OPTICS_TEMPERATURE=[0.5, -999.]),
                               astrometry=True)
    assert obs.dict['OPTICS_TEMPERATURE'] == [0.5, None]


@uses_iss_file
def test_from_file_fill_value_is_kept_elsewhere(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, CENTER_RADIUS=-999.),
                               astrometry=True)
    assert obs.dict['CENTER_RADIUS'] == -999.


@uses_iss_file
def test_from_file_merges_both_sources(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, header={'HEADER_ONLY': 'H'}, LABEL_ONLY='L')
    obs = CassiniISS.from_file(path)
    assert obs.dict['LABEL_ONLY'] == 'L'
    assert obs.dict['HEADER_ONLY'] == 'H'


@uses_iss_file
@pytest.mark.parametrize(('item', 'label', 'header', 'expected'), [
    ('METHOD_DESC', 'A B', 'A;B', 'A;B'),
    ('MISSION_PHASE_NAME', 'SCIENCE CRUISE', 'CHECK_OUT', 'CHECK OUT'),
    ('DESCRIPTION', 'CALIBRATED', 'RAW', 'CALIBRATED'),
    ('DESCRIPTION', 'N/A', 'RAW', 'RAW'),
], ids=['header-wins', 'header-wins-in-label-form', 'label-wins', 'null-falls-through'])
def test_from_file_resolves_differences(tmp_path: pathlib.Path, item: str, label: str,
                                        header: str, expected: str) -> None:
    changes: dict[str, Any] = {item: label}
    path = write_pds3_image(tmp_path, header={item: header}, **changes)
    assert CassiniISS.from_file(path).dict[item] == expected


@uses_iss_file
def test_from_file_header_time_has_the_label_type(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path, START_TIME='2005-016T13:01:59.864Z')
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.dict['START_TIME'] == datetime.datetime(2005, 1, 16, 13, 1, 59, 864000)


@pytest.mark.parametrize(('item', 'header', 'expected'), [
    ('START_TIME', '2005-016T13:01:59.864Z', datetime.datetime(2005, 1, 16, 13, 1, 59,
                                                               864000)),
    ('IMAGE_TIME', '2000-157T20:23:28.509', datetime.datetime(2000, 6, 5, 20, 23, 28,
                                                              509000)),
    ('PRODUCT_CREATION_TIME', '2005-01-17T09:46:41Z', datetime.datetime(2005, 1, 17, 9,
                                                                        46, 41)),
    ('EARTH_RECEIVED_START_TIME', '0000-000T00:00:00.000Z', '0000-000T00:00:00.000Z'),
    ('TARGET_DESC', 'Zeta 1Z', 'Zeta 1Z'),
    ('IMAGE_NUMBER', 1484573295, '1484573295'),
    ('IMAGE_OBSERVATION_TYPE', 'SCIENCE', {'SCIENCE'}),
    ('IMAGE_OBSERVATION_TYPE', ['SCIENCE', 'SUPPORT'], {'SCIENCE', 'SUPPORT'}),
    ('OPTICS_TEMPERATURE', 0.712693, [0.712693, None]),
    ('OPTICS_TEMPERATURE', [0.79789, 1.90571], [0.79789, 1.90571]),
    ('MISSION_PHASE_NAME', 'APPROACH_SCIENCE', 'APPROACH SCIENCE'),
], ids=['time', 'time-without-z', 'calendar-time', 'day-000', 'not-a-time',
        'image-number', 'observation-type', 'observation-types', 'optics-temperature',
        'optics-temperatures', 'mission-phase'])
def test_vicar_header_takes_pds3_forms(item: str, header: Any, expected: Any) -> None:
    header_dict = label_dict(**{item: header})
    assert CassiniISS._pds3_dict_from_vicar(header_dict)[item] == expected


# An early cruise VICAR header, as in COISS_1001/data/1338756756_1338935702/W1338928383_1.
# The keywords whose names or values differ from the PDS3 label's are listed with what the
# label says.
_EARLY_TIME = '2000-157T20:23:28.509'
_EARLY_DATETIME = datetime.datetime(2000, 6, 5, 20, 23, 28, 509000)
_EARLY_VICAR = {
    'MISSION_NAME'         : 'CASSINI',
    'INSTRUMENT_ID'        : 'ISSWA',      # INSTRUMENT_NAME = WAC_NAME
    'IMAGE_NUMBER'         : 1338928383,
    'IMAGE_TIME'           : _EARLY_TIME,  # START_TIME = STOP_TIME = IMAGE_TIME
    'INSTRUMENT_MODE_ID'   : 'SUM4',
    'FILTER1_NAME'         : 'CL1',        # FILTER_NAME = ("CL1", "GRN")
    'FILTER2_NAME'         : 'GRN',
    'EXPOSURE_DURATION'    : 500.0,
    'GAIN_MODE_ID'         : '12 e/DN',    # "12 ELECTRONS PER DN"
    'TARGET_NAME'          : 'SATURN',
    'OBSERVATION_ID'       : 'ICO2_DARK_CURRENT_2',
    'SHUTTER_MODE_ID'      : 'DISABLED',   # SHUTTER_STATE_ID
    'SHUTTER_EVENT_TYPE'   : 'BOTSIM',     # SHUTTER_MODE_ID
    'BIAS'                 : 55.08,        # BIAS_STRIP_MEAN
    'COMPRESSION_RATIO'    : 3.363,        # INST_CMPRS_RATIO
    'CALIB_LAMP_STATE_FLAG': 'OFF',        # CALIBRATION_LAMP_STATE_FLAG
    'CONVERSION_TYPE'      : '8LSB',       # DATA_CONVERSION_TYPE
    'ENCODING_TYPE'        : 'LOSSLESS',   # INST_CMPRS_TYPE
    'EXTENDED_PIXEL_VALUE' : 55.09,        # DARK_STRIP_MEAN
    'SENSOR_HEAD_ELEC_TEMP': 5.27707,      # SENSOR_HEAD_ELEC_TEMPERATURE
}
_EARLY_PDS3 = {
    'MISSION_NAME'                : 'CASSINI',
    'INSTRUMENT_ID'               : 'ISSWA',
    'IMAGE_NUMBER'                : '1338928383',
    'IMAGE_TIME'                  : _EARLY_DATETIME,
    'INSTRUMENT_MODE_ID'          : 'SUM4',
    'EXPOSURE_DURATION'           : 500.0,
    'GAIN_MODE_ID'                : '12 ELECTRONS PER DN',
    'TARGET_NAME'                 : 'SATURN',
    'OBSERVATION_ID'              : 'ICO2_DARK_CURRENT_2',
    'SHUTTER_STATE_ID'            : 'DISABLED',
    'SHUTTER_MODE_ID'             : 'BOTSIM',
    'INSTRUMENT_HOST_NAME'        : 'CASSINI ORBITER',
    'INSTRUMENT_NAME'             : WAC_NAME,
    'FILTER_NAME'                 : ['CL1', 'GRN'],
    'START_TIME'                  : _EARLY_DATETIME,
    'STOP_TIME'                   : _EARLY_DATETIME,
    'BIAS_STRIP_MEAN'             : 55.08,
    'INST_CMPRS_RATIO'            : 3.363,
    'CALIBRATION_LAMP_STATE_FLAG' : 'OFF',
    'DATA_CONVERSION_TYPE'        : '8LSB',
    'INST_CMPRS_TYPE'             : 'LOSSLESS',
    'DARK_STRIP_MEAN'             : 55.09,
    'SENSOR_HEAD_ELEC_TEMPERATURE': 5.27707,
}


@pytest.mark.parametrize(('label', 'expected'), [
    ({'MISSION_NAME': 'CASSINI', 'INSTRUMENT_ID': 'ISSNA'}, True),
    ({'MISSION_NAME': 'CASSINI', 'INSTRUMENT_ID': 'VIMS'}, False),
    ({'MISSION_NAME': 'GALILEO', 'INSTRUMENT_ID': 'ISSNA'}, False),
], ids=['nac', 'other-instrument', 'other-mission'])
def test_detection_in_early_vicar_header(label: dict[str, str], expected: bool) -> None:
    assert CassiniISS._detect_in_vicar(label) is expected


def test_early_vicar_header_takes_pds3_names() -> None:
    assert CassiniISS._pds3_dict_from_vicar(dict(_EARLY_VICAR)) == _EARLY_PDS3


@pytest.mark.parametrize(('missing', 'absent'), [
    ('FILTER1_NAME', 'FILTER_NAME'),
    ('GAIN_MODE_ID', 'GAIN_MODE_ID'),
    ('IMAGE_TIME', 'START_TIME'),
    ('INSTRUMENT_ID', 'INSTRUMENT_NAME'),
])
def test_early_vicar_header_without_a_keyword(missing: str, absent: str) -> None:
    header = {key: value for (key, value) in _EARLY_VICAR.items() if key != missing}
    assert absent not in CassiniISS._pds3_dict_from_vicar(header)


@uses_iss_file
def test_from_file_early_vicar_header(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path, keywords=_EARLY_VICAR))
    tstart = julian.tdb_from_iso(_EARLY_TIME)
    assert obs.time == approx_time((tstart, tstart + 0.5))
    assert obs.detector == 'WAC'
    assert obs.filter == 'GRN'
    assert obs.gain_mode == 3
    assert obs.dict['SHUTTER_MODE_ID'] == 'BOTSIM'
    assert obs.header_dict['SHUTTER_MODE_ID'] == 'DISABLED'


@uses_iss_file
def test_from_file_early_vicar_header_with_the_shutter_enabled(
        tmp_path: pathlib.Path) -> None:
    keywords = _EARLY_VICAR | {'SHUTTER_MODE_ID': 'ENABLED'}
    obs = CassiniISS.from_file(write_vicar_image(tmp_path, keywords=keywords))
    tstop = julian.tdb_from_iso(_EARLY_TIME)
    assert obs.time == approx_time((tstop - 0.5, tstop))

##########################################################################################
# Clock counts of the early cruise images
##########################################################################################

# The test clock reads a count as seconds, so the start count computed from the stop
# count and an exposure of 2000 ms is two seconds before it.
_EARLY_COUNTS: dict[str, Any] = {'IMAGE_NUMBER': '1294562651',
                                 'EXPOSURE_DURATION': 2000.,
                                 'SPACECRAFT_CLOCK_START_COUNT': '1294560651.000',
                                 'SPACECRAFT_CLOCK_STOP_COUNT': '1294562651.000'}


@uses_iss_file
def test_from_file_computes_the_early_cruise_counts(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, **_EARLY_COUNTS),
                               astrometry=True)
    assert obs.spacecraft_clock_start_count == '1294562649.000'
    assert obs.dict['SPACECRAFT_CLOCK_STOP_COUNT'] == '1294562651.000'
    assert obs.label_dict['SPACECRAFT_CLOCK_START_COUNT'] == '1294560651.000'


@uses_iss_file
def test_from_file_early_cruise_counts_with_the_shutter_disabled(
        tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, SHUTTER_STATE_ID='DISABLED', **_EARLY_COUNTS)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.spacecraft_clock_start_count == '1294562651.000'


@uses_iss_file
def test_from_file_keeps_a_later_start_count(tmp_path: pathlib.Path) -> None:
    counts = _EARLY_COUNTS | {'IMAGE_NUMBER': '1347928997'}
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, **counts), astrometry=True)
    assert obs.spacecraft_clock_start_count == '1294560651.000'

##########################################################################################
# Label-text corrections
##########################################################################################


@pytest.mark.parametrize(('text', 'expected'), [
    ('  ^DESCRIPTION = "../../label/vicar2.txt"\n  ^STRUCTURE = "../../label/t.fmt"\n',
     '  ^DESCRIPTION = "vicar2.txt"\n  ^STRUCTURE = "t.fmt"\n'),
    ('PRODUCT_CREATION_TIME = 2019-06- 9T09:58:43\n',
     'PRODUCT_CREATION_TIME = 2019-06-09T09:58:43\n'),
    ('MISSING_LINES = N/A\n', 'MISSING_LINES = "N/A"\n'),
    ('DESCRIPTION = "\nCAL\n"\n      header. Finalized using N1_2.IMG."\nX = 1\n',
     'DESCRIPTION = "\nCAL\n"\nX = 1\n'),
    ('DESCRIPTION = "\nCAL\n"\n    "Extended header missing.\n    header."\nX = 1\n',
     'DESCRIPTION = "\nCAL\n    Extended header missing.\n    header."\nX = 1\n'),
    ('EARTH_RECEIVED_START_TIME = 0000-000T00:00:00.000\n',
     'EARTH_RECEIVED_START_TIME = "0000-000T00:00:00.000"\n'),
], ids=['pointer', 'creation-time', 'missing-lines', 'orphan-line', 'extended-header',
        'day-000'])
def test_correct_label_text(text: str, expected: str) -> None:
    assert CassiniISS._correct_label_text(text, FCPath('x.LBL')) == expected


def test_correct_label_text_leaves_a_good_label() -> None:
    text = '\n'.join(pds3_lines(label_dict())) + '\nEND\n'
    assert CassiniISS._correct_label_text(text, FCPath('x.LBL')) == text


@uses_iss_file
def test_from_file_corrects_a_label_that_strict_refuses(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, extra_lines=['MISSING_LINES = N/A'])
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.label_dict['MISSING_LINES'] == 'N/A'


@uses_iss_file
def test_from_file_parses_with_strict(tmp_path: pathlib.Path) -> None:
    # The strict method refuses an unquoted N/A that it is given without the correction
    path = write_pds3_image(tmp_path, extra_lines=['X = N/A'])
    with pytest.raises(oops.host.PdsHostError, match="Expected 'END', found 'X'"):
        CassiniISS.from_file(path, astrometry=True)

##########################################################################################
# from_index()
##########################################################################################

_NAC_FILE = 'data/1484506648_1484573295/N1484506648_1.IMG'
_WAC_FILE = 'data/1484506648_1484573295/W1484506649_1.IMG'


def index_columns(**changes: list[Any]) -> dict[str, list[Any]]:
    """The columns of a two-row ISS index, with `changes` replacing columns."""

    columns: dict[str, list[Any]] = {
        'VOLUME_ID'                   : ['COISS_2009', 'COISS_2009'],
        'FILE_SPECIFICATION_NAME'     : [_NAC_FILE, _WAC_FILE],
        'INSTRUMENT_HOST_NAME'        : ['CASSINI ORBITER', 'CASSINI ORBITER'],
        'INSTRUMENT_NAME'             : [NAC_NAME, WAC_NAME],
        'START_TIME'                  : ['2005-01-15T18:30:00.000',
                                         '2005-01-15T18:31:00.000'],
        'IMAGE_TIME'                  : ['2005-01-15T18:30:01.000',
                                         '2005-01-15T18:31:00.500'],
        'SHUTTER_STATE_ID'            : ['ENABLED', 'ENABLED'],
        'EXPOSURE_DURATION'           : [1000.0, 500.0],
        'INSTRUMENT_MODE_ID'          : ['FULL', 'SUM2'],
        'FILTER1_NAME'                : ['CL1', 'RED'],
        'FILTER2_NAME'                : ['GRN', 'CL2'],
        'GAIN_MODE_ID'                : ['12 ELECTRONS PER DN', '29 ELECTRONS PER DN'],
        'TARGET_NAME'                 : ['SATURN', 'ERRIAPO'],
        'OBSERVATION_ID'              : ['ISS_000SA_TEST001_PRIME',
                                         'ISS_000SA_TEST002_PRIME'],
        'SPACECRAFT_CLOCK_START_COUNT': ['1484506647.12', '1484506649'],
        'SPACECRAFT_CLOCK_STOP_COUNT' : ['1484506648.122', '1484506649.5'],
    }
    columns.update(changes)
    return columns


def write_index(directory: pathlib.Path, **changes: list[Any]) -> pathlib.Path:
    """Write a two-row Cassini ISS index table and label; return the label's path."""

    return write_table(directory, 'COISS_2009_index', index_columns(**changes))


@pytest.fixture
def index_snapshots(tmp_path: pathlib.Path) -> list[oops.observation.Snapshot]:
    return list(CassiniISS.from_index(write_index(tmp_path)))


def test_from_index_returns_an_iterator(tmp_path: pathlib.Path) -> None:
    snapshots = CassiniISS.from_index(write_index(tmp_path))
    assert next(snapshots).detector == 'NAC'


def test_from_index_makes_one_snapshot_per_row(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert len(index_snapshots) == 2


def test_from_index_filepaths(index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert [obs.filepath for obs in index_snapshots] == ['COISS_2009/' + _NAC_FILE,
                                                         'COISS_2009/' + _WAC_FILE]
    assert [obs.basename for obs in index_snapshots] == ['N1484506648_1.IMG',
                                                         'W1484506649_1.IMG']
    assert not hasattr(index_snapshots[0], 'image_url')


def test_from_index_rows(index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert [obs.detector for obs in index_snapshots] == ['NAC', 'WAC']
    assert [obs.sampling for obs in index_snapshots] == ['FULL', 'SUM2']
    assert [obs.filter for obs in index_snapshots] == ['GRN', 'RED']
    assert [obs.gain_mode for obs in index_snapshots] == [3, 2]
    assert [obs.target for obs in index_snapshots] == ['SATURN', 'ERRIAPUS']
    assert [obs.observation_id for obs in index_snapshots] == [
        'ISS_000SA_TEST001_PRIME', 'ISS_000SA_TEST002_PRIME']


def test_from_index_times(index_snapshots: list[oops.observation.Snapshot]) -> None:
    tstart = julian.tdb_from_iso('2005-01-15T18:31:00.000')
    assert index_snapshots[1].time == approx_time((tstart, tstart + 0.5))


def test_from_index_pads_short_clock_counts(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert [obs.spacecraft_clock_start_count for obs in index_snapshots] == [
        '1484506647.120', '1484506649.000']
    assert [obs.dict['SPACECRAFT_CLOCK_STOP_COUNT'] for obs in index_snapshots] == [
        '1484506648.122', '1484506649.500']


def test_from_index_keeps_the_early_cruise_counts(tmp_path: pathlib.Path) -> None:
    # Unlike a label or an image file, an early cruise index row gives its own counts
    path = write_index(tmp_path, IMAGE_NUMBER=['1294562651', '1484506649'],
                       SPACECRAFT_CLOCK_START_COUNT=['1294560651.000', '1484506649'],
                       SPACECRAFT_CLOCK_STOP_COUNT=['1294562651.000', '1484506649.5'])
    obs = next(CassiniISS.from_index(path))
    assert obs.spacecraft_clock_start_count == '1294560651.000'


def test_from_index_dict_is_the_row(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert index_snapshots[0].dict['TARGET_NAME_mask'] is np.False_


def test_from_index_pointing_subfields(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert index_snapshots[0].ck_object_id == -82000


def test_from_index_defines_camera_frames(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert oops.Frame.frame_id_exists('CASSINI_ISS_WAC')


def test_from_index_fov_applies_to_every_row(tmp_path: pathlib.Path) -> None:
    fov = oops.fov.FlatFOV(1.e-5, (PIXELS, PIXELS))
    snapshots = CassiniISS.from_index(write_index(tmp_path), fov=fov)
    assert [obs.fov for obs in snapshots] == [fov, fov]


def test_from_index_parallel_applies_to_every_row(
        tmp_path: pathlib.Path, index_snapshots: list[oops.observation.Snapshot]) -> None:
    parallel = index_snapshots[0]
    snapshots = CassiniISS.from_index(write_index(tmp_path), parallel=parallel)
    assert [obs.frame.reference.frame_id for obs in snapshots] == [
        parallel.frame.frame_id, parallel.frame.frame_id]


def test_from_index_select(tmp_path: pathlib.Path) -> None:
    rows = list(CassiniISS.from_index(write_index(tmp_path), select=slice(None)))
    assert [[obs.detector for obs in row] for row in rows] == [['NAC'], ['WAC']]


def test_from_index_filter(tmp_path: pathlib.Path) -> None:
    snapshots = CassiniISS.from_index(write_index(tmp_path),
                                      filter={'INSTRUMENT_MODE_ID': {'SUM2'}})
    assert [obs.detector for obs in snapshots] == ['WAC']


def test_from_index_uses_given_rows(tmp_path: pathlib.Path) -> None:
    rows = list(oops.Host._index_rows(write_index(tmp_path), supplement=None))
    snapshots = CassiniISS.from_index(tmp_path / 'missing.lbl', row_dicts=rows[:1])
    assert [obs.detector for obs in snapshots] == ['NAC']


def test_from_index_merges_a_given_supplement(tmp_path: pathlib.Path) -> None:
    supplement = write_table(tmp_path, 'extra',
                             {'VOLUME_ID': ['COISS_2009', 'COISS_2009'],
                              'FILE_SPECIFICATION_NAME': [_NAC_FILE, _WAC_FILE],
                              'EXTRA': ['A', 'B']})
    snapshots = CassiniISS.from_index(write_index(tmp_path), supplement=supplement)
    assert [obs.dict['EXTRA'] for obs in snapshots] == ['A', 'B']


def test_from_index_supplement_must_agree(tmp_path: pathlib.Path) -> None:
    supplement = write_table(tmp_path, 'extra',
                             {'VOLUME_ID': ['COISS_2009', 'COISS_2009'],
                              'FILE_SPECIFICATION_NAME': [_NAC_FILE, _WAC_FILE],
                              'EXPOSURE_DURATION': [1000.0, 600.0]})
    snapshots = CassiniISS.from_index(write_index(tmp_path), supplement=supplement)
    next(snapshots)
    with pytest.raises(oops.host.HostIndexError,
                       match='index row 1 does not match its supplement row'):
        next(snapshots)


def test_from_index_missing_supplement(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        CassiniISS.from_index(write_index(tmp_path), supplement=tmp_path / 'none.lbl')


def test_from_index_rejects_unknown_keywords(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'enclose'"):
        CassiniISS.from_index(write_index(tmp_path), enclose=True)


def test_host_from_index_dispatches_to_iss(tmp_path: pathlib.Path) -> None:
    snapshots = oops.Host.from_index(write_index(tmp_path))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']


@pytest.mark.parametrize(('count', 'expected'), [
    ('1484506647.1', '1484506647.100'),
    ('1484506647.', '1484506647.000'),
    ('1484506647.123', '1484506647.123'),
    ('UNK', 'UNK'),
], ids=['one-digit', 'point', 'three-digits', 'not-a-count'])
def test_padded_clock_counts(count: str, expected: str) -> None:
    row = CassiniISS._padded_clock_counts({'SPACECRAFT_CLOCK_START_COUNT': count})
    assert row['SPACECRAFT_CLOCK_START_COUNT'] == expected


def test_padded_clock_counts_leaves_the_row_unchanged() -> None:
    row = {'SPACECRAFT_CLOCK_START_COUNT': '1.1'}
    CassiniISS._padded_clock_counts(row)
    assert row == {'SPACECRAFT_CLOCK_START_COUNT': '1.1'}

##########################################################################################
