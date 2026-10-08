##########################################################################################
# tests/host_cassini/test_iss.py
##########################################################################################
"""Tests for the Cassini ISS host, host_cassini.iss."""

import os
import pathlib
import subprocess
import sys
from typing import Any

import julian
import numpy as np
import oops
import pytest
import vicar
from conftest import (NAC_HALF_FOV_DEG, PIXELS, WAC_HALF_FOV_DEG, FakeSpice,
                      FakeSpiceDB, tdb_in_month)

import host_cassini
import host_cassini.iss._oops as iss
from host_cassini._oops import _Cassini
from host_cassini.iss import CassiniISS
from host_cassini.iss._host import _CassiniISSHost

START_TIME = '2005-01-01T00:00:00.000'
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
        'EXPOSURE_DURATION'   : 1000.0,
        'INSTRUMENT_MODE_ID'  : 'FULL',
        'FILTER_NAME'         : ['CL1', 'GRN'],
        'GAIN_MODE_ID'        : '12 ELECTRONS PER DN',
        'TARGET_NAME'         : 'SATURN',
        'OBSERVATION_ID'      : 'ISS_000SA_TEST001_PRIME',
    }
    label.update(changes)
    return label


def snapshot(**changes: Any) -> oops.observation.Snapshot:
    """A Snapshot made from `label_dict(**changes)` with default options."""

    obs: oops.observation.Snapshot = CassiniISS._make_snapshot(label_dict(**changes),
                                                               filepath=INDEX_PATH)
    return obs

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
    assert oops.Host._LOOKUP['Cassini ISS'] is _CassiniISSHost


def test_cassini_iss_inherits_registered_host() -> None:
    assert issubclass(CassiniISS, _CassiniISSHost)


_LAZY_IMPORT_SCRIPT = '''
import sys
import oops
import host_cassini
print('host_cassini.iss._oops' in sys.modules)
print(oops.Host._LOOKUP['Cassini ISS'].__name__)
_ = host_cassini.CassiniISS
print('host_cassini.iss._oops' in sys.modules)
'''


def test_import_registers_host_without_loading_implementation() -> None:
    src = pathlib.Path(__file__).parents[2] / 'src'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(src), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-c', _LAZY_IMPORT_SCRIPT], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ['False', '_CassiniISSHost', 'True']


@pytest.mark.parametrize('module', [host_cassini, host_cassini.iss],
                         ids=['host_cassini', 'host_cassini.iss'])
def test_unknown_attribute(module: Any) -> None:
    with pytest.raises(AttributeError, match=f"module '{module.__name__}' has no "
                                             "attribute 'Nope'"):
        _ = module.Nope


@pytest.mark.parametrize('module', [host_cassini, host_cassini.iss, iss],
                         ids=['host_cassini', 'host_cassini.iss', 'iss._oops'])
def test_module_exports(module: Any) -> None:
    assert module.__all__ == ['CassiniISS']


@pytest.mark.parametrize('module', [host_cassini, host_cassini.iss],
                         ids=['host_cassini', 'host_cassini.iss'])
def test_package_exposes_host(module: Any) -> None:
    assert module.CassiniISS is iss.CassiniISS


@pytest.mark.parametrize(('label', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'ISSNA'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'ISSWA'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'VIMS'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_ID': 'ISSNA'}, False),
    ({'INSTRUMENT_ID': 'ISSNA'}, False),
    ({}, False),
], ids=['nac', 'wac', 'other-instrument', 'other-host', 'no-host', 'empty'])
def test_detection(label: dict[str, str], expected: bool) -> None:
    assert _CassiniISSHost._detect_in_pds3(label) is expected
    assert _CassiniISSHost._detect_in_vicar(label) is expected


@pytest.mark.parametrize(('label', 'expected'), [
    ({'^IMAGE_INDEX_TABLE': 'index.tab', 'IMAGE_INDEX_TABLE': {}}, None),
    ({'^INDEX_TABLE': 'index.tab', 'INDEX_TABLE': {}}, False),
    ({}, False),
], ids=['image-index-table', 'index-table', 'empty'])
def test_detection_in_index(label: dict[str, Any], expected: bool | None) -> None:
    assert _CassiniISSHost._detect_in_index(label) is expected


@pytest.mark.parametrize(('row', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_NAME': NAC_NAME}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_NAME': WAC_NAME}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER',
      'INSTRUMENT_NAME': 'VISUAL AND INFRARED MAPPING SPECTROMETER'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_NAME': NAC_NAME}, False),
    ({}, False),
], ids=['nac', 'wac', 'other-instrument', 'other-host', 'empty'])
def test_detection_in_row(row: dict[str, str], expected: bool) -> None:
    assert _CassiniISSHost._detect_in_row(row) is expected

##########################################################################################
# _fix_cassini_iss_target()
##########################################################################################


@pytest.mark.parametrize(('observation_id', 'tstart', 'expected'), [
    ('ISS_053RI_PHOTOMDRK002_PRIME', _Cassini.TOUR, 'SATURN_RING_PLANE'),
    ('ISS_C23RG_SPOKES001_PRIME', _Cassini.TOUR, 'SATURN_RING_PLANE'),
    ('ISS_C23RA_RINGS001_PRIME', _Cassini.TOUR - 1., 'JUPITER_RING_PLANE'),
    ('ISS_053EN_ENCELADUS001_PRIME', _Cassini.TOUR, 'ENCELADUS'),
    ('ISS_053RH_TEST001_PRIME', _Cassini.TOUR, 'ENCELADUS'),
    ('ISS_053RI', _Cassini.TOUR, 'ENCELADUS'),
    ('', _Cassini.TOUR, 'ENCELADUS'),
], ids=['ring-RI', 'ring-RG', 'jupiter-ring', 'not-ring', 'RH-not-ring', 'too-short',
        'empty'])
def test_fix_target(observation_id: str, tstart: float, expected: str) -> None:
    dict_ = {'OBSERVATION_ID': observation_id}
    assert CassiniISS._fix_cassini_iss_target('ENCELADUS', dict_, tstart) == expected


def test_fix_target_without_observation_id() -> None:
    target = CassiniISS._fix_cassini_iss_target('ENCELADUS', {}, _Cassini.TOUR)
    assert target == 'ENCELADUS'

##########################################################################################
# _initialize()
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


def test_initialize_loads_default_instruments(initialized_iss: FakeSpiceDB) -> None:
    assert _Cassini.loaded_instruments == ['ISS', 'VIMS', 'CIRS', 'UVIS']


def test_initialize_passes_options_to_mission(fake_spicedb: FakeSpiceDB,
                                              fake_spice: FakeSpice) -> None:
    CassiniISS._initialize(ck='predicted', spk='predicted', planets=[6],
                           asof='2010-02-03', mst_pck=False, irregulars=False)
    assert fake_spicedb.named('select_spk')[0][1]['name'] == 'CAS-SPK-PREDICTED'
    assert fake_spicedb.named('select_ck')[0][1]['name'] == 'CAS-CK-PREDICTED'
    assert fake_spice.solar_system[0][1] == {'asof': '2010-02-03', 'planets': [6],
                                             'mst_pck': False, 'irregulars': False}


def test_initialize_ignores_later_calls(initialized_iss: FakeSpiceDB) -> None:
    fovs = CassiniISS._FOVS
    count = len(initialized_iss.calls)
    CassiniISS._initialize(spk='predicted')
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
    oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000, frame_id='CASSINI_ISS_NAC')
    CassiniISS._define_camera_frames()
    assert fake_spice.spice_frames == ['CASSINI_ISS_WAC']

##########################################################################################
# _make_snapshot()
##########################################################################################


uses_iss = pytest.mark.usefixtures('initialized_iss')


@uses_iss
def test_snapshot_times() -> None:
    tstart = julian.tdb_from_iso(START_TIME)
    assert snapshot().time == pytest.approx((tstart, tstart + 1.))


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
@pytest.mark.parametrize('fast', [True, False, None])
@pytest.mark.parametrize('mode', ['FULL', 'SUM2', 'SUM4'])
def test_snapshot_fov(mode: str, fast: bool | None) -> None:
    obs = CassiniISS._make_snapshot(label_dict(INSTRUMENT_MODE_ID=mode),
                                    filepath=INDEX_PATH, fast_distortion=fast)
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
    assert CassiniISS._make_snapshot(dict_, filepath=INDEX_PATH).filter == 'RED'


@uses_iss
@pytest.mark.parametrize(('gain', 'expected'), [
    ('215 ELECTRONS PER DN', 0),
    ('95 ELECTRONS PER DN', 1),
    ('29 ELECTRONS PER DN', 2),
    ('12 ELECTRONS PER DN', 3),
    ('UNKNOWN', None),
])
def test_snapshot_gain_mode(gain: str, expected: int | None) -> None:
    assert snapshot(GAIN_MODE_ID=gain).gain_mode == expected


@uses_iss
@pytest.mark.parametrize(('label_target', 'target'), [
    ('SATURN', 'SATURN'),
    ('ERRIAPO', 'ERRIAPUS'),
    ('K07S4', 'AEGAEON'),
    ('S12_2004', 'S/2004_S_12'),
    ('DARK SKY', 'NONE'),
])
def test_snapshot_target(label_target: str, target: str) -> None:
    obs = snapshot(TARGET_NAME=label_target)
    assert obs.target == target
    assert obs.label_target == label_target
    assert obs.lightsource == 'SUN'


@uses_iss
def test_snapshot_ring_target() -> None:
    obs = snapshot(OBSERVATION_ID='ISS_053RI_PHOTOMDRK002_PRIME')
    assert obs.target == 'SATURN_RING_PLANE'


@uses_iss
@pytest.mark.parametrize('star', ['FOMALHAUT', 'SPICA'])
def test_snapshot_star_target(star: str) -> None:
    obs = snapshot(TARGET_NAME=star)
    assert obs.target == 'NONE'
    assert obs.lightsource is oops.lightsource.star_lookup(star)


@uses_iss
def test_snapshot_subfields() -> None:
    dict_ = label_dict()
    obs = CassiniISS._make_snapshot(dict_, filepath=INDEX_PATH)
    assert obs.host == 'Cassini'
    assert obs.instrument == 'ISS'
    assert obs.texp == 1.
    assert obs.dict is dict_
    assert obs.filepath == INDEX_PATH
    assert obs.basename == 'N1484506648_1.IMG'
    assert obs.spice_to_frame is iss._CMATRIX_ROTATION
    assert obs.calibrations == []


@uses_iss
def test_snapshot_data() -> None:
    data = np.zeros((PIXELS, PIXELS), dtype='uint8')
    obs = CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, data=data)
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
    obs = CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, target=target,
                                    calibrations=calibrations)
    assert obs.target == target
    assert obs.calibrations is calibrations


@pytest.fixture
def custom_frame(initialized_iss: FakeSpiceDB) -> oops.Frame:
    return oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000, frame_id='TEST_CMATRIX')


def test_custom_frame_is_used(custom_frame: oops.Frame) -> None:
    obs = CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, frame=custom_frame)
    assert obs.frame == custom_frame.wayframe


def test_custom_frame_loads_no_cks(custom_frame: oops.Frame) -> None:
    CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, frame=custom_frame)
    assert not _Cassini.CK_LOADED.any()
    assert _Cassini.SPK_LOADED.any()


def test_custom_frame_defines_no_camera_frames(custom_frame: oops.Frame) -> None:
    CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, frame=custom_frame)
    assert not oops.Frame.frame_id_exists('CASSINI_ISS_NAC')


def test_custom_frame_reports_no_cks(custom_frame: oops.Frame,
                                     initialized_iss: FakeSpiceDB) -> None:
    CassiniISS._make_snapshot(label_dict(), filepath=INDEX_PATH, frame=custom_frame)
    assert 'CK' not in initialized_iss.named('used_basenames')[0][1]['types']

##########################################################################################
# from_file()
##########################################################################################


def _pds3_value(value: Any) -> str:
    if isinstance(value, list):
        return '(' + ', '.join(_pds3_value(item) for item in value) + ')'
    if isinstance(value, str) and value != START_TIME:
        return f'"{value}"'
    return str(value)


def write_pds3_image(directory: pathlib.Path, *, lines: int = 256,
                     **changes: Any) -> pathlib.Path:
    """Write a detached PDS3 label and its 8-bit image; return the label's path."""

    image = np.arange(lines * lines, dtype='uint32').reshape(lines, lines) % 256
    (directory / 'N0000000001_1.IMG').write_bytes(image.astype('uint8').tobytes())

    keywords = label_dict(INSTRUMENT_MODE_ID='SUM4', **changes)
    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {lines}', f'FILE_RECORDS = {lines}',
            '^IMAGE = "N0000000001_1.IMG"']
    text += [f'{key} = {_pds3_value(value)}' for (key, value) in keywords.items()]
    text += ['OBJECT = IMAGE', f'  LINES = {lines}', f'  LINE_SAMPLES = {lines}',
             '  SAMPLE_BITS = 8', '  SAMPLE_TYPE = MSB_UNSIGNED_INTEGER',
             'END_OBJECT = IMAGE', 'END', '']
    path = directory / 'N0000000001_1.LBL'
    path.write_text('\r\n'.join(text))
    return path


def write_vicar_image(directory: pathlib.Path, *, lines: int = 256,
                      **changes: Any) -> pathlib.Path:
    """Write a VICAR image with an ISS label; return its path."""

    image = np.arange(lines * lines, dtype='uint32').reshape(lines, lines) % 256
    vic = vicar.VicarImage.from_array(image.astype('uint8'))
    for (key, value) in label_dict(INSTRUMENT_MODE_ID='SUM4', **changes).items():
        vic[key] = value
    path = directory / 'N0000000001_1.IMG'
    vic.write_file(path)
    return path


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_pds3(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path))
    assert obs.data.shape == (256, 256)
    assert obs.data[1, 2] == 258 % 256
    assert obs.basename == 'N0000000001_1.LBL'
    assert obs.fov is CassiniISS._FOVS['NAC', 'SUM4', True]


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_pds3_astrometry(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert not hasattr(obs, 'data')
    assert obs.filter == 'GRN'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path, INSTRUMENT_NAME=WAC_NAME))
    assert obs.data.shape == (256, 256)
    assert obs.data[1, 2] == 258 % 256
    assert obs.detector == 'WAC'
    assert obs.basename == 'N0000000001_1.IMG'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar_astrometry(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path), astrometry=True)
    assert not hasattr(obs, 'data')
    assert obs.filter == 'GRN'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_sets_local_paths(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.abspath == path.resolve()


def test_from_file_initializes_iss(tmp_path: pathlib.Path) -> None:
    CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert CassiniISS._initialized


@pytest.mark.usefixtures('initialized_iss')
def test_host_from_file_dispatches_to_iss(tmp_path: pathlib.Path) -> None:
    obs = oops.Host.from_file(write_pds3_image(tmp_path), astrometry=True)
    assert obs.instrument == 'ISS'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_rejects_other_hosts(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, INSTRUMENT_HOST_NAME='GALILEO ORBITER')
    with pytest.raises(oops.host.HostError, match='unrecognized host'):
        CassiniISS.from_file(path, astrometry=True)

##########################################################################################
# from_index()
##########################################################################################


_INDEX_COLUMNS = [
    ('VOLUME_ID', 'CHARACTER', 10),
    ('FILE_SPECIFICATION_NAME', 'CHARACTER', 45),
    ('INSTRUMENT_HOST_NAME', 'CHARACTER', 15),
    ('INSTRUMENT_NAME', 'CHARACTER', 38),
    ('START_TIME', 'CHARACTER', 23),
    ('EXPOSURE_DURATION', 'ASCII_REAL', 10),
    ('INSTRUMENT_MODE_ID', 'CHARACTER', 4),
    ('FILTER1_NAME', 'CHARACTER', 3),
    ('FILTER2_NAME', 'CHARACTER', 3),
    ('GAIN_MODE_ID', 'CHARACTER', 20),
    ('TARGET_NAME', 'CHARACTER', 10),
    ('OBSERVATION_ID', 'CHARACTER', 30),
]

_INDEX_ROWS = [
    ('COISS_2009', 'data/1484506648_1484573295/N1484506648_1.IMG', 'CASSINI ORBITER',
     NAC_NAME,
     '2005-01-15T18:30:00.000', 1000.0, 'FULL', 'CL1', 'GRN', '12 ELECTRONS PER DN',
     'SATURN', 'ISS_000SA_TEST001_PRIME'),
    ('COISS_2009', 'data/1484506648_1484573295/W1484506649_1.IMG', 'CASSINI ORBITER',
     WAC_NAME,
     '2005-01-15T18:31:00.000', 500.0, 'SUM2', 'RED', 'CL2', '29 ELECTRONS PER DN',
     'ERRIAPO', 'ISS_000SA_TEST001_PRIME'),
]


def write_index(directory: pathlib.Path) -> pathlib.Path:
    """Write a two-row Cassini ISS index table and label; return the label's path."""

    widths = [width + 2 if kind == 'CHARACTER' else width
              for (_, kind, width) in _INDEX_COLUMNS]
    row_bytes = sum(widths) + len(widths) - 1 + 2

    lines = []
    for row in _INDEX_ROWS:
        fields = []
        for (value, (_, kind, width)) in zip(row, _INDEX_COLUMNS, strict=True):
            if kind == 'CHARACTER':
                fields.append('"' + str(value).ljust(width) + '"')
            else:
                fields.append(f'{value:{width}.3f}')
        lines.append(','.join(fields) + '\r\n')
    (directory / 'INDEX.TAB').write_text(''.join(lines), newline='')

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {row_bytes}', f'FILE_RECORDS = {len(_INDEX_ROWS)}',
            '^IMAGE_INDEX_TABLE = "INDEX.TAB"', 'OBJECT = IMAGE_INDEX_TABLE',
            '  INTERCHANGE_FORMAT = ASCII', f'  ROWS = {len(_INDEX_ROWS)}',
            f'  COLUMNS = {len(_INDEX_COLUMNS)}', f'  ROW_BYTES = {row_bytes}']
    start = 1
    for ((name, kind, width), field_width) in zip(_INDEX_COLUMNS, widths, strict=True):
        offset = 1 if kind == 'CHARACTER' else 0
        text += ['  OBJECT = COLUMN', f'    NAME = {name}', f'    DATA_TYPE = {kind}',
                 f'    START_BYTE = {start + offset}', f'    BYTES = {width}',
                 '  END_OBJECT = COLUMN']
        start += field_width + 1
    text += ['END_OBJECT = IMAGE_INDEX_TABLE', 'END', '']
    path = directory / 'INDEX.LBL'
    path.write_text('\r\n'.join(text))
    return path


@pytest.fixture
def index_snapshots(tmp_path: pathlib.Path) -> list[oops.observation.Snapshot]:
    snapshots: list[oops.observation.Snapshot] = CassiniISS.from_index(
        write_index(tmp_path))
    return snapshots


def test_from_index_makes_one_snapshot_per_row(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert len(index_snapshots) == 2


def test_from_index_filepaths(index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert [obs.filepath for obs in index_snapshots] == [
        'COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG',
        'COISS_2009/data/1484506648_1484573295/W1484506649_1.IMG']
    assert [obs.basename for obs in index_snapshots] == ['N1484506648_1.IMG',
                                                         'W1484506649_1.IMG']


def test_from_index_rows(index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert [obs.detector for obs in index_snapshots] == ['NAC', 'WAC']
    assert [obs.sampling for obs in index_snapshots] == ['FULL', 'SUM2']
    assert [obs.filter for obs in index_snapshots] == ['GRN', 'RED']
    assert [obs.gain_mode for obs in index_snapshots] == [3, 2]
    assert [obs.target for obs in index_snapshots] == ['SATURN', 'ERRIAPUS']


def test_from_index_times(index_snapshots: list[oops.observation.Snapshot]) -> None:
    tstart = julian.tdb_from_iso('2005-01-15T18:31:00.000')
    assert index_snapshots[1].time == pytest.approx((tstart, tstart + 0.5))


def test_from_index_defines_camera_frames(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert oops.Frame.frame_id_exists('CASSINI_ISS_WAC')


@pytest.mark.parametrize('option', ['fov', 'parallel'])
def test_from_index_rejects_per_row_options(tmp_path: pathlib.Path, option: str) -> None:
    with pytest.raises(ValueError, match=f'disallowed Cassini ISS.from_index.. option '
                                         f'{option}'):
        CassiniISS.from_index(write_index(tmp_path), **{option: 'anything'})


def test_from_index_filter(tmp_path: pathlib.Path) -> None:
    snapshots = CassiniISS.from_index(write_index(tmp_path),
                                      filter={'INSTRUMENT_MODE_ID': {'SUM2'}})
    assert [obs.detector for obs in snapshots] == ['WAC']


def test_from_index_uses_given_rows(tmp_path: pathlib.Path) -> None:
    rows = oops.Host._read_index_rows(write_index(tmp_path))
    snapshots = CassiniISS.from_index(tmp_path / 'missing.lbl', row_dicts=rows[:1])
    assert [obs.detector for obs in snapshots] == ['NAC']


def test_registered_host_from_index_delegates(tmp_path: pathlib.Path) -> None:
    snapshots = _CassiniISSHost.from_index(write_index(tmp_path))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']


def test_host_from_index_dispatches_to_iss(tmp_path: pathlib.Path) -> None:
    snapshots = oops.Host.from_index(write_index(tmp_path))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']

##########################################################################################
