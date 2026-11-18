##########################################################################################
# tests/oops/test_iss.py
##########################################################################################
"""Tests for the Cassini ISS host, host_cassini.oops.iss."""

import os
import pathlib
import subprocess
import sys
from typing import Any, cast

from filecache import FCPath
import julian
import numpy as np
import oops
from pdsparser import Pds3Label
import pytest
import vicar
from conftest import (NAC_HALF_FOV_DEG, PIXELS, WAC_HALF_FOV_DEG, FakeSpice,
                      FakeSpiceDB, ik_dict, pds3_lines, tdb_in_month)

import host_cassini
import host_cassini.oops
import host_cassini.oops.iss as iss
from host_cassini.oops._cassini import _Cassini
from host_cassini.oops.iss      import CassiniISS

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


OVERRIDE_NAMES = ('target', 'lightsource', 'path', 'frame', 'cadence', 'fov',
                  'calibrations', 'timeshift', 'navigation', 'tracker', 'parallel')


def make_snapshot(dict_: dict[str, Any], *, data: Any = None,
                  fast_distortion: bool | None = True,
                  **overrides: Any) -> oops.observation.Snapshot:
    """A Snapshot from `_make_snapshot`, with every override not given set to None."""

    all_overrides = {name: overrides.get(name) for name in OVERRIDE_NAMES}
    obs: oops.observation.Snapshot = CassiniISS._make_snapshot(
        dict_, filepath=INDEX_PATH, data=data, fast_distortion=fast_distortion,
        return_all_planets=False, overrides=all_overrides, source=INDEX_PATH,
        check_pointing=True)
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
    """A GAIN_MODE_ID with no known number of electrons per DN gives no gain_mode."""

    assert 'gain_mode' not in snapshot(GAIN_MODE_ID='UNKNOWN').subfields


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
    assert obs.observation_id == 'ISS_053RI_PHOTOMDRK002_PRIME'


@uses_iss
@pytest.mark.parametrize('star', ['FOMALHAUT', 'SPICA'])
def test_snapshot_star_target(star: str) -> None:
    obs = snapshot(TARGET_NAME=star)
    assert obs.target == 'NONE'
    assert obs.lightsource is oops.lightsource.star_lookup(star)


@uses_iss
def test_snapshot_subfields() -> None:
    dict_ = label_dict()
    obs = make_snapshot(dict_)
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


def write_vicar_image(directory: pathlib.Path, *, lines: int = 256, prefix_bytes: int = 0,
                      binary_header: bool = False, keywords: dict[str, Any] | None = None,
                      **changes: Any) -> pathlib.Path:
    """Write a VICAR image with an ISS label; return its path.

    The image is an 8-bit ramp. `prefix_bytes` gives each line a binary prefix holding its
    line number, and `binary_header` adds a one-record binary header after the label, as
    the archive's raw and calibrated files have. The label holds `keywords`, or by default
    `label_dict(INSTRUMENT_MODE_ID='SUM4', **changes)`; a keyword whose value is None is
    left out.
    """

    image = np.arange(lines * lines, dtype='uint32').reshape(lines, lines) % 256
    prefix = None
    if prefix_bytes:
        prefix = np.zeros((1, lines, prefix_bytes), dtype='uint8')
        prefix[0, :, 1] = np.arange(1, lines + 1) % 256
    header = bytes(lines + prefix_bytes) if binary_header else None
    vic = vicar.VicarImage(array=image.astype('uint8'), prefix=prefix, binheader=header)
    if keywords is None:
        keywords = label_dict(**({'INSTRUMENT_MODE_ID': 'SUM4'} | changes))
    for (key, value) in keywords.items():
        if value is not None:
            vic[key] = value
    path = directory / 'N0000000001_1.IMG'
    vic.write_file(path)
    return path


def write_pds3_image(directory: pathlib.Path, *, lines: int = 256, prefix_bytes: int = 0,
                     binary_header: bool = False, pointer_error: int = 0,
                     header: dict[str, Any] | None = None,
                     pointer_name: str | None = None,
                     extra_lines: list[str] | None = None,
                     **changes: Any) -> pathlib.Path:
    """Write a VICAR image and a detached PDS3 label for it; return the label's path.

    Both labels hold `label_dict(INSTRUMENT_MODE_ID='SUM4', **changes)`, except that the
    VICAR header takes the keywords in `header` instead, leaving out those whose value is
    None. The label's ^IMAGE pointer names `pointer_name`, by default the image file's
    name, and is `pointer_error` records away from the array, which the archive's labels
    give as -1 for a calibrated file and +1 for some raw files. `extra_lines` are added to
    the label text as they are, after the keywords.
    """

    keywords = label_dict(**({'INSTRUMENT_MODE_ID': 'SUM4'} | changes))
    image_path = write_vicar_image(directory, lines=lines, prefix_bytes=prefix_bytes,
                                   binary_header=binary_header,
                                   keywords=keywords | (header or {}))
    vic = vicar.VicarLabel(image_path)
    recsize = cast(int, vic['RECSIZE'])
    record = (cast(int, vic['LBLSIZE']) // recsize + cast(int, vic['NLB']) + 1
              + pointer_error)

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {recsize}', f'FILE_RECORDS = {lines + record}',
            f'^IMAGE = ("{pointer_name or image_path.name}", {record})']
    text += pds3_lines({key: value for (key, value) in keywords.items()
                        if value is not None})
    text += extra_lines or []
    text += ['OBJECT = IMAGE', f'  LINES = {lines}', f'  LINE_SAMPLES = {lines}',
             '  SAMPLE_BITS = 8', '  SAMPLE_TYPE = MSB_UNSIGNED_INTEGER',
             f'  LINE_PREFIX_BYTES = {prefix_bytes}', 'END_OBJECT = IMAGE', 'END', '']
    path = directory / 'N0000000001_1.LBL'
    path.write_text('\r\n'.join(text))
    return path


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_pds3(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path))
    assert obs.data.shape == (256, 256)
    assert obs.data[1, 2] == 258 % 256
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.fov is CassiniISS._FOVS['NAC', 'SUM4', True]


_RAMP = np.arange(256 * 256).reshape(256, 256) % 256


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('prefix_bytes', 'binary_header', 'pointer_error'), [
    (0, False, 0),
    (0, True, -1),
    (24, True, 1),
], ids=['plain', 'calibrated', 'raw-coiss-1006'])
@pytest.mark.parametrize('source', ['label', 'image'])
def test_from_file_reads_the_array_the_vicar_header_places(tmp_path: pathlib.Path,
                                                           source: str,
                                                           prefix_bytes: int,
                                                           binary_header: bool,
                                                           pointer_error: int) -> None:
    """Either input gives the array that the VICAR header locates, not the label."""

    label_path = write_pds3_image(tmp_path, prefix_bytes=prefix_bytes,
                                  binary_header=binary_header,
                                  pointer_error=pointer_error)
    path = label_path if source == 'label' else label_path.with_suffix('.IMG')
    assert np.array_equal(CassiniISS.from_file(path).data, _RAMP)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_pds3_astrometry(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    (tmp_path / 'N0000000001_1.IMG').unlink()
    obs = CassiniISS.from_file(path, astrometry=True)
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
def test_from_file_vicar_astrometry(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_vicar_image(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('the VICAR image array was read')

    monkeypatch.setattr(vicar.VicarImage, 'from_file', fail)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert not hasattr(obs, 'data')
    assert obs.filter == 'GRN'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar_image_astrometry(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    vic = cast(vicar.VicarImage, vicar.VicarImage.from_file(path, strict=False))
    vic.filepath = path
    obs = CassiniISS.from_file(vic, astrometry=True)
    assert not hasattr(obs, 'data')


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar_label_reads_data(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(vicar.VicarLabel(write_vicar_image(tmp_path)))
    assert obs.data[1, 2] == 258 % 256


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('astrometry', [False, True])
def test_from_file_label_names_the_image(tmp_path: pathlib.Path,
                                         astrometry: bool) -> None:
    """Given a label, the four path subfields name the image file, not the label."""

    image_path = write_pds3_image(tmp_path).with_suffix('.IMG')
    obs = CassiniISS.from_file(tmp_path / 'N0000000001_1.LBL', astrometry=astrometry)
    assert obs.filepath == FCPath(image_path)
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.abspath == image_path.resolve()
    assert obs.image_url == image_path.as_posix()


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
    with pytest.raises(oops.host.WrongHostError, match='not a Cassini ISS PDS3 label'):
        CassiniISS.from_file(path, astrometry=True)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_rejects_label_naming_no_image_file(tmp_path: pathlib.Path) -> None:
    """A label whose ^IMAGE pointer names no file raises PdsHostError."""

    path = write_pds3_image(tmp_path)
    lines = ['^IMAGE = 2' if line.startswith('^IMAGE') else line
             for line in path.read_text().splitlines()]
    path.write_text('\r\n'.join(lines))
    with pytest.raises(oops.host.PdsHostError, match=r'\^IMAGE names no image file in '):
        CassiniISS.from_file(path, astrometry=True)


# An early cruise VICAR header, as in COISS_1001/data/1338756756_1338935702/W1338928383_1.
# The keywords whose names or values differ from the PDS3 label's are listed with what the
# label says.
_EARLY_TIME = '2000-157T20:23:28.509'
_EARLY_VICAR = {
    'MISSION_NAME'         : 'CASSINI',
    'INSTRUMENT_ID'        : 'ISSWA',      # INSTRUMENT_NAME = WAC_NAME
    'IMAGE_NUMBER'         : 1338928383,   # SPACECRAFT_CLOCK_STOP_COUNT
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
    'IMAGE_NUMBER'                : 1338928383,
    'IMAGE_TIME'                  : _EARLY_TIME,
    'INSTRUMENT_MODE_ID'          : 'SUM4',
    'FILTER1_NAME'                : 'CL1',
    'FILTER2_NAME'                : 'GRN',
    'EXPOSURE_DURATION'           : 500.0,
    'GAIN_MODE_ID'                : '12 ELECTRONS PER DN',
    'TARGET_NAME'                 : 'SATURN',
    'OBSERVATION_ID'              : 'ICO2_DARK_CURRENT_2',
    'SHUTTER_STATE_ID'            : 'DISABLED',
    'SHUTTER_MODE_ID'             : 'BOTSIM',
    'INSTRUMENT_HOST_NAME'        : 'CASSINI ORBITER',
    'INSTRUMENT_NAME'             : WAC_NAME,
    'FILTER_NAME'                 : ['CL1', 'GRN'],
    'START_TIME'                  : _EARLY_TIME,
    'STOP_TIME'                   : _EARLY_TIME,
    'SPACECRAFT_CLOCK_STOP_COUNT' : '1338928383.000',
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
    """An early header is detected by its MISSION_NAME and INSTRUMENT_ID."""

    assert CassiniISS._detect_in_vicar(label) is expected


def test_early_vicar_header_takes_pds3_names() -> None:
    """An early header's keywords are given the names and spellings of its label."""

    assert CassiniISS._pds3_dict_from_vicar(dict(_EARLY_VICAR)) == _EARLY_PDS3


def test_current_vicar_header_is_unchanged() -> None:
    """A header that already uses the PDS3 keywords is returned unchanged."""

    assert CassiniISS._pds3_dict_from_vicar(label_dict()) == label_dict()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_early_vicar_header(tmp_path: pathlib.Path) -> None:
    """An early VICAR file loads, with the times and subfields its label would give."""

    obs = CassiniISS.from_file(write_vicar_image(tmp_path, keywords=_EARLY_VICAR))
    tstart = julian.tdb_from_iso(_EARLY_TIME)
    assert obs.time == pytest.approx((tstart, tstart + 0.5), abs=1.e-6)
    assert obs.detector == 'WAC'
    assert obs.filter == 'GRN'
    assert obs.gain_mode == 3
    assert obs.dict['SHUTTER_MODE_ID'] == 'BOTSIM'
    assert obs.spacecraft_clock_stop_count == '1338928383.000'

##########################################################################################
# Metadata from the PDS3 label and the VICAR header
##########################################################################################


def label_path_of(tmp_path: pathlib.Path) -> pathlib.Path:
    """The path of the detached label that write_pds3_image writes in `tmp_path`."""

    return tmp_path / 'N0000000001_1.LBL'


def image_path_of(tmp_path: pathlib.Path) -> pathlib.Path:
    """The path of the image file that write_pds3_image writes in `tmp_path`."""

    return tmp_path / 'N0000000001_1.IMG'


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('source', 'astrometry', 'expected'), [
    ('image', False, {'data', 'header_dict'}),
    ('image', True, {'header_dict'}),
    ('label', False, {'data', 'label_dict', 'header_dict'}),
    ('label', True, {'label_dict'}),
], ids=['image', 'image-astrometry', 'label', 'label-astrometry'])
def test_from_file_reads_the_sources_of_its_input(tmp_path: pathlib.Path, source: str,
                                                  astrometry: bool,
                                                  expected: set[str]) -> None:
    """Each input reads what the operator's rules name, and publishes what it read."""

    write_pds3_image(tmp_path)
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path, astrometry=astrometry)
    present = {name for name in ('data', 'label_dict', 'header_dict')
               if name in obs.subfields}
    assert present == expected


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_dict_and_header_dict_are_verbatim(
        tmp_path: pathlib.Path) -> None:
    """label_dict and header_dict hold the two sources as their parsers give them."""

    path = write_pds3_image(tmp_path, header={'TELEMETRY_FORMAT_ID': 'UNK'})
    obs = CassiniISS.from_file(path)
    assert obs.label_dict == Pds3Label(path, method='fast').as_dict()
    assert obs.header_dict == vicar.VicarLabel(image_path_of(tmp_path)).as_dict(
        unique=False)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_astrometry_needs_no_image(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    image_path_of(tmp_path).unlink()
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.label_dict['TARGET_NAME'] == 'SATURN'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_merges_items_from_both_sources(tmp_path: pathlib.Path) -> None:
    """An item that one source alone gives is in dict, under its own name."""

    path = write_pds3_image(tmp_path, header={'SHUTTER_MODE_ID': 'NACONLY',
                                              'DATA_SET_ID': None},
                            DATA_SET_ID='CO-S-ISSNA/ISSWA-2-EDR-V1.0')
    obs = CassiniISS.from_file(path)
    assert obs.dict['SHUTTER_MODE_ID'] == 'NACONLY'
    assert obs.dict['DATA_SET_ID'] == 'CO-S-ISSNA/ISSWA-2-EDR-V1.0'
    assert obs.dict['PDS_VERSION_ID'] == 'PDS3'
    assert obs.dict['LBLSIZE'] == obs.header_dict['LBLSIZE']


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('item', 'label', 'header', 'expected'), [
    ('SEQUENCE_TITLE', 'ISS_023SA_1X2PLRNCL001_PRIME', 'ISS_)23SA_1X2PLRNCL001_PRIME',
     'ISS_023SA_1X2PLRNCL001_PRIME'),
    ('DESCRIPTION', 'Calibrated using CISSCAL 4.0', 'DSN gap truncation.',
     'Calibrated using CISSCAL 4.0'),
    ('METHOD_DESC', 'ISSPT2.6.5 STAR:TYPE=A MAG=0.1', 'ISSPT2.6.5;STAR:TYPE=A,MAG=0.1',
     'ISSPT2.6.5;STAR:TYPE=A,MAG=0.1'),
    ('BIAS_STRIP_MEAN', 14.950098, 14.9501, 14.950098),
    ('BIAS_STRIP_MEAN', 14.95, 14.950098, 14.950098),
    ('OPTICS_TEMPERATURE', [0.712693, 1.905708], [0.712693, 1.90571],
     [0.712693, 1.905708]),
    ('TELEMETRY_FORMAT_ID', 'UNK', 'S_N_ER_3', 'S_N_ER_3'),
    ('TELEMETRY_FORMAT_ID', 'S_N_ER_3', 'N/A', 'S_N_ER_3'),
], ids=['label-by-default', 'label-by-rule', 'header-by-rule', 'header-rounded',
        'label-rounded', 'list-rounded',
        'label-null', 'header-null'])
def test_from_file_resolves_disagreements(tmp_path: pathlib.Path, item: str, label: Any,
                                          header: Any, expected: Any) -> None:
    """Where the two sources disagree, dict holds the value the item's rule chooses."""

    path = write_pds3_image(tmp_path, header={item: header}, **{item: label})
    assert CassiniISS.from_file(path).dict[item] == expected


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_computes_from_merged_items(tmp_path: pathlib.Path) -> None:
    """Values computed from an item follow the item as resolved, here the label's."""

    path = write_pds3_image(tmp_path, header={'TARGET_NAME': 'TITAN',
                                              'EXPOSURE_DURATION': 2000.},
                            TARGET_NAME='ENCELADUS', EXPOSURE_DURATION=500.)
    obs = CassiniISS.from_file(path)
    assert obs.target == 'ENCELADUS'
    assert obs.texp == 0.5


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_takes_items_the_label_lacks_from_the_header(
        tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path,
                            header={'OBSERVATION_ID': 'ISS_000SA_TEST001_PRIME'},
                            OBSERVATION_ID=None)
    assert CassiniISS.from_file(path).observation_id == 'ISS_000SA_TEST001_PRIME'


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('source', ['label', 'image'])
def test_from_file_leaves_out_null_items(tmp_path: pathlib.Path, source: str) -> None:
    """An item that is a PDS3 null value in every source read is absent, never None."""

    write_pds3_image(tmp_path, TARGET_NAME='UNK', OBSERVATION_ID='N/A',
                     GAIN_MODE_ID='NULL', FILTER_NAME=['UNK', 'UNK'])
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path)
    absent = {'label_target', 'observation_id', 'gain_mode', 'filter', 'filter1',
              'filter2'}
    assert absent.isdisjoint(obs.subfields)
    assert {'TARGET_NAME', 'OBSERVATION_ID', 'GAIN_MODE_ID', 'FILTER_NAME'}.isdisjoint(
        obs.dict)
    assert all(value is not None for value in obs.subfields.values())
    assert obs.target == 'NONE'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_keeps_null_items_in_the_raw_sources(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, TARGET_NAME='UNK'))
    assert obs.label_dict['TARGET_NAME'] == 'UNK'
    assert obs.header_dict['TARGET_NAME'] == 'UNK'


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('item', ['START_TIME', 'EXPOSURE_DURATION', 'INSTRUMENT_MODE_ID',
                                  'INSTRUMENT_NAME'])
@pytest.mark.parametrize('value', [None, 'UNK'], ids=['missing', 'null'])
def test_from_file_requires_items(tmp_path: pathlib.Path, item: str,
                                  value: str | None) -> None:
    """A missing required item raises HostFormatError naming the item and the file."""

    path = write_pds3_image(tmp_path, **cast(dict[str, Any], {item: value}))
    with pytest.raises(oops.host.HostFormatError,
                       match=f'{item} is missing from .*N0000000001_1.LBL'):
        CassiniISS.from_file(path)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('changes', 'message'), [
    ({'START_TIME': 'yesterday'}, "invalid START_TIME 'yesterday' in "),
    ({'INSTRUMENT_MODE_ID': 'SUM8'}, "invalid INSTRUMENT_MODE_ID 'SUM8' in "),
], ids=['time', 'mode'])
def test_from_file_rejects_invalid_items(tmp_path: pathlib.Path, changes: dict[str, Any],
                                         message: str) -> None:
    path = write_vicar_image(tmp_path, **changes)
    with pytest.raises(oops.host.HostFormatError, match=message):
        CassiniISS.from_file(path)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('source', ['label', 'image'])
def test_from_file_clock_counts(tmp_path: pathlib.Path, source: str) -> None:
    """The clock counts are subfields, spelled as the labels spell them."""

    write_pds3_image(tmp_path, SPACECRAFT_CLOCK_START_COUNT='1459509269.122',
                     SPACECRAFT_CLOCK_STOP_COUNT='1459509270.122')
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path)
    assert obs.spacecraft_clock_start_count == '1459509269.122'
    assert obs.spacecraft_clock_stop_count == '1459509270.122'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_without_clock_counts(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path))
    assert {'spacecraft_clock_start_count',
            'spacecraft_clock_stop_count'}.isdisjoint(obs.subfields)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_finds_image_in_other_case(tmp_path: pathlib.Path) -> None:
    """A label whose ^IMAGE names the file in the other case still finds it."""

    path = write_pds3_image(tmp_path, pointer_name='n0000000001_1.img')
    obs = CassiniISS.from_file(path)
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.data[1, 2] == 258 % 256


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_missing_image_raises(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path)
    image_path_of(tmp_path).unlink()
    with pytest.raises(FileNotFoundError, match=r'N0000000001_1\.IMG'):
        CassiniISS.from_file(path)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_with_bad_vicar_header(tmp_path: pathlib.Path) -> None:
    """A label whose image file has an unparsable VICAR header raises VicarHostError."""

    path = write_pds3_image(tmp_path)
    image_path_of(tmp_path).write_bytes(b'LBLSIZE=40      FORMAT=  garbage' + bytes(400))
    with pytest.raises(oops.host.VicarHostError, match=r'N0000000001_1\.IMG'):
        CassiniISS.from_file(path)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_remote_label_astrometry_has_no_abspath(tmp_path: pathlib.Path) -> None:
    """With astrometry, a remote label's image is never fetched, so abspath is absent."""

    label = Pds3Label(write_pds3_image(tmp_path), method='fast')
    label.filepath = FCPath('gs://rms-test-bucket/COISS_2001/N0000000001_1.LBL')
    obs = CassiniISS.from_file(label, astrometry=True)
    assert 'abspath' not in obs.subfields
    assert obs.image_url == 'gs://rms-test-bucket/COISS_2001/N0000000001_1.IMG'

##########################################################################################
# Parsing the PDS3 label
##########################################################################################


# A DESCRIPTION whose quoted value starts on the next line, which the fast parser
# rejects; a METHOD_DESC whose value starts on the next line and holds "=", which it
# misreads; and an unquoted N/A, which only the strict parser rejects
_NEXT_LINE_VALUE = ['DESCRIPTION = ', '    "Extended header missing.',
                    '    DSN gap truncation."']
_NEXT_LINE_EQUALS = ['METHOD_DESC = ', '    "ISSPT2.6.5 STAR:TYPE=A MAG=0.100000',
                     '    ISS_056ST_MONITOR001_PRIME_2"']
_NEXT_LINE_DESCRIPTION = 'Extended header missing.\n    DSN gap truncation.'
_UNQUOTED_NULL = ['NOTE = N/A']

# The unterminated DESCRIPTION of the calibrated COISS_1xxx_v1 labels
_UNTERMINATED = ['DESCRIPTION = "', 'Calibrated using CISSCAL 3.6:',
                 '  Gain applied = 30.2700']


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('extra_lines', 'key', 'value'), [
    (_NEXT_LINE_VALUE, 'DESCRIPTION', _NEXT_LINE_DESCRIPTION),
    (_NEXT_LINE_EQUALS, 'METHOD_DESC',
     'ISSPT2.6.5 STAR:TYPE=A MAG=0.100000\n    ISS_056ST_MONITOR001_PRIME_2'),
    (_UNTERMINATED, 'DESCRIPTION',
     '\nCalibrated using CISSCAL 3.6:\n  Gain applied = 30.2700'),
], ids=['next-line', 'next-line-misread', 'unterminated'])
def test_from_file_label_repairs(tmp_path: pathlib.Path, extra_lines: list[str], key: str,
                                 value: str) -> None:
    """A label that the fast parser rejects or misreads is repaired and read in full."""

    path = write_pds3_image(tmp_path, extra_lines=extra_lines)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.label_dict[key] == value
    assert all(iss._PDS3_NAME.fullmatch(name) for name in obs.label_dict)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('extra_lines', 'method'), [
    (_NEXT_LINE_VALUE, 'strict'),
    (_NEXT_LINE_VALUE + _UNQUOTED_NULL, 'loose'),
], ids=['strict', 'loose'])
def test_from_file_parser_fallbacks(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    extra_lines: list[str], method: str) -> None:
    """Without the next-line repair, such a label falls back to a slower parser."""

    monkeypatch.setattr(iss, '_NEXT_LINE_VALUE', (r'^\b$', ''))
    methods = []
    read_fileinfo = CassiniISS._read_fileinfo

    def recording_read_fileinfo(*args: Any, **kwargs: Any) -> Any:
        methods.append(kwargs['pds3_options']['method'])
        return read_fileinfo(*args, **kwargs)

    monkeypatch.setattr(CassiniISS, '_read_fileinfo', recording_read_fileinfo)
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=extra_lines),
                               astrometry=True)
    assert obs.label_dict['DESCRIPTION'] == _NEXT_LINE_DESCRIPTION
    assert methods[-1] == method


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_explicit_parser_method_has_no_fallback(tmp_path: pathlib.Path) -> None:
    """A method other than "fast" is used alone, apart from the DESCRIPTION repair."""

    path = write_pds3_image(tmp_path, extra_lines=_NEXT_LINE_VALUE + _UNQUOTED_NULL)
    with pytest.raises(oops.host.PdsHostError, match="Expected 'END', found 'NOTE'"):
        CassiniISS.from_file(path, astrometry=True, pds3_options={'method': 'strict'})


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_unparsable_label(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, extra_lines=['X = (1, 2'])
    with pytest.raises(oops.host.PdsHostError, match="found 'X'"):
        CassiniISS.from_file(path, astrometry=True)

##########################################################################################
# Pointing
##########################################################################################


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_without_pointing(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """An image with no C kernel pointing raises HostError ending in the SPICE code."""

    def no_pointing(*args: Any, **kwargs: Any) -> None:
        raise OSError('SPICE(CKINSUFFDATA) -- insufficient C kernel data')

    CassiniISS._define_camera_frames()
    spice_frame = oops.Frame.as_frame('CASSINI_ISS_NAC_FLIPPED')
    monkeypatch.setattr(spice_frame, 'transform_at_time', no_pointing)
    with pytest.raises(oops.host.HostError,
                       match=r'no Cassini ISS NAC pointing at 2005-01-01T00:00:00\.500 '
                             r'for .*N0000000001_1.IMG: SPICE\(CKINSUFFDATA\)$'):
        CassiniISS.from_file(write_vicar_image(tmp_path))


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_with_frame_override_skips_pointing(tmp_path: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch
                                                      ) -> None:
    def no_pointing(*args: Any, **kwargs: Any) -> None:
        raise OSError('SPICE(CKINSUFFDATA)')

    CassiniISS._define_camera_frames()
    spice_frame = oops.Frame.as_frame('CASSINI_ISS_NAC_FLIPPED')
    monkeypatch.setattr(spice_frame, 'transform_at_time', no_pointing)
    frame = oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                               frame_id='TEST_CMATRIX')
    obs = CassiniISS.from_file(write_vicar_image(tmp_path), frame=frame)
    assert obs.frame == frame.wayframe

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
     'ERRIAPO', 'ISS_000SA_TEST002_PRIME'),
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
    assert [obs.observation_id for obs in index_snapshots] == [
        'ISS_000SA_TEST001_PRIME', 'ISS_000SA_TEST002_PRIME']


def test_from_index_times(index_snapshots: list[oops.observation.Snapshot]) -> None:
    tstart = julian.tdb_from_iso('2005-01-15T18:31:00.000')
    assert index_snapshots[1].time == pytest.approx((tstart, tstart + 0.5))


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


def test_host_from_index_dispatches_to_iss(tmp_path: pathlib.Path) -> None:
    snapshots = oops.Host.from_index(write_index(tmp_path))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']


def test_host_from_index_dispatches_by_coiss_path(tmp_path: pathlib.Path) -> None:
    directory = tmp_path / 'COISS_2001' / 'index'
    directory.mkdir(parents=True)
    snapshots = oops.Host.from_index(write_index(directory))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']


def test_host_from_index_passes_parallel(
        tmp_path: pathlib.Path, index_snapshots: list[oops.observation.Snapshot]) -> None:
    parallel = index_snapshots[0]
    snapshots = oops.Host.from_index(write_index(tmp_path), parallel=parallel)
    assert [obs.frame.reference.frame_id for obs in snapshots] == [
        parallel.frame.frame_id, parallel.frame.frame_id]


def test_from_index_filter(tmp_path: pathlib.Path) -> None:
    snapshots = CassiniISS.from_index(write_index(tmp_path),
                                      filter={'INSTRUMENT_MODE_ID': {'SUM2'}})
    assert [obs.detector for obs in snapshots] == ['WAC']


def test_from_index_leaves_out_null_items(tmp_path: pathlib.Path) -> None:
    rows = oops.Host._read_index_rows(write_index(tmp_path))
    rows[0]['TARGET_NAME'] = 'UNK'
    obs = CassiniISS.from_index(tmp_path / 'missing.lbl', row_dicts=rows[:1])[0]
    assert 'label_target' not in obs.subfields
    assert obs.target == 'NONE'


def test_from_index_uses_given_rows(tmp_path: pathlib.Path) -> None:
    rows = oops.Host._read_index_rows(write_index(tmp_path))
    snapshots = CassiniISS.from_index(tmp_path / 'missing.lbl', row_dicts=rows[:1])
    assert [obs.detector for obs in snapshots] == ['NAC']


##########################################################################################
