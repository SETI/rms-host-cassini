##########################################################################################
# tests/oops/test_iss.py
##########################################################################################
"""Tests for the Cassini ISS host, host_cassini.oops.iss."""

from collections.abc import Iterator
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, cast

import cspyce
from filecache import FCPath
import julian
import numpy as np
import oops
from pdsparser import Pds3Label
import pytest
import vicar
from conftest import (NAC_HALF_FOV_DEG, PIXELS, WAC_HALF_FOV_DEG, FakeSpice,
                      FakeSpiceDB, claim_more_rows, ik_dict, pds3_lines, tdb_in_month,
                      write_table)

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
        return_all_planets=False, overrides=all_overrides, check_pointing=True)
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
    ('metadata/COISS_2xxx/COISS_2001/COISS_2001_index.lbl', True),
    ('metadata/index.lbl', None),
    ('coiss_2001_index.lbl', True),
    ('COUVIS_0040_index.lbl', False),
    ('COVIMS_0999_index.lbl', False),
], ids=['coiss', 'no-coiss', 'lowercase', 'uvis', 'vims'])
def test_detection_in_index(tmp_path: pathlib.Path, relpath: str,
                            expected: bool | None) -> None:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\r\n'.join(_INDEX_LABEL), newline='')
    assert CassiniISS._detect_in_index(Pds3Label(path)) is expected


def test_detection_in_index_without_a_path() -> None:
    assert CassiniISS._detect_in_index(Pds3Label(_INDEX_LABEL[:-1])) is None


def test_detection_in_index_of_another_instrument() -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3', 'INSTRUMENT_ID = "VIMS"', 'END'])
    assert CassiniISS._detect_in_index(label) is False


def test_merge_supplement_by_position() -> None:
    """No ISS metadata index has a supplement, so one that is given is read in step."""

    assert CassiniISS.MERGE_SUPPLEMENT_BY_FILE is False


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
    """A GAIN_MODE_ID with an unknown number of electrons per DN leaves out gain_mode."""

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


@uses_iss
def test_snapshot_loads_kernels_for_a_cadence_override() -> None:
    """A cadence override in another month loads that month's kernels too, so the
    pointing is checked at its time with them."""

    month = 3 + int((julian.tdb_from_iso(START_TIME) - _Cassini.TDB0) // _Cassini.DTDB)
    make_snapshot(label_dict(),
                  cadence=oops.cadence.SnapCadence(tdb_in_month(month + 0.5), 2.))
    assert _Cassini.SPK_LOADED[month]
    assert _Cassini.CK_LOADED[month]


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
    path.write_text('\r\n'.join(text), newline='')
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
    """Given a label, the path subfields name the image file, not the label."""

    image_path = write_pds3_image(tmp_path).with_suffix('.IMG')
    obs = CassiniISS.from_file(tmp_path / 'N0000000001_1.LBL', astrometry=astrometry)
    assert obs.filepath == FCPath(image_path)
    assert obs.basename == 'N0000000001_1.IMG'
    assert obs.image_url == image_path.as_posix()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_gives_the_abspath_of_the_image_it_read(
        tmp_path: pathlib.Path) -> None:
    image_path = write_pds3_image(tmp_path).with_suffix('.IMG')
    obs = CassiniISS.from_file(tmp_path / 'N0000000001_1.LBL')
    assert obs.abspath == image_path.resolve()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_astrometry_has_no_abspath(tmp_path: pathlib.Path) -> None:
    """An astrometry read does not read the image file, so it gives no abspath."""

    write_pds3_image(tmp_path)
    obs = CassiniISS.from_file(tmp_path / 'N0000000001_1.LBL', astrometry=True)
    assert 'abspath' not in obs.subfields


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_remote_label_gives_the_abspath_that_retrieve_returned(
        tmp_path: pathlib.Path) -> None:
    """The image file of a remote label is retrieved, and abspath is its local copy."""

    label = Pds3Label(write_pds3_image(tmp_path), method='fast')
    label.filepath = FCPath('https://rms-test.invalid/COISS_2001/N0000000001_1.LBL')
    remote = FCPath('https://rms-test.invalid/COISS_2001/N0000000001_1.IMG')
    local = cast(pathlib.Path, remote.get_local_path())
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(image_path_of(tmp_path).read_bytes())
    obs = CassiniISS.from_file(label)
    assert obs.abspath == local.resolve()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_image_file_has_its_abspath(tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    obs = CassiniISS.from_file(path)
    assert obs.abspath == path.resolve()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_image_file_astrometry_has_no_abspath(tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_vicar_image(tmp_path), astrometry=True)
    assert 'abspath' not in obs.subfields


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar_label_gives_the_abspath_of_the_image_it_read(
        tmp_path: pathlib.Path) -> None:
    path = write_vicar_image(tmp_path)
    obs = CassiniISS.from_file(vicar.VicarLabel(path))
    assert obs.abspath == path.resolve()


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_vicar_image_given_has_no_abspath(tmp_path: pathlib.Path) -> None:
    """An image that the caller read was not retrieved by this call: no abspath."""

    path = write_vicar_image(tmp_path)
    vic = cast(vicar.VicarImage, vicar.VicarImage.from_file(path, strict=False))
    vic.filepath = path
    obs = CassiniISS.from_file(vic)
    assert 'abspath' not in obs.subfields


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
@pytest.mark.parametrize('kind', ['pds3', 'vicar-label', 'vicar-image'])
def test_from_file_rejects_other_hosts_parsed(tmp_path: pathlib.Path, kind: str) -> None:
    """Parsed content of another host's file raises WrongHostError, as its file would."""

    path = write_pds3_image(tmp_path, INSTRUMENT_HOST_NAME='GALILEO ORBITER')
    parsed: Any
    if kind == 'pds3':
        parsed = Pds3Label(path, method='fast')
    elif kind == 'vicar-label':
        parsed = vicar.VicarLabel(image_path_of(tmp_path))
    else:
        parsed = vicar.VicarImage(image_path_of(tmp_path))
    with pytest.raises(oops.host.WrongHostError, match='not a Cassini ISS '):
        CassiniISS.from_file(parsed, astrometry=True)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_truncated_image(tmp_path: pathlib.Path) -> None:
    """An image file that ends before its array does raises VicarHostError."""

    path = write_vicar_image(tmp_path)
    path.write_bytes(path.read_bytes()[:-1000])
    with pytest.raises(oops.host.VicarHostError, match=r'N0000000001_1\.IMG'):
        CassiniISS.from_file(path)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_rejects_label_without_an_image_file_name(
        tmp_path: pathlib.Path) -> None:
    """A label whose ^IMAGE pointer does not give a file name raises PdsHostError."""

    path = write_pds3_image(tmp_path)
    lines = ['^IMAGE = 2' if line.startswith('^IMAGE') else line
             for line in path.read_text().splitlines()]
    path.write_text('\r\n'.join(lines), newline='')
    with pytest.raises(oops.host.PdsHostError,
                       match=r'\^IMAGE does not give the name of the image file in '):
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
    'IMAGE_NUMBER'                : '1338928383',
    'IMAGE_TIME'                  : _EARLY_TIME,
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


@pytest.mark.parametrize(('missing', 'absent'), [
    ('FILTER1_NAME', 'FILTER_NAME'),
    ('GAIN_MODE_ID', 'GAIN_MODE_ID'),
    ('IMAGE_TIME', 'START_TIME'),
    ('IMAGE_NUMBER', 'SPACECRAFT_CLOCK_STOP_COUNT'),
])
def test_early_vicar_header_without_a_keyword(missing: str, absent: str) -> None:
    """A keyword missing from an early header leaves out what is made from it."""

    header = {key: value for (key, value) in _EARLY_VICAR.items() if key != missing}
    assert absent not in CassiniISS._pds3_dict_from_vicar(header)


def test_early_vicar_header_with_null_image_number() -> None:
    header = _EARLY_VICAR | {'IMAGE_NUMBER': 'UNK'}
    assert 'SPACECRAFT_CLOCK_STOP_COUNT' not in CassiniISS._pds3_dict_from_vicar(header)


def test_early_vicar_header_keeps_a_filter_it_cannot_pair() -> None:
    """A lone FILTER1_NAME keeps its own name, since FILTER_NAME needs both filters."""

    header = {key: value for (key, value) in _EARLY_VICAR.items()
              if key != 'FILTER2_NAME'}
    assert CassiniISS._pds3_dict_from_vicar(header)['FILTER1_NAME'] == 'CL1'


def test_current_vicar_header_is_unchanged() -> None:
    """A header that already uses the PDS3 keywords and forms is returned unchanged."""

    assert CassiniISS._pds3_dict_from_vicar(label_dict()) == label_dict()


@pytest.mark.parametrize(('item', 'header', 'expected'), [
    ('START_TIME', '2005-016T13:01:59.864Z', '2005-016T13:01:59.864'),
    ('PRODUCT_CREATION_TIME', '2005-01-17T09:46:41.000Z', '2005-01-17T09:46:41.000'),
    ('TARGET_DESC', 'Zeta 1Z', 'Zeta 1Z'),
    ('IMAGE_NUMBER', 1484573295, '1484573295'),
    ('IMAGE_OBSERVATION_TYPE', 'SCIENCE', ['SCIENCE']),
    ('IMAGE_OBSERVATION_TYPE', ['SCIENCE', 'SUPPORT'], ['SCIENCE', 'SUPPORT']),
    ('OPTICS_TEMPERATURE', 0.712693, [0.712693, None]),
    ('OPTICS_TEMPERATURE', [0.79789, 1.90571], [0.79789, 1.90571]),
    ('MISSION_PHASE_NAME', 'APPROACH_SCIENCE', 'APPROACH SCIENCE'),
], ids=['time', 'calendar-time', 'not-a-time', 'image-number', 'observation-type',
        'observation-types', 'optics-temperature', 'optics-temperatures',
        'mission-phase'])
def test_vicar_header_takes_pds3_forms(item: str, header: Any, expected: Any) -> None:
    """A header value is written in the form that the PDS3 labels give the item."""

    header_dict = label_dict(**{item: header})
    assert CassiniISS._pds3_dict_from_vicar(header_dict)[item] == expected


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
# Clock counts of the early cruise images
##########################################################################################

# The counts of an early cruise image as its label gives them: the start count is the
# stop count minus the exposure of 2000 ms counted as 2000 seconds. The test SCLK reads a
# count as seconds, so the start count computed from the stop count and the exposure is
# two seconds before it.
_EARLY_COUNTS: dict[str, Any] = {'IMAGE_NUMBER': '1294562651',
                                 'EXPOSURE_DURATION': 2000.,
                                 'SPACECRAFT_CLOCK_START_COUNT': '1294560651.000',
                                 'SPACECRAFT_CLOCK_STOP_COUNT': '1294562651.000'}
_EARLY_START_COUNT = '1294562649.000'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_computes_the_early_cruise_start_count(tmp_path: pathlib.Path) -> None:
    """The start count is computed from the stop count and the exposure."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path, **_EARLY_COUNTS))
    assert obs.spacecraft_clock_start_count == _EARLY_START_COUNT


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_dict_keeps_the_archived_start_count(
        tmp_path: pathlib.Path) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, **_EARLY_COUNTS))
    assert obs.label_dict['SPACECRAFT_CLOCK_START_COUNT'] == '1294560651.000'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_early_vicar_header_gets_a_start_count(tmp_path: pathlib.Path) -> None:
    """An early header lacks the start count, which is computed from its stop count."""

    obs = CassiniISS.from_file(write_vicar_image(tmp_path, keywords=_EARLY_VICAR))
    assert obs.spacecraft_clock_start_count == '1338928382.500'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_keeps_a_later_start_count(tmp_path: pathlib.Path) -> None:
    """An image after the early cruise keeps the start count that its label gives."""

    counts = _EARLY_COUNTS | {'IMAGE_NUMBER': '1347928997',
                              'SPACECRAFT_CLOCK_START_COUNT': '1347928994.223',
                              'SPACECRAFT_CLOCK_STOP_COUNT': '1347928997.123'}
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, **counts))
    assert obs.spacecraft_clock_start_count == '1347928994.223'


def test_from_index_computes_the_early_cruise_counts(tmp_path: pathlib.Path) -> None:
    """Both counts of an early cruise row follow from its image number and exposure.

    The metadata index gives the label's counts.
    """

    rows = index_rows(tmp_path)
    row = rows[0] | _EARLY_COUNTS | {'SPACECRAFT_CLOCK_START_COUNT': '1294560651.000',
                                     'SPACECRAFT_CLOCK_STOP_COUNT': '1294562651.000'}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.spacecraft_clock_stop_count == '1294562651.000'
    assert obs.spacecraft_clock_start_count == _EARLY_START_COUNT

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
def test_from_file_label_dict_holds_the_label_keywords(tmp_path: pathlib.Path) -> None:
    """label_dict holds every keyword of the label, with its time as the label's text."""

    path = write_pds3_image(tmp_path)
    obs = CassiniISS.from_file(path)
    parsed = Pds3Label(path, method='fast').as_dict()
    keywords = [key for key in parsed
                if key.upper() == key and parsed[key] is not None]
    assert list(obs.label_dict) == keywords
    assert obs.label_dict['START_TIME'] == START_TIME


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_header_dict_is_verbatim(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, header={'TELEMETRY_FORMAT_ID': 'UNK'})
    obs = CassiniISS.from_file(path)
    assert obs.header_dict == vicar.VicarLabel(image_path_of(tmp_path)).as_dict()
    assert obs.header_dict['TELEMETRY_FORMAT_ID'] == 'UNK'


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('key', ['END', 'objects', 'IMAGE_TIME_fmt', 'IMAGE_TIME_day',
                                 'IMAGE_OBSERVATION_TYPE_list', 'IMAGE_TIME_source',
                                 '^IMAGE_offset'])
def test_from_file_label_dict_leaves_out_parser_keys(tmp_path: pathlib.Path,
                                                     key: str) -> None:
    """label_dict leaves out END and the keys the parser derives from a keyword."""

    path = write_pds3_image(tmp_path, extra_lines=_PDS3_FORMS)
    obs = CassiniISS.from_file(path, pds3_options={'sources': True})
    assert key not in obs.label_dict


# Label lines in the PDS3 forms that the parser does not give as text: a time, a time
# written without its milliseconds, a set, and an unquoted -999 fill value
_PDS3_FORMS = ['IMAGE_TIME = 2005-016T13:02:08.064', 'IMAGE_MID_TIME = 2005-016T13:02:04',
               'IMAGE_OBSERVATION_TYPE = {"SCIENCE"}', 'FILTER_TEMPERATURE = -999.0']


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_dict_writes_times_and_sets_as_the_label_does(
        tmp_path: pathlib.Path) -> None:
    """label_dict gives each time as its label text and each set as a list."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=_PDS3_FORMS))
    assert obs.label_dict['START_TIME'] == START_TIME
    assert obs.label_dict['IMAGE_TIME'] == '2005-016T13:02:08.064'
    assert obs.label_dict['IMAGE_MID_TIME'] == '2005-016T13:02:04'
    assert obs.label_dict['IMAGE_OBSERVATION_TYPE'] == ['SCIENCE']
    assert obs.label_dict['FILTER_TEMPERATURE'] == -999.


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_header_dict_keeps_repeated_parameters(tmp_path: pathlib.Path) -> None:
    """A repeated VICAR parameter appears once per occurrence, as NAME[n]."""

    image_path = write_vicar_image(tmp_path)
    vic = cast(vicar.VicarImage, vicar.VicarImage.from_file(image_path, strict=False))
    vic.label.append([('TASK', 'COPY'), ('TASK', 'CISSCAL')])
    vic.write_file(image_path)
    obs = CassiniISS.from_file(image_path)
    assert obs.header_dict['TASK[0]'] == 'COPY'
    assert obs.header_dict['TASK[1]'] == 'CISSCAL'
    assert obs.dict['TASK[1]'] == 'CISSCAL'
    assert 'TASK' not in obs.header_dict


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('source', ['label', 'image'])
@pytest.mark.parametrize('name', ['dict', 'label_dict', 'header_dict'])
def test_from_file_dicts_are_json_serializable(tmp_path: pathlib.Path, source: str,
                                               name: str) -> None:
    write_pds3_image(tmp_path, extra_lines=_PDS3_FORMS)
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path)
    if name in obs.subfields:
        json.dumps(obs.subfields[name])


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('item', 'header'), [
    ('START_TIME', START_TIME + 'Z'),
    ('IMAGE_TIME', '2005-016T13:02:08.064Z'),
    ('IMAGE_OBSERVATION_TYPE', 'SCIENCE'),
    ('IMAGE_NUMBER', 1484573295),
])
def test_from_file_gives_an_item_one_form_from_either_input(tmp_path: pathlib.Path,
                                                            item: str,
                                                            header: Any) -> None:
    """An item the two sources agree on has the same value and type from either input."""

    extra_lines = [*_PDS3_FORMS, 'IMAGE_NUMBER = "1484573295"']
    label_path = write_pds3_image(tmp_path, header={item: header},
                                  extra_lines=extra_lines)
    from_label = CassiniISS.from_file(label_path).dict[item]
    from_image = CassiniISS.from_file(image_path_of(tmp_path)).dict[item]
    assert from_label == from_image
    assert type(from_label) is type(from_image)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_dict_holds_only_items(tmp_path: pathlib.Path) -> None:
    """dict leaves out END and the keys that the label parser derives from a keyword."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=_PDS3_FORMS))
    assert {'END', 'objects', 'IMAGE_TIME_fmt', 'IMAGE_TIME_day', 'IMAGE_TIME_sec',
            'IMAGE_OBSERVATION_TYPE_list'}.isdisjoint(obs.dict)
    assert 'LINES' in obs.dict['IMAGE']
    assert all(value is not None for value in obs.dict.values())


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
    ('MISSION_PHASE_NAME', 'SCIENCE CRUISE', 'CHECK_OUT_2', 'CHECK OUT 2'),
    ('BIAS_STRIP_MEAN', 14.950098, 14.9501, 14.950098),
    ('BIAS_STRIP_MEAN', 14.95, 14.950098, 14.95),
    ('EXPECTED_MAXIMUM', [0.072124, 0.079518], [0.0721239, 0.0795176],
     [0.072124, 0.079518]),
    ('OPTICS_TEMPERATURE', [0.71269, 'UNK'], 0.712693, [0.71269, None]),
    ('EXPOSURE_DURATION', 6.0, 10.0, 6.0),
    ('TELEMETRY_FORMAT_ID', 'UNK', 'S_N_ER_3', 'S_N_ER_3'),
    ('TELEMETRY_FORMAT_ID', 'S_N_ER_3', 'N/A', 'S_N_ER_3'),
    ('MISSION_PHASE_NAME', 'TOUR', ' unk ', 'TOUR'),
    ('DETECTOR_TEMPERATURE', -999.0, -89.2435, -89.2435),
], ids=['label-by-default', 'label-by-rule', 'header-by-rule', 'header-phase',
        'label-more-digits', 'label-fewer-digits', 'label-list', 'label-pair',
        'label-unlike-header', 'label-null', 'header-null', 'header-padded-null',
        'label-fill'])
def test_from_file_resolves_disagreements(tmp_path: pathlib.Path, item: str, label: Any,
                                          header: Any, expected: Any) -> None:
    """Where the two sources disagree, dict holds the value of the item's fixed source.

    A value missing from that source, as a null or fill value, is taken from the other.
    """

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

    write_pds3_image(tmp_path, TARGET_NAME='unk', OBSERVATION_ID=' N/A ',
                     GAIN_MODE_ID='NULL', FILTER_NAME=['UNK', 'UNK'],
                     DETECTOR_TEMPERATURE=-999.0, OPTICS_TEMPERATURE=[-999.0, -999.0])
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path)
    absent = {'label_target', 'observation_id', 'gain_mode', 'filter', 'filter1',
              'filter2'}
    assert absent.isdisjoint(obs.subfields)
    assert {'TARGET_NAME', 'OBSERVATION_ID', 'GAIN_MODE_ID', 'FILTER_NAME',
            'DETECTOR_TEMPERATURE', 'OPTICS_TEMPERATURE'}.isdisjoint(obs.dict)
    assert all(value is not None for value in obs.subfields.values())
    assert obs.target == 'NONE'


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('source', ['label', 'image'])
def test_from_file_marks_a_missing_list_element_none(tmp_path: pathlib.Path,
                                                     source: str) -> None:
    """A null or fill element of a list is None, keeping the others in their places."""

    write_pds3_image(tmp_path, OPTICS_TEMPERATURE=[5.914901, -999.0],
                     FILTER_NAME=['CL1', 'UNK'])
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    obs = CassiniISS.from_file(path)
    assert obs.dict['OPTICS_TEMPERATURE'] == [5.914901, None]
    assert obs.dict['FILTER_NAME'] == ['CL1', None]
    assert {'filter', 'filter1', 'filter2'}.isdisjoint(obs.subfields)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('keyword', ['FILTER_TEMPERATURE',
                                     'INSTRUMENT_DATA_RATE', 'INST_CMPRS_RATE',
                                     'INST_CMPRS_RATIO', 'SENSOR_HEAD_ELEC_TEMPERATURE'])
def test_from_file_leaves_out_minus_999_in_a_fill_keyword(tmp_path: pathlib.Path,
                                                          keyword: str) -> None:
    changes: dict[str, Any] = {keyword: -999.0}
    path = write_pds3_image(tmp_path, **changes)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert keyword not in obs.dict


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_keeps_minus_999_where_it_can_be_real(tmp_path: pathlib.Path) -> None:
    """-999 is missing only in the temperatures, rates and ratios, which it cannot be."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path, DARK_STRIP_MEAN=-999.0))
    assert obs.dict['DARK_STRIP_MEAN'] == -999.


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
                       match=f'{item} is missing from .*N0000000001_1.IMG'):
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
    assert obs.data[1, 2] == 258 % 256
    # A file system that ignores case finds the label's spelling first
    assert obs.basename.upper() == 'N0000000001_1.IMG'


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
    label.filepath = FCPath('https://rms-test.invalid/COISS_2001/N0000000001_1.LBL')
    obs = CassiniISS.from_file(label, astrometry=True)
    assert 'abspath' not in obs.subfields
    assert obs.image_url == 'https://rms-test.invalid/COISS_2001/N0000000001_1.IMG'

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

# The DESCRIPTION of N1640662528_3_CALIB.LBL in calibrated/COISS_2059, closed on a line of
# its own and followed by the raw label's next-line description
_SPLIT = ['DESCRIPTION = "', 'Calibrated using CISSCAL 4.0:', "  UNITS = 'I/F'", '"',
          '    "Extended header missing.', '    DSN gap truncation." ']


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('extra_lines', 'key', 'value'), [
    (_NEXT_LINE_VALUE, 'DESCRIPTION', _NEXT_LINE_DESCRIPTION),
    (_NEXT_LINE_EQUALS, 'METHOD_DESC',
     'ISSPT2.6.5 STAR:TYPE=A MAG=0.100000\n    ISS_056ST_MONITOR001_PRIME_2'),
    (_UNTERMINATED, 'DESCRIPTION',
     '\nCalibrated using CISSCAL 3.6:\n  Gain applied = 30.2700'),
    (_SPLIT, 'DESCRIPTION',
     "\nCalibrated using CISSCAL 4.0:\n  UNITS = 'I/F'\n    Extended header missing.\n"
     '    DSN gap truncation.'),
], ids=['next-line', 'next-line-misread', 'unterminated', 'split'])
def test_from_file_label_repairs(tmp_path: pathlib.Path, extra_lines: list[str], key: str,
                                 value: str) -> None:
    """A label that the fast parser rejects or misreads is repaired and read in full."""

    path = write_pds3_image(tmp_path, extra_lines=extra_lines)
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.label_dict[key] == value
    assert all(iss._PDS3_NAME.fullmatch(name) for name in obs.label_dict)


@pytest.mark.usefixtures('initialized_iss')
def test_host_from_file_reads_a_label_only_iss_repairs(tmp_path: pathlib.Path) -> None:
    """oops.Host.from_file reaches the host for a label that only its repairs read."""

    path = write_pds3_image(tmp_path, extra_lines=_SPLIT)
    assert oops.Host.from_file(path, astrometry=True).detector == 'NAC'


def test_label_correction_leaves_a_label_without_defects(tmp_path: pathlib.Path) -> None:
    text = write_pds3_image(tmp_path).read_text().replace('\r\n', '\n')
    assert CassiniISS._correct_label_text(text, FCPath(tmp_path)) == text


def test_label_correction_leaves_a_closed_description_with_a_keyword_line(
        tmp_path: pathlib.Path) -> None:
    """A DESCRIPTION closed on a later line keeps a line inside it that starts like a
    keyword."""

    text = 'DESCRIPTION = "Line one\nBIAS = 0 is expected."\nEND\n'
    assert CassiniISS._correct_label_text(text, FCPath(tmp_path)) == text


def test_label_correction_moves_a_next_line_value_up(tmp_path: pathlib.Path) -> None:
    """A quoted value on the line after its keyword joins that line, in the text with
    <LF> line terminators that oops passes to the correction."""

    text = 'A =\n    "x = 1"\n'
    assert CassiniISS._correct_label_text(text, FCPath(tmp_path)) == 'A = "x = 1"\n'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_reads_a_next_line_value_that_only_the_fast_parser_reads(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A label file with <CR><LF> line terminators whose next-line value holds "=" is read
    in full by the fast parser alone once the value is moved up, as N1683362654_1.LBL of
    COISS_2068 is, which the strict and loose parsers reject for another reason."""

    monkeypatch.setattr(oops.Host, '_parse_pds3', fail_after_fast(oops.Host._parse_pds3))
    path = write_pds3_image(tmp_path, extra_lines=_NEXT_LINE_EQUALS)
    assert b'METHOD_DESC = \r\n' in path.read_bytes()
    obs = CassiniISS.from_file(path, astrometry=True)
    assert obs.label_dict['METHOD_DESC'] == ('ISSPT2.6.5 STAR:TYPE=A MAG=0.100000\n'
                                             '    ISS_056ST_MONITOR001_PRIME_2')


def without_next_line_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the next-line repair out of the label correction."""

    monkeypatch.setattr(iss, '_LABEL_REPAIRS',
                        tuple(repair for repair in iss._LABEL_REPAIRS
                              if repair != iss._NEXT_LINE_VALUE))


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('extra_lines', [_NEXT_LINE_VALUE,
                                         _NEXT_LINE_VALUE + _UNQUOTED_NULL],
                         ids=['strict', 'loose'])
def test_from_file_parser_fallbacks(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    extra_lines: list[str]) -> None:
    """Without the next-line repair, such a label is read by a slower parser."""

    without_next_line_repair(monkeypatch)
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=extra_lines),
                               astrometry=True)
    assert obs.label_dict['DESCRIPTION'] == _NEXT_LINE_DESCRIPTION


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_reads_a_fast_misread_with_a_later_parser(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A label that the fast parser reads with a keyword that is not a PDS3 name is read
    again from the strict parser on."""

    without_next_line_repair(monkeypatch)
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=_NEXT_LINE_EQUALS),
                               astrometry=True)
    assert all(iss._PDS3_NAME.fullmatch(name) for name in obs.label_dict)
    assert obs.label_dict['METHOD_DESC'] == ('ISSPT2.6.5 STAR:TYPE=A MAG=0.100000\n'
                                             '    ISS_056ST_MONITOR001_PRIME_2')


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_keeps_a_fast_misread_that_no_later_parser_reads(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast parser's result is kept only after the strict parser, from which the
    later ones are tried, has failed."""

    without_next_line_repair(monkeypatch)
    methods: list[str] = []
    monkeypatch.setattr(oops.Host, '_parse_pds3',
                        fail_after_fast(oops.Host._parse_pds3, methods))
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, extra_lines=_NEXT_LINE_EQUALS),
                               astrometry=True)
    assert methods == ['fast', 'strict']
    assert obs.label_dict['TARGET_NAME'] == 'SATURN'


def fail_after_fast(parse_pds3: Any, methods: list[str] | None = None) -> Any:
    """A replacement for oops.Host._parse_pds3 under which only the fast method reads a
    label. The method of each call is appended to `methods`, if it is given."""

    def parse(filepath: Any, pds3_options: dict[str, Any], **kwargs: Any) -> Any:
        if methods is not None:
            methods.append(pds3_options['method'])
        if pds3_options['method'] != 'fast':
            raise oops.host.PdsHostError(f'cannot parse the PDS3 label {filepath}')
        return parse_pds3(filepath, pds3_options, **kwargs)

    return staticmethod(parse)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_explicit_parser_method_falls_back_to_later_ones(
        tmp_path: pathlib.Path) -> None:
    """A requested method is followed by the methods after it in fast, strict, loose."""

    path = write_pds3_image(tmp_path, extra_lines=_NEXT_LINE_VALUE + _UNQUOTED_NULL)
    obs = CassiniISS.from_file(path, astrometry=True, pds3_options={'method': 'strict'})
    assert obs.label_dict['NOTE'] == 'N/A'


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_unparsable_label(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path, extra_lines=['X = (1, 2'])
    with pytest.raises(oops.host.PdsHostError,
                       match=r"cannot parse the PDS3 label .*N0000000001_1\.LBL: "
                             r"method fast: .*found 'X'"):
        CassiniISS.from_file(path, astrometry=True)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_takes_one_repair_as_a_tuple(tmp_path: pathlib.Path) -> None:
    """A single repair in pds3_options may be given as one (pattern, replacement)."""

    path = write_pds3_image(tmp_path, TARGET_NAME='TITAN')
    obs = CassiniISS.from_file(path, astrometry=True,
                               pds3_options={'repairs': ('"TITAN"', '"RHEA"')})
    assert obs.label_dict['TARGET_NAME'] == 'RHEA'

##########################################################################################
# Pointing
##########################################################################################


def remove_pointing(monkeypatch: pytest.MonkeyPatch,
                    error: Exception | None = None) -> list[float]:
    """Make the SPICE frame of the NAC raise `error` at every time; return those times.

    By default the error is the OSError that cspyce raises for a time no C kernel covers.
    """

    times: list[float] = []

    def no_pointing(time: float, *args: Any, **kwargs: Any) -> None:
        times.append(time)
        raise error or OSError('SPICE(CKINSUFFDATA) -- insufficient C kernel data')

    CassiniISS._define_camera_frames()
    spice_frame = oops.Frame.as_frame('CASSINI_ISS_NAC_FLIPPED')
    monkeypatch.setattr(spice_frame, 'transform_at_time', no_pointing)
    return times


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('source', ['label', 'image'])
def test_from_file_without_pointing(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    """An image with no C kernel pointing raises OopsSpiceError naming the image file."""

    write_pds3_image(tmp_path)
    path = label_path_of(tmp_path) if source == 'label' else image_path_of(tmp_path)
    remove_pointing(monkeypatch)
    with pytest.raises(oops.OopsSpiceError,
                       match=r'^no Cassini ISS NAC pointing at 2005-01-01T00:00:00\.500 '
                             r'for .*N0000000001_1\.IMG: SPICE\(CKINSUFFDATA\)$') as info:
        CassiniISS.from_file(path)
    assert info.value.spice_code == 'SPICE(CKINSUFFDATA)'
    assert FCPath(info.value.filepath).name == 'N0000000001_1.IMG'
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('error', [KeyError('SPICE(FRAMEIDNOTFOUND)'),
                                   RuntimeError('SPICE(NOFRAMECONNECT)'),
                                   ValueError('SPICE(INVALIDTIME)')],
                         ids=['lookup', 'runtime', 'value'])
def test_from_file_reports_each_spice_error_class(tmp_path: pathlib.Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  error: Exception) -> None:
    remove_pointing(monkeypatch, error)
    with pytest.raises(oops.OopsSpiceError, match=r'SPICE\([A-Z]+\)$'):
        CassiniISS.from_file(write_vicar_image(tmp_path))


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_lets_other_errors_through(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception of a class that SPICE does not raise is not reported as SPICE's."""

    remove_pointing(monkeypatch, TypeError('not a time'))
    with pytest.raises(TypeError, match='not a time'):
        CassiniISS.from_file(write_vicar_image(tmp_path))


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_checks_pointing_at_the_cadence_midtime(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a cadence override, the pointing is checked at the override's midtime."""

    times = remove_pointing(monkeypatch)
    cadence = oops.cadence.SnapCadence(julian.tdb_from_iso('2006-01-01T00:00:00'), 10.)
    with pytest.raises(oops.OopsSpiceError, match=r' at 2006-01-01T00:00:05\.000 '):
        CassiniISS.from_file(write_vicar_image(tmp_path), cadence=cadence)
    assert times == [cadence.midtime]


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('astrometry', [False, True])
def test_from_file_with_frame_override_skips_pointing(tmp_path: pathlib.Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      astrometry: bool) -> None:
    times = remove_pointing(monkeypatch)
    frame = oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                               frame_id='TEST_CMATRIX')
    obs = CassiniISS.from_file(write_vicar_image(tmp_path), frame=frame,
                               astrometry=astrometry)
    assert obs.frame == frame.wayframe
    assert times == []

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


def write_index(directory: pathlib.Path, *,
                extra: tuple[tuple[str, str, int, tuple[Any, ...],
                                   str | tuple[str, ...]], ...] = ()
                ) -> pathlib.Path:
    """Write a two-row Cassini ISS index table and label; return the label's path.

    Each item of `extra` adds a column: its name, data type, width, its value in each
    row, and a line that its label gives it, such as "NULL_CONSTANT = 0.", or a tuple
    of such lines, or "" for none.
    """

    columns = _INDEX_COLUMNS + [(name, kind, width)
                                for (name, kind, width, _, _) in extra]
    rows = [row + tuple(values[k] for (_, _, _, values, _) in extra)
            for (k, row) in enumerate(_INDEX_ROWS)]
    declarations = {name: (lines,) if isinstance(lines, str) else lines
                    for (name, _, _, _, lines) in extra if lines}
    widths = [width + 2 if kind == 'CHARACTER' else width for (_, kind, width) in columns]
    row_bytes = sum(widths) + len(widths) - 1 + 2

    lines = []
    for row in rows:
        fields = []
        for (value, (_, kind, width)) in zip(row, columns, strict=True):
            if kind == 'CHARACTER':
                fields.append('"' + str(value).ljust(width) + '"')
            elif kind == 'ASCII_INTEGER':
                fields.append(f'{value:{width}d}')
            else:
                fields.append(f'{value:{width}.3f}')
        lines.append(','.join(fields) + '\r\n')
    (directory / 'INDEX.TAB').write_text(''.join(lines), newline='')

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {row_bytes}', f'FILE_RECORDS = {len(rows)}',
            '^IMAGE_INDEX_TABLE = "INDEX.TAB"', 'OBJECT = IMAGE_INDEX_TABLE',
            '  INTERCHANGE_FORMAT = ASCII', f'  ROWS = {len(rows)}',
            f'  COLUMNS = {len(columns)}', f'  ROW_BYTES = {row_bytes}']
    start = 1
    for ((name, kind, width), field_width) in zip(columns, widths, strict=True):
        offset = 1 if kind == 'CHARACTER' else 0
        text += ['  OBJECT = COLUMN', f'    NAME = {name}', f'    DATA_TYPE = {kind}',
                 f'    START_BYTE = {start + offset}', f'    BYTES = {width}']
        text += ['    ' + line for line in declarations.get(name, ())]
        text.append('  END_OBJECT = COLUMN')
        start += field_width + 1
    text += ['END_OBJECT = IMAGE_INDEX_TABLE', 'END', '']
    path = directory / 'INDEX.LBL'
    path.write_text('\r\n'.join(text), newline='')
    return path


def index_rows(directory: pathlib.Path) -> list[dict[str, Any]]:
    """The rows of the index that write_index writes, as oops reads them."""

    return list(CassiniISS._index_rows(write_index(directory), supplement=None))


def index_label(directory: pathlib.Path) -> Pds3Label:
    """The parsed label of the index that write_index has written in `directory`, for
    rows given with the path of an index whose label is elsewhere."""

    return Pds3Label(directory / 'INDEX.LBL', method='fast')


@pytest.fixture
def index_snapshots(tmp_path: pathlib.Path) -> list[oops.observation.Snapshot]:
    return list(CassiniISS.from_index(write_index(tmp_path)))


def test_from_index_makes_one_snapshot_per_row(
        index_snapshots: list[oops.observation.Snapshot]) -> None:
    assert len(index_snapshots) == 2


def test_from_index_builds_each_row_when_reached(tmp_path: pathlib.Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """The result is an iterator that builds each row's Snapshot when it reaches the row
    and yields the rows in index order."""

    built: list[Any] = []
    make_snapshot = CassiniISS._make_snapshot

    def spy(dict_: dict[str, Any], **kwargs: Any) -> Any:
        built.append(kwargs['filepath'])
        return make_snapshot(dict_, **kwargs)

    monkeypatch.setattr(CassiniISS, '_make_snapshot', spy)
    snapshots = CassiniISS.from_index(write_index(tmp_path))
    assert isinstance(snapshots, Iterator)
    assert built == []
    first = next(snapshots)
    assert len(built) == 1
    assert [first.basename] + [obs.basename for obs in snapshots] == [
        'N1484506648_1.IMG', 'W1484506649_1.IMG']


def test_from_index_raises_a_row_error_when_reached(tmp_path: pathlib.Path) -> None:
    """A row that cannot give a Snapshot raises when the iteration reaches it, after the
    rows before it have been yielded."""

    rows = index_rows(tmp_path)
    del rows[1]['START_TIME']
    snapshots = CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=rows)
    assert next(snapshots).basename == 'N1484506648_1.IMG'
    with pytest.raises(oops.host.HostFormatError,
                       match=r'START_TIME is missing from .*W1484506649_1\.IMG'):
        next(snapshots)


def test_from_index_filepaths(tmp_path: pathlib.Path,
                              index_snapshots: list[oops.observation.Snapshot]) -> None:
    """An index outside the metadata tree is in the directory that holds the volumes."""

    image_path = tmp_path / 'COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG'
    obs = index_snapshots[0]
    assert obs.filepath == FCPath(image_path)
    assert obs.basename == 'N1484506648_1.IMG'
    assert obs.image_url == image_path.as_posix()
    assert index_snapshots[1].basename == 'W1484506649_1.IMG'


def test_from_index_local_image_has_no_abspath(tmp_path: pathlib.Path) -> None:
    """A row's image file is never read, so it has no abspath, even on the local disk."""

    image_path = tmp_path / 'COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG'
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b'')
    obs = next(CassiniISS.from_index(write_index(tmp_path)))
    assert 'abspath' not in obs.subfields


def test_from_index_creates_no_directories(tmp_path: pathlib.Path) -> None:
    """Building the rows of an index whose volumes are absent creates nothing."""

    index = tmp_path / 'metadata' / 'COISS_2xxx' / 'COISS_2009' / 'COISS_2009_index.lbl'
    index.parent.mkdir(parents=True)
    rows = index_rows(tmp_path)
    before = sorted(tmp_path.rglob('*'))
    list(CassiniISS.from_index(index, row_dicts=rows, label=index_label(tmp_path)))
    assert sorted(tmp_path.rglob('*')) == before


@pytest.mark.parametrize(('index_path', 'image_path'), [
    ('metadata/COISS_2xxx/COISS_2999/COISS_2999_index.lbl',
     'volumes/COISS_2xxx/COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG'),
    ('copy/COISS_2009_index.lbl',
     'copy/COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG'),
], ids=['cumulative-metadata-index', 'elsewhere'])
def test_from_index_finds_the_image_from_the_index(tmp_path: pathlib.Path,
                                                   index_path: str,
                                                   image_path: str) -> None:
    """A row's image file is its specification within the volume of its VOLUME_ID.

    An index in the metadata tree lists rows of any volume of its series, each found in
    the volumes tree beside it; any other index is in the directory that holds the
    volumes.
    """

    rows = index_rows(tmp_path)
    obs = next(CassiniISS.from_index(tmp_path / index_path, row_dicts=rows[:1],
                                     label=index_label(tmp_path)))
    assert obs.filepath == FCPath(tmp_path / image_path)


def test_from_index_remote_image_has_no_abspath(tmp_path: pathlib.Path) -> None:
    """A remote image that was never fetched has a URL but no abspath."""

    rows = index_rows(tmp_path)
    index_url = 'https://rms-test.invalid/holdings/metadata/COISS_2xxx/COISS_2009'
    obs = next(CassiniISS.from_index(index_url + '/COISS_2009_index.lbl',
                                     row_dicts=rows[:1], label=index_label(tmp_path)))
    assert obs.image_url == ('https://rms-test.invalid/holdings/volumes/COISS_2xxx/'
                             'COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG')
    assert 'abspath' not in obs.subfields
    assert obs.basename == 'N1484506648_1.IMG'


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


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_parallel_override(
        tmp_path: pathlib.Path, index_snapshots: list[oops.observation.Snapshot]) -> None:
    other = index_snapshots[0]
    obs = CassiniISS.from_file(write_pds3_image(tmp_path), parallel=other)
    assert obs.frame.reference.frame_id == other.frame.frame_id


def test_host_from_index_dispatches_to_iss(tmp_path: pathlib.Path) -> None:
    snapshots = oops.Host.from_index(write_index(tmp_path))
    assert [obs.detector for obs in snapshots] == ['NAC', 'WAC']


def test_host_from_index_dispatches_by_coiss_path(tmp_path: pathlib.Path) -> None:
    directory = tmp_path / 'metadata' / 'COISS_2xxx' / 'COISS_2001'
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
    rows = index_rows(tmp_path)
    rows[0]['TARGET_NAME'] = 'UNK'
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=rows[:1]))
    assert 'label_target' not in obs.subfields
    assert obs.target == 'NONE'


def test_from_index_uses_given_rows(tmp_path: pathlib.Path) -> None:
    rows = index_rows(tmp_path)
    snapshots = CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=rows[:1])
    assert [obs.detector for obs in snapshots] == ['NAC']


def test_from_index_dict_is_json_serializable(tmp_path: pathlib.Path) -> None:
    """dict is the row in Python types, without the table's "_mask" entries."""

    rows = index_rows(tmp_path)
    row = rows[0] | {'VALID_MAXIMUM': np.array([4095, 9896]),
                     'VALID_MAXIMUM_mask': np.array([False, False]),
                     'SEQUENCE_NUMBER': np.int64(2), 'SEQUENCE_NUMBER_mask': np.False_,
                     'TARGET_DESC': np.str_('Mimas  '), 'TARGET_DESC_mask': np.False_}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    json.dumps(obs.dict)
    assert obs.dict['VALID_MAXIMUM'] == [4095, 9896]
    assert obs.dict['SEQUENCE_NUMBER'] == 2
    assert obs.dict['TARGET_DESC'] == 'Mimas'
    assert {'TARGET_DESC_mask', 'SEQUENCE_NUMBER_mask'}.isdisjoint(obs.dict)
    assert type(obs.label_target) is str


@pytest.mark.parametrize('column', ['BIAS_STRIP_MEAN', 'DARK_STRIP_MEAN'],
                         ids=['bias', 'dark'])
def test_from_index_keeps_a_value_declared_null_as_zero(tmp_path: pathlib.Path,
                                                        column: str) -> None:
    """A null constant of 0, which the metadata indexes declare for these columns while
    the image labels give 0 as the value, is not applied."""

    path = write_index(tmp_path, extra=((column, 'ASCII_REAL', 10, (0., 5.),
                                         'NULL_CONSTANT = 0.'),))
    obs = next(CassiniISS.from_index(path))
    assert obs.dict[column] == 0.


def test_from_index_applies_a_declared_null_other_than_zero(tmp_path: pathlib.Path
                                                            ) -> None:
    """A corrected label, which declares the fill value its table writes, is followed."""

    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (-1., 5.),
                                         'NULL_CONSTANT = -1.'),))
    obs = next(CassiniISS.from_index(path))
    assert 'BIAS_STRIP_MEAN' not in obs.dict


def test_from_index_applies_a_special_constant_other_than_the_null_constant(
        tmp_path: pathlib.Path) -> None:
    """A corrected label may declare its fill value with another key, such as the
    NOT_APPLICABLE_CONSTANT that the labels give other columns."""

    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (-1., 5.),
                                         'NOT_APPLICABLE_CONSTANT = -1.'),))
    obs = next(CassiniISS.from_index(path))
    assert 'BIAS_STRIP_MEAN' not in obs.dict


def test_from_index_applies_a_declared_null_beside_a_declared_zero(
        tmp_path: pathlib.Path) -> None:
    """A column whose label declares 0 and another null constant keeps a 0 and treats
    the other constant as missing."""

    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (-1., 0.),
                                         ('NULL_CONSTANT = 0.',
                                          'INVALID_CONSTANT = -1.')),))
    [first, second] = CassiniISS.from_index(path)
    assert 'BIAS_STRIP_MEAN' not in first.dict
    assert second.dict['BIAS_STRIP_MEAN'] == 0.


def test_from_index_applies_a_null_constant_of_the_supplement(
        tmp_path: pathlib.Path) -> None:
    """A column that only the supplement gives follows the supplement's declaration."""

    index = write_index(tmp_path)
    supplement = write_table(tmp_path, 'supplemental_index', {
        'VOLUME_ID'              : ['COISS_2009'] * 2,
        'FILE_SPECIFICATION_NAME': [row[1] for row in _INDEX_ROWS],
        'EXTRA'                  : [99, 1],
    })
    text = supplement.read_bytes()
    supplement.write_bytes(text.replace(b'NAME = EXTRA\r\n',
                                        b'NAME = EXTRA\r\n    NULL_CONSTANT = 99\r\n'))
    obs = next(CassiniISS.from_index(index, supplement=supplement))
    assert 'EXTRA' not in obs.dict


def test_index_fill_rules_read_a_label_given_as_a_dictionary(
        tmp_path: pathlib.Path) -> None:
    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (0., 5.),
                                         'NULL_CONSTANT = 0.'),))
    (_, zeros, _) = CassiniISS._index_fill_rules(Pds3Label(path,
                                                           method='fast').as_dict())
    assert 'BIAS_STRIP_MEAN' in zeros


def test_from_index_keeps_a_fill_value_where_the_label_declares_another(
        tmp_path: pathlib.Path) -> None:
    """65534 is missing only in a column whose label does not declare a null constant."""

    path = write_index(tmp_path, extra=(('SEQUENCE_NUMBER', 'ASCII_INTEGER', 5,
                                         (65534, 1), 'NULL_CONSTANT = 99999'),))
    obs = next(CassiniISS.from_index(path))
    assert obs.dict['SEQUENCE_NUMBER'] == 65534


def test_from_index_reads_given_rows_with_a_given_label(tmp_path: pathlib.Path) -> None:
    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (-1., 5.),
                                         'NULL_CONSTANT = -1.'),))
    rows = list(CassiniISS._index_rows(path, supplement=None))
    obs = next(CassiniISS.from_index(path, row_dicts=rows,
                                     label=Pds3Label(path, method='fast')))
    assert 'BIAS_STRIP_MEAN' not in obs.dict


def test_host_from_index_reads_given_rows_with_the_index_label(
        tmp_path: pathlib.Path) -> None:
    """oops.Host.from_index gives the host rows without a label; the host reads the label
    of the index, so a corrected label's declared constants apply to the rows."""

    path = write_index(tmp_path, extra=(('BIAS_STRIP_MEAN', 'ASCII_REAL', 10, (-1., 5.),
                                         'NULL_CONSTANT = -1.'),))
    rows = list(CassiniISS._index_rows(path, supplement=None))
    obs = next(oops.Host.from_index(path, row_dicts=rows))
    assert 'BIAS_STRIP_MEAN' not in obs.dict


def test_from_index_given_rows_without_a_label_need_the_index_label(
        tmp_path: pathlib.Path) -> None:
    rows = index_rows(tmp_path)
    with pytest.raises(FileNotFoundError, match=r'missing\.lbl'):
        CassiniISS.from_index(tmp_path / 'missing.lbl', row_dicts=rows)


def test_from_index_with_given_rows_reads_only_the_index_label(
        tmp_path: pathlib.Path) -> None:
    """Given rows are used in place of the tables, so the call reads only the index label,
    for its null constants: a supplement that does not exist, and an index whose name
    gives another instrument's volume, still give a Snapshot for every row."""

    rows = index_rows(tmp_path)
    other = tmp_path / 'COUVIS_0001_index.lbl'
    other.write_bytes((tmp_path / 'INDEX.LBL').read_bytes())
    obs = list(CassiniISS.from_index(other, supplement=tmp_path / 'missing.lbl',
                                     row_dicts=rows))
    assert len(obs) == 2


@pytest.mark.parametrize(('column', 'value'), [
    ('SEQUENCE_NUMBER', 65534),
    ('ORDER_NUMBER', 65534),
    ('MISSING_LINES', -2147483648),
    ('EXPECTED_MAXIMUM', 1.e32),
    ('PHASE_ANGLE', -1.e32),
    ('EMISSION_ANGLE', 0.),
    ('INCIDENCE_ANGLE', 0.),
    ('CENTER_LATITUDE', 0.),
    ('CENTER_LONGITUDE', 0.),
    ('CENTRAL_BODY_DISTANCE', 'NULL'),
    ('OPTICS_TEMPERATURE', -999.),
    ('INST_CMPRS_RATIO', -999.),
], ids=['65534', 'order-number-65534', 'int32-min', '1e32', 'minus-1e32', 'emission-0',
        'incidence-0', 'center-latitude-0', 'center-longitude-0', 'null-text',
        'temperature-999', 'ratio-999'])
def test_from_index_leaves_out_fill_values(tmp_path: pathlib.Path, column: str,
                                           value: Any) -> None:
    """A fill value of the index tables is missing, whether or not the label declares it.

    The image labels give these items as UNK or N/A, or the row has no target geometry.
    """

    rows = index_rows(tmp_path)
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL',
                                     row_dicts=[rows[0] | {column: value}]))
    assert column not in obs.dict


def test_from_index_keeps_a_zero_angle_with_target_geometry(
        tmp_path: pathlib.Path) -> None:
    """A 0 angle is missing only in a row without target geometry, so a corrected table,
    which writes the fill value there, keeps a real 0."""

    rows = index_rows(tmp_path)
    row = rows[0] | {'PHASE_ANGLE': 30., 'EMISSION_ANGLE': 0.}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.dict['EMISSION_ANGLE'] == 0.


def test_from_index_keeps_minus_999_where_it_can_be_real(tmp_path: pathlib.Path) -> None:
    """-999 is missing only in the temperature, rate and compression columns."""

    rows = index_rows(tmp_path)
    row = rows[0] | {'PLANET_CENTER': np.array([-999., 12.])}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.dict['PLANET_CENTER'] == [-999., 12.]


def test_from_index_marks_a_fill_element_none(tmp_path: pathlib.Path) -> None:
    rows = index_rows(tmp_path)
    row = rows[0] | {'VALID_MAXIMUM': np.array([4095, 65534])}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.dict['VALID_MAXIMUM'] == [4095, None]


@pytest.mark.parametrize(('count', 'expected'), [
    ('1473199010.12', '1473199010.120'),
    ('1357765134.2', '1357765134.200'),
    ('1877634636', '1877634636.000'),
    ('1473199010.122', '1473199010.122'),
], ids=['two-digits', 'one-digit', 'no-digits', 'three-digits'])
def test_from_index_spells_clock_counts_as_the_labels(tmp_path: pathlib.Path, count: str,
                                                      expected: str) -> None:
    """The indexes drop a count's trailing zeros; the labels write three digits."""

    rows = index_rows(tmp_path)
    row = rows[0] | {'SPACECRAFT_CLOCK_STOP_COUNT': count}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.spacecraft_clock_stop_count == expected


def test_from_index_skips_the_pointing_check(tmp_path: pathlib.Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """A row is kept, and nothing raises, even when the C kernels lack the pointing."""

    remove_pointing(monkeypatch)
    assert len(list(CassiniISS.from_index(write_index(tmp_path)))) == 2


@pytest.mark.parametrize('select', [0, slice(None), 'NAC'])
def test_from_index_refuses_select(tmp_path: pathlib.Path, select: Any) -> None:
    """The call itself raises, before any iteration."""

    with pytest.raises(oops.OopsValueError,
                       match='each row of a Cassini ISS index is one observation'):
        CassiniISS.from_index(write_index(tmp_path), select=select)


def test_from_index_rejects_an_unknown_keyword(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'supplementary'"):
        CassiniISS.from_index(write_index(tmp_path), supplementary=None)


def test_from_index_uses_a_given_label(tmp_path: pathlib.Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    index = write_index(tmp_path)
    label = Pds3Label(index)
    labels: list[Any] = []
    index_rows_ = CassiniISS._index_rows

    def spy(filepath: Any, **kwargs: Any) -> Any:
        labels.append(kwargs['label'])
        return index_rows_(filepath, **kwargs)

    monkeypatch.setattr(CassiniISS, '_index_rows', spy)
    CassiniISS.from_index(index, label=label)
    assert labels == [label]


def test_from_index_needs_no_supplement(tmp_path: pathlib.Path) -> None:
    """The default looks beside the index for a supplement, which ISS need not have."""

    index = write_index(tmp_path).rename(tmp_path / 'COISS_2009_index.lbl')
    assert len(list(CassiniISS.from_index(index))) == 2


def test_from_index_without_a_supplement(tmp_path: pathlib.Path) -> None:
    assert len(list(CassiniISS.from_index(write_index(tmp_path), supplement=None))) == 2


def test_from_index_reads_the_supplement_given(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match='missing_supplemental_index'):
        CassiniISS.from_index(write_index(tmp_path),
                              supplement=tmp_path / 'missing_supplemental_index.lbl')


def test_from_index_call_reads_no_row(tmp_path: pathlib.Path) -> None:
    """The call reads only the label of an index that its path shows to be of ISS; the
    table is opened when the iteration starts."""

    index = write_index(tmp_path).rename(tmp_path / 'COISS_2009_index.lbl')
    (tmp_path / 'INDEX.TAB').unlink()
    snapshots = CassiniISS.from_index(index)
    with pytest.raises(FileNotFoundError, match=r'INDEX\.TAB'):
        next(snapshots)


def test_from_index_yields_the_rows_before_one_the_table_lacks(
        tmp_path: pathlib.Path) -> None:
    """Rows are read as the iteration proceeds: a table shorter than its label says
    yields every row it holds before the error."""

    index = write_index(tmp_path)
    claim_more_rows(index, 2, 30)
    snapshots = CassiniISS.from_index(index)
    assert [next(snapshots).basename, next(snapshots).basename] == [
        'N1484506648_1.IMG', 'W1484506649_1.IMG']
    with pytest.raises(oops.host.HostIndexError, match='invalid index table'):
        next(snapshots)


def test_from_index_supplement_in_another_order_raises_at_the_row(
        tmp_path: pathlib.Path) -> None:
    """ISS pairs a supplement by position, so a file out of place raises at its row."""

    index = write_index(tmp_path)
    specs = [row[1] for row in _INDEX_ROWS]
    supplement = write_table(tmp_path, 'supplemental_index', {
        'VOLUME_ID'              : ['COISS_2009'] * 2,
        'FILE_SPECIFICATION_NAME': specs[::-1],
        'EXTRA'                  : [1, 2],
    })
    snapshots = CassiniISS.from_index(index, supplement=supplement)
    with pytest.raises(oops.host.HostIndexError,
                       match=r"index row 0 gives 'data/1484506648_1484573295/"
                             r"N1484506648_1\.IMG'"):
        next(snapshots)

##########################################################################################
# The rules that every host follows
##########################################################################################


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_rejects_an_unknown_keyword(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'astrometery'"):
        CassiniISS.from_file(write_pds3_image(tmp_path), astrometery=True)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_refuses_select_before_reading(tmp_path: pathlib.Path) -> None:
    """A select other than None is refused before the file is opened."""

    with pytest.raises(oops.OopsValueError, match='select must be None, not 0'):
        CassiniISS.from_file(tmp_path / 'missing.IMG', select=0)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_accepts_select_none(tmp_path: pathlib.Path) -> None:
    assert CassiniISS.from_file(write_pds3_image(tmp_path), select=None).detector == 'NAC'


def _write_foreign_host(path: pathlib.Path) -> None:
    """Rewrite the HOST of a VICAR file as a machine other than this one."""

    content = path.read_bytes()
    host = f"HOST='{vicar.vicarimage._HOST}'".encode()
    other = b"HOST='SUN-SOLR'".ljust(len(host))
    path.write_bytes(content.replace(host, other, 1))


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('astrometry', [False, True])
def test_from_file_header_dict_keeps_the_files_host(tmp_path: pathlib.Path,
                                                    astrometry: bool) -> None:
    """header_dict gives the HOST that the file writes, never the reading machine's."""

    path = write_vicar_image(tmp_path)
    _write_foreign_host(path)
    obs = CassiniISS.from_file(path, astrometry=astrometry)
    assert obs.header_dict['HOST'] == 'SUN-SOLR'


@pytest.mark.usefixtures('initialized_iss')
def test_snapshot_time_is_not_from_the_clock() -> None:
    assert snapshot().time_from_sclk is False


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('function', ['scs2e', 'sce2s'])
def test_from_file_early_cruise_clock_error_is_a_spice_error(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, function: str) -> None:
    """A clock count the clock kernel cannot convert raises OopsSpiceError."""

    def fail(*args: Any) -> None:
        raise ValueError('SPICE(INVALIDSCLKSTRING) -- bad clock count')

    monkeypatch.setattr(cspyce, function, fail)
    with pytest.raises(oops.OopsSpiceError,
                       match=r'Cassini clock .*N0000000001_1\.IMG: '
                             r'SPICE\(INVALIDSCLKSTRING\)$'):
        CassiniISS.from_file(write_pds3_image(tmp_path, **_EARLY_COUNTS))


@pytest.mark.parametrize(('number', 'start'), [
    ('1343357427', '1343357425.000'),
    ('1343357428', '1343357423.000'),
], ids=['last-early-image', 'next-image'])
def test_from_index_early_cruise_boundary(tmp_path: pathlib.Path, number: str,
                                          start: str) -> None:
    """The last early cruise image gets a computed start count; the next keeps its own."""

    rows = index_rows(tmp_path)
    row = rows[0] | {'IMAGE_NUMBER': number, 'EXPOSURE_DURATION': 2000.,
                     'SPACECRAFT_CLOCK_START_COUNT': '1343357423.000',
                     'SPACECRAFT_CLOCK_STOP_COUNT': number + '.000'}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert obs.spacecraft_clock_start_count == start


def test_from_index_frame_without_pointing_raises_a_spice_error(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A row is built without pointing; its frame raises OopsSpiceError when used."""

    remove_pointing(monkeypatch)
    obs = next(CassiniISS.from_index(write_index(tmp_path)))
    with pytest.raises(oops.OopsSpiceError,
                       match=r'^no Cassini ISS NAC pointing at 2005-01-15T18:30:00\.500 '
                             r'for .*N1484506648_1\.IMG: SPICE\(CKINSUFFDATA\)$'):
        obs.frame.wrt(oops.Frame.J2000).transform_at_time(obs.midtime)


def test_from_index_frame_with_pointing_is_the_camera_frame(
        tmp_path: pathlib.Path) -> None:
    obs = next(CassiniISS.from_index(write_index(tmp_path)))
    assert obs.frame == oops.Frame.as_wayframe('CASSINI_ISS_NAC')


def test_from_index_probes_pointing_at_the_cadence_midtime(
        tmp_path: pathlib.Path, spice_check_times: list[float]) -> None:
    """With a cadence override, a row's pointing is probed at the override's midtime."""

    cadence = oops.cadence.SnapCadence(julian.tdb_from_iso('2006-01-01T00:00:00'), 10.)
    next(CassiniISS.from_index(write_index(tmp_path), cadence=cadence))
    assert spice_check_times == [cadence.midtime]


# The ISS camera frame is a fixed Cmatrix on the SPICE frame, and an "F" time shift wraps
# the camera frame alone, so the SPICE frame is evaluated at the observation's own time,
# moved only by a "T"
@pytest.mark.parametrize('letters', ['T', 't', 'F', 'TF'])
def test_from_index_probes_pointing_after_a_time_shift(
        tmp_path: pathlib.Path, spice_check_times: list[float], letters: str) -> None:
    """The pointing probe is where the SPICE frame is evaluated, after the time shift."""

    obs = next(CassiniISS.from_index(write_index(tmp_path), timeshift=(30., letters)))
    assert spice_check_times == [pytest.approx(obs.midtime, abs=1.e-3)]


@pytest.mark.usefixtures('initialized_iss', 'saturn_path')
def test_from_file_checks_pointing_where_a_tracker_evaluates_it(
        tmp_path: pathlib.Path, spice_check_times: list[float]) -> None:
    """A tracker evaluates the frame once, at the time it selects."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True,
                               tracker='start')
    assert spice_check_times == [pytest.approx(obs.time[0], abs=1.e-3)]


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_checks_pointing_where_a_frame_shift_evaluates_it(
        tmp_path: pathlib.Path, spice_check_times: list[float]) -> None:
    """An "F" shift of the camera frame leaves the SPICE frame at the midtime."""

    obs = CassiniISS.from_file(write_pds3_image(tmp_path), astrometry=True,
                               timeshift=(30., 'F'))
    assert spice_check_times == [pytest.approx(obs.midtime, abs=1.e-3)]


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize('value', ['UNKNOWN', '', '---'])
def test_from_file_leaves_out_each_placeholder(tmp_path: pathlib.Path,
                                               value: str) -> None:
    obs = CassiniISS.from_file(write_pds3_image(tmp_path, OBSERVATION_ID=value))
    assert 'OBSERVATION_ID' not in obs.dict


@pytest.mark.parametrize('value', ['UNKNOWN', '', '---', 'NOT APPLICABLE'])
def test_from_index_leaves_out_each_placeholder(tmp_path: pathlib.Path,
                                                value: str) -> None:
    rows = index_rows(tmp_path)
    row = rows[0] | {'OBSERVATION_ID': np.str_(value)}
    obs = next(CassiniISS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row]))
    assert 'OBSERVATION_ID' not in obs.dict


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_parsed_label_without_a_path_has_no_file_subfields(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_image(tmp_path), method='fast')
    label.filepath = None
    obs = CassiniISS.from_file(label, astrometry=True)
    assert {'filepath', 'basename', 'abspath', 'image_url'}.isdisjoint(obs.subfields)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_parsed_label_without_a_path_cannot_be_read_in_full(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_image(tmp_path), method='fast')
    label.filepath = None
    with pytest.raises(oops.host.HostFormatError, match='no file path is known'):
        CassiniISS.from_file(label)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_parsed_header_without_a_path_cannot_be_read_in_full(
        tmp_path: pathlib.Path) -> None:
    header = vicar.VicarLabel(write_vicar_image(tmp_path))
    header.filepath = None
    with pytest.raises(oops.host.HostFormatError, match='no file path is known'):
        CassiniISS.from_file(header)


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_parsed_image_without_a_path_needs_no_file(
        tmp_path: pathlib.Path) -> None:
    image = vicar.VicarImage(write_vicar_image(tmp_path))
    image.filepath = None
    obs = CassiniISS.from_file(image)
    assert obs.data.shape == (256, 256)
    assert {'filepath', 'basename', 'abspath', 'image_url'}.isdisjoint(obs.subfields)


@pytest.mark.usefixtures('initialized_iss')
@pytest.mark.parametrize(('item', 'label', 'header', 'expected'), [
    ('DARK_STRIP_MEAN', 1.5, 2.5, 1.5),
    ('DETECTOR_TEMPERATURE', -89.5, -89.25, -89.5),
    ('FILTER_TEMPERATURE', 0.5, 0.25, 0.5),
    ('INST_CMPRS_RATIO', 2.5, 2.25, 2.5),
    ('SENSOR_HEAD_ELEC_TEMPERATURE', 1.5, 1.25, 1.5),
    ('INSTRUMENT_DATA_RATE', 182.783997, 182.784, 182.783997),
    ('INST_CMPRS_RATE', [1.5, 2.5], [1.25, 2.25], [1.5, 2.5]),
    ('DATA_SET_ID', 'CO-S-ISSNA/ISSWA-2-EDR-V1.0', 'CO-X-ISSNA/ISSWA-2-EDR-V0.9',
     'CO-S-ISSNA/ISSWA-2-EDR-V1.0'),
    ('MISSION_NAME', 'CASSINI-HUYGENS', 'CASSINI', 'CASSINI-HUYGENS'),
    ('INSTRUMENT_NAME', NAC_NAME, 'ISS NARROW ANGLE CAMERA', NAC_NAME),
    ('PRODUCT_CREATION_TIME', '2010-001T00:00:00.000', '2005-001T00:00:00.000Z',
     '2010-001T00:00:00.000'),
    ('EARTH_RECEIVED_START_TIME', '2005-016T13:01:59.864', '2005-016T13:01:59.865Z',
     '2005-016T13:01:59.864'),
    ('EARTH_RECEIVED_STOP_TIME', '2005-016T13:02:00.864', '2005-016T13:02:00.865Z',
     '2005-016T13:02:00.864'),
])
def test_from_file_takes_each_fixed_source(tmp_path: pathlib.Path, item: str, label: Any,
                                           header: Any, expected: Any) -> None:
    """Each item with a fixed source takes it where the label and the header disagree."""

    path = write_pds3_image(tmp_path, header={item: header}, **{item: label})
    assert CassiniISS.from_file(path).dict[item] == expected


@pytest.mark.usefixtures('initialized_iss')
def test_from_file_label_dict_keeps_a_sets_label_order(tmp_path: pathlib.Path) -> None:
    path = write_pds3_image(tmp_path,
                            extra_lines=['IMAGE_OBSERVATION_TYPE = {"SCIENCE", "OPNAV"}'])
    assert CassiniISS.from_file(path).label_dict['IMAGE_OBSERVATION_TYPE'] == [
        'SCIENCE', 'OPNAV']

##########################################################################################
