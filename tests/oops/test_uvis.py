##########################################################################################
# tests/oops/test_uvis.py
##########################################################################################
"""Tests for the Cassini UVIS host, host_cassini.oops.uvis."""

import pathlib
from typing import Any

import julian
import numpy as np
import oops
import pytest
from conftest import FakeSpice, FakeSpiceDB, pds3_lines

import host_cassini.oops.uvis as uvis
from host_cassini.oops._cassini import _Cassini
from host_cassini.oops.uvis     import CassiniUVIS

START_TIME = '2005-01-01T00:00:00.000'
STOP_TIME = '2005-01-01T00:01:00.000'

# Slit half-widths in degrees, as in the UVIS instrument kernel
SLIT_REF_DEG = 0.75
SLIT_CROSS_DEG = 0.05
CIRCLE_REF_DEG = 0.1

SPECTRAL_IDS = range(-82845, -82839)    # FUV and EUV at each slit state
CIRCULAR_IDS = range(-82848, -82845)    # HSP, HDAC and SOLAR


def uvis_ik_dict(*, shape: str = 'RECTANGLE') -> dict[str, Any]:
    """The parts of the UVIS instrument kernel that the host reads, keyed by NAIF ID."""

    slit = {'FOV_SHAPE': shape, 'FOV_REF_ANGLE': SLIT_REF_DEG,
            'FOV_CROSS_ANGLE': SLIT_CROSS_DEG}
    circle = {'FOV_SHAPE': 'CIRCLE', 'FOV_REF_ANGLE': CIRCLE_REF_DEG}
    ins: dict[int, dict[str, Any]] = dict.fromkeys(SPECTRAL_IDS, slit)
    ins.update(dict.fromkeys(CIRCULAR_IDS, circle))
    return {'INS': ins}


@pytest.fixture(autouse=True)
def uvis_kernel(fake_spicedb: FakeSpiceDB) -> FakeSpiceDB:
    """The fake spicedb, returning the UVIS instrument kernel."""

    fake_spicedb.kernel_dict = uvis_ik_dict()
    return fake_spicedb


def label_keywords(product_id: str, **changes: Any) -> dict[str, Any]:
    """The top-level label keywords that the UVIS host reads."""

    keywords = {
        'PRODUCT_ID'          : product_id,
        'START_TIME'          : START_TIME,
        'STOP_TIME'           : STOP_TIME,
        'INSTRUMENT_HOST_NAME': 'CASSINI_ORBITER',
        'INSTRUMENT_ID'       : 'UVIS',
        'TARGET_NAME'         : 'SATURN',
        'INTEGRATION_DURATION': 2.,
        'SLIT_STATE'          : 'HIGH_RESOLUTION',
    }
    keywords.update(changes)
    return keywords


def write_label(directory: pathlib.Path, name: str, keywords: dict[str, Any],
                obj: list[str]) -> pathlib.Path:
    """Write a detached PDS3 label from its keywords and object lines; return its path."""

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH', 'RECORD_BYTES = 2',
            'FILE_RECORDS = 1']
    text += pds3_lines(keywords) + obj + ['END', '']
    path = directory / (name + '.LBL')
    path.write_text('\r\n'.join(text))
    return path


def qube_values(bands: int, lines: int, samples: int) -> np.ndarray:
    """Distinct qube values in (sample, line, band) order, band fastest as in the file."""

    return np.arange(samples * lines * bands, dtype='>u2').reshape(samples, lines, bands)


def write_qube(directory: pathlib.Path, *, bands: int = 8, lines: int = 1,
               samples: int = 3, window: dict[str, Any] | None = None,
               data_name: str = 'FUV2005_001_00_00.DAT', **changes: Any) -> pathlib.Path:
    """Write a FUV QUBE and its label; return the label's path.

    The default window covers all 64 lines and every band, unbinned in band.
    """

    qube_values(bands, lines, samples).tofile(directory / data_name)
    window = window or {'UL_CORNER_LINE': 0, 'LR_CORNER_LINE': 63, 'LINE_BIN': 64,
                        'UL_CORNER_BAND': 0, 'LR_CORNER_BAND': bands - 1,
                        'BAND_BIN': 1}
    keywords = label_keywords('FUV2005_001_00_00', **changes)
    keywords['^QUBE'] = 'FUV2005_001_00_00.DAT'
    obj = ['OBJECT = QUBE', '  AXES = 3', '  AXIS_NAME = (BAND, LINE, SAMPLE)',
           f'  CORE_ITEMS = ({bands}, {lines}, {samples})', '  CORE_ITEM_BYTES = 2',
           '  CORE_ITEM_TYPE = MSB_UNSIGNED_INTEGER', '  SUFFIX_ITEMS = (0, 0, 0)']
    obj += pds3_lines(window, indent='  ') + ['END_OBJECT = QUBE']
    return write_label(directory, 'FUV2005_001_00_00', keywords, obj)


def write_time_series(directory: pathlib.Path, product_id: str = 'HSP2005_001_00_00',
                      *, rows: int = 5) -> pathlib.Path:
    """Write a TIME_SERIES and its label; return the label's path."""

    np.arange(rows, dtype='>u2').tofile(directory / (product_id + '.DAT'))
    keywords = label_keywords(product_id)
    keywords['^TIME_SERIES'] = product_id + '.DAT'
    obj = ['OBJECT = TIME_SERIES', f'  ROWS = {rows}', '  COLUMNS = 1',
           '  SAMPLING_PARAMETER_UNIT = MILLISECONDS',
           '  SAMPLING_PARAMETER_INTERVAL = 2.000', '  OBJECT = COLUMN',
           '    NAME = PHOTOMETER_COUNTS', '    DATA_TYPE = MSB_UNSIGNED_INTEGER',
           '    BYTES = 2', '  END_OBJECT = COLUMN', 'END_OBJECT = TIME_SERIES']
    return write_label(directory, product_id, keywords, obj)


def write_spectrum(directory: pathlib.Path, *, rows: int = 6) -> pathlib.Path:
    """Write an EUV SPECTRUM of lines 16-31, binned into one, and its label."""

    np.arange(rows, dtype='>u2').tofile(directory / 'EUV2005_001_00_00.DAT')
    keywords = label_keywords('EUV2005_001_00_00')
    keywords['^SPECTRUM'] = 'EUV2005_001_00_00.DAT'
    obj = ['OBJECT = SPECTRUM', f'  ROWS = {rows}', '  COLUMNS = 1',
           '  UL_CORNER_SPATIAL = 16', '  LR_CORNER_SPATIAL = 31', '  BIN_SPATIAL = 16',
           '  UL_CORNER_SPECTRAL = 100', '  LR_CORNER_SPECTRAL = 105',
           '  BIN_SPECTRAL = 1', '  OBJECT = COLUMN', '    NAME = SPECTRUM',
           '    DATA_TYPE = MSB_UNSIGNED_INTEGER', '    BYTES = 2',
           '  END_OBJECT = COLUMN', 'END_OBJECT = SPECTRUM']
    return write_label(directory, 'EUV2005_001_00_00', keywords, obj)

##########################################################################################
# Registration and detection
##########################################################################################


def test_uvis_is_registered_with_oops() -> None:
    assert oops.Host._LOOKUP['Cassini UVIS'] is CassiniUVIS


def test_module_exports() -> None:
    assert uvis.__all__ == ['CassiniUVIS']


@pytest.mark.parametrize(('label', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI_ORBITER', 'INSTRUMENT_ID': 'UVIS'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'UVIS'}, True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'VIMS'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_ID': 'UVIS'}, False),
    ({}, False),
], ids=['underscore-host', 'space-host', 'other-instrument', 'other-host', 'empty'])
def test_detection(label: dict[str, str], expected: bool) -> None:
    assert CassiniUVIS._detect_in_pds3(label) is expected

##########################################################################################
# _initialize() and _define_frames()
##########################################################################################


def test_initialize_builds_every_fov() -> None:
    CassiniUVIS._initialize()
    expected = {(detector, resolution, lines)
                for (detector, resolution) in uvis._ABBREVS.values()
                for lines in (1, 64)}
    assert set(CassiniUVIS._FOVS) == expected


@pytest.mark.parametrize('lines', [1, 64])
def test_slit_fov(lines: int) -> None:
    CassiniUVIS._initialize()
    fov = CassiniUVIS._FOVS['FUV', 'HIGH_RESOLUTION', lines]
    assert tuple(fov.uv_shape.vals) == (1, lines)
    assert tuple(fov.uv_scale.vals) == pytest.approx(
        (2. * np.radians(SLIT_CROSS_DEG), 2. * np.radians(SLIT_REF_DEG) / lines))


def test_circular_fov() -> None:
    CassiniUVIS._initialize()
    scale = 2. * np.radians(CIRCLE_REF_DEG)
    assert tuple(CassiniUVIS._FOVS['HSP', '', 1].uv_scale.vals) == pytest.approx(
        (scale, scale))


def test_initialize_rejects_unknown_fov_shape(uvis_kernel: FakeSpiceDB) -> None:
    uvis_kernel.kernel_dict = uvis_ik_dict(shape='POLYGON')
    with pytest.raises(oops.OopsValueError, match='Unrecognized FOV_SHAPE: POLYGON'):
        CassiniUVIS._initialize()


def test_initialize_leaves_kernel_unchanged(uvis_kernel: FakeSpiceDB) -> None:
    CassiniUVIS._initialize()
    assert set(uvis_kernel.kernel_dict['INS']) == set(SPECTRAL_IDS) | set(CIRCULAR_IDS)


def test_reset_clears_state() -> None:
    CassiniUVIS._initialize()
    CassiniUVIS._reset()
    assert not CassiniUVIS._initialized
    assert CassiniUVIS._FOVS == {}
    assert CassiniUVIS._INSTRUMENT_KERNEL is None
    assert not _Cassini.initialized


def test_define_frames_only_once(fake_spice: FakeSpice) -> None:
    CassiniUVIS._define_frames()
    CassiniUVIS._define_frames()
    assert fake_spice.spice_frames == list(uvis._FRAME_IDS.values())

##########################################################################################
# QUBE
##########################################################################################


def test_one_line_qube(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path))
    assert type(obs) is oops.observation.Pixel
    assert obs.shape == (3, 8)
    assert np.array_equal(obs.data, qube_values(8, 1, 3).reshape(3, 8))
    assert obs.fov is CassiniUVIS._FOVS['FUV', 'HIGH_RESOLUTION', 1]


def test_one_line_qube_subfields(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path))
    assert obs.host == 'Cassini'
    assert obs.instrument == 'UVIS'
    assert obs.detector == 'FUV'
    assert obs.sampling == 'HIGH_RESOLUTION'
    assert obs.product_type == 'QUBE'
    assert obs.line_window == (0, 64)
    assert obs.band_window == (0, 8)
    assert obs.samples == 3
    assert obs.basename == 'FUV2005_001_00_00.LBL'
    assert obs.frame == oops.Frame.as_wayframe('CASSINI_UVIS_FUV')


def test_qube_times(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path))
    tstart = julian.tdb_from_iso(START_TIME)
    assert obs.time == pytest.approx((tstart, tstart + 3 * 2.))


def test_multi_line_qube_window(tmp_path: pathlib.Path) -> None:
    window = {'UL_CORNER_LINE': 8, 'LR_CORNER_LINE': 15, 'LINE_BIN': 2,
              'UL_CORNER_BAND': 2, 'LR_CORNER_BAND': 5, 'BAND_BIN': 2}
    obs = CassiniUVIS.from_file(write_qube(tmp_path, lines=64, window=window))
    assert type(obs) is oops.observation.TimedImage
    assert obs.shape == (4, 3, 2)
    assert tuple(obs.fov.uv_shape.vals) == (1, 4)

    # Binned values fill the start of each window
    full = qube_values(8, 64, 3).swapaxes(0, 1)
    assert np.array_equal(obs.data, full[8:16, :, 2:6][:4, :, :2])


def test_multiple_windows(tmp_path: pathlib.Path) -> None:
    window = {'UL_CORNER_LINE': [0, 32], 'LR_CORNER_LINE': [15, 63],
              'LINE_BIN': [1, 1], 'UL_CORNER_BAND': [0, 4], 'LR_CORNER_BAND': [3, 7],
              'BAND_BIN': [1, 1]}
    observations = CassiniUVIS.from_file(write_qube(tmp_path, lines=64, window=window))
    assert [obs.shape for obs in observations] == [(16, 3, 4), (32, 3, 4)]
    assert [obs.line_window for obs in observations] == [(0, 16), (32, 64)]


def test_enclosed_windows(tmp_path: pathlib.Path) -> None:
    window = {'UL_CORNER_LINE': [0, 32], 'LR_CORNER_LINE': [15, 63],
              'LINE_BIN': [2, 1], 'UL_CORNER_BAND': [0, 4], 'LR_CORNER_BAND': [3, 7],
              'BAND_BIN': [1, 1]}
    obs = CassiniUVIS.from_file(write_qube(tmp_path, lines=64, window=window),
                                enclose=True)
    assert obs.shape == (64, 3, 8)
    assert obs.line_bin == 1


def test_qube_astrometry_reads_no_data(tmp_path: pathlib.Path) -> None:
    path = write_qube(tmp_path)
    (tmp_path / 'FUV2005_001_00_00.DAT').unlink()
    assert not hasattr(CassiniUVIS.from_file(path, astrometry=True), 'data')


def test_lower_case_data_file(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path, data_name='fuv2005_001_00_00.dat'))
    assert obs.data.shape == (3, 8)


def test_fov_override(tmp_path: pathlib.Path) -> None:
    fov = oops.fov.FlatFOV((1.e-3, 1.e-3), (1, 1))
    assert CassiniUVIS.from_file(write_qube(tmp_path), fov=fov).fov is fov

##########################################################################################
# TIME_SERIES and SPECTRUM
##########################################################################################


@pytest.mark.parametrize('detector', ['HSP', 'HDAC'])
def test_time_series(tmp_path: pathlib.Path, detector: str) -> None:
    obs = CassiniUVIS.from_file(write_time_series(tmp_path, detector + '2005_001_00_00'))
    assert type(obs) is oops.observation.Pixel
    assert obs.detector == detector
    assert obs.product_type == 'TIME_SERIES'
    assert obs.shape == (5,)
    assert obs.data.tolist() == [0, 1, 2, 3, 4]
    assert not hasattr(obs, 'sampling')


def test_time_series_cadence(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_time_series(tmp_path))
    tstart = julian.tdb_from_iso(START_TIME)
    assert obs.time == pytest.approx((tstart, tstart + 5 * 0.002))


def test_time_series_rejects_other_detectors(tmp_path: pathlib.Path) -> None:
    with pytest.raises(oops.OopsValueError, match='neither HSP nor HDAC'):
        CassiniUVIS.from_file(write_time_series(tmp_path, 'XYZ2005_001_00_00'))


def test_spectrum(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_spectrum(tmp_path))
    assert type(obs) is oops.observation.Pixel
    assert obs.detector == 'EUV'
    assert obs.product_type == 'SPECTRUM'
    assert obs.shape == (6,)
    assert obs.data.tolist() == [0, 1, 2, 3, 4, 5]
    assert obs.line_window == (16, 32)
    assert obs.band_window == (100, 106)
    assert tuple(obs.fov.uv_shape.vals) == (1, 1)

##########################################################################################
# Options and dispatch
##########################################################################################


@pytest.mark.parametrize(('target_name', 'target'), [('SATURN', 'SATURN'),
                                                     ('N/A', 'NONE'), ('STAR', 'NONE')])
def test_target_from_label(tmp_path: pathlib.Path, target_name: str, target: str) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path, TARGET_NAME=target_name))
    assert obs.target == target
    assert obs.label_target == target_name
    assert obs.lightsource == 'SUN'


@pytest.mark.usefixtures('saturn_path')
def test_tracker_follows_label_target(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path), tracker='midtime')
    assert type(oops.Frame.as_primary_frame(obs.frame)) is oops.frame.TrackerFrame


_TWO_WINDOWS = {'UL_CORNER_LINE': [0, 32], 'LR_CORNER_LINE': [15, 63],
                'LINE_BIN': [1, 1], 'UL_CORNER_BAND': [0, 4], 'LR_CORNER_BAND': [3, 7],
                'BAND_BIN': [1, 1]}


@pytest.mark.parametrize(('select', 'windows'), [
    (1, (32, 64)),
    (slice(None, 1), [(0, 16)]),
    ((1, 0), [(32, 64), (0, 16)]),
], ids=['index', 'slice', 'tuple'])
def test_select_windows(tmp_path: pathlib.Path, select: Any,
                        windows: tuple[int, int] | list[tuple[int, int]]) -> None:
    path = write_qube(tmp_path, lines=64, window=_TWO_WINDOWS)
    result = CassiniUVIS.from_file(path, select=select)
    if isinstance(result, tuple):
        assert [obs.line_window for obs in result] == windows
    else:
        assert result.line_window == windows


def test_select_single_observation(tmp_path: pathlib.Path) -> None:
    result = CassiniUVIS.from_file(write_time_series(tmp_path), select=slice(None))
    assert [obs.detector for obs in result] == ['HSP']


def test_select_rejects_names(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="unrecognized select name: 'VIS'"):
        CassiniUVIS.from_file(write_qube(tmp_path), select='VIS')


def test_custom_frame(tmp_path: pathlib.Path, fake_spice: FakeSpice) -> None:
    frame = oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                               frame_id='TEST_CMATRIX')
    obs = CassiniUVIS.from_file(write_qube(tmp_path), frame=frame)
    assert obs.frame == frame.wayframe
    assert not _Cassini.CK_LOADED.any()
    assert _Cassini.SPK_LOADED.any()
    assert fake_spice.spice_frames == []


def test_host_from_file_dispatches_to_uvis(tmp_path: pathlib.Path) -> None:
    obs = oops.Host.from_file(write_time_series(tmp_path))
    assert obs.instrument == 'UVIS'

##########################################################################################
