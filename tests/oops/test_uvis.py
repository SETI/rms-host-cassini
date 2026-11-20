##########################################################################################
# tests/oops/test_uvis.py
##########################################################################################
"""Tests for the Cassini UVIS host, host_cassini.oops.uvis."""

import json
import pathlib
from collections.abc import Callable
from typing import Any

from filecache import FCPath
import julian
import numpy as np
import oops
import pytest
from conftest import FakeSpice, FakeSpiceDB, claim_more_rows, pds3_lines, write_table
from pdsparser import Pds3Label

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
        'OBSERVATION_ID'      : 'UVIS_000SA_TEST001_PRIME',
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
    path.write_text('\r\n'.join(text), newline='')
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
# initialize() and _define_frames()
##########################################################################################


def test_initialize_builds_every_fov() -> None:
    CassiniUVIS.initialize()
    expected = {(detector, resolution, lines)
                for (detector, resolution) in uvis._ABBREVS.values()
                for lines in (1, 64)}
    assert set(CassiniUVIS._FOVS) == expected


@pytest.mark.parametrize('lines', [1, 64])
def test_slit_fov(lines: int) -> None:
    CassiniUVIS.initialize()
    fov = CassiniUVIS._FOVS['FUV', 'HIGH_RESOLUTION', lines]
    assert tuple(fov.uv_shape.vals) == (1, lines)
    assert tuple(fov.uv_scale.vals) == pytest.approx(
        (2. * np.radians(SLIT_CROSS_DEG), 2. * np.radians(SLIT_REF_DEG) / lines))


def test_circular_fov() -> None:
    CassiniUVIS.initialize()
    scale = 2. * np.radians(CIRCLE_REF_DEG)
    assert tuple(CassiniUVIS._FOVS['HSP', '', 1].uv_scale.vals) == pytest.approx(
        (scale, scale))


def test_initialize_rejects_unknown_fov_shape(uvis_kernel: FakeSpiceDB) -> None:
    uvis_kernel.kernel_dict = uvis_ik_dict(shape='POLYGON')
    with pytest.raises(oops.OopsValueError, match='Unrecognized FOV_SHAPE: POLYGON'):
        CassiniUVIS.initialize()


def test_initialize_leaves_kernel_unchanged(uvis_kernel: FakeSpiceDB) -> None:
    CassiniUVIS.initialize()
    assert set(uvis_kernel.kernel_dict['INS']) == set(SPECTRAL_IDS) | set(CIRCULAR_IDS)


def test_reset_clears_state() -> None:
    CassiniUVIS.initialize()
    CassiniUVIS._reset()
    assert not CassiniUVIS._initialized
    assert CassiniUVIS._FOVS == {}
    assert CassiniUVIS._INSTRUMENT_KERNEL is None
    assert not _Cassini.initialized


def test_define_frames_only_once(fake_spice: FakeSpice) -> None:
    CassiniUVIS._define_frames()
    CassiniUVIS._define_frames()
    assert fake_spice.spice_frames == list(uvis._FRAME_NAMES.values())

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


def test_line_window_not_a_multiple_of_line_bin(tmp_path: pathlib.Path) -> None:
    window = {'UL_CORNER_LINE': 8, 'LR_CORNER_LINE': 14, 'LINE_BIN': 2,
              'UL_CORNER_BAND': 0, 'LR_CORNER_BAND': 7, 'BAND_BIN': 1}
    path = write_qube(tmp_path, lines=64, window=window)
    with pytest.raises(oops.host.HostFormatError,
                       match='line window of 7 lines is not a multiple of LINE_BIN 2'):
        CassiniUVIS.from_file(path)


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
    with pytest.raises(oops.host.HostFormatError, match='neither HSP nor HDAC'):
        CassiniUVIS.from_file(write_time_series(tmp_path, 'XYZ2005_001_00_00'))


@pytest.mark.parametrize(('write', 'data_name'), [
    (write_time_series, 'HSP2005_001_00_00.DAT'),
    (write_spectrum, 'EUV2005_001_00_00.DAT'),
], ids=['time-series', 'spectrum'])
def test_astrometry_reads_no_data(tmp_path: pathlib.Path,
                                  write: Callable[[pathlib.Path], pathlib.Path],
                                  data_name: str) -> None:
    path = write(tmp_path)
    (tmp_path / data_name).unlink()
    assert not hasattr(CassiniUVIS.from_file(path, astrometry=True), 'data')


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


# The frame IDs are those of the Cassini frame kernel, not the instrument codes of the
# UVIS instrument kernel.
@pytest.mark.parametrize(('write', 'frame_name', 'frame_id'), [
    (write_qube, 'CASSINI_UVIS_FUV', -82840),
    (write_spectrum, 'CASSINI_UVIS_EUV', -82842),
    (write_time_series, 'CASSINI_UVIS_HSP', -82844),
    (lambda d: write_time_series(d, 'HDAC2005_001_00_00'), 'CASSINI_UVIS_HDAC', -82845),
], ids=['qube', 'spectrum', 'hsp', 'hdac'])
def test_spice_frame(tmp_path: pathlib.Path,
                     write: Callable[[pathlib.Path], pathlib.Path], frame_name: str,
                     frame_id: int) -> None:
    obs = CassiniUVIS.from_file(write(tmp_path))
    assert obs.spice_frame_name == frame_name
    assert obs.spice_frame_id == frame_id

##########################################################################################
# Options and dispatch
##########################################################################################


@pytest.mark.parametrize(('target_name', 'target'), [('SATURN', 'SATURN'),
                                                     ('STAR', 'NONE')])
def test_target_from_label(tmp_path: pathlib.Path, target_name: str, target: str) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path, TARGET_NAME=target_name))
    assert obs.target == target
    assert obs.label_target == target_name
    assert obs.lightsource == 'SUN'


@pytest.mark.parametrize('target_name', ['N/A', 'UNK', 'UNKNOWN', 'NULL', '---', ''])
def test_null_target_names_no_target(tmp_path: pathlib.Path, target_name: str) -> None:
    """A placeholder TARGET_NAME gives the target NONE and leaves out label_target."""

    obs = CassiniUVIS.from_file(write_qube(tmp_path, TARGET_NAME=target_name))
    assert obs.target == 'NONE'
    assert 'label_target' not in obs.subfields


@pytest.mark.parametrize(('obs_id', 'star'), [
    ('UVIS_034RI_ALPVIROCC001_PRIME', 'alf Vir'),
    ('UVIS_034RI_URZETAORIOCC001_PRIME', 'zet Ori'),
    ('UVIS_034RI_205839TI001_PRIME', 'b Cen'),
], ids=['occultation', 'repaired', 'repaired-with-alt-name'])
def test_star_lightsource_from_observation_id(tmp_path: pathlib.Path, obs_id: str,
                                              star: str) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path, OBSERVATION_ID=obs_id))
    assert obs.lightsource is oops.lightsource.star_lookup(star)
    assert obs.observation_id == obs_id


def test_star_lightsource_gets_alt_name_from_observation_id(
        tmp_path: pathlib.Path) -> None:
    CassiniUVIS.from_file(write_qube(tmp_path,
                                     OBSERVATION_ID='UVIS_034RI_205839OCC001_PRIME'))
    assert 'SAO 205839' in oops.lightsource.star_lookup('b Cen').alt_names


def test_numeric_observation_id_names_no_star(tmp_path: pathlib.Path) -> None:
    """The archive labels give OBSERVATION_ID as a number, such as 888."""

    obs = CassiniUVIS.from_file(write_qube(tmp_path, OBSERVATION_ID=888))
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
    ((1,), (32, 64)),
    ((slice(None, 1),), [(0, 16)]),
], ids=['index', 'slice', 'tuple-index', 'tuple-slice'])
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
# Index detection
##########################################################################################

_INDEX_LABEL = ['PDS_VERSION_ID = PDS3', 'OBJECT = INDEX_TABLE',
                'END_OBJECT = INDEX_TABLE', 'END', '']


@pytest.mark.parametrize(('relpath', 'expected'), [
    ('COUVIS_0001/INDEX/INDEX.LBL', True),
    ('COUVIS_0999_index.lbl', True),
    ('couvis_0001/index/index.lbl', True),
    ('INDEX/INDEX.LBL', None),
    ('COISS_2001_index.lbl', False),
    ('COVIMS_0040/index/index.lbl', False),
], ids=['couvis-directory', 'couvis-file', 'lowercase', 'no-couvis', 'iss', 'vims'])
def test_detection_in_index(tmp_path: pathlib.Path, relpath: str,
                            expected: bool | None) -> None:
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\r\n'.join(_INDEX_LABEL), newline='')
    assert CassiniUVIS._detect_in_index(Pds3Label(path)) is expected


def test_detection_in_index_without_a_path() -> None:
    assert CassiniUVIS._detect_in_index(Pds3Label(_INDEX_LABEL[:-1])) is None


def test_detection_in_index_of_another_instrument() -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3', 'INSTRUMENT_ID = "VIMS"', 'END'])
    assert CassiniUVIS._detect_in_index(label) is False


def test_volumes_own_index_is_refused_alike_through_oops_host(
        tmp_path: pathlib.Path) -> None:
    """oops.Host.from_index finds UVIS from the path of a volume's own index, without
    reading a row, so the refusal is the one the class gives."""

    path = tmp_path / 'COUVIS_0040/INDEX/INDEX.LBL'
    path.parent.mkdir(parents=True)
    path.write_text('\r\n'.join(_INDEX_LABEL), newline='')
    messages = []
    for host in (CassiniUVIS, oops.Host):
        with pytest.raises(oops.host.HostIndexError,
                           match="Cassini UVIS reads the Ring-Moon Systems Node") as info:
            host.from_index(path)
        messages.append(str(info.value))
    assert messages[0] == messages[1]


def test_merge_supplement_by_file() -> None:
    """Most UVIS supplements list the products in another order than their index."""

    assert CassiniUVIS.MERGE_SUPPLEMENT_BY_FILE is True


@pytest.mark.parametrize(('data_set_id', 'expected'), [
    ('CO-S-UVIS-2-CUBE-V1.2', True),
    ('CO-J-UVIS-2-SSB-V1.2', True),
    ('CO-X-UVIS-2-SPEC-V1.2', True),
    ('CO-S-ISSNA/ISSWA-2-EDR-V1.0', False),
    ('GO-J-SSI-2-REDR-V1.0', False),
], ids=['saturn', 'jupiter', 'cruise', 'other-instrument', 'other-mission'])
def test_detection_in_row(data_set_id: str, expected: bool) -> None:
    assert CassiniUVIS._detect_in_row({'DATA_SET_ID': data_set_id}) is expected


def test_detection_in_row_without_data_set_id() -> None:
    assert CassiniUVIS._detect_in_row({}) is False

##########################################################################################
# from_index
##########################################################################################

_UVIS_SPEC = 'DATA/D2005_001/FUV2005_001_00_00.LBL'


def qube_row(**changes: Any) -> dict[str, Any]:
    """A merged index and supplement row describing a one-line FUV QUBE of three
    samples."""

    row = label_keywords('FUV2005_001_00_00')
    row.update({'FILE_NAME': '/COUVIS_0001/' + _UVIS_SPEC, 'VOLUME_ID': 'COUVIS_0001',
                'FILE_SPECIFICATION_NAME': _UVIS_SPEC, 'DATA_OBJECT_TYPE': 'QUBE',
                'MINIMUM_BAND_NUMBER': 0, 'MAXIMUM_BAND_NUMBER': 1023,
                'BAND_BINNING_FACTOR': 1, 'WINDOW_MINIMUM_LINE_NUMBER': 0,
                'WINDOW_MAXIMUM_LINE_NUMBER': 63, 'LINE_BINNING_FACTOR': 64,
                'LINE_SAMPLES': 3, 'DWELL_TIME': -999})
    row.update(changes)
    return row


def _labeled_and_row(directory: pathlib.Path,
                     kind: str) -> tuple[pathlib.Path, dict[str, Any]]:
    """A product label of the given kind and the index row that describes it."""

    window = {'UL_CORNER_LINE': 8, 'LR_CORNER_LINE': 23, 'LINE_BIN': 2,
              'UL_CORNER_BAND': 0, 'LR_CORNER_BAND': 1023, 'BAND_BIN': 1}
    if kind == 'one-line QUBE':
        return (write_qube(directory, bands=1024, lines=1, samples=3), qube_row())
    if kind == 'QUBE':
        return (write_qube(directory, bands=1024, lines=64, samples=2, window=window),
                qube_row(WINDOW_MINIMUM_LINE_NUMBER=8, WINDOW_MAXIMUM_LINE_NUMBER=23,
                         LINE_BINNING_FACTOR=2, LINE_SAMPLES=2))
    if kind == 'TIME_SERIES':
        return (write_time_series(directory),
                qube_row(PRODUCT_ID='HSP2005_001_00_00', DATA_OBJECT_TYPE='TIME_SERIES',
                         INTEGRATION_DURATION=2., LINE_SAMPLES=5))
    return (write_spectrum(directory),
            qube_row(PRODUCT_ID='EUV2005_001_00_00', DATA_OBJECT_TYPE='SPECTRUM',
                     MINIMUM_BAND_NUMBER=100, MAXIMUM_BAND_NUMBER=105,
                     WINDOW_MINIMUM_LINE_NUMBER=16, WINDOW_MAXIMUM_LINE_NUMBER=31,
                     LINE_BINNING_FACTOR=16))


@pytest.mark.parametrize('kind', ['one-line QUBE', 'QUBE', 'TIME_SERIES', 'SPECTRUM'])
def test_from_index_row_gives_the_labels_observation(tmp_path: pathlib.Path,
                                                     kind: str) -> None:
    """A row's observation is the one the product's label gives with astrometry."""

    (label, row) = _labeled_and_row(tmp_path, kind)
    from_label = CassiniUVIS.from_file(label, astrometry=True)
    [from_row] = CassiniUVIS.from_index(tmp_path / 'INDEX.LBL', row_dicts=[row])
    assert (from_row.shape, from_row.fov.uv_shape, from_row.time) == (
        from_label.shape, from_label.fov.uv_shape, from_label.time)


def test_from_index_hdac_interval_is_its_dwell_time() -> None:
    row = qube_row(PRODUCT_ID='HDAC2005_001_00_00', DATA_OBJECT_TYPE='TIME_SERIES',
                   INTEGRATION_DURATION=-999., DWELL_TIME=32, LINE_SAMPLES=5)
    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row])
    assert obs.time[1] - obs.time[0] == pytest.approx(5 * 4.)


def test_from_index_row_without_a_positive_interval_is_a_format_error() -> None:
    row = qube_row(PRODUCT_ID='HSP2005_001_00_00', DATA_OBJECT_TYPE='TIME_SERIES',
                   INTEGRATION_DURATION=0., LINE_SAMPLES=5)
    with pytest.raises(oops.host.HostFormatError, match='that is not positive: '):
        next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row]))


def test_from_index_row_without_a_column_is_a_format_error() -> None:
    row = qube_row()
    del row['LINE_SAMPLES']
    with pytest.raises(oops.host.HostFormatError,
                       match='Cassini UVIS index row has no LINE_SAMPLES: '):
        next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row]))


def write_tables(directory: pathlib.Path) -> pathlib.Path:
    """Write a two-row UVIS index and a supplement that lists its products in the other
    order; return the index label's path."""

    specs = [_UVIS_SPEC, _UVIS_SPEC.replace('00_00', '00_10')]
    index = write_table(directory, 'COUVIS_0001_index', {
        'FILE_NAME'           : ['/COUVIS_0001/' + spec for spec in specs],
        'START_TIME'          : [START_TIME] * 2,
        'STOP_TIME'           : [STOP_TIME] * 2,
        'TARGET_NAME'         : ['SATURN'] * 2,
        'INTEGRATION_DURATION': [2.] * 2,
        'SLIT_STATE'          : ['HIGH_RESOLUTION'] * 2,
    })
    write_table(directory, 'COUVIS_0001_supplemental_index', {
        'VOLUME_ID'                 : ['COUVIS_0001'] * 2,
        'FILE_SPECIFICATION_NAME'   : specs[::-1],
        'PRODUCT_ID'                : ['FUV2005_001_00_10', 'FUV2005_001_00_00'],
        'OBSERVATION_ID'            : ['UVIS_000SA_TEST001_PRIME'] * 2,
        'DATA_OBJECT_TYPE'          : ['QUBE'] * 2,
        'MINIMUM_BAND_NUMBER'       : [0] * 2,
        'MAXIMUM_BAND_NUMBER'       : [1023] * 2,
        'BAND_BINNING_FACTOR'       : [1] * 2,
        'WINDOW_MINIMUM_LINE_NUMBER': [0] * 2,
        'WINDOW_MAXIMUM_LINE_NUMBER': [63] * 2,
        'LINE_BINNING_FACTOR'       : [64] * 2,
        'LINE_SAMPLES'              : [4, 3],
    })
    return index


def test_from_index_pairs_a_supplement_in_another_order(tmp_path: pathlib.Path) -> None:
    """The index names each product by FILE_NAME and the supplement by VOLUME_ID and
    FILE_SPECIFICATION_NAME."""

    snapshots = CassiniUVIS.from_index(write_tables(tmp_path))
    assert [(obs.dict['PRODUCT_ID'], obs.samples) for obs in snapshots] == [
        ('FUV2005_001_00_00', 3), ('FUV2005_001_00_10', 4)]


def test_from_index_yields_the_rows_before_one_the_index_lacks(
        tmp_path: pathlib.Path) -> None:
    """Rows are read as the iteration proceeds: an index shorter than its label says
    yields every row it holds before the error."""

    index = write_tables(tmp_path)
    claim_more_rows(index, 2, 30)
    snapshots = CassiniUVIS.from_index(index)
    assert [next(snapshots).dict['PRODUCT_ID'], next(snapshots).dict['PRODUCT_ID']] == [
        'FUV2005_001_00_00', 'FUV2005_001_00_10']
    with pytest.raises(oops.host.HostIndexError, match='invalid index table'):
        next(snapshots)


def test_from_index_row_without_the_supplement_raises_at_the_row() -> None:
    row = qube_row()
    del row['FILE_SPECIFICATION_NAME']
    with pytest.raises(oops.host.HostIndexError,
                       match=r'Cassini UVIS needs a supplemental index: INDEX\.LBL'):
        next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row]))


# The supplement is required, so None is refused rather than taken as "search for it"
def test_from_index_refuses_no_supplement() -> None:
    with pytest.raises(oops.host.HostIndexError,
                       match='Cassini UVIS index needs its supplemental index: '
                             r'INDEX\.LBL'):
        CassiniUVIS.from_index('INDEX.LBL', supplement=None)


def test_from_index_rows_need_no_supplement(tmp_path: pathlib.Path) -> None:
    assert len(list(CassiniUVIS.from_index(tmp_path / 'INDEX.LBL', supplement=None,
                                           row_dicts=[qube_row()]))) == 1


def test_from_index_uses_a_given_label(monkeypatch: pytest.MonkeyPatch) -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3', 'END'])
    labels: list[Any] = []

    def spy(filepath: Any, **kwargs: Any) -> list[dict[str, Any]]:
        labels.append(kwargs['label'])
        return []

    monkeypatch.setattr(CassiniUVIS, '_index_rows', spy)
    CassiniUVIS.from_index('INDEX.LBL', label=label)
    assert labels == [label]


def test_from_index_rejects_an_unknown_keyword() -> None:
    with pytest.raises(oops.OopsTypeError,
                       match=r"unexpected keyword argument for CassiniUVIS\.from_index: "
                             r"'supplementary'"):
        CassiniUVIS.from_index('INDEX.LBL', supplementary=None)


def test_from_index_names_the_file_in_full(tmp_path: pathlib.Path) -> None:
    """A row's file is its specification within its volume, found from the index."""

    index = tmp_path / 'metadata/COUVIS_0xxx/COUVIS_0001/COUVIS_0001_index.lbl'
    [obs] = CassiniUVIS.from_index(index, row_dicts=[qube_row()])
    path = tmp_path / 'volumes/COUVIS_0xxx/COUVIS_0001' / _UVIS_SPEC
    assert obs.filepath == FCPath(path)


def test_from_index_unfetched_file_has_no_abspath(tmp_path: pathlib.Path) -> None:
    index = ('https://rms-test.invalid/holdings/metadata/COUVIS_0xxx/COUVIS_0001/'
             'COUVIS_0001_index.lbl')
    [obs] = CassiniUVIS.from_index(index, row_dicts=[qube_row()])
    assert 'abspath' not in obs.subfields

##########################################################################################
# The rules that every host follows
##########################################################################################


def test_from_file_parallel_override(tmp_path: pathlib.Path) -> None:
    other = CassiniUVIS.from_file(write_spectrum(tmp_path))
    obs = CassiniUVIS.from_file(write_qube(tmp_path), parallel=other)
    assert obs.frame.reference.frame_id == other.frame.frame_id


def test_from_file_rejects_an_unknown_keyword(tmp_path: pathlib.Path) -> None:
    with pytest.raises(oops.OopsTypeError, match=r"CassiniUVIS\.from_file: 'enclosed'"):
        CassiniUVIS.from_file(write_qube(tmp_path), enclosed=True)


@pytest.mark.parametrize(('select', 'error'), [('VIS', oops.OopsValueError),
                                               (1.5, oops.OopsTypeError)])
def test_from_file_checks_select_before_reading(tmp_path: pathlib.Path, select: Any,
                                                error: type[Exception]) -> None:
    with pytest.raises(error, match='select'):
        CassiniUVIS.from_file(tmp_path / 'missing.LBL', select=select)


@pytest.mark.parametrize(('select', 'error'), [('VIS', oops.OopsValueError),
                                               (1.5, oops.OopsTypeError)])
def test_from_index_checks_select_when_called(select: Any,
                                              error: type[Exception]) -> None:
    with pytest.raises(error, match='select'):
        CassiniUVIS.from_index('INDEX.LBL', row_dicts=[], select=select)


def test_from_index_builds_each_row_when_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Any] = []
    make_observations = CassiniUVIS._make_observations

    def spy(label: dict[str, Any], filepath: Any, **kwargs: Any) -> Any:
        built.append(filepath)
        return make_observations(label, filepath, **kwargs)

    monkeypatch.setattr(CassiniUVIS, '_make_observations', spy)
    rows = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[qube_row(), qube_row()])
    assert built == []
    next(rows)
    assert len(built) == 1


@pytest.mark.parametrize(('column', 'value', 'masked'), [
    ('SUB_SOLAR_LATITUDE', np.float64(-99.), True),
    ('CENTRAL_BODY_DISTANCE', np.float64(-999.), True),
    ('TARGET_NAME', np.str_('UNK'), False),
], ids=['masked', 'masked-999', 'null-text'])
def test_from_index_leaves_out_missing_values(column: str, value: Any,
                                              masked: bool) -> None:
    row = qube_row(**{column: value, column + '_mask': np.bool_(masked)})
    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row])
    assert column not in obs.dict


def test_from_index_keeps_minus_999_where_the_label_declares_no_null() -> None:
    """-999 is missing only where the table's label declares it a null constant."""

    row = qube_row(ODC_ID=np.int64(-999), ODC_ID_mask=np.bool_(False))
    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row])
    assert obs.dict['ODC_ID'] == -999


def test_from_index_dict_is_json_without_masks() -> None:
    row = qube_row(LINE_SAMPLES=np.int64(3), PRODUCT_ID=np.str_('FUV2005_001_00_00  '),
                   LINE_SAMPLES_mask=np.False_)
    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row])
    assert json.loads(json.dumps(obs.dict, allow_nan=False)) == obs.dict
    assert not any(key.endswith('_mask') for key in obs.dict)
    assert obs.dict['PRODUCT_ID'] == 'FUV2005_001_00_00'


@pytest.mark.parametrize('name', ['dict', 'label_dict'])
def test_from_file_dicts_are_json(tmp_path: pathlib.Path, name: str) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path))
    value = getattr(obs, name)
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_from_file_label_dict_is_the_label(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(tmp_path, TARGET_NAME='UNK'))
    assert obs.label_dict['TARGET_NAME'] == 'UNK'
    assert 'TARGET_NAME' not in obs.dict


def test_from_file_clock_counts(tmp_path: pathlib.Path) -> None:
    obs = CassiniUVIS.from_file(write_qube(
        tmp_path, SPACECRAFT_CLOCK_START_COUNT='1/1485311337.0',
        SPACECRAFT_CLOCK_STOP_COUNT='UNK'))
    assert obs.spacecraft_clock_start_count == '1/1485311337.0'
    assert 'spacecraft_clock_stop_count' not in obs.subfields


def test_time_is_not_from_the_clock(tmp_path: pathlib.Path) -> None:
    assert CassiniUVIS.from_file(write_qube(tmp_path)).time_from_sclk is False


def test_from_index_refuses_a_volumes_own_index(tmp_path: pathlib.Path) -> None:
    with pytest.raises(oops.host.HostIndexError,
                       match='reads the Ring-Moon Systems Node metadata index'):
        CassiniUVIS.from_index(tmp_path / 'COUVIS_0001' / 'INDEX' / 'INDEX.LBL')


def remove_fuv_pointing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SPICE frame of the FUV detector raise as cspyce does with no CK data."""

    def no_pointing(time: float, *args: Any, **kwargs: Any) -> None:
        raise OSError('SPICE(CKINSUFFDATA) -- insufficient C kernel data')

    CassiniUVIS._define_frames()
    frame = oops.Frame.as_frame('CASSINI_UVIS_FUV')
    monkeypatch.setattr(frame, 'transform_at_time', no_pointing)


def test_from_file_without_pointing(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    remove_fuv_pointing(monkeypatch)
    with pytest.raises(oops.OopsSpiceError,
                       match=r'^no Cassini UVIS CASSINI_UVIS_FUV pointing at .* for '
                             r'.*FUV2005_001_00_00\.LBL: SPICE\(CKINSUFFDATA\)$'):
        CassiniUVIS.from_file(write_qube(tmp_path))


def test_from_index_frame_without_pointing_raises_when_used(
        monkeypatch: pytest.MonkeyPatch) -> None:
    remove_fuv_pointing(monkeypatch)
    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[qube_row()])
    with pytest.raises(oops.OopsSpiceError, match=r'SPICE\(CKINSUFFDATA\)$'):
        obs.frame.wrt(oops.Frame.J2000).transform_at_time(obs.midtime)


def test_from_file_parsed_label_without_a_path_has_no_file_subfields(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_qube(tmp_path), method='fast')
    label.filepath = None
    obs = CassiniUVIS.from_file(label, astrometry=True)
    assert {'filepath', 'basename', 'abspath', 'image_url'}.isdisjoint(obs.subfields)


def test_from_file_parsed_label_without_a_path_cannot_be_read_in_full(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_qube(tmp_path), method='fast')
    label.filepath = None
    with pytest.raises(oops.host.HostFormatError, match='no file path is known'):
        CassiniUVIS.from_file(label)


def append_to_label(path: pathlib.Path, *lines: str) -> None:
    """Add lines to the end of a label written by write_label, before its END."""

    label = path.read_bytes()
    assert label.endswith(b'\r\nEND\r\n')
    extra = ''.join(line + '\r\n' for line in lines).encode('ascii')
    path.write_bytes(label[:-len(b'END\r\n')] + extra + b'END\r\n')


def test_from_file_falls_back_after_a_value_error(tmp_path: pathlib.Path) -> None:
    """A value that the fast parser cannot convert, day 366 of a year of 365 days, is
    read by the strict parser."""

    path = write_qube(tmp_path)
    append_to_label(path, 'PRODUCT_CREATION_TIME = 2005-366T00:00:00')
    obs = CassiniUVIS.from_file(path, astrometry=True)
    assert 'PRODUCT_CREATION_TIME' in obs.label_dict


def test_from_file_falls_back_to_a_later_parser(tmp_path: pathlib.Path) -> None:
    """A label that the requested parser rejects, an unquoted value with a slash for the
    strict parser, is read by a later one."""

    path = write_qube(tmp_path)
    append_to_label(path, 'PROCESSING_NOTE = A/B')
    obs = CassiniUVIS.from_file(path, astrometry=True, pds3_options={'method': 'strict'})
    assert obs.dict['PROCESSING_NOTE'] == 'A/B'


def test_from_file_applies_a_single_repair(tmp_path: pathlib.Path) -> None:
    """A single (pattern, replacement) pair is a repair, as a list of pairs is."""

    path = write_qube(tmp_path)
    append_to_label(path, 'DESCRIPTION = "Extended header missing."')
    obs = CassiniUVIS.from_file(path, astrometry=True,
                                pds3_options={'repairs': ('Extended', 'Repaired')})
    assert obs.dict['DESCRIPTION'] == 'Repaired header missing.'


def test_from_file_of_another_instruments_parsed_label() -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3',
                       'INSTRUMENT_HOST_NAME = "CASSINI ORBITER"',
                       'INSTRUMENT_ID = "ISS"', 'END'])
    with pytest.raises(oops.host.WrongHostError, match='not a Cassini UVIS PDS3 label'):
        CassiniUVIS.from_file(label, astrometry=True)


def test_from_file_leaves_out_a_null_inside_an_object(tmp_path: pathlib.Path) -> None:
    window = {'UL_CORNER_LINE': 0, 'LR_CORNER_LINE': 63, 'LINE_BIN': 64,
              'UL_CORNER_BAND': 0, 'LR_CORNER_BAND': 7, 'BAND_BIN': 1,
              'CORE_VALID_MINIMUM': 'N/A'}
    obs = CassiniUVIS.from_file(write_qube(tmp_path, window=window), astrometry=True)
    assert 'CORE_VALID_MINIMUM' not in obs.dict['QUBE']


@pytest.mark.parametrize('key', ['START_TIME', 'STOP_TIME'])
def test_from_index_row_without_a_time_is_a_format_error(key: str) -> None:
    row = qube_row()
    del row[key]
    with pytest.raises(oops.host.HostFormatError, match=f'{key} is missing from '):
        next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[row]))


def test_from_index_select_true_is_a_type_error() -> None:
    with pytest.raises(oops.OopsTypeError, match='invalid select: True'):
        CassiniUVIS.from_index('INDEX.LBL', row_dicts=[], select=True)


def test_from_index_probes_pointing_at_the_cadence_midtime(
        spice_check_times: list[float]) -> None:
    """With a cadence override, a row's pointing is probed at the override's midtime."""

    [obs] = CassiniUVIS.from_index('INDEX.LBL', row_dicts=[qube_row()])
    cadence = obs.cadence.time_shift(1000.)
    next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[qube_row()], cadence=cadence))
    assert spice_check_times[-1] == cadence.midtime


def test_from_index_probes_pointing_after_a_time_shift(
        spice_check_times: list[float]) -> None:
    """A time shift of the observation ("T") moves the time of the pointing probe."""

    obs = next(CassiniUVIS.from_index('INDEX.LBL', row_dicts=[qube_row()],
                                      timeshift=(30., 'T')))
    assert spice_check_times == [pytest.approx(obs.midtime, abs=1.e-3)]

##########################################################################################
