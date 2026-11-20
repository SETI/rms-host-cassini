##########################################################################################
# tests/oops/test_vims.py
##########################################################################################
"""Tests for the Cassini VIMS host, host_cassini.oops.vims."""

import json
import pathlib
import re
from collections.abc import Callable
from typing import Any

import cspyce
from filecache import FCPath
import julian
import numpy as np
import oops
import pytest
from conftest import FakeSpice, FakeSpiceDB, pds3_lines, write_table
from oops.host import HostFormatError, HostIndexError, WrongHostError
from pdsparser import Pds3Label

import host_cassini.oops.vims as vims
from host_cassini.oops._cassini import _Cassini
from host_cassini.oops.vims     import CassiniVIMS

START_TIME = '2005-01-01T00:00:00.000'
STOP_TIME = '2005-01-01T00:00:10.000'
RECORD_BYTES = 512
BAND_SUFFIX_NAME = ['X_SCAN_DRIVE_CURRENT', 'Z_SCAN_DRIVE_CURRENT',
                    'X_SCAN_MIRROR_POSITION', 'Z_SCAN_MIRROR_POSITION']
SAMPLE_SUFFIX_ITEMS = 1
BAND_SUFFIX_ITEMS = 4
VIS_BANDS = 96


def vims_keywords(**changes: Any) -> dict[str, Any]:
    """The label keywords that the VIMS host reads, for a 4x2 image of Saturn."""

    keywords = {
        'INSTRUMENT_HOST_NAME'     : 'CASSINI ORBITER',
        'INSTRUMENT_ID'            : 'VIMS',
        'TARGET_NAME'              : 'SATURN',
        'OBSERVATION_ID'           : 'VIMS_000SA_TEST001_PRIME',
        'START_TIME'               : START_TIME,
        'STOP_TIME'                : STOP_TIME,
        'OVERWRITTEN_CHANNEL_FLAG' : 'OFF',
        'INSTRUMENT_MODE_ID'       : 'IMAGE',
        'INTERFRAME_DELAY_DURATION': 520.,
        'INTERLINE_DELAY_DURATION' : 400.,
        'POWER_STATE_FLAG'         : ['ON', 'ON'],
        'EXPOSURE_DURATION'        : [100., 1000.],
        'X_OFFSET'                 : 31,
        'Z_OFFSET'                 : 32,
        'SWATH_WIDTH'              : 4,
        'SWATH_LENGTH'             : 2,
        'PACKING_FLAG'             : 'OFF',
        'SAMPLING_MODE_ID'         : ['NORMAL', 'NORMAL'],
    }
    keywords.update(changes)
    return keywords


def cube_core(samples: int, bands: int, lines: int) -> np.ndarray:
    """Distinct core values in (line, sample, band) order."""

    values = np.arange(lines * bands * samples) % 4000
    return values.reshape(lines, bands, samples).swapaxes(1, 2)


def cube_bytes(samples: int, bands: int, lines: int) -> bytes:
    """A cube with one sample suffix and four band suffixes, as VIMS writes it."""

    core = cube_core(samples, bands, lines)
    chunks = []
    for line in range(lines):
        for band in range(bands):
            chunks.append(core[line, :, band].astype('>i2').tobytes())
            chunks.append(np.zeros(SAMPLE_SUFFIX_ITEMS, dtype='>i4').tobytes())
        chunks.append(np.zeros(BAND_SUFFIX_ITEMS * (samples + SAMPLE_SUFFIX_ITEMS),
                               dtype='>i4').tobytes())
    return b''.join(chunks)


def write_pds3_cube(directory: pathlib.Path, *, samples: int = 4, bands: int = 352,
                    lines: int = 2, **changes: Any) -> pathlib.Path:
    """Write a detached PDS3 label and its cube; return the label's path."""

    data = cube_bytes(samples, bands, lines)
    (directory / 'v0000000001_1.qub').write_bytes(bytes(RECORD_BYTES) + data)

    text = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
            f'RECORD_BYTES = {RECORD_BYTES}',
            f'FILE_RECORDS = {1 + len(data) // RECORD_BYTES}',
            '^QUBE = ("v0000000001_1.qub", 2)']
    text += pds3_lines(vims_keywords(**changes))
    text += ['OBJECT = SPECTRAL_QUBE', '  AXES = 3', '  AXIS_NAME = (SAMPLE, BAND, LINE)',
             f'  CORE_ITEMS = ({samples}, {bands}, {lines})',
             f'  SUFFIX_ITEMS = ({SAMPLE_SUFFIX_ITEMS}, {BAND_SUFFIX_ITEMS}, 0)']
    text += pds3_lines({'BAND_SUFFIX_NAME': BAND_SUFFIX_NAME}, indent='  ')
    text += ['END_OBJECT = SPECTRAL_QUBE', 'END', '']
    path = directory / 'v0000000001_1.lbl'
    path.write_text('\r\n'.join(text))
    return path


def write_isis_cube(directory: pathlib.Path, *, samples: int = 4, bands: int = 352,
                    lines: int = 2, **changes: Any) -> pathlib.Path:
    """Write an ISIS cube, whose attached label describes it in a QUBE object."""

    label_records = 8
    keywords = vims_keywords(**changes)
    keywords['PACKING'] = keywords.pop('PACKING_FLAG')

    text = ['CCSD3ZF0000100000001NJPL3IF0PDS200000001 = SFDU_LABEL',
            'RECORD_TYPE = FIXED_LENGTH', f'RECORD_BYTES = {RECORD_BYTES}',
            f'^QUBE = {label_records + 1}', 'OBJECT = QUBE', '  AXES = 3',
            '  AXIS_NAME = (SAMPLE, BAND, LINE)',
            f'  CORE_ITEMS = ({samples}, {bands}, {lines})', '  CORE_ITEM_BYTES = 2',
            '  CORE_ITEM_TYPE = SUN_INTEGER',
            f'  SUFFIX_ITEMS = ({SAMPLE_SUFFIX_ITEMS}, {BAND_SUFFIX_ITEMS}, 0)',
            '  SAMPLE_SUFFIX_ITEM_TYPE = SUN_INTEGER']
    text += pds3_lines({'BAND_SUFFIX_NAME': BAND_SUFFIX_NAME, **keywords}, indent='  ')
    text += ['END_OBJECT = QUBE', 'END', '']
    header = '\r\n'.join(text).encode('ascii')
    assert len(header) <= label_records * RECORD_BYTES

    path = directory / 'v0000000001_1.qub'
    path.write_bytes(header.ljust(label_records * RECORD_BYTES)
                     + cube_bytes(samples, bands, lines))
    return path


Observations = tuple[oops.Observation, oops.Observation]


def row_label(row: dict[str, Any]) -> dict[str, Any]:
    """The label that the VIMS host builds from an index row."""

    label: dict[str, Any] = CassiniVIMS._label_from_index_row(
        row, FCPath('v0000000001_1.qub'))
    return label


def index_row(**changes: Any) -> dict[str, Any]:
    """A merged index and supplemental index row for the cube of `vims_keywords()`."""

    keywords = vims_keywords()
    row = {name: keywords[name]
           for name in ('TARGET_NAME', 'OBSERVATION_ID', 'START_TIME',
                        'INSTRUMENT_MODE_ID', 'SWATH_WIDTH', 'SWATH_LENGTH',
                        'OVERWRITTEN_CHANNEL_FLAG', 'PACKING_FLAG',
                        'INTERFRAME_DELAY_DURATION', 'INTERLINE_DELAY_DURATION',
                        'X_OFFSET', 'Z_OFFSET')}
    (row['IR_POWER_STATE_FLAG'], row['VIS_POWER_STATE_FLAG']) = (
        keywords['POWER_STATE_FLAG'])
    (row['IR_EXPOSURE'], row['VIS_EXPOSURE']) = keywords['EXPOSURE_DURATION']
    (row['IR_SAMPLING_MODE_ID'], row['VIS_SAMPLING_MODE_ID']) = (
        keywords['SAMPLING_MODE_ID'])
    row.update({'VOLUME_ID': 'COVIMS_0001', 'FILE_NAME': 'v0000000001_1.qub',
                'FILE_SPECIFICATION_NAME': 'data/2005001T000000/v0000000001_1.lbl',
                'STOP_TIME': STOP_TIME})
    row.update(changes)
    return row


@pytest.fixture
def image_pair(tmp_path: pathlib.Path) -> Observations:
    """The (VIS, IR) observations of a 4x2 image cube."""

    (vis, ir) = CassiniVIMS.from_file(write_pds3_cube(tmp_path))
    assert vis is not None
    assert ir is not None
    return (vis, ir)

##########################################################################################
# Registration and detection
##########################################################################################


def test_vims_is_registered_with_oops() -> None:
    assert oops.Host._LOOKUP['Cassini VIMS'] is CassiniVIMS


def test_module_exports() -> None:
    assert vims.__all__ == ['CassiniVIMS']


@pytest.mark.parametrize(('label', 'expected'), [
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'VIMS'}, True),
    ({'QUBE': {'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'VIMS'}},
     True),
    ({'INSTRUMENT_HOST_NAME': 'CASSINI ORBITER', 'INSTRUMENT_ID': 'ISSNA'}, False),
    ({'INSTRUMENT_HOST_NAME': 'GALILEO ORBITER', 'INSTRUMENT_ID': 'VIMS'}, False),
    ({'QUBE': {'CORE_ITEMS': [1, 2, 3]}}, False),
    ({}, False),
], ids=['pds3', 'isis', 'other-instrument', 'other-host', 'other-qube', 'empty'])
def test_detection(label: dict[str, Any], expected: bool) -> None:
    assert CassiniVIMS._detect_in_pds3(label) is expected

##########################################################################################
# initialize() and _define_frames()
##########################################################################################


def test_initialize_reads_instrument_kernel(fake_spicedb: FakeSpiceDB) -> None:
    CassiniVIMS.initialize()
    assert CassiniVIMS._INSTRUMENT_KERNEL is fake_spicedb.kernel_dict
    assert fake_spicedb.named('select_inst')[-1][1]['inst'] == 'VIMS'


def test_initialize_ignores_later_calls(fake_spicedb: FakeSpiceDB) -> None:
    CassiniVIMS.initialize()
    count = len(fake_spicedb.calls)
    CassiniVIMS.initialize(spk='predicted')
    assert len(fake_spicedb.calls) == count


def test_reset_clears_state() -> None:
    CassiniVIMS.initialize()
    CassiniVIMS._reset()
    assert not CassiniVIMS._initialized
    assert CassiniVIMS._INSTRUMENT_KERNEL is None
    assert not _Cassini.initialized


def test_define_frames_only_once(fake_spice: FakeSpice) -> None:
    CassiniVIMS._define_frames()
    CassiniVIMS._define_frames()
    assert fake_spice.spice_frames == ['CASSINI_VIMS_V', 'CASSINI_VIMS_IR',
                                       'CASSINI_VIMS_IR_SOL']

##########################################################################################
# from_file()
##########################################################################################


def test_image_observation_types(image_pair: Observations) -> None:
    assert [type(obs) for obs in image_pair] == [oops.observation.TimedImage,
                                                 oops.observation.TimedImage]


def test_image_data(image_pair: Observations) -> None:
    (vis, ir) = image_pair
    core = cube_core(4, 352, 2)
    assert np.array_equal(vis.data, core[:, :, :VIS_BANDS])
    assert np.array_equal(ir.data, core[:, :, VIS_BANDS:])


def test_image_subfields(image_pair: Observations) -> None:
    (vis, ir) = image_pair
    assert [obs.detector for obs in image_pair] == ['VIS', 'IR']
    assert [obs.sampling for obs in image_pair] == ['NORMAL', 'NORMAL']
    assert {obs.host for obs in image_pair} == {'Cassini'}
    assert {obs.instrument for obs in image_pair} == {'VIMS'}
    assert {obs.basename for obs in image_pair} == {'v0000000001_1.lbl'}
    assert vis.dict is ir.dict


def test_image_frames(image_pair: Observations) -> None:
    (vis, ir) = image_pair
    assert vis.frame == oops.Frame.as_wayframe('CASSINI_VIMS_V')
    assert ir.frame == oops.Frame.as_wayframe('CASSINI_VIMS_IR')


def test_image_fovs(image_pair: Observations) -> None:
    (vis, ir) = image_pair
    assert tuple(vis.fov.uv_shape.vals) == (4, 2)
    assert tuple(ir.fov.uv_shape.vals) == (4, 2)
    assert tuple(ir.fov.uv_scale.vals) == pytest.approx((0.495e-3, 0.495e-3))
    assert tuple(vis.fov.uv_scale.vals) == pytest.approx((0.506e-3, 0.506e-3))


def test_image_times(image_pair: Observations) -> None:
    (vis, ir) = image_pair
    tstart = julian.tdb_from_iso(START_TIME)
    assert vis.time[0] == pytest.approx(tstart)
    assert ir.time[0] == pytest.approx(tstart)


def test_image_label_has_fmt_content(image_pair: Observations) -> None:
    qube = image_pair[0].dict['SPECTRAL_QUBE']
    assert qube['CORE_NAME'] == 'RAW DATA NUMBER'
    assert image_pair[0].dict['QUBE'] == qube


def test_from_file_leaves_parsed_label_unchanged(tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_cube(tmp_path), method='fast')
    CassiniVIMS.from_file(label)
    assert 'QUBE' not in label.dict
    assert 'CORE_NAME' not in label.dict['SPECTRAL_QUBE']


def test_isis_cube(tmp_path: pathlib.Path) -> None:
    (vis, ir) = CassiniVIMS.from_file(write_isis_cube(tmp_path))
    core = cube_core(4, 352, 2)
    assert np.array_equal(vis.data, core[:, :, :VIS_BANDS])
    assert np.array_equal(ir.data, core[:, :, VIS_BANDS:])
    assert ir.dict['PACKING_FLAG'] == 'OFF'


def test_astrometry_reads_no_data(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path)
    (tmp_path / 'v0000000001_1.qub').unlink()
    (vis, ir) = CassiniVIMS.from_file(path, astrometry=True)
    assert not hasattr(vis, 'data')
    assert not hasattr(ir, 'data')


@pytest.mark.parametrize(('power', 'active'), [(['ON', 'OFF'], [False, True]),
                                               (['OFF', 'ON'], [True, False])],
                         ids=['vis-off', 'ir-off'])
def test_inactive_channel(tmp_path: pathlib.Path, power: list[str],
                          active: list[bool]) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path, POWER_STATE_FLAG=power))
    assert [obs is not None for obs in pair] == active


@pytest.mark.parametrize('changes', [{'TARGET_NAME': 'SUN'},
                                     {'OBSERVATION_ID': 'VIMS_000SA_SOLAR_SOL'}],
                         ids=['sun-target', 'sol-observation'])
def test_solar_port_frame(tmp_path: pathlib.Path, changes: dict[str, Any]) -> None:
    (_, ir) = CassiniVIMS.from_file(write_pds3_cube(tmp_path, **changes))
    assert ir.frame == oops.Frame.as_wayframe('CASSINI_VIMS_IR_SOL')


def test_single_line(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, lines=1, SWATH_LENGTH=1)
    (vis, ir) = CassiniVIMS.from_file(path)
    assert type(vis) is oops.observation.Slit1D
    assert type(ir) is oops.observation.RasterSlit1D
    assert vis.data.shape == (4, VIS_BANDS)
    assert ir.data.shape == (4, 256)


def test_point(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, samples=2, lines=1, SWATH_WIDTH=1, SWATH_LENGTH=1,
                           POWER_STATE_FLAG=['ON', 'OFF'])
    (vis, ir) = CassiniVIMS.from_file(path)
    assert vis is None
    assert type(ir) is oops.observation.Pixel
    assert ir.data.shape == (2, 256)


def test_point_with_vis_on(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, samples=2, lines=1, SWATH_WIDTH=1, SWATH_LENGTH=1)
    (vis, ir) = CassiniVIMS.from_file(path)
    assert type(vis) is oops.observation.Pixel
    assert vis.data.shape == (2, 96)
    assert vis.cadence.shape == (2,)
    assert vis.time[0] == pytest.approx(ir.time[0])
    assert vis.cadence.texp == pytest.approx(1. * vims._TIME_FACTOR)


def test_point_with_ir_off(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, samples=2, lines=1, SWATH_WIDTH=1, SWATH_LENGTH=1,
                           POWER_STATE_FLAG=['OFF', 'ON'])
    (vis, ir) = CassiniVIMS.from_file(path)
    assert ir is None
    assert type(vis) is oops.observation.Pixel
    assert vis.data.shape == (2, 96)


def test_vis_off_tolerates_placeholder_exposure(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, POWER_STATE_FLAG=['ON', 'OFF'],
                           EXPOSURE_DURATION=[100., -999.])
    (vis, ir) = CassiniVIMS.from_file(path)
    assert vis is None
    assert type(ir) is oops.observation.TimedImage


def test_unsupported_shape(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, lines=1, SWATH_WIDTH=2, SWATH_LENGTH=2)
    with pytest.raises(HostFormatError, match='unsupported VIMS format'):
        CassiniVIMS.from_file(path, astrometry=True)


def test_misaligned_suffix(tmp_path: pathlib.Path) -> None:
    label = {'SPECTRAL_QUBE': {'CORE_ITEMS': (1, 256, 1), 'SUFFIX_ITEMS': (0, 1)},
             'RECORD_BYTES': 512}
    path = tmp_path / 'cube.qub'
    with pytest.raises(HostFormatError,
                       match=re.escape(f'misaligned array elements in {path}')):
        CassiniVIMS._load_data_and_times(path, 0, label)


def test_validate_index_file_names() -> None:
    index_row = {'VOLUME_ID': 'COVIMS_0001', 'FILE_NAME': 'v1.QUB'}
    supp_row = {'VOLUME_ID': 'COVIMS_0001', 'FILE_SPECIFICATION_NAME': 'data/v2.LBL'}
    with pytest.raises(HostIndexError,
                       match=re.escape('file names do not match: "v1.QUB", "v2.LBL"')):
        CassiniVIMS._validate_index(index_row, supp_row)


def test_from_index_requires_supplement() -> None:
    row = {'VOLUME_ID': 'COVIMS_0001', 'FILE_NAME': 'v1.QUB'}
    with pytest.raises(HostIndexError,
                       match=re.escape('Cassini VIMS needs a supplemental index: '
                                       'INDEX.LBL')):
        CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])


# The supplement is required, so None is refused rather than taken as "search for it"
def test_from_index_refuses_no_supplement() -> None:
    with pytest.raises(HostIndexError,
                       match='Cassini VIMS index needs its supplemental index: '
                             r'INDEX\.LBL'):
        CassiniVIMS.from_index('INDEX.LBL', supplement=None)


def test_from_index_rows_need_no_supplement() -> None:
    assert len(list(CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row()],
                                           supplement=None))) == 1


def test_from_index_uses_a_given_label(monkeypatch: pytest.MonkeyPatch) -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3', 'END'])
    labels: list[Any] = []

    def spy(filepath: Any, **kwargs: Any) -> list[dict[str, Any]]:
        labels.append(kwargs['label'])
        return []

    monkeypatch.setattr(CassiniVIMS, '_read_index_rows', spy)
    CassiniVIMS.from_index('INDEX.LBL', label=label)
    assert labels == [label]


def test_from_index_rejects_an_unknown_keyword() -> None:
    with pytest.raises(oops.OopsTypeError,
                       match=r"unexpected keyword argument for CassiniVIMS\.from_index: "
                             r"'supplementary'"):
        CassiniVIMS.from_index('INDEX.LBL', supplementary=None)


@pytest.mark.parametrize(('key', 'expected'), [
    ('POWER_STATE_FLAG', ('ON', 'ON')),
    ('EXPOSURE_DURATION', (100., 1000.)),
    ('SAMPLING_MODE_ID', ('NORMAL', 'NORMAL')),
])
def test_label_from_index_row_pairs_channels(key: str, expected: tuple[Any, Any]) -> None:
    assert row_label(index_row())[key] == expected


def test_from_index_pairs_a_supplement_in_another_order(tmp_path: pathlib.Path) -> None:
    """The index names each cube's data file by VOLUME_ID, PATH_NAME and FILE_NAME, and
    the supplement names its label by VOLUME_ID and FILE_SPECIFICATION_NAME."""

    rows = [index_row(), index_row(FILE_NAME='v0000000002_1.qub',
                                   OBSERVATION_ID='VIMS_000SA_TEST002_PRIME')]
    columns = {name: [row[name] for row in rows] for name in rows[0]
               if name != 'FILE_SPECIFICATION_NAME'}
    columns['PATH_NAME'] = ['/data/2005001T000000'] * 2
    index = write_table(tmp_path, 'COVIMS_0001_index', columns)
    write_table(tmp_path, 'COVIMS_0001_supplemental_index', {
        'VOLUME_ID'              : ['COVIMS_0001'] * 2,
        'FILE_SPECIFICATION_NAME': ['data/2005001T000000/v0000000002_1.lbl',
                                    'data/2005001T000000/v0000000001_1.lbl'],
        'PRODUCT_CREATION_TIME'  : ['2005-01-02', '2005-01-01'],
    })
    pairs = CassiniVIMS.from_index(index)
    assert [(ir.dict['OBSERVATION_ID'], ir.dict['PRODUCT_CREATION_TIME'])
            for (_, ir) in pairs] == [('VIMS_000SA_TEST001_PRIME', '2005-01-01'),
                                      ('VIMS_000SA_TEST002_PRIME', '2005-01-02')]


def test_label_from_index_row_core_items() -> None:
    label = row_label(index_row())
    assert label['SPECTRAL_QUBE']['CORE_ITEMS'] == (4, 352, 2)
    assert label['QUBE'] is label['SPECTRAL_QUBE']


@pytest.mark.parametrize(('mode', 'core_items'), [
    ('POINT', (1, 352, 64)),
    ('OCCULTATION', (64, 352, 64)),
])
def test_label_from_index_row_full_buffer(mode: str,
                                          core_items: tuple[int, int, int]) -> None:
    row = index_row(INSTRUMENT_MODE_ID=mode, SWATH_WIDTH=1, SWATH_LENGTH=1)
    assert row_label(row)['QUBE']['CORE_ITEMS'] == core_items


def test_label_from_index_row_uses_core_columns() -> None:
    row = index_row(INSTRUMENT_MODE_ID='POINT', SWATH_WIDTH=1, SWATH_LENGTH=1,
                    CORE_SAMPLES=1, CORE_BANDS=352, CORE_LINES=24)
    assert row_label(row)['QUBE']['CORE_ITEMS'] == (1, 352, 24)


def test_label_from_index_row_uses_exposure_columns() -> None:
    row = index_row(IR_EXPOSURE=-999., IR_EXPOSURE_DURATION=160.,
                    VIS_EXPOSURE_DURATION=1000.)
    label = row_label(row)
    assert label['EXPOSURE_DURATION'] == (160., 1000.)
    assert label['POWER_STATE_FLAG'] == ('ON', 'ON')


def test_from_index_uses_backplane_sclk() -> None:
    row = index_row(INSTRUMENT_MODE_ID='POINT', SWATH_WIDTH=1, SWATH_LENGTH=1,
                    VIS_POWER_STATE_FLAG='OFF', CORE_SAMPLES=1, CORE_BANDS=256,
                    CORE_LINES=24, BACKPLANE_SCLK_START_COUNT='1294638282.666')
    [(_, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert ir.cadence.shape == (24,)
    assert ir.time[0] == pytest.approx(1294638282.666)


def test_from_index_blank_backplane_sclk_uses_start_time() -> None:
    row = index_row(BACKPLANE_SCLK_START_COUNT='              ')
    [(_, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert ir.time[0] == pytest.approx(julian.tdb_from_iso(START_TIME))


def test_label_from_index_row_leaves_row_unchanged() -> None:
    row = index_row()
    row_label(row)
    assert row == index_row()


def test_from_index_image_matches_from_file(image_pair: Observations) -> None:
    [pair] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row()])
    for (from_index, from_file) in zip(pair, image_pair, strict=True):
        assert type(from_index) is type(from_file)
        assert from_index.time == pytest.approx(from_file.time)
        assert tuple(from_index.fov.uv_shape.vals) == tuple(from_file.fov.uv_shape.vals)


def test_from_index_filepath_and_no_data(tmp_path: pathlib.Path) -> None:
    """A row's file is its specification within its volume, found from the index."""

    index = tmp_path / 'metadata/COVIMS_0xxx/COVIMS_0001/COVIMS_0001_index.lbl'
    [(_, ir)] = CassiniVIMS.from_index(index, row_dicts=[index_row()])
    path = tmp_path / 'volumes/COVIMS_0xxx/COVIMS_0001/data/2005001T000000'
    assert ir.filepath == FCPath(path / 'v0000000001_1.lbl')
    assert not hasattr(ir, 'data')


def test_from_index_unfetched_file_has_no_abspath() -> None:
    index = ('https://rms-test.invalid/holdings/metadata/COVIMS_0xxx/COVIMS_0001/'
             'COVIMS_0001_index.lbl')
    [(_, ir)] = CassiniVIMS.from_index(index, row_dicts=[index_row()])
    assert 'abspath' not in ir.subfields


@pytest.mark.parametrize(('mode', 'steps'), [('POINT', 64), ('OCCULTATION', 4096)])
def test_from_index_point_assumes_full_buffer(mode: str, steps: int) -> None:
    row = index_row(INSTRUMENT_MODE_ID=mode, SWATH_WIDTH=1, SWATH_LENGTH=1,
                    VIS_POWER_STATE_FLAG='OFF')
    [(vis, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert vis is None
    assert type(ir) is oops.observation.Pixel
    assert ir.cadence.shape == (steps,)
    assert ir.time[0] == pytest.approx(julian.tdb_from_iso(START_TIME))


def test_from_index_point_with_vis_on() -> None:
    row = index_row(INSTRUMENT_MODE_ID='POINT', SWATH_WIDTH=1, SWATH_LENGTH=1)
    [(vis, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert type(vis) is oops.observation.Pixel
    assert vis.cadence.shape == (64,)
    assert vis.time[0] == pytest.approx(ir.time[0])


def test_from_index_point_with_ir_off() -> None:
    row = index_row(INSTRUMENT_MODE_ID='POINT', SWATH_WIDTH=1, SWATH_LENGTH=1,
                    IR_POWER_STATE_FLAG='OFF')
    [(vis, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert ir is None
    assert type(vis) is oops.observation.Pixel


def test_from_index_point_ignores_placeholder_exposure_of_off_channels() -> None:
    row = index_row(INSTRUMENT_MODE_ID='POINT', SWATH_WIDTH=1, SWATH_LENGTH=1,
                    IR_EXPOSURE=-999., VIS_POWER_STATE_FLAG='OFF')
    [pair] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert pair == (None, None)


def test_from_index_vis_off_tolerates_placeholder_exposure() -> None:
    row = index_row(VIS_POWER_STATE_FLAG='OFF', VIS_EXPOSURE=-999.)
    [(vis, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert vis is None
    assert type(ir) is oops.observation.TimedImage


@pytest.mark.parametrize(('changes', 'on_index'), [
    ({'VIS_EXPOSURE': -999.}, 1),
    ({'IR_EXPOSURE': 0.}, 0),
], ids=['VIS', 'IR'])
def test_from_index_treats_channel_without_exposure_as_off(changes: dict[str, Any],
                                                           on_index: int) -> None:
    [pair] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row(**changes)])
    assert pair[1 - on_index] is None
    assert type(pair[on_index]) is oops.observation.TimedImage


def test_from_index_select() -> None:
    [ir] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row()], select='IR')
    assert ir.detector == 'IR'


def test_from_index_leaves_rows_unchanged() -> None:
    rows = [index_row()]
    CassiniVIMS.from_index('INDEX.LBL', row_dicts=rows)
    assert rows == [index_row()]


def test_fov_override(tmp_path: pathlib.Path) -> None:
    fov = oops.fov.FlatFOV((1.e-3, 1.e-3), (4, 2))
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path), fov=fov)
    assert [obs.fov for obs in pair] == [fov, fov]


@pytest.mark.parametrize(('target_name', 'target'), [('SATURN', 'SATURN'),
                                                     ('DARK SKY', 'NONE')])
def test_target_from_label(tmp_path: pathlib.Path, target_name: str, target: str) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path, TARGET_NAME=target_name))
    assert [obs.target for obs in pair] == [target, target]
    assert [obs.label_target for obs in pair] == [target_name, target_name]
    assert [obs.lightsource for obs in pair] == ['SUN', 'SUN']


@pytest.mark.parametrize(('obs_id', 'star'), [
    ('VIMS_000SA_ALPBOOCC001_PRIME', 'alf Boo'),
    ('VIMS_000SA_URGAMCRUETACAL001_PRIME', 'gam Cru'),
], ids=['occultation', 'repaired'])
def test_star_lightsource_from_observation_id(tmp_path: pathlib.Path, obs_id: str,
                                              star: str) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path, OBSERVATION_ID=obs_id))
    source = oops.lightsource.star_lookup(star)
    assert [obs.lightsource for obs in pair] == [source, source]
    assert [obs.observation_id for obs in pair] == [obs_id, obs_id]


def test_star_target_from_label(tmp_path: pathlib.Path) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path, TARGET_NAME='SPICA'))
    source = oops.lightsource.star_lookup('SPICA')
    assert [obs.target for obs in pair] == ['NONE', 'NONE']
    assert [obs.lightsource for obs in pair] == [source, source]


@pytest.mark.usefixtures('saturn_path')
def test_tracker_follows_label_target(tmp_path: pathlib.Path) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path), tracker='midtime')
    frames = [oops.Frame.as_primary_frame(obs.frame) for obs in pair]
    assert [type(frame) for frame in frames] == [oops.frame.TrackerFrame] * 2


def test_target_override(tmp_path: pathlib.Path) -> None:
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path), target='TITAN')
    assert [obs.target for obs in pair] == ['TITAN', 'TITAN']


def test_custom_frame(tmp_path: pathlib.Path, fake_spice: FakeSpice) -> None:
    frame = oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                               frame_id='TEST_CMATRIX')
    pair = CassiniVIMS.from_file(write_pds3_cube(tmp_path), frame=frame)
    assert [obs.frame for obs in pair] == [frame.wayframe, frame.wayframe]
    assert not _Cassini.CK_LOADED.any()
    assert _Cassini.SPK_LOADED.any()
    assert fake_spice.spice_frames == []


def test_host_from_file_dispatches_to_vims(tmp_path: pathlib.Path) -> None:
    pair = oops.Host.from_file(write_pds3_cube(tmp_path), astrometry=True)
    assert [obs.instrument for obs in pair] == ['VIMS', 'VIMS']

##########################################################################################
# Index detection
##########################################################################################

_VIMS_INDEX_KEYWORDS = ['INSTRUMENT_HOST_ID = "CO"', 'INSTRUMENT_ID = "VIMS"']


def write_index(directory: pathlib.Path, keywords: list[str],
                volume_id: str) -> pathlib.Path:
    """Write a one-row, one-column index of VOLUME_ID; return its label's path."""

    row = f'"{volume_id:<11}"\r\n'
    (directory / 'index.tab').write_text(row, newline='')
    label = (['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
              f'RECORD_BYTES = {len(row)}', 'FILE_RECORDS = 1',
              '^INDEX_TABLE = "index.tab"'] + keywords
             + ['OBJECT = INDEX_TABLE', '  INTERCHANGE_FORMAT = ASCII',
                f'  ROW_BYTES = {len(row)}', '  ROWS = 1', '  COLUMNS = 1',
                '  OBJECT = COLUMN', '    NAME = VOLUME_ID', '    DATA_TYPE = CHARACTER',
                '    START_BYTE = 2', '    BYTES = 11', '  END_OBJECT = COLUMN',
                'END_OBJECT = INDEX_TABLE', 'END', ''])
    path = directory / 'index.lbl'
    path.write_text('\r\n'.join(label), newline='')
    return path


@pytest.mark.parametrize(('keywords', 'expected'), [
    (_VIMS_INDEX_KEYWORDS, True),
    (['INSTRUMENT_HOST_ID = "CO"', 'INSTRUMENT_ID = "ISS"'], None),
    (['INSTRUMENT_HOST_ID = "GO"', 'INSTRUMENT_ID = "VIMS"'], None),
    ([], None),
], ids=['vims', 'other-instrument', 'other-host', 'unidentified'])
def test_detection_in_index(keywords: list[str], expected: bool | None) -> None:
    label = Pds3Label(['PDS_VERSION_ID = PDS3', *keywords, 'END'])
    assert CassiniVIMS._detect_in_index(label) is expected


@pytest.mark.parametrize(('row', 'expected'), [
    ({'VOLUME_ID': 'COVIMS_0040'}, True),
    ({'VOLUME_ID': 'COISS_2001'}, False),
    ({}, False),
], ids=['vims', 'other-volume', 'empty'])
def test_detection_in_row(row: dict[str, str], expected: bool) -> None:
    assert CassiniVIMS._detect_in_row(row) is expected


@pytest.mark.parametrize(('keywords', 'volume_id'), [
    (_VIMS_INDEX_KEYWORDS, 'COVIMS_0040'),
    ([], 'COVIMS_0040'),
], ids=['from-label', 'from-row'])
def test_read_index_rows_accepts_vims_index(tmp_path: pathlib.Path, keywords: list[str],
                                            volume_id: str) -> None:
    path = write_index(tmp_path, keywords, volume_id)
    rows = CassiniVIMS._read_index_rows(path, supplement=None)
    assert [row['VOLUME_ID'] for row in rows] == [volume_id]


def test_read_index_rows_rejects_other_index(tmp_path: pathlib.Path) -> None:
    path = write_index(tmp_path, [], 'COISS_2001')
    with pytest.raises(WrongHostError, match='not a Cassini VIMS index'):
        CassiniVIMS._read_index_rows(path, supplement=None)

##########################################################################################
# select
##########################################################################################


@pytest.mark.parametrize(('select', 'detectors'), [
    ('VIS', 'VIS'),
    ('ir', 'IR'),
    (0, 'VIS'),
    (-1, 'IR'),
    (slice(1, None), ['IR']),
    (('IR',), 'IR'),
    ((slice(1, None),), ['IR']),
], ids=['VIS', 'ir', 'index', 'negative', 'slice', 'tuple-name', 'tuple-slice'])
def test_select(tmp_path: pathlib.Path, select: Any, detectors: str | list[str]) -> None:
    result = CassiniVIMS.from_file(write_pds3_cube(tmp_path), select=select)
    if isinstance(result, tuple):
        assert [obs.detector for obs in result] == detectors
    else:
        assert result.detector == detectors


def test_host_from_file_passes_select(tmp_path: pathlib.Path) -> None:
    obs = oops.Host.from_file(write_pds3_cube(tmp_path), select='IR', astrometry=True)
    assert obs.detector == 'IR'


def test_select_inactive_channel(tmp_path: pathlib.Path) -> None:
    path = write_pds3_cube(tmp_path, POWER_STATE_FLAG=['ON', 'OFF'])
    assert CassiniVIMS.from_file(path, select='VIS') is None


@pytest.mark.parametrize(('select', 'error', 'message'), [
    ('UV', ValueError, "unrecognized select name: 'UV'"),
    (2, IndexError, 'select index out of range: 2'),
    (('IR', 'VIS'), IndexError, "invalid select ('IR', 'VIS')"),
    (1.5, TypeError, 'invalid select: 1.5'),
], ids=['name', 'index', 'tuple', 'type'])
def test_select_errors(tmp_path: pathlib.Path, select: Any, error: type[Exception],
                       message: str) -> None:
    with pytest.raises(error, match=re.escape(message)):
        CassiniVIMS.from_file(write_pds3_cube(tmp_path), select=select)

##########################################################################################
# meshgrid_and_times()
##########################################################################################


def test_meshgrid_and_times(image_pair: Observations) -> None:
    (meshgrid, time) = CassiniVIMS.meshgrid_and_times(image_pair[1], oversample=2,
                                                      extend=0.)
    # A 4x2 FOV sampled twice per pixel, including both edges, in (v,u) order
    assert meshgrid.shape == (5, 9)
    assert time.shape == meshgrid.shape


def test_meshgrid_and_times_rejects_other_instruments(image_pair: Observations) -> None:
    obs = image_pair[1]
    obs.insert_subfield('instrument', 'ISS')
    with pytest.raises(WrongHostError, match='not a VIMS observation: ISS'):
        CassiniVIMS.meshgrid_and_times(obs)

##########################################################################################
# The rules that every host follows
##########################################################################################


def test_from_file_rejects_an_unknown_keyword(tmp_path: pathlib.Path) -> None:
    with pytest.raises(oops.OopsTypeError, match=r"CassiniVIMS\.from_file: 'selection'"):
        CassiniVIMS.from_file(write_pds3_cube(tmp_path), selection='IR')


@pytest.mark.parametrize(('select', 'error'), [('UV', oops.OopsValueError),
                                               (1.5, oops.OopsTypeError)])
def test_from_file_checks_select_before_reading(tmp_path: pathlib.Path, select: Any,
                                                error: type[Exception]) -> None:
    with pytest.raises(error, match='select'):
        CassiniVIMS.from_file(tmp_path / 'missing.lbl', select=select)


@pytest.mark.parametrize(('select', 'error'), [('BOGUS', oops.OopsValueError),
                                               (1.5, oops.OopsTypeError)])
def test_from_index_checks_select_when_called(select: Any,
                                              error: type[Exception]) -> None:
    with pytest.raises(error, match='select'):
        CassiniVIMS.from_index('INDEX.LBL', row_dicts=[], select=select)


def test_from_index_builds_each_row_when_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Any] = []
    make_observations = CassiniVIMS._make_observations

    def spy(label: dict[str, Any], filepath: Any, **kwargs: Any) -> Any:
        built.append(filepath)
        return make_observations(label, filepath, **kwargs)

    monkeypatch.setattr(CassiniVIMS, '_make_observations', spy)
    rows = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row(), index_row()])
    assert built == []
    next(rows)
    assert len(built) == 1


@pytest.mark.parametrize(('column', 'value', 'masked'), [
    ('PHASE_ANGLE', np.float64(-1.e32), True),
    ('INSTRUMENT_DATA_RATE', np.float64(-999.), False),
    ('TELEMETRY_FORMAT_ID', np.str_('UNK'), False),
], ids=['masked', 'fill', 'null-text'])
def test_from_index_leaves_out_missing_values(column: str, value: Any,
                                              masked: bool) -> None:
    row = index_row(**{column: value, column + '_mask': np.bool_(masked)})
    [(_, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert column not in ir.dict


def test_from_index_dict_is_json_without_masks() -> None:
    row = index_row(SWATH_WIDTH=np.int64(4), TARGET_NAME=np.str_('SATURN  '),
                    SWATH_WIDTH_mask=np.False_)
    [(_, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert json.loads(json.dumps(ir.dict, allow_nan=False)) == ir.dict
    assert not any(key.endswith('_mask') for key in ir.dict)
    assert ir.dict['TARGET_NAME'] == 'SATURN'


@pytest.mark.parametrize(('write', 'name'), [(write_pds3_cube, 'label_dict'),
                                             (write_isis_cube, 'header_dict')],
                         ids=['label', 'cube'])
def test_from_file_keeps_the_label_read(tmp_path: pathlib.Path,
                                        write: Callable[..., pathlib.Path],
                                        name: str) -> None:
    """The detached label is label_dict, and the cube's own ISIS label header_dict."""

    (_, ir) = CassiniVIMS.from_file(write(tmp_path, TARGET_NAME='UNK'))
    raw = getattr(ir, name)
    keywords = raw if name == 'label_dict' else raw['QUBE']
    assert keywords['TARGET_NAME'] == 'UNK'
    assert 'TARGET_NAME' not in ir.dict
    assert json.loads(json.dumps(raw, allow_nan=False)) == raw


def test_from_file_dict_is_json(image_pair: Observations) -> None:
    value = image_pair[1].dict
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_from_file_clock_counts(tmp_path: pathlib.Path) -> None:
    (_, ir) = CassiniVIMS.from_file(write_pds3_cube(
        tmp_path, SPACECRAFT_CLOCK_START_COUNT='1/1640775127.032',
        SPACECRAFT_CLOCK_STOP_COUNT='1/1640775211.002'))
    assert ir.spacecraft_clock_start_count == '1/1640775127.032'
    assert ir.spacecraft_clock_stop_count == '1/1640775211.002'


def test_from_file_time_is_not_from_the_clock(image_pair: Observations) -> None:
    assert image_pair[1].time_from_sclk is False


def test_from_index_backplane_count_gives_the_time() -> None:
    row = index_row(BACKPLANE_SCLK_START_COUNT='157766400.000')
    [(_, ir)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[row])
    assert ir.time_from_sclk is True


def test_from_index_backplane_count_error_is_a_spice_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any) -> None:
        raise ValueError('SPICE(INVALIDSCLKSTRING) -- bad clock count')

    monkeypatch.setattr(cspyce, 'scs2e', fail)
    rows = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row(
        BACKPLANE_SCLK_START_COUNT='157766400.000')])
    with pytest.raises(oops.OopsSpiceError, match=r'SPICE\(INVALIDSCLKSTRING\)$'):
        next(rows)


def test_from_index_refuses_a_volumes_own_index(tmp_path: pathlib.Path) -> None:
    with pytest.raises(oops.host.HostIndexError,
                       match='reads the Ring-Moon Systems Node metadata index'):
        CassiniVIMS.from_index(tmp_path / 'COVIMS_0001' / 'index' / 'index.lbl')


def remove_ir_pointing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SPICE frames of VIMS raise as cspyce does with no C kernel data."""

    def no_pointing(time: float, *args: Any, **kwargs: Any) -> None:
        raise OSError('SPICE(CKINSUFFDATA) -- insufficient C kernel data')

    CassiniVIMS._define_frames()
    frame = oops.Frame.as_frame('CASSINI_VIMS_V')
    monkeypatch.setattr(frame, 'transform_at_time', no_pointing)


def test_from_file_without_pointing(tmp_path: pathlib.Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    remove_ir_pointing(monkeypatch)
    with pytest.raises(oops.OopsSpiceError,
                       match=r'^no Cassini VIMS CASSINI_VIMS_V pointing at .* for '
                             r'.*v0000000001_1\.lbl: SPICE\(CKINSUFFDATA\)$'):
        CassiniVIMS.from_file(write_pds3_cube(tmp_path))


def test_from_index_frame_without_pointing_raises_when_used(
        monkeypatch: pytest.MonkeyPatch) -> None:
    remove_ir_pointing(monkeypatch)
    [(vis, _)] = CassiniVIMS.from_index('INDEX.LBL', row_dicts=[index_row()])
    with pytest.raises(oops.OopsSpiceError, match=r'SPICE\(CKINSUFFDATA\)$'):
        vis.frame.wrt(oops.Frame.J2000).transform_at_time(vis.midtime)


def test_from_file_parsed_label_without_a_path_has_no_file_subfields(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_cube(tmp_path), method='fast')
    label.filepath = None
    (_, ir) = CassiniVIMS.from_file(label, astrometry=True)
    assert {'filepath', 'basename', 'abspath', 'image_url'}.isdisjoint(ir.subfields)


def test_from_file_parsed_label_without_a_path_cannot_be_read_in_full(
        tmp_path: pathlib.Path) -> None:
    label = Pds3Label(write_pds3_cube(tmp_path), method='fast')
    label.filepath = None
    with pytest.raises(HostFormatError, match='no file path is known'):
        CassiniVIMS.from_file(label)


def test_from_file_reads_a_comment_that_spans_two_lines(tmp_path: pathlib.Path) -> None:
    """A comment that opens on a value's line and closes on the next is repaired."""

    path = write_pds3_cube(tmp_path)
    path.write_bytes(path.read_bytes().replace(
        b'PACKING_FLAG = "OFF"', b'PACKING_FLAG = OFF/* Unpacked from\r\n    QUB*/'))
    (_, ir) = CassiniVIMS.from_file(path, astrometry=True)
    assert ir.dict['PACKING_FLAG'] == 'OFF'

##########################################################################################
