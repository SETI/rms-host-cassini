##########################################################################################
# tests/host_cassini/test_cassini.py
##########################################################################################
"""Tests for the mission-level kernel management in host_cassini._Cassini."""

from typing import Any

import cspyce
import julian
import numpy as np
import pytest
import spicedb
from conftest import (FakeKernel, FakeSpice, FakeSpiceDB, kernel_for_months,
                      tdb_in_month)
from oops.body import Body

from host_cassini._oops import _Cassini

##########################################################################################
# Mission constants
##########################################################################################


def test_mission_spans_240_equal_months() -> None:
    span = _Cassini.DTDB * _Cassini.MONTHS
    assert span == pytest.approx(_Cassini.TDB1 - _Cassini.TDB0)


def test_mission_start_matches_start_time() -> None:
    start = _Cassini.TDB0
    assert start == pytest.approx(julian.tdb_from_iso('1997-10-01'))


def test_tour_falls_at_end_of_2002() -> None:
    assert julian.iso_from_tai(julian.tai_from_tdb(_Cassini.TOUR))[:10] == '2002-12-31'

##########################################################################################
# initialize_kernels()
##########################################################################################


def _lists() -> np.ndarray:
    return np.empty(_Cassini.MONTHS, dtype='object')


def _months_holding(lists: np.ndarray, kernel: FakeKernel) -> list[int]:
    return [m for m in range(_Cassini.MONTHS) if kernel in lists[m]]


def test_initialize_kernels_with_no_kernels_gives_empty_lists() -> None:
    lists = _lists()
    _Cassini.initialize_kernels([], lists)
    assert all(entry == [] for entry in lists)


@pytest.mark.parametrize(('start', 'stop', 'expected'), [
    (10.25, 10.5, [10]),
    (10.25, 12.5, [10, 11, 12]),
    (10.0 + 1000. / _Cassini.DTDB, 10.5, [9, 10]),
    (10.5, 11.0 - 1000. / _Cassini.DTDB, [10, 11]),
    (-5.0, 1.5, [0, 1]),
    (238.5, 250.0, [238, 239]),
], ids=['within-month', 'spans-months', 'slop-reaches-back', 'slop-reaches-forward',
        'starts-before-mission', 'ends-after-mission'])
def test_initialize_kernels_assigns_months(start: float, stop: float,
                                           expected: list[int]) -> None:
    # A kernel is listed from the month its start (less SLOP) falls in through the month
    # its stop (plus SLOP) falls in, clipped to the mission.
    kernel = kernel_for_months('k.bc', start, stop)
    lists = _lists()
    _Cassini.initialize_kernels([kernel], lists)
    assert _months_holding(lists, kernel) == expected


def test_initialize_kernels_keeps_kernel_order_within_a_month() -> None:
    first = kernel_for_months('first.bc', 10.1, 10.2)
    second = kernel_for_months('second.bc', 10.3, 10.4)
    lists = _lists()
    _Cassini.initialize_kernels([first, second], lists)
    assert lists[10] == [first, second]


def test_initialize_kernels_replaces_existing_entries() -> None:
    lists = _lists()
    lists[50] = [FakeKernel('stale.bc')]
    _Cassini.initialize_kernels([], lists)
    assert lists[50] == []

##########################################################################################
# load_kernels()
##########################################################################################


def _month_lists(contents: dict[int, list[FakeKernel]]) -> np.ndarray:
    lists = _lists()
    for m in range(_Cassini.MONTHS):
        lists[m] = contents.get(m, [])
    return lists


def test_load_kernels_loads_only_the_month() -> None:
    lists = _month_lists({9: [FakeKernel('a.bc')], 10: [FakeKernel('b.bc')],
                          11: [FakeKernel('c.bc')]})
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5), loaded, lists, {})
    assert list(np.nonzero(loaded)[0]) == [10]


def test_load_kernels_furnishes_only_the_month(fake_spicedb: FakeSpiceDB) -> None:
    lists = _month_lists({9: [FakeKernel('a.bc')], 10: [FakeKernel('b.bc')],
                          11: [FakeKernel('c.bc')]})
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5),
                          np.zeros(_Cassini.MONTHS, dtype='bool'), lists, {})
    assert fake_spicedb.furnished() == ['b.bc']


def test_load_kernels_furnishes_each_kernel_once(fake_spicedb: FakeSpiceDB) -> None:
    shared = FakeKernel('shared.bc')
    lists = _month_lists({10: [shared, FakeKernel('a.bc')], 11: [shared]})
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(11.5),
                          np.zeros(_Cassini.MONTHS, dtype='bool'), lists, {})
    assert fake_spicedb.furnished() == ['shared.bc', 'a.bc']


def test_load_kernels_records_furnished_kernels() -> None:
    kernel = FakeKernel('a.bc')
    kernel_dict: dict[str, Any] = {}
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5),
                          np.zeros(_Cassini.MONTHS, dtype='bool'),
                          _month_lists({10: [kernel]}), kernel_dict)
    assert kernel_dict == {'a.bc': kernel}


def test_load_kernels_skips_kernels_already_furnished(fake_spicedb: FakeSpiceDB) -> None:
    kernel = FakeKernel('a.bc')
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5),
                          np.zeros(_Cassini.MONTHS, dtype='bool'),
                          _month_lists({10: [kernel]}), {'a.bc': kernel})
    assert fake_spicedb.furnished() == []


def test_load_kernels_skips_months_already_loaded(fake_spicedb: FakeSpiceDB) -> None:
    lists = _month_lists({10: [FakeKernel('a.bc')], 11: [FakeKernel('b.bc')]})
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    loaded[10] = True
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(11.5), loaded, lists, {})
    assert fake_spicedb.furnished() == ['b.bc']


def test_load_kernels_marks_a_month_with_no_kernels_loaded() -> None:
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5), loaded,
                          _month_lists({}), {})
    assert loaded[10]


def test_load_kernels_does_not_repeat_a_loaded_month(fake_spicedb: FakeSpiceDB) -> None:
    lists = _month_lists({10: [FakeKernel('a.bc')]})
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    kernel_dict: dict[str, Any] = {}
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5), loaded, lists,
                          kernel_dict)
    kernel_dict.clear()
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(10.5), loaded, lists,
                          kernel_dict)
    assert fake_spicedb.furnished() == ['a.bc']


def test_load_kernels_covers_a_time_range() -> None:
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    _Cassini.load_kernels(tdb_in_month(10.5), tdb_in_month(13.5), loaded,
                          _month_lists({}), {})
    assert list(np.nonzero(loaded)[0]) == [10, 11, 12, 13]


@pytest.mark.parametrize('month', [-10.0, 250.0], ids=['before-mission', 'after-mission'])
def test_load_kernels_ignores_times_outside_mission(month: float) -> None:
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    _Cassini.load_kernels(tdb_in_month(month), tdb_in_month(month), loaded,
                          _month_lists({}), {})
    assert not loaded.any()


def test_load_kernels_clips_to_last_month() -> None:
    loaded = np.zeros(_Cassini.MONTHS, dtype='bool')
    _Cassini.load_kernels(tdb_in_month(239.5), tdb_in_month(239.5), loaded,
                          _month_lists({}), {})
    assert list(np.nonzero(loaded)[0]) == [239]


@pytest.mark.parametrize(('method', 'args', 'kind'), [
    ('load_ck', (tdb_in_month(20.5),), 'CK'),
    ('load_cks', (tdb_in_month(20.5), tdb_in_month(20.5)), 'CK'),
    ('load_spk', (tdb_in_month(20.5),), 'SPK'),
    ('load_spks', (tdb_in_month(20.5), tdb_in_month(20.5)), 'SPK'),
])
def test_load_methods_use_their_own_kernel_tables(method: str, args: tuple[float, ...],
                                                  kind: str) -> None:
    other = 'SPK' if kind == 'CK' else 'CK'
    _Cassini.initialize_kernels([kernel_for_months('k', 20.1, 20.2)],
                                getattr(_Cassini, kind + '_LIST'))
    getattr(_Cassini, method)(*args)
    assert list(getattr(_Cassini, kind + '_DICT')) == ['k']
    assert not getattr(_Cassini, other + '_LOADED').any()

##########################################################################################
# initialize()
##########################################################################################


def test_initialize_defines_the_solar_system(fake_spice: FakeSpice) -> None:
    _Cassini.initialize(planets=[6], asof='2020-01-01', mst_pck=False, irregulars=False)
    assert fake_spice.solar_system == [(('1997-10-01', '2017-10-01'),
                                        {'asof': '2020-01-01', 'planets': [6],
                                         'mst_pck': False, 'irregulars': False})]


def test_initialize_loads_leap_seconds(fake_spice: FakeSpice) -> None:
    _Cassini.initialize()
    assert fake_spice.leap_seconds == 1


def test_initialize_defines_the_cassini_path(fake_spice: FakeSpice) -> None:
    _Cassini.initialize()
    assert fake_spice.spice_paths == [('CASSINI', 'SATURN')]


@pytest.mark.parametrize(('kind', 'select'), [('spk', 'select_spk'), ('ck', 'select_ck')])
@pytest.mark.parametrize('option', ['reconstructed', 'Predicted'])
def test_initialize_selects_kernel_set(fake_spicedb: FakeSpiceDB, kind: str, select: str,
                                       option: str) -> None:
    _Cassini.initialize(asof='2020-01-01', **{kind: option})
    assert fake_spicedb.named(select) == [
        ((-82,), {'name': f'CAS-{kind.upper()}-{option.upper()}',
                  'time': ('1997-10-01', '2017-10-01'), 'asof': '2020-01-01'})]


def test_initialize_distributes_selected_kernels(fake_spicedb: FakeSpiceDB) -> None:
    spk = kernel_for_months('s.bsp', 30.2, 30.4)
    ck = kernel_for_months('c.bc', 40.2, 40.4)
    fake_spicedb.spk_kernels = [spk]
    fake_spicedb.ck_kernels = [ck]
    _Cassini.initialize()
    assert _Cassini.SPK_LIST[30] == [spk]
    assert _Cassini.CK_LIST[40] == [ck]


@pytest.mark.parametrize(('kind', 'select'), [('spk', 'select_spk'), ('ck', 'select_ck')])
def test_initialize_none_skips_kernel_selection(fake_spicedb: FakeSpiceDB, kind: str,
                                                select: str) -> None:
    _Cassini.initialize(**{kind: 'none'})
    assert fake_spicedb.named(select) == []


@pytest.mark.parametrize('kind', ['SPK', 'CK'])
def test_initialize_none_marks_every_month_loaded(kind: str) -> None:
    _Cassini.initialize(**{kind.lower(): 'NONE'})
    assert getattr(_Cassini, kind + '_LOADED').all()
    assert all(entry == [] for entry in getattr(_Cassini, kind + '_LIST'))


@pytest.mark.parametrize(('ck', 'gapfill', 'expected'), [
    ('reconstructed', True, [((-82,), {'name': 'CAS-CK-GAPFILL'})]),
    ('reconstructed', False, []),
    ('predicted', True, []),
    ('none', True, []),
])
def test_initialize_gapfill(fake_spicedb: FakeSpiceDB, ck: str, gapfill: bool,
                            expected: list[Any]) -> None:
    _Cassini.initialize(ck=ck, gapfill=gapfill)
    assert fake_spicedb.named('furnish_ck') == expected


def test_initialize_opens_and_closes_the_database(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.initialize()
    names = fake_spicedb.call_names()
    assert names[0] == 'open_db'
    assert names[-1] == 'close_db'


def test_initialize_sets_initialized() -> None:
    _Cassini.initialize()
    assert _Cassini.initialized


def test_initialize_ignores_later_calls(fake_spicedb: FakeSpiceDB,
                                        fake_spice: FakeSpice) -> None:
    _Cassini.initialize()
    count = len(fake_spicedb.calls)
    _Cassini.initialize(spk='predicted')
    assert len(fake_spicedb.calls) == count
    assert fake_spice.leap_seconds == 1

##########################################################################################
# reset()
##########################################################################################


def test_reset_restores_initial_state() -> None:
    _Cassini.initialize(spk='none', ck='none')
    _Cassini.load_instruments()
    _Cassini.CK_DICT['x'] = FakeKernel('x')
    _Cassini.reset()
    assert not _Cassini.initialized
    assert _Cassini.loaded_instruments == []
    assert not _Cassini.CK_LOADED.any()
    assert not _Cassini.SPK_LOADED.any()
    assert _Cassini.CK_DICT == {}

##########################################################################################
# load_instruments()
##########################################################################################


def _furnished_instruments(fake_spicedb: FakeSpiceDB) -> list[list[str]]:
    return [kwargs['inst'] for (_, kwargs) in fake_spicedb.named('furnish_inst')]


def test_load_instruments_loads_defaults_first(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments()
    assert _furnished_instruments(fake_spicedb) == [['ISS', 'VIMS', 'CIRS', 'UVIS']]


def test_load_instruments_uses_cassini_id(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments()
    assert fake_spicedb.named('furnish_inst')[0][0] == (-82,)


def test_load_instruments_adds_requested_to_defaults(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments(['RSS'])
    assert _furnished_instruments(fake_spicedb) == [['RSS', 'ISS', 'VIMS', 'CIRS',
                                                     'UVIS']]


def test_load_instruments_records_loaded_instruments() -> None:
    _Cassini.load_instruments(['RSS'])
    assert _Cassini.loaded_instruments == ['RSS', 'ISS', 'VIMS', 'CIRS', 'UVIS']


def test_load_instruments_second_call_does_nothing(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments()
    count = len(fake_spicedb.calls)
    _Cassini.load_instruments()
    assert len(fake_spicedb.calls) == count


def test_load_instruments_loads_only_new_instruments(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments()
    _Cassini.load_instruments(['ISS', 'RSS', 'RSS'])
    assert _furnished_instruments(fake_spicedb)[-1] == ['RSS']


def test_load_instruments_leaves_caller_list_unchanged() -> None:
    instruments = ['RSS']
    _Cassini.load_instruments(instruments)
    assert instruments == ['RSS']


def test_load_instruments_normalizes_asof(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments(asof='2010-02-03')
    assert fake_spicedb.named('furnish_inst')[0][1]['asof'] == '2010-02-03T00:00:00'


def test_load_instruments_closes_the_database(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.load_instruments()
    assert fake_spicedb.call_names() == ['open_db', 'furnish_inst', 'close_db']

##########################################################################################
# spice_instrument_kernel() and spice_frames_kernel()
##########################################################################################


def test_spice_instrument_kernel_returns_dict_and_name(fake_spicedb: FakeSpiceDB) -> None:
    assert _Cassini.spice_instrument_kernel('ISS') == (fake_spicedb.kernel_dict,
                                                       'cas_iss_v10.ti')


def test_spice_instrument_kernel_selects_ik(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.spice_instrument_kernel('ISS', asof='2010-02-03')
    assert fake_spicedb.named('select_inst') == [
        ((-82,), {'types': 'IK', 'inst': 'ISS', 'asof': '2010-02-03T00:00:00'})]


@pytest.mark.parametrize('method', ['spice_instrument_kernel', 'spice_frames_kernel'])
def test_text_kernels_are_furnished_fast(fake_spicedb: FakeSpiceDB, method: str) -> None:
    args = ('ISS',) if method == 'spice_instrument_kernel' else ()
    getattr(_Cassini, method)(*args)
    assert fake_spicedb.named('furnish_kernels') == [((fake_spicedb.inst_kernels,),
                                                      {'fast': True})]


def test_spice_frames_kernel_returns_dict_and_names(fake_spicedb: FakeSpiceDB) -> None:
    fake_spicedb.inst_kernels = [FakeKernel('cas_v42.tf'), FakeKernel('cas_v43.tf')]
    assert _Cassini.spice_frames_kernel() == (fake_spicedb.kernel_dict,
                                              ['cas_v42.tf', 'cas_v43.tf'])


def test_spice_frames_kernel_selects_fk(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.spice_frames_kernel(asof='2010-02-03')
    assert fake_spicedb.named('select_inst') == [
        ((-82,), {'types': 'FK', 'asof': '2010-02-03T00:00:00'})]

##########################################################################################
# used_kernels()
##########################################################################################


_JUPITER_TIME = (_Cassini.TOUR - 100., _Cassini.TOUR - 99.)
_SATURN_TIME = (_Cassini.TOUR, _Cassini.TOUR + 1.)
_PLANETS = [1, 199, 2, 299, 3, 399, 4, 499, 5, 599, 6, 699, 7, 799, 8, 899]


@pytest.fixture
def moons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Body, 'JUPITER_MOONS_LOADED', [501, 502])
    monkeypatch.setattr(Body, 'SATURN_MOONS_LOADED', [601, 602])


@pytest.mark.usefixtures('moons')
@pytest.mark.parametrize(('time', 'all_planets', 'bodies'), [
    (_JUPITER_TIME, False, [5, 599, 501, 502]),
    (_SATURN_TIME, False, [6, 699, 601, 602]),
    (_JUPITER_TIME, True, [*_PLANETS, 501, 502]),
    (_SATURN_TIME, True, [*_PLANETS, 601, 602]),
], ids=['jupiter', 'saturn', 'jupiter-all-planets', 'saturn-all-planets'])
def test_used_kernels_bodies(fake_spicedb: FakeSpiceDB, time: tuple[float, float],
                             all_planets: bool, bodies: list[int]) -> None:
    _Cassini.used_kernels(time, 'iss', return_all_planets=all_planets)
    assert fake_spicedb.named('used_basenames')[0][1]['bodies'] == bodies


@pytest.mark.usefixtures('moons')
def test_used_kernels_query(fake_spicedb: FakeSpiceDB) -> None:
    assert _Cassini.used_kernels(_SATURN_TIME, 'iss') == fake_spicedb.used
    assert fake_spicedb.named('used_basenames') == [
        ((), {'types': None, 'time': _SATURN_TIME, 'inst': 'iss', 'sc': -82,
              'bodies': [6, 699, 601, 602]})]


def test_used_kernels_can_exclude_cks(fake_spicedb: FakeSpiceDB) -> None:
    _Cassini.used_kernels(_SATURN_TIME, 'iss', ck=False)
    types = fake_spicedb.named('used_basenames')[0][1]['types']
    assert types == [t for t in spicedb.KERNEL_TYPE_SORT_ORDER if t != 'CK']


@pytest.mark.usefixtures('moons')
@pytest.mark.parametrize('all_planets', [False, True])
def test_used_kernels_leaves_moon_lists_unchanged(all_planets: bool) -> None:
    _Cassini.used_kernels(_SATURN_TIME, 'iss', return_all_planets=all_planets)
    assert Body.SATURN_MOONS_LOADED == [601, 602]


def test_no_kernels_reach_cspyce() -> None:
    # The autouse fixtures replace cspyce.furnsh; confirm the guard is in place.
    with pytest.raises(AssertionError, match='tried to furnish a SPICE kernel'):
        cspyce.furnsh('x.bsp')

##########################################################################################
