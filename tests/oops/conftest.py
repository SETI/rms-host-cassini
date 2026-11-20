##########################################################################################
# tests/oops/conftest.py
##########################################################################################
"""Shared fixtures that let the Cassini host run without SPICE kernels or spicedb.

Every spicedb function the package calls is replaced by :class:`FakeSpiceDB`, and the few
oops and cspyce calls that would read SPICE kernels directly are replaced by
kernel-free equivalents. Everything else, including oops itself, runs for real.
"""

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

import cspyce
import julian
import oops
import pytest
import spicedb
from filecache import FileCache, file_cache_path
from oops.body import Body

from host_cassini.oops._cassini import _Cassini
from host_cassini.oops.iss      import CassiniISS
from host_cassini.oops.uvis     import CassiniUVIS
from host_cassini.oops.vims     import CassiniVIMS

# Nominal Cassini ISS camera geometry: 1024x1024 pixels; NAC 0.35 deg and WAC 3.5 deg
# square fields of view.
NAC_HALF_FOV_DEG = 0.175
WAC_HALF_FOV_DEG = 1.75
PIXELS = 1024


_ISO_TIME = re.compile(r'\d{4}-\d{2}-\d{2}T[\d:.]+')


def pds3_value(value: Any) -> str:
    """A Python value formatted for a PDS3 label; ISO times and numbers stay unquoted."""

    if isinstance(value, (list, tuple)):
        return '(' + ', '.join(pds3_value(item) for item in value) + ')'
    if isinstance(value, str) and not _ISO_TIME.fullmatch(value):
        return f'"{value}"'
    return str(value)


def pds3_lines(keywords: dict[str, Any], indent: str = '') -> list[str]:
    """PDS3 label lines assigning each keyword its formatted value."""

    return [f'{indent}{key} = {pds3_value(value)}' for (key, value) in keywords.items()]


def write_table(directory: pathlib.Path, stem: str,
                columns: dict[str, list[Any]]) -> pathlib.Path:
    """Write an ASCII table and its detached PDS3 label; return the label's path.

    Parameters:
        directory: The directory to write into.
        stem: The file name of the table and label, without the extension.
        columns: The values of each column, by name: str values make a CHARACTER
            column, int an ASCII_INTEGER column, and float an ASCII_REAL column.
    """

    formats = {str: ('CHARACTER', '{!s}'), int: ('ASCII_INTEGER', '{:d}'),
               float: ('ASCII_REAL', '{:.4f}')}
    names = list(columns)
    texts = {name: [formats[type(value)][1].format(value) for value in columns[name]]
             for name in names}
    widths = [max(len(text) for text in texts[name]) for name in names]
    nrows = len(columns[names[0]])
    row_bytes = sum(widths) + len(widths) - 1 + 2
    lines = [','.join(texts[name][i].ljust(width)
                      for (name, width) in zip(names, widths, strict=True)) + '\r\n'
             for i in range(nrows)]
    (directory / f'{stem}.tab').write_text(''.join(lines), newline='')

    label = ['PDS_VERSION_ID = PDS3', 'RECORD_TYPE = FIXED_LENGTH',
             f'RECORD_BYTES = {row_bytes}', f'FILE_RECORDS = {nrows}',
             f'^INDEX_TABLE = "{stem}.tab"', 'OBJECT = INDEX_TABLE',
             '  INTERCHANGE_FORMAT = ASCII', f'  ROWS = {nrows}',
             f'  ROW_BYTES = {row_bytes}', f'  COLUMNS = {len(names)}']
    start = 1
    for (name, width) in zip(names, widths, strict=True):
        label += ['  OBJECT = COLUMN', f'    NAME = {name}',
                  f'    DATA_TYPE = {formats[type(columns[name][0])][0]}',
                  f'    START_BYTE = {start}', f'    BYTES = {width}',
                  '  END_OBJECT = COLUMN']
        start += width + 1
    label += ['END_OBJECT = INDEX_TABLE', 'END', '']
    path = directory / f'{stem}.lbl'
    path.write_text('\r\n'.join(label), newline='')
    return path


def claim_more_rows(label: pathlib.Path, rows: int, claimed: int) -> None:
    """Rewrite a table label to say that its table holds more rows than it does.

    Parameters:
        label: The label's path.
        rows: The number of rows that the label gives now.
        claimed: The number of rows that it is to give.
    """

    text = label.read_bytes()
    for keyword in (b'ROWS', b'FILE_RECORDS'):
        old = keyword + b' = %d\r\n' % rows
        assert old in text
        text = text.replace(old, keyword + b' = %d\r\n' % claimed)
    label.write_bytes(text)


def iso_from_tdb(tdb: float) -> str:
    """The ISO date-time, to the millisecond, of a time in seconds TDB."""

    return str(julian.iso_from_tai(julian.tai_from_tdb(tdb), digits=3))


def tdb_in_month(month: float) -> float:
    """The time in seconds TDB at a fractional "month" of the Cassini mission."""

    return float(_Cassini.TDB0 + month * _Cassini.DTDB)


@dataclass
class FakeKernel:
    """A stand-in for a spicedb KernelInfo object."""

    filespec: str
    start_time: str = '2000-01-01T00:00:00.000'
    stop_time: str = '2000-01-01T00:00:00.000'


def kernel_for_months(name: str, start_month: float, stop_month: float) -> FakeKernel:
    """A FakeKernel covering the given fractional months of the mission."""

    return FakeKernel(name, iso_from_tdb(tdb_in_month(start_month)),
                      iso_from_tdb(tdb_in_month(stop_month)))


def ik_dict(*, units: str = 'DEGREES') -> dict[str, Any]:
    """The parts of the Cassini ISS instrument kernel that the host reads."""

    def camera(half_fov: float) -> dict[str, Any]:
        return {'PIXEL_LINES': PIXELS, 'PIXEL_SAMPLES': PIXELS,
                'FOV_REF_ANGLE': half_fov, 'FOV_CROSS_ANGLE': half_fov,
                'FOV_ANGLE_UNITS': units}

    return {'INS': {'CASSINI_ISS_NAC': camera(NAC_HALF_FOV_DEG),
                    'CASSINI_ISS_WAC': camera(WAC_HALF_FOV_DEG)}}


@dataclass
class FakeSpiceDB:
    """Records every spicedb call and returns configurable results.

    Attributes:
        calls (list[tuple[str, tuple, dict]]): Each call as (name, args, kwargs).
        spk_kernels (list[FakeKernel]): Returned by `select_spk`.
        ck_kernels (dict[str, list[FakeKernel]]): Returned by `select_ck`, by kernel set
            name; a set name that is absent gives an empty list.
        inst_kernels (list[FakeKernel]): Returned by `select_inst`.
        kernel_dict (dict): Returned by `as_dict`.
        used (list[str]): Returned by `used_basenames`.
    """

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    spk_kernels: list[FakeKernel] = field(default_factory=list)
    ck_kernels: dict[str, list[FakeKernel]] = field(default_factory=dict)
    inst_kernels: list[FakeKernel] = field(
        default_factory=lambda: [FakeKernel('cas_iss_v10.ti')])
    kernel_dict: dict[str, Any] = field(default_factory=ik_dict)
    used: list[str] = field(default_factory=lambda: ['cas_v43.tf', 'cas_iss_v10.ti'])

    def _record(self, func: str, /, *args: Any, **kwargs: Any) -> None:
        self.calls.append((func, args, kwargs))

    def named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        """The (args, kwargs) of every recorded call to the named function."""

        return [(args, kwargs) for (n, args, kwargs) in self.calls if n == name]

    def call_names(self) -> list[str]:
        """The names of the recorded calls, in order."""

        return [name for (name, _, _) in self.calls]

    def furnished(self) -> list[str]:
        """The filespecs passed to `furnish_kernels`, in order."""

        return [kernel.filespec for (args, _) in self.named('furnish_kernels')
                for kernel in args[0]]

    def open_db(self, *args: Any, **kwargs: Any) -> None:
        self._record('open_db', *args, **kwargs)

    def close_db(self, *args: Any, **kwargs: Any) -> None:
        self._record('close_db', *args, **kwargs)

    def select_spk(self, *args: Any, **kwargs: Any) -> list[FakeKernel]:
        self._record('select_spk', *args, **kwargs)
        return self.spk_kernels

    def select_ck(self, *args: Any, **kwargs: Any) -> list[FakeKernel]:
        self._record('select_ck', *args, **kwargs)
        return self.ck_kernels.get(kwargs['name'], [])

    def select_inst(self, *args: Any, **kwargs: Any) -> list[FakeKernel]:
        self._record('select_inst', *args, **kwargs)
        return self.inst_kernels

    def furnish_ck(self, *args: Any, **kwargs: Any) -> list[str]:
        self._record('furnish_ck', *args, **kwargs)
        return []

    def furnish_inst(self, *args: Any, **kwargs: Any) -> list[str]:
        self._record('furnish_inst', *args, **kwargs)
        return []

    def furnish_kernels(self, *args: Any, **kwargs: Any) -> list[str]:
        self._record('furnish_kernels', *args, **kwargs)
        return []

    def as_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._record('as_dict', *args, **kwargs)
        return self.kernel_dict

    def as_names(self, kernels: list[FakeKernel]) -> list[str]:
        self._record('as_names', kernels)
        return [kernel.filespec for kernel in kernels]

    def used_basenames(self, *args: Any, **kwargs: Any) -> list[str]:
        self._record('used_basenames', *args, **kwargs)
        return self.used


_SPICEDB_FUNCTIONS = ('open_db', 'close_db', 'select_spk', 'select_ck', 'select_inst',
                      'furnish_ck', 'furnish_inst', 'furnish_kernels', 'as_dict',
                      'as_names', 'used_basenames')


@dataclass
class FakeSpice:
    """Records the oops calls that would otherwise read SPICE kernels.

    Attributes:
        leap_seconds (int): The number of calls to `oops.spice.load_leap_seconds`.
        solar_system (list[tuple[tuple, dict]]): The (args, kwargs) of each call to
            `Body.define_solar_system`.
        spice_paths (list[tuple[str, str]]): The (path ID, origin) of each SpicePath.
        spice_frames (list[str]): The frame ID given to each SpiceFrame.
    """

    leap_seconds: int = 0
    solar_system: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list)
    spice_paths: list[tuple[str, str]] = field(default_factory=list)
    spice_frames: list[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def fake_spicedb(monkeypatch: pytest.MonkeyPatch) -> FakeSpiceDB:
    """Replace every spicedb function the package calls with a recording fake."""

    fake = FakeSpiceDB()
    for name in _SPICEDB_FUNCTIONS:
        monkeypatch.setattr(spicedb, name, getattr(fake, name))
    return fake


@pytest.fixture(autouse=True)
def fake_spice(monkeypatch: pytest.MonkeyPatch) -> FakeSpice:
    """Keep SPICE kernels out of every test and isolate the oops registries.

    Kernel-reading calls are replaced: time conversion uses julian, a SpicePath becomes
    a FixedPath at the origin, and a SpiceFrame becomes an identity Cmatrix on J2000.
    Any attempt to furnish a kernel through cspyce fails the test. The Path and Frame
    registries, caches, and waypoint and wayframe pools are copied, and the Cmatrix
    wayframes start empty, so that paths and frames made by one test never reach
    another.
    """

    fake = FakeSpice()

    def load_leap_seconds() -> None:
        fake.leap_seconds += 1

    def define_solar_system(*args: Any, **kwargs: Any) -> None:
        fake.solar_system.append((args, kwargs))

    def spice_path(spice_id: str, origin: str) -> oops.Path:
        fake.spice_paths.append((spice_id, origin))
        return oops.path.FixedPath((0., 0., 0.), oops.Path.SSB, oops.Frame.J2000,
                                   path_id=spice_id)

    def spice_frame(spice_id: str, *, frame_id: str) -> oops.Frame:
        fake.spice_frames.append(spice_id)
        return oops.frame.Cmatrix(oops.Matrix3.IDENTITY, oops.Frame.J2000,
                                  frame_id=frame_id)

    def furnsh(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'a test tried to furnish a SPICE kernel: {args}')

    def scs2e(code: int, sclk: str) -> float:
        return float(sclk)

    # The inverse of scs2e, spelled as cspyce spells a count, partition first
    def sce2s(code: int, et: float) -> str:
        return f'1/{et:.3f}'

    monkeypatch.setattr(cspyce, 'str2et', julian.tdb_from_iso)
    monkeypatch.setattr(cspyce, 'scs2e', scs2e)
    monkeypatch.setattr(cspyce, 'sce2s', sce2s)
    monkeypatch.setattr(cspyce, 'furnsh', furnsh)
    monkeypatch.setattr(oops.spice, 'load_leap_seconds', load_leap_seconds)
    monkeypatch.setattr(Body, 'define_solar_system', staticmethod(define_solar_system))
    monkeypatch.setattr(oops.path, 'SpicePath', spice_path)
    monkeypatch.setattr(oops.frame, 'SpiceFrame', spice_frame)

    monkeypatch.setattr(oops.Path, '_PATH_REGISTRY', dict(oops.Path._PATH_REGISTRY))
    monkeypatch.setattr(oops.Path, '_PATH_CACHE', dict(oops.Path._PATH_CACHE))
    monkeypatch.setattr(oops.Frame, '_FRAME_REGISTRY', dict(oops.Frame._FRAME_REGISTRY))
    monkeypatch.setattr(oops.Frame, '_FRAME_CACHE', dict(oops.Frame._FRAME_CACHE))

    # A path or frame defined as an earlier one was shares its waypoint or wayframe, and
    # so its ID, so each class's pool is copied too; otherwise a fixed path made by one
    # test could become the path of the next, every fake SPICE path being fixed at the
    # origin. These copies stay until oops offers one helper that isolates all of its
    # global state for a test, SETI/rms-oops#250.
    for path_class in oops.Path._PATH_SUBCLASSES:
        if '_WAYPOINTS' in vars(path_class):
            monkeypatch.setattr(path_class, '_WAYPOINTS', dict(path_class._WAYPOINTS))
    for frame_class in oops.Frame._FRAME_SUBCLASSES:
        for pool in ('_WAYFRAMES', '_FOR_NAME', '_FRAME_LOOKUP'):
            if pool in vars(frame_class):
                monkeypatch.setattr(frame_class, pool, dict(getattr(frame_class, pool)))

    # Every fake SPICE frame is the same Cmatrix, so without a pool of its own a test's
    # frame would share the wayframe, and the evaluation, of one an earlier test made
    monkeypatch.setattr(oops.frame.Cmatrix, '_WAYFRAMES', {})
    monkeypatch.setattr(Body, 'BODY_REGISTRY', dict(Body.BODY_REGISTRY))
    return fake


@pytest.fixture(autouse=True)
def private_filecache(tmp_path: pathlib.Path,
                      monkeypatch: pytest.MonkeyPatch) -> FileCache:
    """Give every test its own file cache, so no test sees what another cached.

    FCPath objects that name no FileCache of their own share one default cache, normally
    the global one under the system's temporary directory.
    """

    cache = FileCache(cache_root=tmp_path / 'filecache')
    monkeypatch.setattr(file_cache_path, '_DEFAULT_FILECACHE', cache)
    return cache


@pytest.fixture(autouse=True)
def reset_host() -> Any:
    """Start and end every test with the Cassini and instrument class states reset."""

    hosts = (CassiniISS, CassiniUVIS, CassiniVIMS)
    for host in hosts:
        host._reset()
    yield
    for host in hosts:
        host._reset()


@pytest.fixture
def saturn_path() -> oops.Path:
    """A fixed path and a body registered as SATURN, which a TrackerFrame can track."""

    path = oops.path.FixedPath((1.e9, 0., 0.), oops.Path.SSB, oops.Frame.J2000,
                               path_id='SATURN')
    Body('SATURN', path, oops.Frame.J2000)  # type: ignore[no-untyped-call]  # unannotated
    return path


@pytest.fixture
def initialized_iss(fake_spicedb: FakeSpiceDB) -> FakeSpiceDB:
    """CassiniISS initialized with default options against the fake spicedb."""

    CassiniISS.initialize()
    return fake_spicedb


# Time-shift letters, and the seconds after the observation's midtime at which its frame
# is evaluated for a shift of 30 s: "T" moves the observation's times, and "F" evaluates
# the frame that much later than them; the letters are read in either case
TIME_SHIFTS = [('T', 0.), ('t', 0.), ('F', 30.), ('TF', 30.)]


@pytest.fixture
def spice_check_times(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The times at which the hosts check the SPICE pointing, recorded as they check."""

    times: list[float] = []
    check = oops.Host._check_spice_frame

    def spy(frame: Any, time: Any, **kwargs: Any) -> Any:
        times.append(float(time))
        return check(frame, time, **kwargs)

    monkeypatch.setattr(oops.Host, '_check_spice_frame', staticmethod(spy))
    return times

##########################################################################################
