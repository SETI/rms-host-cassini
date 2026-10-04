##########################################################################################
# tests/conftest.py
##########################################################################################
"""Shared fixtures that let the Cassini host run without SPICE kernels or spicedb.

Every spicedb function the package calls is replaced by :class:`FakeSpiceDB`, and the few
oops and cspyce calls that would read SPICE kernels directly are replaced by
kernel-free equivalents. Everything else, including oops itself, runs for real.
"""

from dataclasses import dataclass, field
from typing import Any

import cspyce
import julian
import numpy as np
import oops
import pytest
import spicedb
from oops.body import Body

from host_cassini import _Cassini
from host_cassini.iss import ISS

# Nominal Cassini ISS camera geometry: 1024x1024 pixels; NAC 0.35 deg and WAC 3.5 deg
# square fields of view.
NAC_HALF_FOV_DEG = 0.175
WAC_HALF_FOV_DEG = 1.75
PIXELS = 1024


def iso_from_tdb(tdb: float) -> str:
    """The ISO date-time, to the millisecond, of a time in seconds TDB."""

    return julian.iso_from_tai(julian.tai_from_tdb(tdb), digits=3)


def tdb_in_month(month: float) -> float:
    """The time in seconds TDB at a fractional "month" of the Cassini mission."""

    return _Cassini.TDB0 + month * _Cassini.DTDB


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
        ck_kernels (list[FakeKernel]): Returned by `select_ck`.
        inst_kernels (list[FakeKernel]): Returned by `select_inst`.
        kernel_dict (dict): Returned by `as_dict`.
        used (list[str]): Returned by `used_basenames`.
    """

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    spk_kernels: list[FakeKernel] = field(default_factory=list)
    ck_kernels: list[FakeKernel] = field(default_factory=list)
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
        return self.ck_kernels

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
    registries are copied so that frames registered by one test never reach another.
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
        return oops.frame.Cmatrix(np.eye(3), oops.Frame.J2000, frame_id=frame_id)

    def furnsh(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'a test tried to furnish a SPICE kernel: {args}')

    monkeypatch.setattr(cspyce, 'str2et', julian.tdb_from_iso)
    monkeypatch.setattr(cspyce, 'furnsh', furnsh)
    monkeypatch.setattr(oops.spice, 'load_leap_seconds', load_leap_seconds)
    monkeypatch.setattr(Body, 'define_solar_system', staticmethod(define_solar_system))
    monkeypatch.setattr(oops.path, 'SpicePath', spice_path)
    monkeypatch.setattr(oops.frame, 'SpiceFrame', spice_frame)

    monkeypatch.setattr(oops.Path, '_PATH_REGISTRY', dict(oops.Path._PATH_REGISTRY))
    monkeypatch.setattr(oops.Path, '_PATH_CACHE', dict(oops.Path._PATH_CACHE))
    monkeypatch.setattr(oops.Frame, '_FRAME_REGISTRY', dict(oops.Frame._FRAME_REGISTRY))
    monkeypatch.setattr(oops.Frame, '_FRAME_CACHE', dict(oops.Frame._FRAME_CACHE))
    return fake


@pytest.fixture(autouse=True)
def reset_host() -> Any:
    """Start and end every test with the Cassini and ISS class state reset."""

    ISS._reset()
    yield
    ISS._reset()


@pytest.fixture
def initialized_iss(fake_spicedb: FakeSpiceDB) -> FakeSpiceDB:
    """ISS initialized with default options against the fake spicedb."""

    ISS._initialize()
    return fake_spicedb

##########################################################################################
