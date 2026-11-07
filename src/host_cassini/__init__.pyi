##########################################################################################
# host_cassini/__init__.pyi
##########################################################################################
"""Type stub for :mod:`host_cassini`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.

The detectors and support methods that a Host subclass overrides or calls, such as
`_register` and `_detect_in_pds3`, are private and are not published here.
"""

import pathlib
from typing import Any

from filecache import FCPath
from pdsparser import Pds3Label
from polymath import Scalar
from vicar import VicarImage, VicarLabel
from oops import Body, Meshgrid
from oops.cadence import Cadence
from oops.calibration import Calibration
from oops.fov import FOV
from oops.frame import Frame
from oops.host import Host, ObsArray
from oops.lightsource import LightSource
from oops.observation import Observation, Snapshot
from oops.path import Path

__all__ = ['CassiniISS', 'CassiniUVIS', 'CassiniVIMS']

class CassiniISS(Host):
    # The runtime names the first parameter `fileinfo`, where Host names it `filepath`.
    @staticmethod
    def from_file(  # type: ignore[override]
        fileinfo: str | pathlib.Path | FCPath | Pds3Label | VicarLabel
        | VicarImage, *,
        fast_distortion: bool | None = True, return_all_planets: bool = False,
        pds3_options: dict[str, Any] | None = None, astrometry: bool = False,
        target: str | Body | None = None, lightsource: LightSource | None = None,
        path: str | Path | None = None, frame: str | Frame | None = None,
        cadence: Cadence | None = None, fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None, **kwargs: Any) -> Snapshot: ...
    # Every row is one Snapshot, so the list is typed more narrowly than Host's, which a
    # list's invariance does not allow.
    @staticmethod
    def from_index(  # type: ignore[override]
        filepath: str | pathlib.Path | FCPath, *,
        row_dicts: list[dict[str, Any]] | None = None,
        pds3_options: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None, fast_distortion: bool | None = True,
        return_all_planets: bool = False, target: str | Body | None = None,
        lightsource: LightSource | None = None, path: str | Path | None = None,
        frame: str | Frame | None = None, cadence: Cadence | None = None,
        fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None, **kwargs: Any) -> list[Snapshot]: ...
    @staticmethod
    def initialize(*, ck: str = 'reconstructed', planets: list[Any] | None = None,
        asof: str | None = None, spk: str = 'reconstructed', gapfill: bool = True,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...

class CassiniUVIS(Host):
    # The runtime names the first parameter `fileinfo`, where Host names it `filepath`,
    # and `select` accepts no channel name, because UVIS has none.
    @staticmethod
    def from_file(  # type: ignore[override]
        fileinfo: str | pathlib.Path | FCPath | Pds3Label, *,
        enclose: bool = False, pds3_options: dict[str, Any] | None = None,
        select: int | slice | tuple[int | slice, ...] | None = None,
        astrometry: bool = False, target: str | Body | None = None,
        lightsource: LightSource | None = None, path: str | Path | None = None,
        frame: str | Frame | None = None, cadence: Cadence | None = None,
        fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None,
        **kwargs: Any) -> Observation | tuple[Observation, ...]: ...
    # `select` accepts no channel name, and the list is typed more narrowly than Host's,
    # which a list's invariance does not allow.
    @staticmethod
    def from_index(  # type: ignore[override]
        filepath: str | pathlib.Path | FCPath, *, enclose: bool = False,
        supplement: str | pathlib.Path | FCPath | object | None = None,
        label: Pds3Label | None = None, row_dicts: list[dict[str, Any]] | None = None,
        pds3_options: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        select: int | slice | tuple[int | slice, ...] | None = None,
        target: str | Body | None = None, lightsource: LightSource | None = None,
        path: str | Path | None = None, frame: str | Frame | None = None,
        cadence: Cadence | None = None, fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None,
        **kwargs: Any) -> list[Observation | ObsArray]: ...
    @staticmethod
    def initialize(*, ck: str = 'reconstructed', planets: list[Any] | None = None,
        asof: str | None = None, spk: str = 'reconstructed', gapfill: bool = True,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...

class CassiniVIMS(Host):
    # The runtime names the first parameter `fileinfo`, where Host names it `filepath`.
    @staticmethod
    def from_file(  # type: ignore[override]
        fileinfo: str | pathlib.Path | FCPath | Pds3Label, *,
        pds3_options: dict[str, Any] | None = None,
        select: int | slice | str | tuple[int | slice | str, ...] | None = None,
        astrometry: bool = False, target: str | Body | None = None,
        lightsource: LightSource | None = None, path: str | Path | None = None,
        frame: str | Frame | None = None, cadence: Cadence | None = None,
        fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None, **kwargs: Any
        ) -> tuple[Observation | None, Observation | None] | Observation | None: ...
    # Every row is a (vis, ir) pair or a selection from one, so the list is typed more
    # narrowly than Host's, which a list's invariance does not allow.
    @staticmethod
    def from_index(  # type: ignore[override]
        filepath: str | pathlib.Path | FCPath, *,
        supplement: str | pathlib.Path | FCPath | object | None = None,
        label: Pds3Label | None = None, row_dicts: list[dict[str, Any]] | None = None,
        pds3_options: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        select: int | slice | str | tuple[int | slice | str, ...] | None = None,
        target: str | Body | None = None, lightsource: LightSource | None = None,
        path: str | Path | None = None, frame: str | Frame | None = None,
        cadence: Cadence | None = None, fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None, **kwargs: Any
        ) -> list[tuple[Observation | None, Observation | None] | Observation
                  | None]: ...
    @staticmethod
    def meshgrid_and_times(obs: Observation, *, oversample: float = 6,
        extend: float = 1.5) -> tuple[Meshgrid, Scalar]: ...
    @staticmethod
    def initialize(*, ck: str = 'reconstructed', planets: list[Any] | None = None,
        asof: str | None = None, spk: str = 'reconstructed', gapfill: bool = True,
        mst_pck: bool = True, irregulars: bool = True) -> None: ...

##########################################################################################
