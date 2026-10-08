# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rms-host-cassini` is the Cassini host plug-in for `oops` (rms-oops). It is a port of
`rms-oops/src/hosts/cassini/` to the newer `oops.Host` / `oops.host` API, part of splitting
the mission hosts out of rms-oops (see the sibling `rms-host-galileo`). Only ISS is ported so
far; rms-oops still has VIMS and UVIS.

## Standards

Tooling and conventions follow **rms-polymath**. `.claude/rules/*.md`, which were copied from
it, are the authoritative coding, testing, documentation, dependency and environment
standards. When adding packaging, CI or scripts, model them on rms-polymath's
`pyproject.toml`, `.flake8`, `.github/workflows/` and `scripts/` (`setup-venv.sh`,
`run-all-checks.sh`). Some details in the rules are polymath-specific, for example the
two published `.pyi` stubs and `stubtest`. Ignore those until this repo has the equivalent.

- Hand-aligned columns (assignments, dict colons, `from x     import`) are deliberate. Never
  run `ruff format` or any other auto-formatter.
- Keep the file banners: a line of 90 `#` characters at the top and bottom of each module,
  with a `# host_cassini/<file>.py` header.

## Current state

- `scripts/run-all-checks.sh` doesn't exist yet, so the `run-all-checks` skill has nothing
  to run. CI (`.github/workflows/run-tests.yml`) runs ruff, flake8 `E12`/`E13`, mypy on
  `tests/`, pip-audit, codespell, PyMarkdown and pytest.
- The `oops.Host` API is unreleased. CI installs rms-oops from its `mrs_260925_host_reorg`
  branch; drop that step once a release has it.
- Tests never touch SPICE kernels or spicedb. `tests/conftest.py` replaces them with fakes
  for every test, so new tests get this automatically.

## Architecture gotchas

- Each instrument is a subpackage (`iss/`) split in two so that the host registers at
  import time while its implementation loads lazily. `Host.from_file()` dispatches through
  the registered hosts' `_detect_in_*` methods, so a host is invisible until registered.
  - `iss/_host.py` is imported eagerly. It defines the registered `_CassiniISSHost`
    (`NAME`, the detectors, and `from_file`/`from_index` that import and call the full
    class) and calls `_register()`. Keep its imports cheap.
  - `iss/_oops.py` defines the full `CassiniISS`, which subclasses `_CassiniISSHost` and is
    not registered itself. The `__getattr__` in `host_cassini/__init__.py` and
    `iss/__init__.py` imports it on first access to `CassiniISS`.
- `_Cassini` in `host_cassini/_oops.py` is never instantiated; all its state lives in class
  attributes. CK and SPK kernels load lazily by "month": the 1997-10-01 to 2017-10-01
  mission is split into 240 equal periods, each padded by `SLOP`. Before `TOUR` the Jupiter
  system is used, after it the Saturn system.
- Running anything needs a SPICE kernel database through `spicedb` (`SPICE_PATH`, or
  `$OOPS_RESOURCES/SPICE`, plus `SPICE_SQLITE_DB_NAME`). It must contain the kernel sets
  `CAS-SPK-*`, `CAS-CK-*` and `CAS-CK-GAPFILL`. `spicedb` currently ships inside rms-oops.

## Repo etiquette

- Commit subjects are a plain, capitalized, imperative sentence with no type prefix and no
  trailing period. PRs are squash-merged onto `main`.
- Branch names: `<initials>_<YYMMDD>_<topic>`.
- Versions come from `setuptools_scm`; never hand-write a version.
