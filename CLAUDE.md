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

- No `pyproject.toml`, tests, CI or `scripts/run-all-checks.sh` exist yet, so the
  `run-all-checks` skill has nothing to run.
- The import name is `host_cassini`. The directory is still `src/host-cassini/` and must be
  renamed to `src/host_cassini/`, because a hyphenated name can't be imported and `iss.py`
  uses relative imports.

## Architecture gotchas

- Each `oops.Host` subclass must call `cls._register()` right after its class definition, as
  `iss.py` does. `Host.from_file()` dispatches through the registered hosts' `_detect_in_*`
  methods, so a host is invisible until its module has been imported.
- `_Cassini` in `__init__.py` is never instantiated; all its state lives in class attributes.
  CK and SPK kernels load lazily by "month": the 1997-10-01 to 2017-10-01 mission is split into
  240 equal periods, each padded by `SLOP`. Before `TOUR` the Jupiter system is used, after it
  the Saturn system.
- Running anything needs a SPICE kernel database through `spicedb` (`SPICE_PATH`, or
  `$OOPS_RESOURCES/SPICE`, plus `SPICE_SQLITE_DB_NAME`). It must contain the kernel sets
  `CAS-SPK-*`, `CAS-CK-*` and `CAS-CK-GAPFILL`. `spicedb` currently ships inside rms-oops.

## Repo etiquette

- Commit subjects are a plain, capitalized, imperative sentence with no type prefix and no
  trailing period. PRs are squash-merged onto `main`.
- Branch names: `<initials>_<YYMMDD>_<topic>`.
- Versions come from `setuptools_scm`; never hand-write a version.
