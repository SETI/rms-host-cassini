# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rms-host-cassini` is the Cassini host plug-in for `oops` (rms-oops). It is a port of
`rms-oops/src/hosts/cassini/` to the newer `oops.Host` / `oops.host` API, part of splitting
the mission hosts out of rms-oops (see the sibling `rms-host-galileo`). ISS, VIMS and UVIS
are ported, each with both `from_file()` and `from_index()`.

## Standards

Tooling and conventions follow **rms-polymath**. `.claude/rules/*.md`, which were copied from
it, are the authoritative coding, testing, documentation, dependency and environment
standards. The repo layout, CI and scripts come from `rms-devenv/repo_template`, adapted as
rms-polymath adapted them. Model changes to packaging, CI or scripts on rms-polymath's
`pyproject.toml`, `.flake8`, `.github/workflows/` and `scripts/`. The rules' references to
the two published `.pyi` stubs and `stubtest` apply here to the stubs described below.

- Hand-aligned columns (assignments, dict colons, `from x     import`) are deliberate. Never
  run `ruff format` or any other auto-formatter.
- Keep the file banners: a line of 90 `#` characters at the top and bottom of each module,
  with a `# host_cassini/<file>.py` header (`.pyi` for a stub).

## Current state

- `scripts/run-all-checks.sh` is the single source of truth for the checks: ruff, flake8
  `E12`/`E13`, mypy on `tests/`, pytest, pyroma, stubtest, pip-audit, the Sphinx build
  (`-W -n`), codespell and PyMarkdown. `ruff format` is disabled.
  `.github/workflows/run-tests.yml` runs the same set. `scripts/setup-venv.sh` builds the
  `venv/` that the scripts expect; set `VENV` to use another.
- The `oops.Host` API first appears in rms-oops 0.4.0, and `oops.OopsSpiceError`, which
  `CassiniISS` raises for missing pointing, in 0.5.0, the minimum version in
  `pyproject.toml`. Until 0.5.0 is released, install rms-oops from its
  `rf_261118_oops_spice_error` branch to develop or test. The `ObsArray` entry in
  `nitpick_ignore_regex` in `docs/conf.py` stays: oops never documents that type alias.
- The type information is published in two stubs: `host_cassini/__init__.pyi` declares
  the three hosts in full, and `host_cassini/oops/__init__.pyi` re-exports them. `stubtest`
  checks them against the runtime in the check script and in CI, so adding, renaming or
  re-signing a public member means updating `host_cassini/__init__.pyi` in the same change.
  The `[tool.mypy] exclude`, the override list in `pyproject.toml` and
  `.stubtest-allowlist` keep it off the stub-less modules. Each `type: ignore[override]`
  marks a real mismatch with `oops.Host`: every `from_file` names its first parameter
  `fileinfo`, every `from_index` returns a list narrower than the base class's, and
  CassiniUVIS's `select` accepts no channel name.
- `docs/conf.py` resolves the short class names in docstrings (`FCPath`, `Observation`,
  ...) through intersphinx with `_SHORT_NAMES`. Add a name there when a nitpicky build
  reports a class from another RMS package.
- Coverage must stay at or above 65% (`fail_under` in `pyproject.toml`, and the project and
  patch targets in `codecov.yml`); it is at 87%, held down by `vims.py` at 71%. The ubuntu /
  Python 3.13 CI job uploads to Codecov and fails if the upload fails. The upload
  authenticates through GitHub OIDC (`use_oidc`), so it needs no `CODECOV_TOKEN` secret.
- Tests never touch SPICE kernels or spicedb. `tests/oops/conftest.py` replaces them with
  fakes for every test, and gives each test its own copy of the oops path, frame and body
  registries, so new tests get this automatically. A test that tracks a target with
  `tracker` needs a registered body; the `saturn_path` fixture provides SATURN.

## Architecture gotchas

- The oops support lives in the `host_cassini.oops` subpackage, one module per instrument
  (`iss.py`, `vims.py`, `uvis.py`). Each host class calls `_register()` at import
  time, and `oops/__init__.py` imports every instrument module. `Host.from_file()`
  dispatches through the registered hosts' `_detect_in_*` methods, so a host is invisible
  until `host_cassini.oops` (or its module) is imported.
- The hosts apply the standard overrides through `oops.Host._apply_overrides()` and `select`
  through `oops.Host._apply_select()`. `_apply_select()` needs a tuple, so UVIS wraps a
  single observation in one first. What the instruments share lives in `oops/_cassini.py`:
  `_CMATRIX_ROTATION`, `_pds3_options_with_defaults()`, and the OBSERVATION_ID helpers
  `_cassini_obs_id_code()` and `_fix_cassini_ring_target()`. Keep them out of
  `oops/__init__.py`, which the instrument modules can't import from without a circular
  import.
- Each host's public `initialize()` chooses the kernels. It is optional: `from_file()` and
  `from_index()` call it with the defaults, and after the first call later calls are
  ignored until `_reset()`.
- The `spice_frame_id` subfield is the frame ID from the Cassini frame kernel, not the
  instrument code from the instrument kernel; the two differ for UVIS HSP, HDAC and the
  solar ports. UVIS keeps them apart: `_FRAME_IDS` (frame kernel) for `spice_frame_id`,
  `_INSTRUMENT_IDS` (instrument kernel) for the FOV lookup in `initialize()`.
- TARGET_NAME repairs are per instrument: each module keeps its own
  `_TARGET_NAME_REPAIRS`. Don't merge them into `_cassini.py`.
- `CassiniISS.from_file()` reads only the VICAR image file when given it, and the PDS3
  label plus the whole image file when given the label (the label alone with
  `astrometry`). `_merged_dict()` combines the two into `dict`: PDS3 null values (any
  case) and the fill value -999 count as missing, the header's values are first written
  in the label's forms (`_pds3_form()`), and `_RESOLUTION_RULES` names each item's one
  fixed source, the label by default; there is no per-image choice. Each rule's comment
  gives the evidence and cites issue #10, which lists every item on which the two
  disagree, with counts and the images on which the fixed source is wrong; update both
  together. `dict`, `label_dict` and `header_dict` must stay JSON-serializable: times as
  the label's text, sets as lists, repeated VICAR parameters as `NAME[n]` keys. Labels
  are always parsed with `sources=False`, and `label_dict` keeps only the upper-case
  keywords whose values are not None: no END and none of the lower-case keys the parser
  adds (`START_TIME_fmt`, `^IMAGE_offset`, `objects`, ...). The array always comes from
  the VICAR header's layout, never the label's `^IMAGE` offset, which is wrong for
  calibrated files. Labels the parser rejects, cannot convert, or misreads go
  through `_parse_fileinfo()`'s fallbacks (from the requested method on through fast,
  strict, loose) and in-memory repairs (#11).
- Missing C-kernel pointing raises `oops.OopsSpiceError`, checked only in `from_file`
  and only when the frame comes from SPICE (no `frame` override); `from_index` never
  checks.
- Every `from_index()` names each row's file by its full path or URL:
  `_index_file_path()` in `_cassini.py` takes the volume root from where the index
  lives (the directory above `index/`, or `volumes/<series>/<VOLUME_ID>` for an index
  in the `metadata/` tree), and `_drop_unfetched_abspath()` drops `abspath` when a
  remote file was never fetched. All three follow the `oops.Host.from_index` contract
  (#20): `supplement` defaults to `oops.Host.SUPPLEMENT` (UVIS and VIMS require it, so
  None raises `HostIndexError` unless `row_dicts` is given; ISS has none and ignores a
  missing one), `label=` reaches `_read_index_rows()`, and an unknown keyword raises
  `OopsTypeError`. ISS refuses `select`, since each row is one observation; UVIS and VIMS
  apply it, since a row gives several windows or a (vis, ir) pair.
- `CassiniISS._row_items()` does not apply the table reader's masks: the metadata index
  labels declare 0 the null constant of BIAS_STRIP_MEAN and DARK_STRIP_MEAN, real values
  in the image labels. `_INDEX_FILL_VALUES` and the bound 1e32 list the fill values
  instead, for the volume indexes (which declare none) and the metadata indexes alike. It
  also writes the clock counts with three digits after the point, which the tables drop.
- `CassiniISS._early_cruise_counts()` fixes the clock counts of the first 1250 images of
  COISS_1001 (through `_LAST_EARLY_CRUISE_IMAGE`): the stop count is IMAGE_NUMBER, the
  start count is computed from it and the exposure with the SCLK kernel (operator,
  2026-11-19). The tests' fake `cspyce.scs2e`/`sce2s` read a count as seconds.
- `host_cassini/__init__.py` imports the hosts lazily, through a PEP 562 `__getattr__`, so
  that importing `host_cassini` does not import oops. Keep any import of
  `host_cassini.oops` out of `host_cassini/__init__.py`.
- `_Cassini` in `host_cassini/oops/_cassini.py` is never instantiated; all its state
  lives in class attributes. CK and SPK kernels load lazily by "month": the 1997-10-01 to
  2017-10-01 mission is split into 240 equal periods, each padded by `SLOP`. Before `TOUR`
  the Jupiter system is used, after it the Saturn system.
- Running anything needs a SPICE kernel database through `spicedb` (`SPICE_PATH`, or
  `$OOPS_RESOURCES/SPICE`, plus `SPICE_SQLITE_DB_NAME`). It must contain the kernel sets
  `CAS-SPK-*`, `CAS-CK-*` and `CAS-CK-GAPFILL`. `spicedb` currently ships inside rms-oops.

## Repo etiquette

- Commit subjects are a plain, capitalized, imperative sentence with no type prefix and no
  trailing period. PRs are squash-merged onto `main`.
- Branch names: `<initials>_<YYMMDD>_<topic>`.
- Versions come from `setuptools_scm`; never hand-write a version.
