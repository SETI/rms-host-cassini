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
  marks a real mismatch with `oops.Host`: every `from_file` and `from_index` accepts only
  the keywords it names, where `oops.Host`'s takes `**kwargs`; every `from_file` names its
  first parameter `fileinfo`; and CassiniUVIS's `select` accepts no channel name.
- `docs/conf.py` resolves the short class names in docstrings (`FCPath`, `Observation`,
  ...) through intersphinx with `_SHORT_NAMES`. Add a name there when a nitpicky build
  reports a class from another RMS package.
- Coverage must stay at or above 65% (`fail_under` in `pyproject.toml`, and the project and
  patch targets in `codecov.yml`); it is at 93%, held down by `vims.py` at 85%. The ubuntu /
  Python 3.13 CI job uploads to Codecov and fails if the upload fails. The upload
  authenticates through GitHub OIDC (`use_oidc`), so it needs no `CODECOV_TOKEN` secret.
- Tests never touch SPICE kernels or spicedb. `tests/oops/conftest.py` replaces them with
  fakes for every test, and gives each test its own copy of the oops path, frame and body
  registries and of each path and frame class's waypoint and wayframe pools, so new tests
  get this automatically. A test that tracks a target with
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
  case), and -999 in the keywords of `_FILL_KEYWORDS`, count as missing, the header's
  values are first written in the label's forms (`_pds3_form()`), and
  `_RESOLUTION_RULES` names each item's one fixed source, the label by default; there is
  no per-image choice. Each rule's comment gives the evidence and cites the issue that
  lists every item on which the two disagree, with counts and the images on which the
  fixed source is wrong; update both together. `dict`, `label_dict` and `header_dict`
  must stay JSON-serializable: times as the label's text, sets as lists, repeated VICAR
  parameters as `NAME[n]` keys. Labels
  are always parsed with `sources=False`, and `label_dict` keeps only the upper-case
  keywords whose values are not None: no END and none of the lower-case keys the parser
  adds (`START_TIME_fmt`, `^IMAGE_offset`, `objects`, ...). The array always comes from
  the VICAR header's layout, never the label's `^IMAGE` offset, which is wrong for
  calibrated files. Every repair of label text lives in each host's
  `_correct_label_text()` (ISS: `_LABEL_REPAIRS`; VIMS: `_TWO_LINE_COMMENT`), which
  oops runs before any parser method, for product and index
  labels alike, so each pattern must match only the defect; the comment beside each
  cites its issue. The fast/strict/loose fallback is oops's
  (`_read_fileinfo(..., pds3_fallback=True)`); ISS keeps only the check that sends a
  label that the fast parser misreads (a keyword that is not a PDS3 name) to the later
  methods. `oops.Host.from_file`/`from_index` offer a label that no method reads to each
  host's correction, so the hosts need no `_detect_in_file()`.
- Missing C-kernel pointing raises `oops.OopsSpiceError`, checked in `from_file` with
  `oops.Host._check_spice_frame()` only when the frame comes from SPICE (no `frame`
  override). `from_index` calls it with `defer=True`, so a row whose pointing SPICE lacks
  gets an `oops.frame.SpiceErrorFrame`, which raises `OopsSpiceError` when evaluated.
  `_check_spice_frame(frame, cadence, overrides, ...)` is called before
  `_apply_overrides()` with the cadence before the overrides and the whole overrides
  dict, and evaluates the frame as the observation will (cadence override, tracker, "T",
  "F"). ISS passes its camera frame (`CASSINI_ISS_NAC`, a fixed Cmatrix on the
  `_FLIPPED` SpiceFrame), so an "F" shift leaves the C-kernel lookup at the
  observation's time; UVIS and VIMS pass the SpiceFrame itself, which "F" shifts. Clock
  conversions go through `_tdb_from_sclk()`/`_sclk_from_tdb()`, which use oops's one
  rule for cspyce failures (`OopsSpiceError.CSPYCE_ERRORS` and `is_spice_error`): only
  an error with a SPICE code becomes `OopsSpiceError`; any other propagates.
- No `from_file()` or `from_index()` takes `**kwargs`, so Python itself raises
  `TypeError` for an unknown keyword. Every `from_file()` checks `select` before reading
  with `oops.Host._check_select()` (ISS: `unit=`, so it must be None; VIMS: the channel
  names). `abspath` is passed to `oops.Host._apply_overrides(abspath=...)` only by a full
  `from_file` read that retrieved the data file in this call: the local path that
  `FCPath.retrieve()` returned (ISS: `_read_vicar_image()`, or a retrieve after oops read
  a VICAR path; UVIS: `_load_data()`; VIMS: the cube's `data_file`). Astrometry reads,
  a caller's `VicarImage`, and every `from_index` row give none. UVIS and VIMS name the
  data file in `abspath` while their other file subfields name the label (see below).
  ISS builds
  `header_dict` from a `VicarLabel` read from the file, since rms-vicar writes the
  reading machine into HOST and BHOST when it reads the array.
- The null rules (`_is_null`, `_without_nulls`, `_row_items`, `_label_dict` in
  `_cassini.py`) are shared by the three instruments: `oops.host.is_pds3_null` text
  ("UNK", "UNKNOWN", "N/A", "NULL", "NOT APPLICABLE", "---", "", any case, padded) is
  missing; index rows also drop 1e32 and -1e32. -999 is missing only where it is a
  declared null or cannot be a real value: in the keywords and ISS index columns of each
  module's `_FILL_KEYWORDS` (temperatures, rates, ratios, VIMS exposures); UVIS and VIMS
  rows get it from their labels' declared null constants. A placeholder TARGET_NAME
  gives the target NONE in every instrument. UVIS and VIMS rows apply the table masks,
  since their labels declare real null constants (-999, -99, -9.9e99, -1, N/A); ISS rows
  apply them per column as `_index_fill_rules()` decides (see below). UVIS and VIMS
  `from_file` publish `dict` (no nulls), `label_dict` or (VIMS cube) `header_dict`, the
  clock counts, and `time_from_sclk`; their file subfields still name the label, an
  open question in the issues. They read through `_read_pds3_label()`, which also
  refuses another instrument's parsed label with `WrongHostError`.
- Every `from_index()` reads its rows through `oops.Host._index_rows()`, which streams
  the index and supplement with `PdsTable(defer_load=True).iter_dicts` 100 rows at a
  time, and pairs, validates and filters them. The hosts read the Ring-Moon Systems Node
  metadata indexes and their supplements. Nothing in
  the host reads a whole table, builds a list of rows, or indexes `row_dicts`; a check
  on a row (UVIS/VIMS: the supplement's columns in `row_dicts`) happens in the generator
  at that row. Each row's file is its full path or URL: `oops.Host._index_file_path()`
  puts `<VOLUME_ID>/<FILE_SPECIFICATION_NAME>` in `oops.Host._volumes_directory()` of
  the index. No row has `abspath`, and nothing creates a directory; parsed content with
  no path (`_known_path()` is None) has no file subfields. All three follow the
  `oops.Host.from_index` contract: `supplement` defaults to `oops.Host.SUPPLEMENT` (UVIS
  and VIMS require it, so None raises `HostIndexError` unless `row_dicts` is given; ISS
  has none and ignores a missing one), and an unknown keyword raises `TypeError`. The
  default pairing key is FILE_SPECIFICATION_NAME alone, with VOLUME_ID still compared as
  a shared column. ISS refuses `select`, since each row is one
  observation; UVIS and VIMS apply it, since a row gives several windows or a (vis, ir)
  pair.
- `MERGE_SUPPLEMENT_BY_FILE` on each class fixes how oops pairs the two tables, from the
  survey of every metadata index cited beside it: ISS False (no ISS supplement exists, so
  a given one is paired by position), UVIS and VIMS True (54 of 61 and 25 of 95 out of
  order until the supplements are corrected; the comment beside each flag cites the
  issues). UVIS passes
  `file_key='FILE_NAME'` and `supplement_key=('VOLUME_ID', 'FILE_SPECIFICATION_NAME')`;
  VIMS passes `file_key=('VOLUME_ID', 'PATH_NAME', 'FILE_NAME')` and the same
  `supplement_key` (the index names the `.qub`, the supplement the `.lbl`, and oops
  compares paths without the extension). Each host's `_validate_index` checks every
  pair. `oops.Host.from_index()` asks every host's `_detect_in_index()` about the label
  first: ISS and UVIS claim their paths (COISS, COUVIS) and return False for another
  instrument's path or INSTRUMENT_ID, VIMS claims INSTRUMENT_ID VIMS or a COVIMS path, so
  no Cassini index needs its first row read to be identified.
- `CassiniUVIS._label_from_index_row()` builds the QUBE, TIME_SERIES or SPECTRUM object
  that a product label has from a row's DATA_OBJECT_TYPE and window, binning and
  LINE_SAMPLES columns. A QUBE whose line window bins to one line has one line;
  every QUBE has 1024 bands. An HSP sampling interval is INTEGRATION_DURATION (ms); an
  HDAC one is 125 ms times DWELL_TIME, since its INTEGRATION_DURATION is -999 (true of
  1,555 of 1,559 sampled labels; 4 in COUVIS_0031/0032 give DWELL_TIME 72 and 125 ms). A
  row gives one window, the enclosing one for a multi-window QUBE, binned by the smallest
  binning (`from_file(enclose=True)` leaves it unbinned).
- Every `from_index()` returns an iterator. It checks its arguments and calls
  `_index_rows()`, which reads only the labels at call time, then returns a generator
  that builds each row's observation when the iteration reaches it, so an error in one
  row waits for that row. It must not be a generator function, which would defer every
  argument error to the first `next()`.
- Every index workaround must still give the right answer once the index files are
  corrected (the holdings' product labels and data files cannot change; the index and
  supplement files can, and each instrument's index problems are listed in one issue in
  SETI/rms-data-projects, which the `#` comment beside each workaround cites): apply it
  only where the defect shows, decided once per table, say in its comment that it works
  around the problem until the index is corrected, and test the corrected form.
- `CassiniISS._index_fill_rules(label)` decides, once per table from the index label,
  which masks apply: those of columns that declare a null constant other than 0 (the
  metadata labels declare 0 for BIAS_STRIP_MEAN and DARK_STRIP_MEAN, real values in the
  image labels, and for EMISSION_ANGLE and INCIDENCE_ANGLE). `_UNDECLARED_FILL_VALUES`
  (65534, -2147483648) apply only to columns whose label does not declare a constant
  (the labels of COISS_2115 and COISS_2116 do not declare any); -999 in `_FILL_KEYWORDS`
  always. With `row_dicts` the label comes from `label`, if given, and otherwise from
  `filepath`, because `oops.Host.from_index` passes only the caller's `label`, often
  None, to a host that corrects label text. A 0 angle (`_ZERO_WITHOUT_GEOMETRY`) is
  missing only in a row without PHASE_ANGLE, so a corrected table keeps a real 0.
  `_row_items()` also writes the clock counts with three digits after the point, which
  the tables drop.
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
