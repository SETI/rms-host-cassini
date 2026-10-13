<!-- pyml disable MD025 -->
# rms-host-cassini

[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-host-cassini/run-tests.yml?branch=main)](https://github.com/SETI/rms-host-cassini/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-host-cassini/badge/?version=latest)](https://rms-host-cassini.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-host-cassini/main?logo=codecov)](https://codecov.io/gh/SETI/rms-host-cassini)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-host-cassini)](https://pypi.org/project/rms-host-cassini)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-host-cassini)](https://pypi.org/project/rms-host-cassini)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-host-cassini)](https://pypi.org/project/rms-host-cassini)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-host-cassini)](https://pypi.org/project/rms-host-cassini)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-host-cassini/latest)](https://github.com/SETI/rms-host-cassini/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-host-cassini)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-host-cassini)](https://github.com/SETI/rms-host-cassini/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-host-cassini)
<!-- start-after-point -->

# Introduction

`rms-host-cassini` is the Cassini mission host for
[OOPS](https://github.com/SETI/rms-oops), the Object-Oriented Python SPICE toolkit for
observation geometry. It turns Cassini ISS, UVIS, and VIMS data products into OOPS
`Observation` objects, with the frames, fields of view, timing, targets, and light sources
that each instrument requires. It is a product of the
[PDS Ring-Moon Systems Node](https://pds-rings.seti.org).

# Installation

The `host_cassini` module is available via the `rms-host-cassini` package on PyPI and can
be installed with:

```sh
pip install rms-host-cassini
```

The host needs the `oops.Host` interface, which is not yet in a released version of
`rms-oops`. Until it is, install `rms-oops` from its development branch:

```sh
pip install "rms-oops @ git+https://github.com/SETI/rms-oops@mrs_260925_host_reorg"
```

Geometry calculations need a SPICE kernel database, located through `spicedb` by the
`SPICE_PATH` environment variable (or `$OOPS_RESOURCES/SPICE`) and `SPICE_SQLITE_DB_NAME`.
It must contain the kernel sets `CAS-SPK-*`, `CAS-CK-*`, and `CAS-CK-GAPFILL`.

# Getting Started

Importing `host_cassini.oops` registers the Cassini hosts with `oops.Host`, which can
then identify and read any supported Cassini file:

```python
import oops
import host_cassini.oops

obs = oops.Host.from_file('N1454725799_1.IMG')
```

Each instrument also has its own host class:

```python
from host_cassini import CassiniISS, CassiniUVIS, CassiniVIMS

snapshot = CassiniISS.from_file('N1454725799_1.IMG')
(vis, ir) = CassiniVIMS.from_file('v1484505000_1.qub')
ir = CassiniVIMS.from_file('v1484505000_1.qub', select='IR')
```

# Features

The hosts are:

- `CassiniISS`: The Imaging Science Subsystem narrow- and wide-angle cameras. A file
  becomes a `Snapshot`; an image index can also be read with `CassiniISS.from_index()`.
- `CassiniUVIS`: The Ultraviolet Imaging Spectrograph. A file becomes one observation, or
  one for each window of a QUBE with several windows.
- `CassiniVIMS`: The Visual and Infrared Mapping Spectrometer. A cube becomes a pair of
  observations, one for each channel (None for an inactive channel), and `select='VIS'` or
  `select='IR'` returns just one.

SPICE kernels are loaded only as needed, a month of the mission at a time. A target named
in the label is repaired where the label is known to be wrong, and an observation of a
star uses that star as its light source.

Details are available in the
[module documentation](https://rms-host-cassini.readthedocs.io/en/latest/module.html).

# Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-host-cassini/blob/main/CONTRIBUTING.md).

# Links

- [Documentation](https://rms-host-cassini.readthedocs.io)
- [Repository](https://github.com/SETI/rms-host-cassini)
- [Issue tracker](https://github.com/SETI/rms-host-cassini/issues)
- [PyPi](https://pypi.org/project/rms-host-cassini)

# Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-host-cassini/blob/main/LICENSE).
