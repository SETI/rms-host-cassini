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
observation geometry. It reads data products of three Cassini instruments, the Imaging
Science Subsystem (ISS) cameras, the Ultraviolet Imaging Spectrograph (UVIS), and the
Visual and Infrared Mapping Spectrometer (VIMS), as the Planetary Data System (PDS)
archives them. It turns each product into OOPS observations: the data array, together
with the time, field of view, pointing, target, and light source that OOPS needs to
compute its geometry. It is a product of the
[PDS Ring-Moon Systems Node](https://pds-rings.seti.org).

# Features

- **Either file of an ISS product**: an ISS image can be opened from its VICAR image file
  or from its PDS3 label, and the pixels are identical either way.
- **Metadata from both labels**: when an ISS image's PDS3 label is read together with
  its image file, the values in the label and in the VICAR header are merged under the
  PDS3 names, with one fixed choice for each value on which they disagree.
- **Spectrometer products**: a UVIS product becomes one observation per window, and a
  VIMS cube becomes a pair of observations, one for each channel.
- **Whole volumes at once**: a Ring-Moon Systems Node metadata index gives the
  observations of every product in a volume, without opening the products.
- **Pointing from SPICE**: the pointing comes from the Cassini C kernels, loaded a month
  of the mission at a time, or from a frame that the caller supplies.

# Installation

`rms-host-cassini` needs Python 3.11 or later. Install it from PyPI with:

```sh
pip install rms-host-cassini
```

The host also needs the Cassini SPICE kernels. The
[setup chapter](https://rms-host-cassini.readthedocs.io/en/latest/user_guide/user_guide_setup.html)
of the user guide describes the kernels and the environment variables that locate them.

# Quick Start

Importing `host_cassini.oops` registers the hosts with `oops.Host`, which can then open
any supported Cassini file:

```python
import oops
import host_cassini.oops

obs = oops.Host.from_file('COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG')
```

Each instrument also has its own host class, for one product or for every product in a
volume:

```python
from host_cassini import CassiniISS, CassiniVIMS

snapshot = CassiniISS.from_file('COISS_2009/data/1484506648_1484573295/N1484506648_1.LBL')
snapshots = list(CassiniISS.from_index('metadata/COISS_2xxx/COISS_2009/COISS_2009_index.lbl'))
(vis, ir) = CassiniVIMS.from_file('v1484505000_1.qub')
```

Replace each path with the location of the file in your copy of the archive, or with
its URL.

# Documentation

The [documentation](https://rms-host-cassini.readthedocs.io) holds the user guide and
the API reference. To build it locally, run `make html` in the `docs` directory after
installing the package with its `docs` extra.

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
