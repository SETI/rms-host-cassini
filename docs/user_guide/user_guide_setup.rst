=====
Setup
=====

Installation
============

The package needs Python 3.11 or later and installs from PyPI:

.. code-block:: sh

   pip install rms-host-cassini

It brings in OOPS (``rms-oops``) and the readers that the hosts use for PDS3 labels,
VICAR files, and PDS3 index tables.

SPICE kernels
=============

The geometry of every observation comes from the Cassini SPICE kernels. The hosts read
them through ``spicedb``, which is distributed with OOPS and finds them with two
environment variables:

``SPICE_PATH``
    The directory that holds the kernels, as a local path or a URL. If it is not set,
    ``$OOPS_RESOURCES/SPICE`` is used.

``SPICE_SQLITE_DB_NAME``
    The SQLite database that lists the kernels. If it is not set, ``SPICE.db`` in the
    ``SPICE_PATH`` directory is used.

The kernels and the database are the ones that OOPS itself uses. The
`rms-oops README <https://github.com/SETI/rms-oops#environment-variables>`_ describes
these variables and the ``SPICE`` directory of ``$OOPS_RESOURCES`` that holds them.
The Ring-Moon Systems Node publishes these resources, the kernels and ``SPICE.db``
included, at ``https://storage.googleapis.com/rms-node-oops-resources``, a URL that
``OOPS_RESOURCES`` can name directly.

The hosts choose the kernels through the database, by kernel set: ``CAS-SPK-*`` for the
spacecraft's trajectory, ``CAS-CK-*`` for its attitude, and ``CAS-CK-GAPFILL`` for the
C kernels that fill gaps in the attitude record. The database must hold these sets, and
the instrument and frame kernels of each instrument.

The C and SP kernels are loaded only when an observation needs them. The hosts divide
the mission, from 1997-10-01 to 2017-10-01, into 240 periods of equal length, about a
month each, and the first observation that falls in a period loads that period's
kernels, with 12 hours to spare on either side. An observation with a ``cadence``
override also loads the kernels of the periods that the override's times fall in.

Registering the hosts
=====================

The host classes import lazily. ``import host_cassini`` gives access to
:class:`~host_cassini.oops.iss.CassiniISS`, :class:`~host_cassini.oops.uvis.CassiniUVIS`,
and :class:`~host_cassini.oops.vims.CassiniVIMS` without importing OOPS until a class is
first used. Importing ``host_cassini.oops`` registers all three hosts with
:class:`~oops.host.Host`, so that :meth:`oops.host.Host.from_file` recognizes a Cassini
ISS, UVIS, or VIMS file:

.. code-block:: python

   import oops
   import host_cassini.oops

   obs = oops.Host.from_file('COISS_2009/data/1484506648_1484573295/N1484506648_1.IMG')

Initialization
==============

The first call to a host's ``from_file()`` or ``from_index()`` loads the instrument
kernel and defines the instrument's fields of view. To choose the kernel sets, the
bodies that OOPS defines, or to use the kernels released by a given date, call the
host's ``initialize()`` before opening the first product. This call loads the
predicted C kernels in place of the reconstructed ones, and defines Saturn, planet 6,
and its moons, and no other planet:

.. code-block:: python

   from host_cassini import CassiniISS

   CassiniISS.initialize(ck='predicted', planets=(6,))

The options are listed with :meth:`~host_cassini.oops.iss.CassiniISS.initialize`.
``ck`` and ``spk`` choose the reconstructed or predicted kernels, or ``'none'`` to leave
the loading of those kernels to the caller; ``gapfill`` adds the gap-filling C kernels;
and ``mst_pck`` adds the updated rotation models of some small moons. The three hosts
share one set of mission kernels, so the first host to be initialized makes these
choices for all three. Later calls to ``initialize()`` have no effect.

``asof`` takes a date, such as ``'2008-01-01'``, and each kernel is then chosen from
those released no later than the start (00:00) of that date. Where every kernel that a
query could choose was released after the start of the date, the query silently uses
the kernel it would choose without a date. For example, ``asof='2008-01-01'`` still
loads the newest spacecraft clock kernel, released in 2018, because all three clock
kernels in the kernel database were released in 2017 or 2018. That happens only when
the whole query comes back empty: a set of SP or C kernels that covers the mission
keeps just the kernels released by the start of the date, so a date inside the mission
can leave the later part of the mission uncovered. With ``asof='2008-01-01'``, for
example, the reconstructed SP kernels end on 2007-09-19 and the reconstructed C kernels
on 2007-12-28. The C kernels never fall back to the newest, so for any date up to and
including 2003-11-04, ``initialize()`` raises :exc:`ValueError`. When Saturn is among
the planets, as it is by default, any date from 2003-11-05 through 2004-04-26 makes
``initialize()`` raise :exc:`RuntimeError`, because the frames kernel of Saturn's small
moons chosen for those dates has a malformed line that SPICE rejects. The date decides
the leap seconds too: with ``asof='2008-01-01'``, the leap seconds kernel released on
2005-08-03 is used, which lacks the leap second at the end of 2008. Two sets of kernels
are loaded without the date: the gap-filling C kernels, and each host's own instrument
kernel, from which ISS and UVIS read their fields of view.

Data files and indexes
======================

The hosts need no environment variable for the archive itself. Each image file, cube,
PDS3 label, or index is passed by its path or URL, and a remote file is retrieved and
cached when it is read.
