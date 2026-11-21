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
bodies that OOPS defines, or to use only the kernels that existed before a given date,
call the host's ``initialize()`` before opening the first product. This call loads the
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

Data files and indexes
======================

The hosts need no environment variable for the archive itself. Each image file, cube,
PDS3 label, or index is passed by its path or URL, and a remote file is retrieved and
cached when it is read.
