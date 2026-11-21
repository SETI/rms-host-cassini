=====================
Reading an Image File
=====================

Each host's ``from_file()`` makes the observations of one data product. This chapter
describes ISS first, then UVIS and VIMS.

ISS images
==========

:meth:`~host_cassini.oops.iss.CassiniISS.from_file` makes one
:class:`~oops.observation.Snapshot` from one ISS image, raw or calibrated. It accepts
either file of the product, the VICAR image file or its PDS3 label:

.. code-block:: python

   from host_cassini import CassiniISS

   snapshot = CassiniISS.from_file('COISS_2009/data/1484573295_1484664788/N1484573295_1.IMG')
   snapshot = CassiniISS.from_file('COISS_2009/data/1484573295_1484664788/N1484573295_1.LBL')

What is read
------------

What the host reads depends on the file it is given and on the ``astrometry`` option.
An astrometry read is for work that needs only the image's time and geometry, so it
leaves the image array unread.

.. list-table::
   :header-rows: 1

   * - Input
     - Normal read
     - With ``astrometry=True``
   * - VICAR image file
     - The image file alone: its header and its array.
     - The image file's header alone.
   * - PDS3 label
     - The label, and the whole image file beside it.
     - The label alone. The image file need not exist.

The metadata come from whatever was read, as :doc:`user_guide_metadata` describes. The
``data`` subfield holds the image array except in an astrometry read.

The host always finds the image array through the VICAR file's own header, never
through the record number in the label's ``^IMAGE`` pointer. That number is one record
off in every calibrated label and in some raw labels of COISS_1006. Reading the array
through the header keeps it identical from either input.

Finding the image file
----------------------

Given a PDS3 label, the host looks in the label's directory for the image file that the
label's ``^IMAGE`` pointer gives. It tries the name as the label spells it, then in
upper and in lower case, so that a copy of the archive with renamed files still works.
An astrometry read does not look for the file, so its path subfields give the name as
the label spells it.
A path that ends in ``.lbl``, in any case, is read as a PDS3 label, and any other path
as a VICAR image file.

The ``filepath``, ``basename``, ``abspath``, and ``image_url`` subfields always give the
image file, also when the input is the label. ``abspath`` is the local path of the image
file, and only a read that retrieves the image file to read its array gives it: a
normal read of a path, of a parsed PDS3 label, or of a :class:`~vicar.VicarLabel`. For a
remote file, it is the copy in the local cache. An astrometry read, and a read of a
:class:`~vicar.VicarImage` passed in, have no ``abspath``, even when the image file is
on the local disk. Reading a file never creates a directory.

Parsed input
------------

A label or image that has already been parsed can be passed in its place: a
:class:`~pdsparser.Pds3Label` is read as a PDS3 label, and a :class:`~vicar.VicarLabel`
or :class:`~vicar.VicarImage` as the image file.

Parsed content that came from no file, such as a label parsed from text, has no
``filepath``, ``basename``, ``abspath``, or ``image_url``. An astrometry read of it
works. So does a normal read of a :class:`~vicar.VicarImage`, which holds its array. A
normal read of a PDS3 label or a :class:`~vicar.VicarLabel` needs the image file, so it
raises :exc:`~oops.host.HostFormatError`, saying that no file path is known.

Reading the PDS3 label
----------------------

Before the parser reads a label, the host repairs in memory three defects of archive
labels that the parser rejects or misreads. A label without them is read as it is.

- A quoted value that starts on the line after its keyword, as in
  ``N1569827183_4.LBL`` of COISS_2037, is moved up to the keyword's line. The fast
  method rejects such a value, or, if it holds an equals sign, misreads it.
- The ``DESCRIPTION`` of every label in the first version of the calibrated COISS_1xxx
  volumes, and of some in the first version of the calibrated COISS_2xxx volumes, opens
  a quote that it never closes. The quote is closed at the end of the line before the
  next keyword.
- The ``DESCRIPTION`` of some other calibrated labels, as in
  ``N1640662528_3_CALIB.LBL`` of COISS_2059, is split into two quoted strings. The two
  are joined into one.

The host then starts with the parser's fast method and, if that cannot read the label,
tries the strict method and then the loose one. A method given in ``pds3_options``
starts that sequence at the method given. A label that the fast method reads with a
keyword that is not a PDS3 name is read again from the strict method on.
``pds3_options`` can add repairs of its own as ``repairs``, a list of (pattern,
replacement) pairs or a single pair, which are applied after the host's.

Options
-------

These options are particular to the ISS host:

``fast_distortion``
    How the distortion of the field of view is inverted, as
    :doc:`user_guide_products` describes. True, the default, uses a polynomial fitted in
    advance.

``return_all_planets``
    True lists in the ``spice_kernels`` subfield the kernels of every planet, and not
    only those of Jupiter or Saturn, whichever the mission was observing at the time.

``pds3_options``
    Options for the PDS3 label parser, as described above.

``astrometry``
    True for an astrometry read, as described above.

``select``
    Must be None. An image is one observation, so there is nothing to select, and any
    other value raises :exc:`~oops.OopsValueError`.

The standard OOPS overrides, ``target``, ``lightsource``, ``path``, ``frame``,
``cadence``, ``fov``, ``calibrations``, ``timeshift``, ``navigation``, ``tracker``, and
``parallel``, replace what the host would otherwise build. They are listed with
:meth:`~host_cassini.oops.iss.CassiniISS.from_file`. Any other keyword raises
:exc:`TypeError`, so that a misspelled argument is not silently ignored. The arguments
are checked before any file is read.

UVIS products
=============

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_file` accepts the PDS3 label of a UVIS
product, or the label already parsed as a :class:`~pdsparser.Pds3Label`. It reads the
label, and unless ``astrometry`` is True, it reads the data file that the label's
``^QUBE``, ``^TIME_SERIES``, or ``^SPECTRUM`` pointer gives, in the label's directory, in
the label's spelling or else in lower case.

.. code-block:: python

   from host_cassini import CassiniUVIS

   obs = CassiniUVIS.from_file('COUVIS_0009/DATA/D2004_274/EUV2004_274_01_39.LBL')

A QUBE with several windows gives a tuple of observations, one per window, and any other
product gives a single observation. Two options control this:

``enclose``
    True returns a single observation covering every window: its lines and bands are
    the enclosing limits of the windows, and its binning is 1.

``select``
    An index, a slice, or a tuple of indices and slices that picks from the
    observations, with a single observation treated as a tuple of one. ``select=1``
    returns only the observation of the second window. A ``select`` of a type that it
    does not accept, or a name, raises an error before the label is read.

The ``label_dict`` subfield holds the PDS3 label and ``dict`` the same keywords without
their missing values, as :doc:`user_guide_metadata` describes. ``filepath``,
``basename``, and ``image_url`` give the label, not the data file; whether they should
give the data file instead is an open question
(`rms-host-cassini#32 <https://github.com/SETI/rms-host-cassini/issues/32>`_).
``abspath`` is the local path of the data file, given only by a normal read, which
retrieves it.
``pds3_options``, ``astrometry``, and the standard overrides work as they do for ISS,
and the parser falls back from the fast method to the strict and the loose ones in the
same way. Parsed content that came from no file works as it does for ISS: an astrometry
read has no file subfields, and a normal read raises :exc:`~oops.host.HostFormatError`.
Any other keyword raises :exc:`TypeError`.

VIMS cubes
==========

:meth:`~host_cassini.oops.vims.CassiniVIMS.from_file` accepts a VIMS cube in ISIS format,
its detached PDS3 label, or the label already parsed as a :class:`~pdsparser.Pds3Label`.
Given the cube, it reads the cube's own ISIS label and, unless ``astrometry`` is True,
the cube's data. Given the PDS3 label, it reads the label and, unless ``astrometry`` is
True, the cube that the label's ``^QUBE`` pointer gives, in the label's directory.

.. code-block:: python

   from host_cassini import CassiniVIMS

   (vis, ir) = CassiniVIMS.from_file('v1484505000_1.qub')
   ir = CassiniVIMS.from_file('v1484505000_1.qub', select='IR')

The result is the pair (vis, ir), with None for a channel that was off. ``select`` picks
from the pair by an index, a slice, a tuple of these, or the channel's name, ``'VIS'`` or
``'IR'`` in any case. A ``select`` of a type that it does not accept, or another name,
raises an error before the file is read.

The label that was read is ``label_dict`` for a detached PDS3 label and ``header_dict``
for the cube's own ISIS label, and ``dict`` holds its keywords without their missing
values, as :doc:`user_guide_metadata` describes. ``filepath``, ``basename``, and
``image_url`` give the file that was passed in, which is the label when the label is
given; whether they should give the cube instead is an open question
(`rms-host-cassini#32 <https://github.com/SETI/rms-host-cassini/issues/32>`_).
``abspath`` is the local path of the cube, given only by a normal read, which retrieves
it.
``pds3_options``, ``astrometry``, parsed content, and the standard overrides work as they
do for UVIS. Override ``fov`` only when ``select`` picks a single channel, because the
two channels have different fields of view. Any other keyword raises :exc:`TypeError`.

Some VIMS labels, such as ``v1481842120_1_048.lbl`` of COVIMS_0005, have a comment that
opens on the line of a value and closes on the next line:

.. code-block:: text

   SNAPSHOT_MODE_FLAG               = OFF/* IR high resolution,
                                        IR low resolution, Visible (degree K) */

A PDS3 comment ends on the line where it starts, so no method of the parser can read
such a label. The host joins the two lines of the comment in memory before parsing.

Reading a label through oops.Host
=================================

:meth:`oops.host.Host.from_file` recognizes an ISS image file or PDS3 label, a UVIS
label, or a VIMS cube or label, and passes the file's path, with its other arguments, to
the host's ``from_file()``, so the two give the same observations. To recognize a label,
it tries the parser's fast, strict, and loose methods in turn, as the hosts do. A label
that none of them can read is repaired by each host in turn, as described above, and
goes to the host whose repairs make it readable and that recognizes it: the calibrated
ISS labels whose ``DESCRIPTION`` is split in two, such as ``N1640662528_3_CALIB.LBL`` of
COISS_2059, go to the ISS host, and the VIMS labels with a comment that spans two lines
to the VIMS host. A file or label that more than one registered host claims raises
``oops.host.AmbiguousHostError``, which names the hosts.
