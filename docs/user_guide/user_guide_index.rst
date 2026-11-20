====================
Reading a PDS3 Index
====================

Each host's ``from_index()`` makes the observations of every row of a PDS3 index. It
reads only the index tables and opens none of the products, which makes it much faster
than opening a volume's products one at a time. Its observations have no ``data``. This
chapter describes ISS first, then UVIS and VIMS.

``from_index()`` returns an iterator, which gives the observations row by row, in the
order of the index. The call itself checks the arguments and reads the PDS3 labels of
the index and its supplement, so a wrong argument or a label that cannot be read raises
an error at once. The rows of the two tables are read as the iteration proceeds, a few
at a time, and each row's observation is built when the iteration reaches that row. The
memory used stays small however long the index is, and the first observation comes
without waiting for the rest of the table. An error in a row is raised when the
iteration reaches it, after the rows before it, and the iteration ends with that error.
``filter`` can leave such a row out. To build every row at once, pass the iterator to
``list()``.

The hosts read the indexes of the Ring-Moon Systems Node holdings, in their ``metadata``
tree, such as ``metadata/COISS_2xxx/COISS_2009/COISS_2009_index.lbl``. A volume's own
index, in the volume's ``index`` directory, is not supported: given one, ``from_index()``
raises :exc:`~oops.host.HostIndexError` at once, before it reads anything, and the
message gives an example of a metadata index to use instead.

ISS indexes
===========

:meth:`~host_cassini.oops.iss.CassiniISS.from_index` makes one
:class:`~oops.observation.Snapshot` for each row of an ISS index:

.. code-block:: python

   from host_cassini import CassiniISS

   index = 'metadata/COISS_2xxx/COISS_2009/COISS_2009_index.lbl'
   for snapshot in CassiniISS.from_index(index):
       print(snapshot.basename, snapshot.time)

   snapshots = list(CassiniISS.from_index(index))

The index and its supplement
----------------------------

The Ring-Moon Systems Node holdings keep an index of each COISS volume in their
``metadata`` tree, such as ``COISS_2009_index.lbl``, with one row per image, beside a
cumulative index of the whole series, such as ``COISS_2999_index.lbl``. The host reads
either. It recognizes an ISS index by ``COISS`` in its path, or failing that by the
instrument that its first row gives.

The archive has no supplemental index for ISS, so the host needs none. The
``supplement`` argument chooses one if it exists:

- By default, the host reads the file beside the index whose name has
  ``_supplemental_index`` in place of ``_index``, in upper or lower case, if there is
  one.
- A path or URL reads that file instead, and the file must exist. Its rows must list
  the same images as the index, in the same order, and where both tables give a column,
  the index's value is kept. When both tables have ``VOLUME_ID`` and
  ``FILE_SPECIFICATION_NAME``, a supplement row whose image is not the one in the same
  row of the index raises :exc:`~oops.host.HostIndexError` when the iteration reaches
  it, and so does a supplement that ends first or has rows left over.
- None reads the index alone.

Other arguments
---------------

``label``
    The index's PDS3 label, if it has already been parsed.

``row_dicts``
    The rows themselves, one dictionary per row, if the tables have already been read,
    in a list or any other iterable. The host then uses these rows in place of the
    tables, ``supplement`` and ``label`` are ignored, and each row is filtered and built
    when the iteration reaches it.

``filter``
    A dictionary of column names and the values to keep. A row is kept when each column
    equals its value, is one of a set of values, or falls in a range given as a tuple of
    two values. A column that neither table has raises :exc:`~oops.OopsValueError` when
    the method is called, and its message gives the column. A range that cannot be
    compared with a row's value raises :exc:`~oops.OopsTypeError` when the iteration
    reaches that row.

``select``
    Must be None. Each row is one image, so there is nothing to select, and any other
    value raises :exc:`~oops.OopsValueError`.

``fast_distortion``, ``return_all_planets``, ``pds3_options``, and the standard overrides
    The same as for :meth:`~host_cassini.oops.iss.CassiniISS.from_file`, applied to every
    row. Override ``fov`` only when every row is of the same camera, because the two
    cameras have different fields of view.

Any other keyword raises :exc:`~oops.OopsTypeError`, so that a misspelled argument is
not silently ignored.

What one row gives
------------------

Each row gives a :class:`~oops.observation.Snapshot` with no ``data``. Its ``dict``
subfield holds the row, in Python types, with the padding stripped from its text and a
list for each array column. The index tables write fill values for values that an image
does not have, and these are left out:

- the text that the archive writes for no value, ``UNK``, ``UNKNOWN``, ``N/A``,
  ``NULL``, ``---``, or an empty string;
- 1e32 and -1e32, in every column;
- -999 in the temperature, data rate, and compression columns, where the index labels
  declare it a null value or it cannot be a real value; elsewhere, as in a position, it
  is kept;
- 65534 in ``COMMAND_SEQUENCE_NUMBER``, ``ELECTRONICS_BIAS``, ``EXPECTED_PACKETS``,
  ``ORDER_NUMBER``, ``PARALLEL_CLOCK_VOLTAGE_INDEX``, ``RECEIVED_PACKETS``,
  ``SEQUENCE_NUMBER``, and ``VALID_MAXIMUM``, and -2147483648 in ``INST_CMPRS_PARAM``
  and ``MISSING_LINES``, where the image labels give ``UNK`` or ``N/A``;
- an ``EMISSION_ANGLE``, ``INCIDENCE_ANGLE``, ``CENTER_LATITUDE``, or
  ``CENTER_LONGITUDE`` of 0 in a row that has no target geometry, that is, no
  ``PHASE_ANGLE``. Until the index is corrected, the tables write 0 there for a missing
  angle. A 0 in a row that has target geometry is kept.

A missing element of an array column is None. The index labels in the ``metadata``
tree declare 0 the null value of ``BIAS_STRIP_MEAN`` and ``DARK_STRIP_MEAN``, but the
image labels give 0 as a real value there, so the host does not use the null values that
the index labels declare.

The tables drop the trailing zeros of the clock counts, so the host writes each count
with three digits after the point, as the image labels do: ``1473199010.120`` for the
table's ``1473199010.12``. A count that the table already writes with three digits is
left as it is. The clock counts of the early cruise images are computed as
:doc:`user_guide_metadata` describes.

The subfields are those of an image read from its label (see
:doc:`user_guide_metadata`), except ``data``, ``label_dict``, and ``header_dict``, and
except that each item the row does not give leaves out its subfield. If the C kernels
hold no pointing for an image at the middle of its exposure, its row still gives a
Snapshot, whose frame raises :exc:`oops.OopsSpiceError` when it is evaluated. The
message is the one that :meth:`~host_cassini.oops.iss.CassiniISS.from_file` gives for
the same image (see :doc:`user_guide_errors`).

File paths
----------

A row's ``FILE_SPECIFICATION_NAME`` is the path of the image file within the volume
that its ``VOLUME_ID`` gives. ``filepath``, ``basename``, and ``image_url`` give that file
as a full path or URL, formed from the location of the index:

- For an index in the ``metadata`` tree of the Ring-Moon Systems Node holdings, such as
  ``.../metadata/COISS_2xxx/COISS_2009/COISS_2009_index.lbl``, the volume is in the
  matching ``volumes`` tree, ``.../volumes/COISS_2xxx/COISS_2009``. A cumulative index
  lists images of many volumes this way.
- For an index anywhere else, the volume is the directory of that name beside the
  index, as in a copy of the holdings that keeps each index next to its volumes.

``abspath`` is present only when the image file is on the local file system, or is a
remote file that is already in the local cache. Reading an index never creates a
directory, even when the volume is not where the index places it.

UVIS indexes
============

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_index` reads a COUVIS index together
with its supplemental index, such as ``COUVIS_0009_index.lbl`` and
``COUVIS_0009_supplemental_index.lbl``. The supplement is required, because the file path
and most of each observation's values come from it:

- By default, the host reads the supplement beside the index, which must exist.
- A path or URL reads that file instead.
- None raises :exc:`~oops.host.HostIndexError`, unless ``row_dicts`` is given. Each row
  given in ``row_dicts`` must include the supplement's columns, or it raises
  :exc:`~oops.host.HostIndexError` when the iteration reaches it.

The two tables must list the same products, and until the supplements are corrected,
54 of the 61 supplements list them in another order than their indexes. The host
therefore merges each index row with the supplement row for the same product: the index
gives it by ``FILE_NAME``, as in ``/COUVIS_0001/DATA/D1999_007/HSP1999_007_16_53.LBL``,
and the supplement by ``VOLUME_ID`` and ``FILE_SPECIFICATION_NAME``, as in
``COUVIS_0001`` and ``DATA/D1999_007/HSP1999_007_16_53.LBL``. The paths are compared
without regard to case or a leading slash. This gives the same rows when the two tables
are in the same order.

The supplement is read only as far ahead as the product of the current index row. The
supplement rows read on the way are held until their own index rows are reached, so if
no row is more than 100 places from where the index puts it, no more than 100 rows are
held. The host raises :exc:`~oops.host.HostIndexError`:

- at an index row whose product the supplement does not list, which is known only once
  the whole supplement has been read;
- at a supplement row that gives a product a second time while its first row is still
  held;
- after the last index row, if the supplement lists a product that the index does not,
  or lists one more often; the error gives the first few such products and both tables.

For each pair of rows, the index's ``FILE_NAME`` must also contain the supplement's
``VOLUME_ID`` and ``FILE_SPECIFICATION_NAME``.

Each row gives the observation that
:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_file` would give for the product with
``astrometry=True``, built from the row's columns:

- ``DATA_OBJECT_TYPE`` gives the kind of product: ``QUBE``, ``TIME_SERIES``, or
  ``SPECTRUM``.
- For a ``QUBE`` or ``SPECTRUM``, ``MINIMUM_BAND_NUMBER``, ``MAXIMUM_BAND_NUMBER``,
  ``BAND_BINNING_FACTOR``, ``WINDOW_MINIMUM_LINE_NUMBER``,
  ``WINDOW_MAXIMUM_LINE_NUMBER``, and ``LINE_BINNING_FACTOR`` give the window and the
  binning. A row gives a single window, so for a ``QUBE`` with several windows it gives
  the window that encloses them all, binned by the smallest of their binnings. For the
  product's label, ``enclose=True`` gives the same window unbinned.
- For a ``QUBE`` or ``TIME_SERIES``, ``LINE_SAMPLES`` gives the number of samples.
- For an HSP time series, ``INTEGRATION_DURATION`` gives the sampling interval in
  milliseconds. For an HDAC time series, whose ``INTEGRATION_DURATION`` is a null value,
  ``DWELL_TIME`` gives it as a number of 125-millisecond integrations. This matches the
  product labels of nearly all HDAC products; a few in COUVIS_0031 and COUVIS_0032 give
  a ``DWELL_TIME`` of 72 and an interval of 125 milliseconds.

A row that lacks one of these columns, or gives a sampling interval that is not
positive, raises :exc:`~oops.host.HostFormatError` when the iteration reaches it. ``select`` applies to
each row as it does for the product's label, and a ``select`` of a type that it does not
accept, or a name, raises an error when the method is called.

The ``dict`` subfield holds the row, with the ``QUBE``, ``TIME_SERIES``, or ``SPECTRUM``
object built from it, in Python types and without the table's masks. These are missing
and left out: each value that the table's label declares a null constant, such as -99 in
the supplement's angles and -999 in the index's geometry columns; the text that the
archive writes for no value; and 1e32 and -1e32. ``label_target``, ``observation_id``,
and the clock counts follow from the row as they do from the label. If the C kernels
hold no pointing for an observation, its frame raises :exc:`oops.OopsSpiceError` when it
is evaluated.

The ``filepath`` and ``image_url`` subfields give the product's PDS3 label, from the
supplement's ``FILE_SPECIFICATION_NAME``, as a full path or URL formed as for ISS.
Whether they should give the data file instead is an open question
(`rms-host-cassini#32 <https://github.com/SETI/rms-host-cassini/issues/32>`_).
``label``, ``row_dicts``, ``filter``, ``pds3_options``, and the standard overrides work
as they do for ISS, and any other keyword raises :exc:`~oops.OopsTypeError`.

VIMS indexes
============

:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` reads a COVIMS index together
with its supplemental index, such as ``COVIMS_0009_index.lbl`` and
``COVIMS_0009_supplemental_index.lbl``. The supplement is required, and the
``supplement`` argument works as it does for UVIS. Until the supplements are corrected,
25 of the 95 supplements, that of the cumulative ``COVIMS_0999`` among them, list the
cubes in another order than their indexes, so the host merges each index row with the
supplement row for the same cube, as it does for UVIS. The index gives the cube's data
file by ``VOLUME_ID``, ``PATH_NAME``, and ``FILE_NAME``, as in
``/data/1999010T054026_1999010T060958`` and ``v1294638283_1.qub``, and the supplement
gives its label by ``VOLUME_ID`` and ``FILE_SPECIFICATION_NAME``, as in
``data/1999010T054026_1999010T060958/v1294638283_1.lbl``. The paths are compared without
regard to case, a leading slash, or the extension. The supplement is read only as far
ahead as it must be, and the errors are those described for UVIS. In ``COVIMS_0999``,
with 464,752 rows, no more than about 900 supplement rows are held at a time. For each
pair of rows, the two tables must also give the same ``VOLUME_ID``, and the index's
``FILE_NAME`` and the supplement's ``FILE_SPECIFICATION_NAME`` must give the same file
name apart from its extension.

Each row gives the (vis, ir) pair of observations that
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_file` would give for the cube with
``astrometry=True``, with None for a channel that was off, and ``select`` picks from
each pair by index or by name. A row does not hold everything that the cube's label
does, so the observations are built from these columns:

- The shape of the cube is the index's swath with a nominal 352 bands, and a ``POINT``
  or ``OCCULTATION`` cube is taken to fill the 64-line buffer, as nearly every one does:
  1 by 64 pixels for ``POINT`` and 64 by 64 for ``OCCULTATION``.
- A channel whose recorded exposure is missing, zero, or negative is treated as off,
  because only the cube's label holds its real exposure.
- A supplement with the columns ``CORE_SAMPLES``, ``CORE_BANDS``, and ``CORE_LINES``,
  ``IR_EXPOSURE_DURATION`` and ``VIS_EXPOSURE_DURATION``, or
  ``BACKPLANE_SCLK_START_COUNT`` would give the cube's actual shape, its exposures, or
  the clock count of the first sample of a cube with time backplanes, which then
  replaces ``START_TIME`` and makes ``time_from_sclk`` True. No supplement in the
  archive has these columns yet.

The ``dict`` subfield holds the row, cleaned as for UVIS, with a ``SPECTRAL_QUBE`` object
whose ``CORE_ITEMS`` give the shape that the observations use. The ``filepath`` and
``image_url`` subfields give the cube's PDS3 label, from the supplement's
``FILE_SPECIFICATION_NAME``, as a full path or URL formed as for ISS; whether they should
give the cube instead is the same open question as for UVIS. The other arguments work as
they do for UVIS.

Reading an index through oops.Host
==================================

:meth:`oops.host.Host.from_index` asks each host about the index's PDS3 label first. ISS
claims a label whose path holds ``COISS``, UVIS one whose path holds ``COUVIS``, and VIMS
one that gives the instrument ``VIMS`` or whose path holds ``COVIMS``; ISS and UVIS turn
down a label that gives another instrument or the path of another instrument's volume.
Only when no host claims the label is the first row read, once, to find the host from
its instrument, ``DATA_SET_ID``, or ``VOLUME_ID``. The index then goes, with all of the
other arguments, to the host's ``from_index()``, which reads and pairs the rows itself,
so the two give the same observations, also when ``supplement`` gives a path.

A volume's own ISS or UVIS index raises the same :exc:`~oops.host.HostIndexError`
either way. The label of a volume's own VIMS index, such as
``volumes/COVIMS_0xxx/COVIMS_0040/index/index.lbl``, holds a line of plain text that is
not valid PDS3, so :meth:`oops.host.Host.from_index`, which parses it to find the host,
raises :exc:`~oops.host.PdsHostError`, while
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` raises its
:exc:`~oops.host.HostIndexError` without reading the label.
