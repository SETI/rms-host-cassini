====================
Reading a PDS3 Index
====================

Each host's ``from_index()`` makes the observations of every row of a PDS3 index. It
reads only the index tables and opens none of the products, which makes it much faster
than opening a volume's products one at a time. Its observations have no ``data``. This
chapter describes ISS first, then UVIS and VIMS.

``from_index()`` returns an iterator, which gives the observations row by row, in the
order of the index. The call itself checks the arguments and reads the index, so a wrong
argument or an index that cannot be read raises an error at once. Each row's observation
is built only when the iteration reaches that row, so an error in a row is raised there,
after the rows before it. To build every row at once, pass the iterator to ``list()``.

ISS indexes
===========

:meth:`~host_cassini.oops.iss.CassiniISS.from_index` makes one
:class:`~oops.observation.Snapshot` for each row of an ISS index:

.. code-block:: python

   from host_cassini import CassiniISS

   for snapshot in CassiniISS.from_index('COISS_2009/index/index.lbl'):
       print(snapshot.basename, snapshot.time)

   snapshots = list(CassiniISS.from_index('COISS_2009/index/index.lbl'))

The index and its supplement
----------------------------

Each COISS volume has a PDS3 index, ``index/index.lbl``, with one row per image. The
Ring-Moon Systems Node holdings keep a copy of it in their ``metadata`` tree, such as
``COISS_2009_index.lbl``, beside a cumulative index of the whole series, such as
``COISS_2999_index.lbl``. The host reads any of these. It recognizes an ISS index by
``COISS`` in its path, or failing that by the instrument that its rows give.

The archive has no supplemental index for ISS, so the host needs none. The
``supplement`` argument chooses one if it exists:

- By default, the host reads the file beside the index whose name has
  ``_supplemental_index`` in place of ``_index``, in upper or lower case, if there is
  one.
- A path or URL reads that file instead, and the file must exist. Its rows must list
  the same images as the index, in the same order or paired by ``VOLUME_ID`` and
  ``FILE_SPECIFICATION_NAME`` as described for UVIS below, and where both tables give a
  column, the index's value is kept.
- None reads no supplement.

Other arguments
---------------

``label``
    The index's PDS3 label, if it has already been parsed.

``row_dicts``
    The rows themselves, one dictionary per row, if the tables have already been read.
    The host then reads no table, and ``supplement`` and ``label`` are ignored.

``filter``
    A dictionary of column names and the values to keep. A row is kept when each column
    equals its value, is one of a set of values, or falls in a range given as a tuple of
    two values.

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

- the PDS3 null constants, such as ``UNK`` and ``NULL``, and the fill value -999;
- 1e32 and -1e32, in every column;
- 65534 in ``COMMAND_SEQUENCE_NUMBER``, ``ELECTRONICS_BIAS``, ``EXPECTED_PACKETS``,
  ``ORDER_NUMBER``, ``PARALLEL_CLOCK_VOLTAGE_INDEX``, ``RECEIVED_PACKETS``,
  ``SEQUENCE_NUMBER``, and ``VALID_MAXIMUM``, and -2147483648 in ``INST_CMPRS_PARAM``
  and ``MISSING_LINES``, where the image labels give ``UNK`` or ``N/A``;
- an ``EMISSION_ANGLE``, ``INCIDENCE_ANGLE``, ``CENTER_LATITUDE``, or
  ``CENTER_LONGITUDE`` of 0, which the tables write for an image that has no target
  geometry.

A missing element of an array column is None. The index labels in the ``metadata``
tree declare 0 the null value of ``BIAS_STRIP_MEAN`` and ``DARK_STRIP_MEAN``, but the
image labels give 0 as a real value there, so the host does not use the null values that
the index labels declare.

The tables drop the trailing zeros of the clock counts, so the host writes each count
with three digits after the point, as the image labels do: ``1473199010.120`` for the
table's ``1473199010.12``. The clock counts of the early cruise images are computed as
:doc:`user_guide_metadata` describes.

The subfields are those of an image read from its label (see
:doc:`user_guide_metadata`), except ``data``, ``label_dict``, and ``header_dict``, and
except that each item the row does not give leaves out its subfield. The host does not
check the C kernels for the pointing of an index row. If the kernels hold no pointing
for an image, its row still gives a Snapshot, and the error appears only when the
Snapshot's frame is evaluated.

File paths
----------

A row's ``FILE_SPECIFICATION_NAME`` is the path of the image file within its volume.
``filepath``, ``basename``, and ``image_url`` give that file as a full path or URL,
formed from the location of the index:

- For an index in a volume's ``index`` directory, in either case, the volume is the
  directory above it.
- For an index in the ``metadata`` tree of the Ring-Moon Systems Node holdings, such as
  ``.../metadata/COISS_2xxx/COISS_2009/COISS_2009_index.lbl``, the volume is the row's
  ``VOLUME_ID`` in the matching ``volumes`` tree, ``.../volumes/COISS_2xxx/COISS_2009``.
  A cumulative index lists images of many volumes this way.
- For an index anywhere else, the volume is the index's own directory.

``abspath`` is present only when the image file is local or already in the local cache.

UVIS indexes
============

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_index` reads a COUVIS index together
with its supplemental index, such as ``COUVIS_0009_index.lbl`` and
``COUVIS_0009_supplemental_index.lbl``. The supplement is required, because the file path
and most of each observation's values come from it:

- By default, the host reads the supplement beside the index, which must exist.
- A path or URL reads that file instead.
- None raises :exc:`~oops.host.HostIndexError`, unless ``row_dicts`` is given. Rows
  given in ``row_dicts`` must include the supplement's columns.

The two tables must list the same products, but in most volumes the supplement lists
them in another order. When the two tables list the products in the same order, each
index row is merged with the supplement row in the same position. Otherwise, each index
row is merged with the supplement row for the same product: the index names it by
``FILE_NAME``, as in ``/COUVIS_0001/DATA/D1999_007/HSP1999_007_16_53.LBL``, and the
supplement by ``VOLUME_ID`` and ``FILE_SPECIFICATION_NAME``, as in ``COUVIS_0001`` and
``DATA/D1999_007/HSP1999_007_16_53.LBL``. The paths are compared without regard to
case or a leading slash. The host raises :exc:`~oops.host.HostIndexError` if a product
appears in only one of the tables, or if, when the orders differ, either table lists a
product twice; the error names the first few such products and both tables. For each
pair of rows, the index's ``FILE_NAME`` must also contain the supplement's ``VOLUME_ID``
and ``FILE_SPECIFICATION_NAME``.

Each row gives the observation that
:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_file` would give for the product with
``astrometry=True``, built from the row's columns:

- ``DATA_OBJECT_TYPE`` gives the kind of product: ``QUBE``, ``TIME_SERIES``, or
  ``SPECTRUM``.
- For a ``QUBE`` or ``SPECTRUM``, ``MINIMUM_BAND_NUMBER``, ``MAXIMUM_BAND_NUMBER``,
  ``BAND_BINNING_FACTOR``, ``WINDOW_MINIMUM_LINE_NUMBER``,
  ``WINDOW_MAXIMUM_LINE_NUMBER``, and ``LINE_BINNING_FACTOR`` give the window and the
  binning. A row gives a single window, so for a ``QUBE`` with several windows it gives
  the window that encloses them all, with the smallest binning, as ``enclose=True``
  does for the product's label.
- For a ``QUBE`` or ``TIME_SERIES``, ``LINE_SAMPLES`` gives the number of samples.
- For an HSP time series, ``INTEGRATION_DURATION`` gives the sampling interval in
  milliseconds. For an HDAC time series, whose ``INTEGRATION_DURATION`` is a null value,
  ``DWELL_TIME`` gives it as a number of 125-millisecond integrations.

A row that lacks one of these columns, or gives no positive sampling interval, raises
:exc:`~oops.host.HostFormatError` when the iteration reaches it. ``select`` applies to
each row as it does for the product's label. The
``filepath`` and ``image_url`` subfields give the product's PDS3 label, from the
supplement's ``FILE_SPECIFICATION_NAME``, as a full path or URL formed as for ISS.
``label``, ``row_dicts``, ``filter``, ``pds3_options``, and the standard overrides work
as they do for ISS, and any other keyword raises :exc:`~oops.OopsTypeError`.

VIMS indexes
============

:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` reads a COVIMS index together
with its supplemental index, such as ``COVIMS_0009_index.lbl`` and
``COVIMS_0009_supplemental_index.lbl``. The supplement is required, and the
``supplement`` argument works as it does for UVIS. The supplements of 24 volumes and of
``COVIMS_0999`` list the cubes in another order than their indexes. When the orders
differ, each index row is merged with the supplement row for the same cube: the index
names the cube's data file by ``VOLUME_ID``, ``PATH_NAME``, and ``FILE_NAME``, as in
``/data/1999010T054026_1999010T060958`` and ``v1294638283_1.qub``, and the supplement
names its label by ``VOLUME_ID`` and ``FILE_SPECIFICATION_NAME``, as in
``data/1999010T054026_1999010T060958/v1294638283_1.lbl``. The paths are compared without
regard to case, a leading slash, or the extension. The errors are those described for
UVIS. For each pair of rows, the two tables must also give the same ``VOLUME_ID``, and
the index's ``FILE_NAME`` and the supplement's ``FILE_SPECIFICATION_NAME`` must give the
same file name apart from its extension.

Each row gives the (vis, ir) pair of observations that
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_file` would give for the cube with
``astrometry=True``, with None for a channel that was off, and ``select`` picks from
each pair by index or by name. A row does not hold everything that the cube's label
does, so the observations are built from these columns:

- The supplement's ``CORE_SAMPLES``, ``CORE_BANDS``, and ``CORE_LINES`` give the shape
  of the cube, and its ``IR_EXPOSURE_DURATION`` and ``VIS_EXPOSURE_DURATION`` give the
  exposures, when it has those columns.
- Without them, the shape is the index's swath with a nominal 352 bands, and a ``POINT``
  or ``OCCULTATION`` cube is taken to fill the 64-line buffer, as nearly every one does:
  1 by 64 pixels for ``POINT`` and 64 by 64 for ``OCCULTATION``.
- A channel whose recorded exposure is zero or negative is treated as off, because only
  the cube's label holds its real exposure.
- A nonblank ``BACKPLANE_SCLK_START_COUNT``, the clock count of the first sample of a
  cube with time backplanes, replaces ``START_TIME`` as the start of the observations.

The ``filepath`` and ``image_url`` subfields give the cube's PDS3 label, from the
supplement's ``FILE_SPECIFICATION_NAME``, as a full path or URL formed as for ISS. The
other arguments work as they do for UVIS.
