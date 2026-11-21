======
Errors
======

The hosts report each kind of failure with one exception class. The classes come from
OOPS, so code that reads data from several missions can catch them the same way.

Reading an ISS image file
=========================

:meth:`~host_cassini.oops.iss.CassiniISS.from_file` raises:

:exc:`oops.OopsSpiceError`
    The C kernels lack the camera's pointing at the time at which the observation
    evaluates its frame: the middle of the exposure, unless a ``cadence`` override, a
    ``tracker``, or a ``timeshift`` moves that time (see :doc:`user_guide_geometry`).
    The message reads "no Cassini ISS NAC pointing at <time> for <image file>: <code>",
    ending with the SPICE error code, such as ``SPICE(CKINSUFFDATA)``, and the
    exception's ``spice_code`` and ``filepath`` attributes hold the code and the image
    file. It is never raised for missing pointing when the caller supplies the pointing
    with the ``frame`` override. It is also raised when the spacecraft clock kernel
    cannot convert the clock counts of an early cruise image.

:exc:`~oops.host.HostFormatError`
    No label that was read gives ``START_TIME``, ``EXPOSURE_DURATION``,
    ``INSTRUMENT_MODE_ID``, or ``INSTRUMENT_NAME``, or ``START_TIME`` or
    ``INSTRUMENT_MODE_ID`` is not valid.
    The message gives the item and the image file. It is also raised when a normal read
    needs the image file of parsed content that came from no file.

:exc:`TypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.OopsValueError`
    ``select`` is not None.

:exc:`~oops.host.PdsHostError`
    No method of the parser can read the PDS3 label, even after the repairs that
    :doc:`user_guide_files` describes, or the label's ``^IMAGE`` pointer does not give
    the name of a file.
    The message gives the label and the error of each method of the parser that was
    tried.

:exc:`~oops.host.VicarHostError`
    The VICAR header cannot be read, or the image file ends before its array. The
    message gives the image file.

:exc:`~oops.host.WrongHostError`
    The file, or the parsed label or image passed in, is not of a Cassini ISS image.

:exc:`FileNotFoundError`
    The file does not exist, or, in a read that needs it, neither does the image file
    that a PDS3 label points to, in the label's spelling, in upper case, or in lower case.

:exc:`OSError`
    A file cannot be read.

Reading an ISS index
====================

:meth:`~host_cassini.oops.iss.CassiniISS.from_index` raises the errors listed below.
When the rows are given in ``row_dicts``, the call reads only the index's PDS3 label, for
the null values it declares, and not even that if ``label`` is given. The errors that
come from the supplement, from a table, or from an index of another instrument are then
never raised: the supplement need not exist, the index label is not checked to describe
a table that can be read, and the index is not checked to be a Cassini ISS index. The checks of ``select`` and
``filter`` are still made, as their entries say, and so are those of each row.

:exc:`~oops.host.HostFormatError`
    A row lacks ``START_TIME``, ``EXPOSURE_DURATION``, ``INSTRUMENT_MODE_ID``, or
    ``INSTRUMENT_NAME``, or gives a ``START_TIME`` or ``INSTRUMENT_MODE_ID`` that is not
    valid. This error is raised when the iteration reaches that row.

:exc:`~oops.host.HostIndexError`
    When the method is called: the PDS3 label of the index or the supplement does not
    describe a table that can be read. During the iteration: a supplement that was given
    does not list the same image as the index in that row; a table ends before its label
    says, after every row it holds; or a row is longer than the label's row length,
    raised at the first of the 100 rows read with it, none of which gives an
    observation. After the last row: the supplement has rows left over, or a table holds
    more rows than its label says. A value that cannot be read as its column's number is
    not reported as an error: the table reader issues a warning, and the row is read with
    that value as text, or missing in a column whose declared null values apply.

:exc:`oops.OopsSpiceError`
    The spacecraft clock kernel cannot convert the clock counts of an early cruise
    image. This error is raised when the iteration reaches that row. The method does not
    check the pointing; an observation whose pointing the C kernels lack raises this
    error only when its frame is evaluated.

:exc:`FileNotFoundError`
    The label of a supplement that was given does not exist, when the method is called;
    or the table of the index or the supplement does not exist, when the iteration
    starts.

:exc:`TypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.OopsTypeError`
    When the iteration reaches a row, a range in ``filter`` cannot be compared with the
    row's value.

:exc:`~oops.OopsValueError`
    ``select`` is not None, or ``filter`` asks for a column that the index does not
    have. For rows given in ``row_dicts`` as an iterator rather than a list, the
    ``filter`` error comes when the iteration reads the first row.

:exc:`~oops.host.PdsHostError`
    The PDS3 label of the index or the supplement cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not a Cassini ISS index.

:exc:`OSError`
    The label of the index cannot be read, when the method is called; or its table
    cannot be read, when the iteration starts.

UVIS and VIMS
=============

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_file` and
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_file` raise:

:exc:`oops.OopsSpiceError`
    The C kernels lack the instrument's pointing at the time at which an observation
    evaluates its frame, and no ``frame`` override is given. That time is the middle of
    the observation, unless a ``cadence`` override, a ``tracker``, or a ``timeshift``
    moves it (see :doc:`user_guide_geometry`). The message reads "no Cassini UVIS
    CASSINI_UVIS_FUV pointing at <time> for <file>: <code>", with the SPICE frame.
    For a VIMS cube, it is also raised when the spacecraft clock kernel cannot convert
    the counts of the cube's time backplanes.

:exc:`~oops.host.HostFormatError`
    The label or the data file is not valid, it describes an option that the host does
    not support, or a normal read needs the data file of parsed content that came from
    no file.

:exc:`~oops.host.PdsHostError`
    No method of the parser can read the PDS3 label, or the ISIS label of a VIMS cube.
    The message gives the label and the error of each method that was tried.

:exc:`~oops.host.WrongHostError`
    The file, or the parsed label given in its place, is not of the host's instrument.

:exc:`~oops.OopsIndexError`, :exc:`~oops.OopsTypeError`, :exc:`~oops.OopsValueError`
    ``select`` is out of range, is of a type that it does not accept, or gives a name
    that it does not accept. A wrong type or name is found before any file is read.

:exc:`TypeError`
    A keyword argument is not one that the method accepts.

:exc:`OSError`
    A file cannot be read.

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_index` and
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` raise the errors listed below.
When the rows are given in ``row_dicts``, the call reads only those rows, so the errors
that come from reading a label or a table are never raised: the labels need not exist or
be readable, the index is not checked to be one of the host's instrument, and
``pds3_options`` is not checked. The checks of ``select`` and ``filter`` are still made,
as their entries say, and so are those of each row.

:exc:`~oops.host.HostIndexError`
    When the method is called: the PDS3 label of the index or the supplement does not
    describe a table that can be read, or lacks a column that gives a row's file; or
    ``supplement`` is None and ``row_dicts`` is not given. When the iteration reaches the
    row: the supplement does not list the index row's product, which is known once the
    whole supplement has been read; the supplement gives a product a second time while
    its first row is still held; a pair of rows gives different files; a row of
    ``row_dicts`` does not include the supplement's columns; a table ends before its
    label says, after every row it holds; or a row is longer than the label's row
    length, raised at the first of the 100 rows read with it. After the last row: the
    supplement lists a product that the index does not, or lists it more often, or a
    table holds more rows than its label says. A value that cannot be read as its
    column's number is not reported as an error: the table reader issues a warning, and
    the value is missing from the row.

:exc:`~oops.host.HostFormatError`
    A row is not valid, it describes an option that the host does not support, or, for
    UVIS, it lacks a column that its observation needs or gives a time series a
    sampling interval that is not positive. This error is raised when the iteration
    reaches that row.

:exc:`~oops.OopsIndexError`
    ``select`` is out of range for a row. This error is raised when the iteration
    reaches that row.

:exc:`oops.OopsSpiceError`
    For VIMS, the spacecraft clock kernel cannot convert a row's
    ``BACKPLANE_SCLK_START_COUNT``. This error is raised when the iteration reaches that
    row. Neither method checks the pointing; an observation whose pointing the C kernels
    lack raises this error only when its frame is evaluated.

:exc:`TypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.OopsTypeError`
    ``select`` is not of a type that it accepts; or, when the iteration reaches a row, a
    range in ``filter`` cannot be compared with the row's value.

:exc:`~oops.OopsValueError`
    ``select`` gives a name that it does not accept, or ``filter`` asks for a column that
    the index does not have. For rows given in ``row_dicts`` as an iterator rather than
    a list, the ``filter`` error comes when the iteration reads the first row.

:exc:`~oops.host.PdsHostError`
    The PDS3 label of the index or the supplement cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not an index of the host's instrument.

:exc:`OSError`
    The label of the index or the supplement cannot be read, when the method is called;
    or a table cannot be read, when the iteration starts.
