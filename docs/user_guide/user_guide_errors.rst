======
Errors
======

The hosts report each kind of failure with one exception class. The classes come from
OOPS, so code that reads data from several missions can catch them the same way.

Reading an ISS image file
=========================

:meth:`~host_cassini.oops.iss.CassiniISS.from_file` raises:

:exc:`oops.OopsSpiceError`
    The C kernels hold no pointing for the camera at the middle of the exposure. The
    message reads "no Cassini ISS NAC pointing at <time> for <image file>: <code>",
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

:exc:`~oops.OopsTypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.OopsValueError`
    ``select`` is not None.

:exc:`~oops.host.PdsHostError`
    No method of the parser can read the PDS3 label, even after the repairs that
    :doc:`user_guide_files` describes, or the label's ``^IMAGE`` pointer does not give
    the name of a file.
    The message gives the label.

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

:meth:`~host_cassini.oops.iss.CassiniISS.from_index` raises:

:exc:`~oops.host.HostFormatError`
    A row lacks ``START_TIME``, ``EXPOSURE_DURATION``, ``INSTRUMENT_MODE_ID``, or
    ``INSTRUMENT_NAME``, or gives a ``START_TIME`` or ``INSTRUMENT_MODE_ID`` that is not
    valid. This error is raised when the iteration reaches that row.

:exc:`~oops.host.HostIndexError`
    When the method is called: the index is a volume's own index rather than a metadata
    index, or the PDS3 label of the index or the supplement does not describe a table
    that can be read. When the iteration reaches the row: a row does not match the
    table's label, as when the table holds fewer rows than its label says, or a
    supplement that was given does not list the same image as the index in that row.
    After the last row: the supplement has rows left over.

:exc:`oops.OopsSpiceError`
    The spacecraft clock kernel cannot convert the clock counts of an early cruise
    image. This error is raised when the iteration reaches that row. The method does not
    check the pointing; an observation whose pointing the C kernels lack raises this
    error only when its frame is evaluated.

:exc:`FileNotFoundError`
    The label of a supplement that was given does not exist, when the method is called;
    or the table of the index or the supplement does not exist, when the iteration
    starts.

:exc:`~oops.OopsTypeError`
    A keyword argument is not one that the method accepts, or, when the iteration
    reaches a row, a range in ``filter`` cannot be compared with the row's value.

:exc:`~oops.OopsValueError`
    ``select`` is not None, or ``filter`` asks for a column that the index does not
    have.

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
    The C kernels hold no pointing for the instrument at the middle of an observation,
    and no ``frame`` override is given. The message reads "no Cassini UVIS
    CASSINI_UVIS_FUV pointing at <time> for <file>: <code>", with the SPICE frame.
    For a VIMS cube, it is also raised when the spacecraft clock kernel cannot convert
    the counts of the cube's time backplanes.

:exc:`~oops.host.HostFormatError`
    The label or the data file is not valid, it describes an option that the host does
    not support, or a normal read needs the data file of parsed content that came from
    no file.

:exc:`~oops.host.PdsHostError`
    No method of the parser can read the PDS3 label, or the ISIS label of a VIMS cube.

:exc:`~oops.host.WrongHostError`
    The file, or the parsed label given in its place, is not of the host's instrument.

:exc:`~oops.OopsIndexError`, :exc:`~oops.OopsTypeError`, :exc:`~oops.OopsValueError`
    ``select`` is out of range, is of a type that it does not accept, or gives a name
    that it does not accept. A wrong type or name is found before any file is read.
    :exc:`~oops.OopsTypeError` is also raised for a keyword argument that the method
    does not accept.

:exc:`OSError`
    A file cannot be read.

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_index` and
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` raise:

:exc:`~oops.host.HostIndexError`
    When the method is called: the index is a volume's own index rather than a metadata
    index; the PDS3 label of the index or the supplement does not describe a table that
    can be read, or lacks a column that gives a row's file; or ``supplement`` is None
    and ``row_dicts`` is not given. When the iteration reaches the row: a row does not
    match the table's label; the supplement does not list the index row's product,
    which is known once the whole supplement has been read; the supplement gives a
    product a second time while its first row is still held; a pair of rows gives
    different files; or a row of ``row_dicts`` does not include the supplement's
    columns. After the last row: the supplement lists a product that the index does
    not, or lists it more often.

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

:exc:`~oops.OopsTypeError`
    A keyword argument is not one that the method accepts, or ``select`` is not of a
    type that it accepts; or, when the iteration reaches a row, a range in ``filter``
    cannot be compared with the row's value.

:exc:`~oops.OopsValueError`
    ``select`` gives a name that it does not accept, or ``filter`` asks for a column that
    the index does not have.

:exc:`~oops.host.PdsHostError`
    The PDS3 label of the index or the supplement cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not an index of the host's instrument.

:exc:`OSError`
    The label of the index or the supplement cannot be read, when the method is called;
    or a table cannot be read, when the iteration starts.
