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
    file. It is never raised when the caller supplies the pointing with the ``frame``
    override.

:exc:`~oops.host.HostFormatError`
    No label read gives ``START_TIME``, ``EXPOSURE_DURATION``, ``INSTRUMENT_MODE_ID``,
    or ``INSTRUMENT_NAME``, or ``START_TIME`` or ``INSTRUMENT_MODE_ID`` is not valid.
    The message gives the item and the image file.

:exc:`~oops.host.PdsHostError`
    No method of the parser can read the PDS3 label, even after the repairs that
    :doc:`user_guide_files` describes, or the label's ``^IMAGE`` pointer gives no file.
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
    A supplement that was given does not list the same images as the index, or, when
    the two tables list the images in different orders, one of them lists an image twice.

:exc:`FileNotFoundError`
    A supplement that was given does not exist.

:exc:`~oops.OopsTypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.OopsValueError`
    ``select`` is not None.

:exc:`~oops.host.PdsHostError`
    The index's PDS3 label cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not a Cassini ISS index.

:exc:`OSError`
    The index cannot be read.

The method does not check the pointing and never raises :exc:`oops.OopsSpiceError`.

UVIS and VIMS
=============

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_file` and
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_file` raise:

:exc:`~oops.host.HostFormatError`
    The label or the data file is not valid, or it describes an option that the host
    does not support.

:exc:`~oops.host.PdsHostError`
    The PDS3 label, or the ISIS label of a VIMS cube, cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not of the host's instrument.

:exc:`~oops.OopsIndexError`, :exc:`~oops.OopsTypeError`, :exc:`~oops.OopsValueError`
    ``select`` is out of range or not of a type that it accepts.

:exc:`OSError`
    A file cannot be read.

:meth:`~host_cassini.oops.uvis.CassiniUVIS.from_index` and
:meth:`~host_cassini.oops.vims.CassiniVIMS.from_index` raise:

:exc:`~oops.host.HostIndexError`
    ``supplement`` is None and ``row_dicts`` is not given; a product appears in only one
    of the index and the supplement; the two tables list the products in different orders
    and one of them lists a product twice; a pair of rows names different files; or
    ``row_dicts`` does not include the supplement's columns.

:exc:`~oops.host.HostFormatError`
    A row is not valid, it describes an option that the host does not support, or, for
    UVIS, it lacks a column that its observation needs or gives no positive sampling
    interval for a time series. This error is raised when the iteration reaches that
    row.

:exc:`~oops.OopsTypeError`
    A keyword argument is not one that the method accepts.

:exc:`~oops.host.PdsHostError`
    The index's PDS3 label cannot be parsed.

:exc:`~oops.host.WrongHostError`
    The file is not an index of the host's instrument.

:exc:`OSError`
    The index or the supplement cannot be read.

None of these four methods checks the pointing or raises :exc:`oops.OopsSpiceError`.
