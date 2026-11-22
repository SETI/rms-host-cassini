=======================
Metadata and Subfields
=======================

This chapter describes the metadata of an ISS image first, and then that of a UVIS
product or a VIMS cube, which have a single label each.

The two labels
==============

An ISS product describes itself twice. The *PDS3 label* is the detached ``.LBL`` file
beside the image. The *VICAR header* is the label at the start of the image file itself.
The two were written at different times and do not always agree, and each holds values
that the other lacks.

The host takes its metadata from whichever of the two it read (see
:doc:`user_guide_files`): from the VICAR header alone when given the image file, from
the PDS3 label alone in an astrometry read of the label, and from both when given the
label otherwise.

Merging the two labels
======================

Most VICAR headers use the PDS3 keywords already. The host writes each header value in
the form that the PDS3 label uses:

- A time loses the ``Z`` that the header writes at its end.
- ``IMAGE_NUMBER`` is text, as the label quotes it.
- ``IMAGE_OBSERVATION_TYPE`` is a list, as the label's set is given.
- A single ``OPTICS_TEMPERATURE``, the front temperature, becomes the pair of it and a
  missing back temperature.
- ``MISSION_PHASE_NAME`` has spaces where the header writes underscores.

The headers of the early cruise images, the first 1250 images of COISS_1001, taken from
January 1999 to July 2000, use older keywords. The host gives each of them the name and
spelling that the image's PDS3 label gives the same value. For example, ``BIAS`` becomes
``BIAS_STRIP_MEAN``, ``FILTER1_NAME`` and ``FILTER2_NAME`` together become
``FILTER_NAME``, ``GAIN_MODE_ID`` is spelled in full, such as ``12 ELECTRONS PER DN``,
and ``IMAGE_TIME`` gives ``START_TIME`` and ``STOP_TIME``. A converted value appears in
the merged metadata only under its PDS3 name, and the original keyword stays in
``header_dict``.

Where only one label gives an item, its value is used. Where both give it and agree,
that value is used. Where both give it and disagree, the item always takes its value
from the same label, for every image. The VICAR header supplies these items:

.. list-table::
   :header-rows: 1

   * - Items
     - Why the VICAR header
   * - ``MISSION_PHASE_NAME``
     - Every early cruise label gives ``SCIENCE CRUISE``, which the mission catalog
       dates 2002 to 2004, while each header gives the phase of the image's own date.
   * - ``METHOD_DESC``
     - The labels replace its semicolons and commas with spaces, and its ``@`` with a
       line break.

The PDS3 label supplies every other item. Some of the differences are worth knowing:

- Many numbers, such as ``BIAS_STRIP_MEAN`` and the temperatures, differ only in the
  last digit, because the label prints a single-precision value to six decimal places
  and the header rounds it to six significant digits.
- The header of a calibrated image keeps the raw image's ``DESCRIPTION`` and
  ``PRODUCT_CREATION_TIME``, while the label gives the calibrated image's own. The raw
  values stay in ``header_dict``.
- The early cruise headers give an older ``DATA_SET_ID``, and ``CASSINI`` as the
  ``MISSION_NAME``, where the labels and the archive's catalog give
  ``CASSINI-HUYGENS``.

If the chosen label lacks a value for an item, the other label's value is used.

Missing values
==============

If no label that was read gives a value, the value is left out: there is no ``dict`` key
and no subfield for it, and never a None or a placeholder in its place. These count as
no value:

- the text that the archive writes for no value, ``UNK``, ``UNKNOWN``, ``N/A``,
  ``NULL``, ``NOT APPLICABLE``, ``---``, or an empty string, in any case;
- in an ISS image or a VIMS cube, the number -999 in a temperature, a data rate, a
  compression ratio, or, for VIMS, an exposure, which cannot take that value. The labels
  write it for a reading that was not made, such as the back optics temperature of an
  ISS camera that has no back sensor, or the exposure of a VIMS channel that was off. In
  any other keyword, -999 is kept. A UVIS label keeps -999 in every keyword;
- a list that holds only these.

In a list whose positions carry meaning, such as the front and back temperatures of
``OPTICS_TEMPERATURE`` or the two filters of ``FILTER_NAME``, a missing element is kept
as None, so that the other elements keep their places.

The dictionaries
================

Three subfields hold the metadata as dictionaries. All three can be written with
:func:`json.dump`: times are the label's text, sets are lists, and an item has the same
type whichever label it came from.

``dict``
    The merged metadata: every item of either label read, under the PDS3 name where the
    host converts it, and under its own name otherwise. The label's keywords come
    first, in label order, then the header's other keywords.

``label_dict``
    The PDS3 label as its parser reads it, present only when the label was read. Times
    are the label's own text, and sets are lists in the label's order. The ``END``
    keyword and the extra keys that the parser adds beside each keyword, such as
    ``START_TIME_fmt`` and ``^IMAGE_offset``, are left out.

``header_dict``
    Every parameter of the VICAR header as the file writes it, present only when the
    image file was read. A normal read and an astrometry read give the same values,
    including ``HOST`` and ``BHOST``, the machine that wrote the file. A parameter that
    occurs more than once, such as the ``TASK`` of each step of the processing history,
    is keyed by its name and its occurrence counted from zero: ``TASK[0]``, ``TASK[1]``,
    and so on.

Subfields
=========

Besides ``dict``, ``label_dict``, ``header_dict``, and ``data``, every Snapshot has
these subfields. One whose item no label gives is absent.

.. list-table::
   :header-rows: 1

   * - Subfield
     - Contents
   * - ``host``, ``instrument``
     - ``"Cassini"`` and ``"ISS"``.
   * - ``detector``
     - ``"NAC"`` or ``"WAC"``, from ``INSTRUMENT_NAME``.
   * - ``sampling``
     - ``INSTRUMENT_MODE_ID``: ``"FULL"``, ``"SUM2"``, or ``"SUM4"``.
   * - ``texp``
     - ``EXPOSURE_DURATION`` in seconds.
   * - ``time_from_sclk``
     - Always False: the time of an ISS image comes from ``START_TIME``, never from a
       spacecraft clock count.
   * - ``filter``, ``filter1``, ``filter2``
     - ``filter1`` and ``filter2`` are the two names of ``FILTER_NAME``, such as
       ``"CL1"`` and ``"GRN"``. ``filter`` is the filter that is not clear (``CL1`` or
       ``CL2``), ``"CLEAR"`` when both are clear, or the two names in alphabetical order
       joined by ``+`` when neither is. All three are absent unless both names are
       given.
   * - ``gain_mode``
     - 0, 1, 2, or 3, for a ``GAIN_MODE_ID`` of 215, 95, 29, or 12 electrons per DN.
   * - ``label_target``
     - ``TARGET_NAME`` as the label spells it.
   * - ``observation_id``
     - ``OBSERVATION_ID``.
   * - ``spacecraft_clock_start_count``, ``spacecraft_clock_stop_count``
     - ``SPACECRAFT_CLOCK_START_COUNT`` and ``SPACECRAFT_CLOCK_STOP_COUNT``, described
       below.
   * - ``spice_kernels``
     - The SPICE kernels used for the observation.
   * - ``spice_frame_name``, ``spice_frame_id``, ``spice_to_frame``
     - The SPICE frame of the camera, ``CASSINI_ISS_NAC`` (-82360) or
       ``CASSINI_ISS_WAC`` (-82361), and the rotation from it to the camera frame of
       OOPS, a half turn about the optic axis.
   * - ``filepath``, ``basename``, ``abspath``, ``image_url``
     - The image file (see :doc:`user_guide_files`).

The spacecraft clock counts
===========================

``spacecraft_clock_start_count`` and ``spacecraft_clock_stop_count`` are the counts at
the start and end of the exposure, written as the PDS3 label writes them, with three
digits after the point, such as ``1459509269.122``.

The early cruise images need a correction. Their labels give a start count equal to the
stop count less the exposure in milliseconds counted as seconds, and their headers give
no start count. For these images the stop count is ``IMAGE_NUMBER``, the whole seconds
of the clock at the close of the shutter, with zero ticks, as the labels give it. The
start count is computed from it and ``EXPOSURE_DURATION`` with the Cassini spacecraft
clock kernel. The same correction applies whether the image is read from its label, from
its image file, or from an index, and ``dict`` holds the corrected counts. If the clock
kernel cannot convert a count, the read raises :exc:`oops.OopsSpiceError`.

How the start count of an early cruise image whose shutter was disabled should be
computed is an open question. The host computes it the same way for every early cruise
image, as the stop count less the exposure.

UVIS and VIMS metadata
======================

A UVIS product has one label, its detached PDS3 label. A VIMS cube has two forms of
one label: the detached PDS3 label and the ISIS label at the start of the cube. The
host reads the label it is given, and no merging applies.

``label_dict``
    The detached PDS3 label as its parser reads it, when the label was read, with times
    as the label's text and sets as lists, and without ``END`` or the extra keys that
    the parser adds.

``header_dict``
    The ISIS label of a VIMS cube, in the same form, when the cube itself was given.

``dict``
    The keywords of the label read without their missing values, as described above
    for ISS. For a VIMS cube, both ``QUBE`` and ``SPECTRAL_QUBE`` hold the cube's
    object. From the ISIS label, the object's keywords also appear at the top level;
    from the PDS3 label, the object holds the fixed values of the archive's format
    files.

Each observation has the subfields ``label_target``, ``observation_id``,
``spacecraft_clock_start_count``, and ``spacecraft_clock_stop_count`` when the label
gives their items, and ``time_from_sclk``. The clock counts are the label's text, such
as ``1/1640775127.032``. ``time_from_sclk`` is True only for a VIMS cube whose time
backplanes gave its times, and False otherwise. A backplane gives a clock reading in
seconds and a decimal fraction, which the host converts as whole seconds plus that
fraction of a second. All three dictionaries can be written with :func:`json.dump`.
