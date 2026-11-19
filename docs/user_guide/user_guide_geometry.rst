=============================
Time, Pointing, and Targets
=============================

Time
====

An ISS image's exposure begins at ``START_TIME``, in UTC at the spacecraft, and lasts
for ``EXPOSURE_DURATION``. The Snapshot's cadence covers exactly that interval. An image
cannot be read if no label read gives ``START_TIME`` or ``EXPOSURE_DURATION`` (see
:doc:`user_guide_errors`).

A UVIS observation begins at the label's ``START_TIME``. Each sample of a QUBE or
SPECTRUM lasts for ``INTEGRATION_DURATION``, and each sample of a TIME_SERIES for the
label's sampling interval.

A VIMS observation begins at ``START_TIME``. Its pixels follow one another at the
label's exposure of each channel and its delays between frames and between lines, each
scaled by the instrument's timing factor of 1.01725. When the cube is read and holds
time backplanes, those give the time of each pixel, or at least a more accurate start.

Pointing
========

Each instrument's pointing comes from the Cassini C kernels, through the SPICE frame of
the instrument: ``CASSINI_ISS_NAC`` or ``CASSINI_ISS_WAC`` for the ISS cameras, the
frame of the detector for UVIS, such as ``CASSINI_UVIS_FUV``, and ``CASSINI_VIMS_V`` and
``CASSINI_VIMS_IR`` for the VIMS channels. A VIMS observation of the Sun, or one whose
``OBSERVATION_ID`` contains ``_SOL``, uses the frame of the IR channel's solar port,
``CASSINI_VIMS_IR_SOL``. The C kernels for the time of the observation are loaded when
it is made, as :doc:`user_guide_setup` describes.

When :meth:`~host_cassini.oops.iss.CassiniISS.from_file` builds the frame from the C
kernels, it checks at once that they hold the camera's pointing at the middle of the
exposure, after any ``cadence`` override. If they do not, it raises
:exc:`oops.OopsSpiceError` (see :doc:`user_guide_errors`). The UVIS and VIMS hosts do not
check the pointing when they make an observation; if the kernels lack it, the error
appears only when the observation's frame is evaluated.

A caller that has better pointing can pass it with the ``frame`` override, for example
as a :class:`~oops.frame.Cmatrix`. The host then uses that frame, does not load or
consult the C kernels, and raises no error about missing pointing.

Targets and light sources
=========================

The target body comes from ``TARGET_NAME``. An ISS image whose labels give no
``TARGET_NAME`` has the target ``NONE``. Each host repairs the names that its labels
give wrongly or that are not bodies. The ISS host repairs these:

.. list-table::
   :header-rows: 1

   * - ``TARGET_NAME``
     - Target
   * - ``ERRIAPO``, ``K07S4``, ``S8_2004``, ``S14_2004``, ``S18_2004``, ``SKADI``,
       ``SUTTUNG``, ``THRYM``
     - ``ERRIAPUS``, ``AEGAEON``, ``FORNJOT``, ``HATI``, ``BESTLA``, ``SKATHI``,
       ``SUTTUNGR``, and ``THRYMR``
   * - ``S12_2004``, ``S13_2004``
     - ``S/2004_S_12`` and ``S/2004_S_13``
   * - ``DARK SKY``, ``SKY``
     - ``NONE``

The UVIS host gives the target ``NONE`` for ``CALIB``, ``IPH``, ``N/A``, ``SKY``,
``SOLAR WIND``, ``STAR``, and ``UNK``, and expands the two-letter codes of small moons,
such as ``DA`` for ``DAPHNIS``. The VIMS host gives the target ``NONE`` for ``DARK SKY``,
``DUST``, ``OTHER``, ``SKY``, and ``UNK``.

An ISS or VIMS observation of the rings has a ring plane as its target. The host
recognizes it by the target code in ``OBSERVATION_ID``, ``RI`` or ``RA`` through ``RG``,
as in ``ISS_053RI_PHOTOMDRK002_PRIME``. Its target is ``JUPITER_RING_PLANE`` before the
Saturn tour, and ``SATURN_RING_PLANE`` during it.

The light source is the Sun, except in an observation of a star:

- When an ISS or VIMS target is ``FOMALHAUT`` or ``SPICA``, that star becomes the light
  source and the target becomes ``NONE``.
- A UVIS or VIMS observation whose ``OBSERVATION_ID`` contains the name of a star known
  to OOPS, as a stellar occultation does, has that star as its light source.

The ``label_target`` subfield keeps ``TARGET_NAME`` as the label spells it.
