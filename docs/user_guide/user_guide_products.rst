==================
Supported Products
==================

Instruments
===========

The package has one host for each of three instruments of the Cassini orbiter:

- :class:`~host_cassini.oops.iss.CassiniISS` reads the images of the Imaging Science
  Subsystem (ISS), whose narrow angle camera (NAC) and wide angle camera (WAC) took
  images of 1024 by 1024 pixels. It reads the raw images of the PDS3 volumes COISS_1001
  through COISS_1009, from the cruise and the Jupiter flyby, and of COISS_2001 through
  COISS_2116, from the Saturn tour. It also reads the calibrated images made from them.
- :class:`~host_cassini.oops.uvis.CassiniUVIS` reads the products of the Ultraviolet
  Imaging Spectrograph (UVIS), in the PDS3 volumes COUVIS_0001 through COUVIS_0060.
- :class:`~host_cassini.oops.vims.CassiniVIMS` reads the cubes of the Visual and Infrared
  Mapping Spectrometer (VIMS), in the PDS3 volumes COVIMS_0001 through COVIMS_0094.

ISS products
============

Each raw ISS product is a VICAR image file, such as ``N1484573295_1.IMG``, with a
detached PDS3 label beside it, such as ``N1484573295_1.LBL``. A calibrated product has
the same pair of files, such as ``N1484573295_1_CALIB.IMG`` and its label. The VICAR file
holds a header of its own, so a product carries two labels. :doc:`user_guide_metadata`
describes how the host combines them.

Every image becomes an OOPS :class:`~oops.observation.Snapshot`, with the image in its
``data`` subfield and the image's lines and samples as its *v* and *u* axes.

ISS sampling modes and fields of view
=====================================

The sampling mode, given by ``INSTRUMENT_MODE_ID``, decides the field of view. ``FULL``
images have the full field of view of 1024 by 1024 pixels. ``SUM2`` images combine each
2 by 2 block of pixels into one, so their field of view is 512 by 512 pixels, and
``SUM4`` images combine each 4 by 4 block, for 256 by 256 pixels.

The field of view models each camera's distortion with the polynomials of the 2003
geometric calibration of the cameras by W. M. Owen. The ``fast_distortion`` option of
:meth:`~host_cassini.oops.iss.CassiniISS.from_file` and
:meth:`~host_cassini.oops.iss.CassiniISS.from_index` chooses how the polynomial is
inverted:

- True, the default, uses a polynomial fitted in advance to the inverse.
- False solves the polynomial numerically for each position.
- None uses a field of view without distortion.

UVIS products
=============

A UVIS product is a data file with a detached PDS3 label. The label describes one of
three kinds of data, and each becomes a different OOPS observation:

- A ``QUBE`` from the far ultraviolet (FUV) or extreme ultraviolet (EUV) channel becomes
  a :class:`~oops.observation.TimedImage`, or a :class:`~oops.observation.Pixel` when it
  has a single line. A QUBE can hold several windows, each a range of lines and bands
  with its own binning; each window becomes its own observation.
- A ``TIME_SERIES`` from the High Speed Photometer (HSP) or the Hydrogen Deuterium
  Absorption Cell (HDAC) becomes a :class:`~oops.observation.Pixel`.
- A ``SPECTRUM`` from the FUV or EUV channel becomes a
  :class:`~oops.observation.Pixel`.

The field of view of each detector and resolution comes from the UVIS instrument kernel.

VIMS products
=============

A VIMS product is a cube in ISIS format, such as ``v1484505000_1.qub``, with a detached
PDS3 label. The cube holds both channels of the instrument, the visual channel (VIS)
and the infrared channel (IR), and the host makes one observation for each. The kind of
observation depends on the shape of the swath:

- A swath of one pixel, as in a ``POINT`` or ``OCCULTATION`` cube, gives a
  :class:`~oops.observation.Pixel` for each channel.
- A swath of one line gives a :class:`~oops.observation.Slit1D` for VIS and a
  :class:`~oops.observation.RasterSlit1D` for IR.
- A full swath gives a :class:`~oops.observation.TimedImage` for each channel.

A channel that was off gives None in place of its observation. In the Nyquist sampling
mode, where ``SAMPLING_MODE_ID`` is ``UNDER``, each pixel is twice the normal size and
the pixels overlap. An OOPS field of view cannot represent that, so the boundary of the
field of view is 1/64 too small along the line direction.

Calibration
===========

The hosts give no observation a calibration. The observation's list of calibrations is
empty unless the caller supplies one with the ``calibrations`` override.
