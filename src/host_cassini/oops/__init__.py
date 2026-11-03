##########################################################################################
# host_cassini/oops/__init__.py
##########################################################################################
"""The Cassini hosts for oops.

The hosts are :class:`~host_cassini.oops.iss.CassiniISS`,
:class:`~host_cassini.oops.uvis.CassiniUVIS` and
:class:`~host_cassini.oops.vims.CassiniVIMS`, each documented with its module below.

Importing this package registers all three hosts with :class:`~oops.host.host_.Host`, so
that its ``from_file()`` and ``from_index()`` recognize Cassini ISS, UVIS and VIMS files
and indexes.
"""

# Importing each instrument module registers its host with oops.Host
from .iss  import CassiniISS
from .uvis import CassiniUVIS
from .vims import CassiniVIMS

__all__ = ['CassiniISS', 'CassiniUVIS', 'CassiniVIMS']

##########################################################################################
