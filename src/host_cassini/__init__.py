##########################################################################################
# host_cassini/__init__.py
##########################################################################################
"""The Cassini mission hosts for the OOPS observation geometry toolkit.

The package exports :class:`~host_cassini.oops.iss.CassiniISS`,
:class:`~host_cassini.oops.uvis.CassiniUVIS` and
:class:`~host_cassini.oops.vims.CassiniVIMS`. Importing ``host_cassini`` does not import
oops; the first access to one of these names imports :mod:`host_cassini.oops`, which
registers all three hosts with :class:`~oops.host.host_.Host`.
"""

import importlib

__all__ = ['CassiniISS', 'CassiniUVIS', 'CassiniVIMS']

# The oops hosts are imported on first access, so that importing host_cassini does not
# import oops. Importing host_cassini.oops registers every host with oops.Host.
_LAZY_NAMES = {
    'CassiniISS' : 'host_cassini.oops',
    'CassiniUVIS': 'host_cassini.oops',
    'CassiniVIMS': 'host_cassini.oops',
}


def __getattr__(name):
    """Import an oops host class on first access (PEP 562)."""

    if name in _LAZY_NAMES:
        return getattr(importlib.import_module(_LAZY_NAMES[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

##########################################################################################
