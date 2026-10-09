##########################################################################################
# host_cassini/__init__.py
##########################################################################################

import importlib

__all__ = ['CassiniISS']

# The oops hosts are imported on first access so that host_cassini.spyceman can be used
# without oops, and the hosts without spyceman. Importing host_cassini.oops registers
# every host with oops.Host.
_LAZY_NAMES = {
    'CassiniISS': 'host_cassini.oops',
}


def __getattr__(name):
    """Import an oops host class on first access (PEP 562)."""

    if name in _LAZY_NAMES:
        return getattr(importlib.import_module(_LAZY_NAMES[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

##########################################################################################
