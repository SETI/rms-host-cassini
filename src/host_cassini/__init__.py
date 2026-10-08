##########################################################################################
# host_cassini/__init__.py
##########################################################################################

import importlib

# Importing each instrument subpackage registers its host with oops.Host right away; the
# full host classes are imported only when their names are first used.
from . import iss     # noqa: F401 (imported to register the host)

__all__ = ['CassiniISS']

# Public name -> the instrument subpackage that provides it
_LAZY_NAMES = {
    'CassiniISS': 'host_cassini.iss',
}


def __getattr__(name):
    """Import a full host class on first access (PEP 562)."""

    if name in _LAZY_NAMES:
        return getattr(importlib.import_module(_LAZY_NAMES[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

##########################################################################################
