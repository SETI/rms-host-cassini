##########################################################################################
# host_cassini/iss/__init__.py
##########################################################################################

import importlib

# Importing _host registers Cassini ISS with oops.Host right away. The full CassiniISS
# implementation in _oops is imported only when the name is first used.
from . import _host

__all__ = ['CassiniISS']


def __getattr__(name):
    """Import the full CassiniISS class on first access (PEP 562)."""

    if name == 'CassiniISS':
        return importlib.import_module(_host._IMPL_MODULE).CassiniISS
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

##########################################################################################
