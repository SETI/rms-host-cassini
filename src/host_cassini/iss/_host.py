##########################################################################################
# host_cassini/iss/_host.py
##########################################################################################

import importlib

import oops

# The module holding the full CassiniISS implementation, loaded on first use
_IMPL_MODULE = 'host_cassini.iss._oops'


def _load_cassini_iss():
    """The full CassiniISS class, importing its module on the first call.

    Returns:
        type: :class:`~host_cassini.iss._oops.CassiniISS`.
    """

    return importlib.import_module(_IMPL_MODULE).CassiniISS


class _CassiniISSHost(oops.Host):
    """The registered Cassini ISS host, cheap enough to import eagerly.

    It holds only what :class:`oops.Host` needs to identify Cassini ISS data: the name and
    the detectors. Its constructors import the full implementation,
    :class:`~host_cassini.iss._oops.CassiniISS`, on first use and pass the call on to it.
    CassiniISS subclasses this class, so the detectors are defined once.
    """

    NAME = 'Cassini ISS'

    @staticmethod
    def from_file(fileinfo, **kwargs):
        """A Snapshot object based on a given Cassini ISS image file.

        Parameters:
            fileinfo (str | pathlib.Path | FCPath | PdsLabel | VicarLabel | VicarImage):
                The file path or its parsed content.
            **kwargs: Keyword options, as described for
                :meth:`~host_cassini.iss._oops.CassiniISS.from_file`.

        Returns:
            Snapshot: The observation.
        """

        return _load_cassini_iss().from_file(fileinfo, **kwargs)

    @staticmethod
    def from_index(filepath, **kwargs):
        """A list of Snapshot objects, one for each row of a Cassini ISS index file.

        Parameters:
            filepath (str | pathlib.Path | FCPath): The full path to a Cassini ISS index
                file or its PDS label.
            **kwargs: Keyword options, as described for
                :meth:`~host_cassini.iss._oops.CassiniISS.from_index`.

        Returns:
            list[Snapshot]: One observation per row of the index.
        """

        return _load_cassini_iss().from_index(filepath, **kwargs)

    ######################################################################################
    # Detectors
    ######################################################################################

    @staticmethod
    def _detect_in_pds3(label):
        """True if the given parsed PDS3 label describes data from this host/instrument.
        """

        return (label.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and label.get('INSTRUMENT_ID', '').startswith('ISS'))

    @staticmethod
    def _detect_in_vicar(label):
        """True if the given VicarLabel describes data from this host/instrument."""

        # PDS3 and VICAR use the same names
        return _CassiniISSHost._detect_in_pds3(label)

    @staticmethod
    def _detect_in_index(label):
        """True if the given index label refers to this host/instrument.

        Return None if the host/instrument cannot be inferred from the label, only from
        individual records.
        """

        return label.get('IMAGE_INDEX_TABLE', '').startswith('COISS')

    @staticmethod
    def _detect_in_row(row_dict):
        """True if the given row of an index file label refers to this host/instrument."""

        return (row_dict.get('INSTRUMENT_HOST_NAME', '').startswith('CASSINI')
                and row_dict.get('INSTRUMENT_NAME', '').startswith('IMAGING SCIENCE'))


_CassiniISSHost._register()

##########################################################################################
