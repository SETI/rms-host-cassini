##########################################################################################
# spyceman/hosts/Cassini/ck.py
##########################################################################################
"""Cassini CK constructor functions:

Methods:
    gapfill_ck: A low-precedence "gapfill" CK for the Saturn tour.
    jupiter_ck: A CK for the Jupiter flyby.
    ck: A function returning a C (pointing) kernel for any part of the mission.
"""

from spyceman.kernelfile import KernelFile
from spyceman.rule       import Rule
from spyceman._spicefunc import _spicefunc
from spyceman._utils     import _input_set

from . import _source_url, _DEFAULT_TIMES, _DEFAULT_BODY_IDS
from ._RECONSTRUCTED_CKS import _RECONSTRUCTED_CKS

# This pattern will match any Cassini CK except the original Jupiter CKs, which use YYMMDD
# dates.
Rule(r'(YYDOY)_(YYDOY)(p[a-z]|r[abc]]).*\.bc', naif_ids={-82000}, mission='CASSINI',
     source=_source_url('CK'))

##########################################################################################
# Gapfill CK
##########################################################################################

from ._GAPFILL_CKS import _GAPFILL_CKS

_rule = Rule(r'(YYDOY)_(YYDOY)pa_gapfill_v(NN)\.bc', family='Cassini-gapfill-CK',
             dest='Cassini/CK-gapfill', planet='SATURN', gapfill=True)
KernelFile.veto(_rule.pattern, r'\1_\2pa_gapfill.*\.bc')
# Veto other gapfill CKs for the same dates, different version.

_gapfill_ck_pattern = r'\d{5}_\d{5}p[a-z]_gapfill_v\d\d\.tf'

_gapfill_ck_notes = """\
        The Cassini "gapfill" CKs should always be furnished at a lower precedence than
        any other selected CKs.

        The final version released by the team is 14 except during 2003, when the only
        version is 1.
"""

gapfill_ck = _spicefunc('gapfill_ck',
                        title = 'Cassini gapfill CKs',
                        known = _GAPFILL_CKS,
                        unknown = _gapfill_ck_pattern,
                        source = _source_url('CK'),
                        exclude = False,
                        reduce = True,
                        default_times = ('2003-01-01', '2017-09-16'),
                        default_ids = None,         # to match any Cassini frame ID
                        notes = _gapfill_ck_notes)

##########################################################################################
# Jupiter CKs
##########################################################################################

from ._JUPITER_CKS import _JUPITER_CKS

Rule(r'(YYMMDD)_(YYMMDD)(|r[ab])\.bc', family='Cassini-Jupiter-CK', naif_ids={-82000},
     version={1,2}, source=_source_url('CK'), dest='Cassini/CK-Jupiter',
     mission='CASSINI', planet='JUPITER')

# These five files in _JUPITER_CKS are overrides of version=1 kernels and are specific to
# version 2.
_jupiter_ck_shadows = [('001113_001116ra.bc', '001113_001118ra.bc'),
                       ('001129_001130ra.bc', '001123_001130ra.bc'),
                       ('001213_001213ra.bc', '001213_001215ra.bc'),
                       ('010121_010122ra.bc', '010121_010123ra.bc'),
                       ('010123_010124ra.bc', '010123_010128ra.bc')]
for _basename, _ in _jupiter_ck_shadows:
    KernelFile(_basename, version=2)

# This pattern will match any "reconstructed" CK; version=3
Rule(r'(YYDOY)_(YYDOY)r[ab]\.bc', family='Cassini-reconstructed-CK', version=3)
Rule(r'(00[23]|010)\d\d_\d{5}r[ab]\.bc', planet='JUPITER')  # start time indicates Jupiter

def _jupiter_ck_sort_key(basename):
    """Sort key for the Jupiter flyby CKs, in increasing order of precedence.

    The predicted-pointing kernels sort lowest, then the five reconstructed kernels that
    do not follow the "YYDOY_" naming convention, then the remaining reconstructions.

    Parameters:
        basename (str): Basename of a Cassini Jupiter CK.

    Returns:
        (int, str): The sort key for this basename.
    """

    if basename[5:] == '_':     # reconstructed begin with "YYDOY_"
        return (3, basename)
    if basename in {'001113_001116ra.bc',
                    '001129_001130ra.bc',
                    '001213_001213ra.bc',
                    '010121_010122ra.bc',
                    '010123_010124ra.bc'}:
        return (2, basename)
    return (1, basename)

_jupiter_ck_notes = """\
    This kernel contains the reconstructed pointing for the Jupiter flyby of late 2000 and
    early 2001.

    * Version 1 uses the initial pointing reconstruction from telemetry.
    * Version 2, from during and shortly after the flyby, augments Version 1 with a few
      updates to the pointing reconstruction.
    * Version 3 contains the mission's final pointing reconstruction from 2018.
"""

jupiter_ck = _spicefunc('jupiter_ck',
                        title = 'Cassini Jupiter CKs',
                        known = _JUPITER_CKS + _RECONSTRUCTED_CKS,
                        sort = _jupiter_ck_sort_key,
                        exclude = False,
                        reduce = False,
                        shadows = _jupiter_ck_shadows,
                        default_times = ('2000-10-01', '2001-04-01'),
                        default_ids = None,         # to match any Cassini frame ID
                        notes = _jupiter_ck_notes)

##########################################################################################
# General CKs
##########################################################################################

from ._LIVE_CKS     import _LIVE_CKS
from ._AS_FLOWN_CKS import _AS_FLOWN_CKS

# This pattern will match any "live" CK but not a gapfill CK; version=1
Rule(r'(YYDOY)_(YYDOY)p[a-x](?!_gapfill).*\.bc', family='Cassini-live-CK', version=1,
     dest='Cassini/CK-live', planet='SATURN')

# This pattern will match any "as flown" CK; version=2
Rule(r'(YYDOY)_(YYDOY)py_as_flown\.bc', family='Cassini-as-flown-CK', version=2,
     dest='Cassini/CK-as-flown', planet='SATURN')

_ck_pattern = r'\d{5}_\d{5}[pr][a-z].*\.bc'

def _ck_sort_key(basename):
    """Sort prioritizing the version code pa-pz, ra-rz first.

    Parameters:
        basename (str): Basename of a Cassini CK, whose characters 12 and 13 hold the
            two-letter version code.

    Returns:
        (str, str): The sort key for this basename.
    """

    basename = basename.lower()
    return (basename[11:13], basename)

# Adapt other functions to receive the same inputs as ck()
def _gapfill_ck_adapted(version=None, basename=None, planet=None, gapfill=True,
                        **keywords):
    """Wrapper letting `gapfill_ck()` accept the same inputs as `ck()`.

    The gapfill CK applies only to Saturn, so this returns None whenever the request
    names planets but not Saturn, or names basenames that are not gapfill basenames

    Parameters:
        version (int, str, tuple, set or list, optional): Ignored; the gapfill CK has no
            versions.
        basename (str, list, set or tuple, optional): Only include kernel files matching
            one or more basenames or regular expressions.
        planet (str, set, list or tuple, optional): The mission phases requested; None for
            all of them.
        gapfill (bool, optional): False to suppress the gapfill CK entirely.
        **keywords: Additional constraints passed through to `gapfill_ck()`.

    Returns:
        Kernel, None: The gapfill CK, or None if it does not apply to this request.
    """

    # If input property gapfill is False, no CK is needed
    if not gapfill:
        return None

    # If not Saturn, no CK is needed.
    planet = _input_set(planet, default={'SATURN'})
    if not planet & {'SATURN'}:
        return None

    # Identify explicit basenames that are part of the gapfill CK.
    basename = _input_set(basename)
    if basename:
        basename = {b for b in basename if '_gapfill' in b}
        if not basename:
            return None

    return gapfill_ck(version=None, basename=basename, **keywords)

def _jupiter_ck_adapted(version=None, basename=None, planet=None, **keywords):
    """Wrapper letting `jupiter_ck()` accept the same inputs as `ck()`.

    Version 3 of the pointing reconstruction is shared by Jupiter and Saturn, so a
    Jupiter-specific CK is needed only for versions 1 and 2. This returns None whenever
    the request excludes those versions, or names planets that exclude Jupiter.

    Parameters:
        version (int, str, tuple, set, list, optional): The versions requested; None for
            all of them.
        basename (str, list, set, tuple, optional): Only include kernel files matching
            this basename or regular expression, or one of these.
        planet (str, set, list, tuple, optional): The mission phases requested; None for
            all of them.
        **keywords: Additional constraints passed through to `jupiter_ck()`.

    Returns:
        Kernel, None: The Jupiter flyby CK, or None if it does not apply to this request.
    """

    # Version 3 is the same for Jupiter and Saturn, so no special CK is needed.
    version = _input_set(version, default={1, 2, 3}, ranges=True)
    if not version & {1, 2}:
        return None

    # If an explicit planet is listed and JUPITER is not among them, no CK is needed.
    planet = _input_set(planet, default=set(_DEFAULT_BODY_IDS.keys()))
    if not planet & {'JUPITER'}:
        return None

    # Identify explicit basenames that are part of the V1 or V2 Jupiter CK
    basename = _input_set(basename)
    basename = {b for b in basename if b[6:7] == '_'}

    return jupiter_ck(version=version, basename=basename, **keywords)

_ck_notes = """\
    * Version 1 uses the predicted pointing during mission operations.
    * Version 2 uses the final regeneration of "as flown" predicted pointing.
    * Version 3 contains the mission's final pointing reconstruction from telemetry. This
      version covers the entire mission, whereas Versions 1 and 2 only apply to the Saturn
      tour.

    Note that Versions 1 and 2 have the property that pointing changes are smooth.
    Although the reconstructed pointing should be, in general, more accurate in absolute
    terms, it contains unrealistic, high-frequency "jitter", which was an artifact of the
    processing.
"""

_ck_docstrings = {}
_ck_docstrings['planet'] = """\
        planet (str, set, list, tuple, optional): One or more of "VENUS", "EARTH",
            "MASURSKY", "JUPITER", or "SATURN", naming the phase or phases of the Cassini
            mission that this CK should cover. "SATURN" is the default.
"""
_ck_docstrings['gapfill'] = """\
        gapfill (bool, optional): True to include the prerequisite "gapfill" kernel during
            the Saturn tour; False otherwise. Default is True.
"""

ck = _spicefunc('ck',
                title = 'Cassini mission CKs',
                known = _LIVE_CKS + _AS_FLOWN_CKS + _RECONSTRUCTED_CKS,
                unknown = _ck_pattern,
                source = _source_url('CK'),
                sort = _ck_sort_key,
                exclude = False,
                reduce = True,
                require = (_gapfill_ck_adapted, _jupiter_ck_adapted),
                default_times = _DEFAULT_TIMES,
                default_times_key = 'planet',
                default_ids = None,         # to match any Cassini frame ID
                default_properties = {'planet': 'SATURN', 'gapfill': True},
                notes = _ck_notes,
                docstrings = _ck_docstrings)

##########################################################################################
