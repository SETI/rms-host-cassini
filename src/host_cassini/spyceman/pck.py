##########################################################################################
# spyceman/hosts/Cassini/pck.py
##########################################################################################
"""Cassini PCK constructor functions:

Methods:
    mission_pck: A function returning a mission PCK, containing GM, gravity harmonics,
        radii, body orientations, ring geometry, atmospheric and magnetic pole info.
    mission_nav_pck: A function returning a mission PCK containing a subset of the
        `mission_pck` information: just gravity fields and body poles.
    mission_sci_pck: A function returning a mission PCK containing all the `mission_pck`
        information _not_ found in the `mision_nav_pck` files.
    rock_pck: A function returning a PCK file describing the shapes, poles, and rotation
        states of the small Saturnian satellites.
    rock_merged_pck: A function returning a PCK file containing all the information of the
        `rock_pck` files but also defining frames for the small satellites.
"""

from spyceman.kernelfile import KernelFile
from spyceman.rule       import Rule
from spyceman._spicefunc import _spicefunc
from spyceman._utils     import _input_set

from . import _source_url

##########################################################################################
# Mission "Nav" PCKs
##########################################################################################

from ._CASSINI_NAV_PCKS import _CASSINI_NAV_PCKS

_nav_rule = Rule(r'cpck(DDMONYYYY)_Nav\.tpc', mission='CASSINI',
                 source=_source_url('PCK'), dest='Cassini/PCK', family='Cassini-Nav-PCK')
KernelFile.mutual_veto(_nav_rule.pattern)

_mission_nav_pck_notes = """\
    This function returns a mission "Nav" PCK, containing a subset of the information
    found in the corresponding `mission_pck` kernel: just gravity fields and body poles.

    The final Cassini Nav PCK was released on 2017-12-15.
"""

mission_nav_pck = _spicefunc('mission_nav_pck',
                             title = 'Cassini Nav PCK',
                             known = _CASSINI_NAV_PCKS,
                             unknown = _nav_rule.pattern,
                             source = _source_url('PCK'),
                             exclude = True,   # never more than one
                             notes = _mission_nav_pck_notes)

##########################################################################################
# Mission "Sci" PCKs
##########################################################################################

from ._CASSINI_SCI_PCKS import _CASSINI_SCI_PCKS

_sci_rule = Rule(r'cpck(DDMONYYYY)_Sci\.tpc', mission='CASSINI',
                 source=_source_url('PCK'), dest='Cassini/PCK', family='Cassini-Sci-PCK')
KernelFile.mutual_veto(_sci_rule.pattern)

_mission_sci_pck_notes = """\
    This function returns a mission "Sci" PCK, containing a subset of the information
    found in the corresponding `mission_pck` kernel: just radii, body orientations,
    ring geometry, atmospheric and magnetic pole info. It excludes the gravity fields
    found in the "Nav" PCKs.

    The final Cassini Sci PCK was released on 2010-09-02.
"""

mission_nav_pck = _spicefunc('mission_sci_pck',
                             title = 'Cassini Sci PCK',
                             known = _CASSINI_SCI_PCKS,
                             unknown = _sci_rule.pattern,
                             source = _source_url('PCK'),
                             exclude = True,   # never more than one
                             notes = _mission_sci_pck_notes)

##########################################################################################
# General PCKs
##########################################################################################

from ._CASSINI_ROCK_PCKS import _CASSINI_ROCK_PCKS

_rule = Rule(r'cpck(DDMONYYYY)\.tpc', mission='CASSINI', source=_source_url('PCK'),
             dest='Cassini/PCK', family='Cassini-PCK')
KernelFile.mutual_veto(_rule.pattern)
KernelFile.veto(_nav_rule.pattern)
KernelFile.veto(_sci_rule.pattern)

_rock_pck_notes = """\
    This function returns a PCK defining the shapes, poles, and rotation states of the
    small Saturnian satellites. It was produced to support the Cassini tour of the Saturn
    system.

    The final Cassini Sci PCK was released on 2010-09-02.
"""

rock_pck = _spicefunc('rock_pck',
                      title = 'Cassini "rock" PCK',
                      known = _CASSINI_ROCK_PCKS,
                      unknown = _rule.pattern,
                      source = _source_url('PCK'),
                      exclude = True,   # never more than one
                      notes = _rock_pck_notes)

##########################################################################################
# Rock PCKs
##########################################################################################

from ._CASSINI_ROCK_PCKS import _CASSINI_ROCK_PCKS

_rule = Rule(r'cpck_rock_(DDMONYYYY)\.tpc', mission='CASSINI', source=_source_url('PCK'),
             dest='Cassini/PCK', family='Cassini-rock-PCK')
KernelFile.mutual_veto(_rule.pattern)
# KernelFile.veto(_nav_rule.pattern)
# KernelFile.veto(_sci_rule.pattern)

_rock_pck_notes = """\
    This function returns a general PCK for the Cassini mission, containing GM, gravity
    harmonics, radii, body orientations, ring geometry, atmospheric and magnetic pole
    info.

    The final Cassini Sci PCK was released on 2010-09-02.
"""

mission_pck = _spicefunc('mission_pck',
                         title = 'General Cassini mission PCK',
                         known = _CASSINI_PCKS,
                         unknown = _rule.pattern,
                         source = _source_url('PCK'),
                         exclude = True,   # never more than one
                         notes = _mission_pck_notes)



_rule = Rule(r'cpck(DDMONYYYY)\.tpc', mission='CASSINI', source=_source_url('PCK'),
             dest='Cassini/PCK', family='Cassini-PCK')
KernelFile.mutual_veto(_rule.pattern)

_rule = Rule(r'cpck(DDMONYYYY)_Nav\.tpc', mission='CASSINI', source=_source_url('PCK'),
             dest='Cassini/PCK', family='Cassini-Nav-PCK')
KernelFile.mutual_veto(_rule.pattern)
_rule = Rule(r'cpck(DDMONYYYY)_Sci\.tpc', mission='CASSINI', source=_source_url('PCK'),
             dest='Cassini/PCK', family='Cassini-Sci-PCK')
KernelFile.mutual_veto(_rule.pattern)




# PCKs
##########################################################################################

from ._CASSINI_ROCK_PCKS import _CASSINI_ROCK_PCKS
from ._SATURN_MST_PCKS import _SATURN_MST_PCKS
from ._SATURN_SSD_PCKS import _SATURN_SSD_PCKS

_pck_source = _SOURCE_URL + 'pck'

_name_to_body_id_set = {k.lower():{v} for k,v in BODY_IDS.items() if '_' not in k}

_rule1 = _Rule(r'cpck_rock_(DDMONYYYY)\.tpc', version=1, origin='CASSINI',
               planet='SATURN', source=_pck_source, dest='Saturn/PCK-Cassini',
               family='Saturn-Cassini-PCK')
_KernelFile.mutual_veto(_rule1.pattern)  # never more than one furnished

_rule2 = _Rule(r'(?P<naif_ids>[a-z]+)_mst201[38]\.bpc', version=1,
               naif_ids=_name_to_body_id_set, origin='MST', planet='SATURN',
               source=_pck_source, dest='Saturn/PCK-MST', family='Saturn-MST-PCK')
_KernelFile.shadow(_rule2.pattern, r'pck\d{5}\.tpc')  # supersedes any standard PCK

_rule3 = _Rule(r'(?P<naif_ids>[a-z]+)_ssd_(YYMMDD)_v(N+)\.tpc',
               naif_ids=_name_to_body_id_set, source=_pck_source, dest='Saturn/PCK-SSD',
               origin='SSD', planet='SATURN', family='Saturn-SSD-PCK')
_KernelFile.shadow(r'enceladus_ssd_.*\.tpc', r'pck\d{5}.*\.tpc')
# According to the documentation, the Enceladus file can only be furnished above the
# general PCK.

# Precedence order is Cassini, MST, SSD.
def _pck_sort_order(basename):
    """Sort key for Saturn PCKs, in increasing order of precedence.

    The Cassini-origin kernels sort lowest, then the MST kernels describing the irregular
    rotations of the inner moons, then everything else.

    Parameters:
        basename (str): Basename of a Saturn PCK.

    Returns:
        (int, str): The sort key for this basename.
    """

    if basename[:4] == 'cpck':
        return (0, basename)
    if '_mst201' in basename:
        return (1, basename)
    return (2, basename)

_pck_notes = """\
    This function generates a Planetary Constants Kernel object for one or more of
    Saturn's satellites. It should be furnished after one of the general PCKs from NAIF,
    e.g., "pck00011.tpc", so that it takes precedence.

    The "CASSINI"-origin kernels include masses, shapes, and rotation states of all the
    small, inner moons and all of the irregular satellites known at the time of the
    mission.

    The "MST" kernels provide binary descriptions of the irregular rotations of the inner
    moons Aegaeon, Atlas, Calypso, Daphnis, Enceladus, Epimetheus, Helene, Janus, Methone,
    Pallene, Pan, and Telesto.

    The "SSD" kernels currently only provides a new rotation model for Enceladus.
"""

_pck_docstrings = {'origin': """\
        origin (str, set, list, tuple, optional): "CASSINI" for one of the Cassini "rocks"
            PCKs; "MST" for one or more of the inner satellite rotation models by Matt
            Tiscareno; "SSD" for a later rotation model from JPL's Solar System Dynamics
            team. To combine multiple sources, use a set, list, or tuple; None uses all
            sources.
"""}

pck = _spicefunc('pck',
                 title = 'Saturn satellite PCK',
                 known = _CASSINI_ROCK_PCKS + _SATURN_MST_PCKS + _SATURN_SSD_PCKS,
                 unknown = (_rule1.pattern, _rule3.pattern),
                 source = _pck_source,
                 sort = _pck_sort_order,
                 exclude = False,
                 reduce = True,
                 notes = _pck_notes,
                 docstrings = _pck_docstrings)


##########################################################################################
