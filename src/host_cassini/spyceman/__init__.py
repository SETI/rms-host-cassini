##########################################################################################
# host_cassini/_spyceman/__init__.py
##########################################################################################
"""host_cassini._spyceman: Support for Cassini-specific kernels.

Attributes:
    NAME (str): "CASSINI".
    HOST_ID (int): The body ID of Cassini, -82.
    FRAME_IDS (dict): A mapping from frame name or (instrument, component) to frame ID.
    FRAME_NAMES (dict[int, str]): A mapping from frame ID to frame name.
    INSTRUMENT_NAMES (set): All the instrument names.
    INSTRUMENT_IDS (dict[str, set[int]]): A mapping from each instrument to the associated
        frame IDs.

Methods:
    ck: Function returning a C (pointing) kernel for any part of the mission.
    fk: Function returning a frames kernel.
    ik: Function returning an instrument kernel.
    sclk: Function returning a spacecraft clock kernel.
    spk: Function returning an SP (trajectory) kernel for any part of the mission.
    meta: Function returning a Metakernel containing all of the above.
    gapfill_ck: A low-precedence "gapfill" CK for the Saturn tour.
    jupiter_ck: A CK for the Jupiter flyby.
    cruise_spk: A SPK for the cruise to Saturn including the Jupiter flyby.
    small_satellite_spk: A SPK for Saturn's small satellites.
    irregular_satellite_spk: A SPK for Saturn's irregular satellites.
"""

from spyceman._spicefunc  import _spicefunc
from spyceman.kernelfile  import KernelFile
from spyceman.metakernel  import Metakernel
from spyceman.rule        import Rule
from spyceman.solarsystem import Jupiter, Saturn

NAME = 'CASSINI'
HOST_ID = -82

INSTRUMENT_NAMES = {'CAPS', 'CDA', 'CIRS', 'INMS', 'ISS', 'MAG', 'MIMI', 'RADAR', 'RPWS',
                    'RSS', 'UVIS', 'VIMS'}

FRAME_IDS = {   # from cas_v43.tf
    'SC_COORD'     : -82000,
    'HGA'          : -82101,
    'LGA1'         : -82102,
    'LGA2'         : -82103,
    'XBAND'        : -82104,
    'KABAND'       : -82105,
    'KUBAND'       : -82106,
    'SBAND'        : -82107,
    'XBAND_TRUE'   : -82108,
    'ISS_NAC'      : -82360,
    'ISS_WAC'      : -82361,
    'CIRS_FP1'     : -82890,
    'CIRS_FP3'     : -82891,
    'CIRS_FP4'     : -82892,
    'CIRS_FPB'     : -82893,
    'UVIS_FUV'     : -82840,
    'UVIS_EUV'     : -82842,
    'UVIS_SOLAR'   : -82843,
    'UVIS_SOL_OFF' : -82849,
    'UVIS_HSP'     : -82844,
    'UVIS_HDAC'    : -82845,
    'VIMS_V'       : -82370,
    'VIMS_IR'      : -82371,
    'VIMS_IR_SOL'  : -82372,
    'CAPS'         : -82820,
    'CDA'          : -82790,
    'INMS'         : -82740,
    'MAG_PLUS'     : -82350,
    'MAG_MINUS'    : -82351,
    'MIMI_CHEMS'   : -82760,
    'MIMI_INCA'    : -82761,
    'MIMI_LEMMS1'  : -82762,
    'MIMI_LEMMS2'  : -82763,
    'RADAR_1'      : -82810,
    'RADAR_2'      : -82811,
    'RADAR_3'      : -82812,
    'RADAR_4'      : -82813,
    'RADAR_5'      : -82814,
    'RPWS'         : -82730,
    'RPWS_EXPLUS'  : -82731,
    'RPWS_EXMINUS' : -82732,
    'RPWS_EZPLUS'  : -82733,
    'RPWS_LP'      : -82734,
    'RPWS_EDIPOLE' : -82735,
    'RPWS_EXZPLANE': -82736,
    'RPWS_EXPZP'   : -82737,
    'RPWS_EXMZP'   : -82738,
    'RPWS_EDPZP'   : -82739,
}

FRAME_NAMES = {v:k for k,v in FRAME_IDS.items()}

FRAME_IDS['SC'] = FRAME_IDS['SC_COORD']     # alt name

# Add keys that are tuples (instrument, component) and then clean up
for _key in list(FRAME_IDS.keys()):
    _parts = _key.partition('_')
    if _parts[1] == '_':
        FRAME_IDS[_parts[0], _parts[2]] = FRAME_IDS[_key]

del FRAME_IDS['SC', 'COORD']    # "SC_COORD" has a different use of underscore

# RSS is not in the frame names for the antennas
for _key in ('HGA', 'LGA1', 'LGA2', 'XBAND', 'KABAND', 'KUBAND', 'SBAND', 'XBAND_TRUE'):
    FRAME_IDS['RSS', _key] = FRAME_IDS[_key]

INSTRUMENT_IDS = {_key: set() for _key in INSTRUMENT_NAMES}
for _key, _frame_id in FRAME_IDS.items():
    if _key in INSTRUMENT_NAMES:
        INSTRUMENT_IDS[_key].add(_frame_id)
    elif isinstance(_key, tuple) and _key[0] in INSTRUMENT_NAMES:
        INSTRUMENT_IDS[_key[0]].add(_frame_id)

# mission_phase -> time limits
_DEFAULT_TIMES = {
    'VENUS'   : ('1998-04-18', '1999-06-25'),
    'EARTH'   : ('1999-08-16', '1999-09-15'),
    'MASURSKY': ('2000-01-23', '2000-01-24'),
    'JUPITER' : ('1999-08-19', '2001-03-24'),
    'SATURN'  : ('2004-01-01', '2017-09-16'),
}

# mission_phase -> set of body IDs in cruise SPKs
# During the Jupiter flyby, the inner satellites plus Himalia, Elara, Pasiphae, Sinope,
# Lysithea, Carme, Ananke, and Leda are always included.
_DEFAULT_CRUISE_BODY_IDS = {
    'VENUS'   : {HOST_ID, 2, 299},
    'EARTH'   : {HOST_ID, 3, 301, 399},
    'MASURSKY': {HOST_ID, 2002685},
    'JUPITER' : {HOST_ID, 5, 599} | set(range(501, 517)),
    'SATURN'  : {HOST_ID} | Saturn.SYSTEM
}

_DEFAULT_BODY_IDS = _DEFAULT_CRUISE_BODY_IDS.copy()
_DEFAULT_BODY_IDS['SATURN'] = {HOST_ID} | Saturn.SYSTEM

# [planet, irregulars] -> set of body IDs with or without irregulars
_DEFAULT_BODY_IDS_W_IRREGULARS = {}
for _planet, _ids in _DEFAULT_BODY_IDS.items():
    _DEFAULT_BODY_IDS_W_IRREGULARS[_planet, False] = _ids
    _DEFAULT_BODY_IDS_W_IRREGULARS[_planet, True] = _ids.copy()

# Overrides...
_DEFAULT_BODY_IDS_W_IRREGULARS['JUPITER', True] |= Jupiter.ALL_IDS
_DEFAULT_BODY_IDS_W_IRREGULARS['SATURN', True] |= Saturn.ALL_IDS

##########################################################################################
# Utilities
##########################################################################################

def _source_url(ktype):
    """The online directories that hold Cassini kernels of the given type.

    Parameters:
        ktype (str): Kernel type, e.g., "SPK" or "CK". Case is not significant; the value
            is lower-cased to form the final path element of each URL.

    Returns:
        (str, str): The NAIF Cassini kernel directory and the PDS archive directory for
            this kernel type, in that order of preference.
    """

    _SOURCE1 = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/'
    _SOURCE2 = ('https://naif.jpl.nasa.gov/pub/naif/pds/data'
                '/co-s_j_e_v-spice-6-v1.0/cosp_1000/data/')
    ktype = ktype.lower()
    return (_SOURCE1 + ktype, _SOURCE2 + ktype)

##########################################################################################
# FKs
##########################################################################################

from ._CASSINI_FKS import _CASSINI_FKS

_rule = Rule(r'cas_v(NN)\.tf', mission='CASSINI', source=_source_url('FK'),
             dest='Cassini/FK')
KernelFile.mutual_veto(_rule.pattern)

# `_source_url()` returns one URL per archive, so the release subdirectories have
# to be built for each of them.
_fk_bases = _source_url('FK')
_fk_source = list(_fk_bases) + [f'{_url}/release.{i:02d}'
                                for _url in _fk_bases
                                for i in range(1, 14)]

_fk_notes = """\
    The final Cassini frames kernel is version 43.
"""

fk = _spicefunc('fk',
                title = 'Cassini FK',
                known = _CASSINI_FKS,
                unknown = _rule.pattern,
                source = _fk_source,
                exclude = True,      # never more than one
                notes = _fk_notes)

##########################################################################################
# IKs
##########################################################################################

from ._CASSINI_IKS import _CASSINI_IKS

_rule = Rule(r'cas_(?P<instrument>[a-z]+)_v(NN)\.ti', mission='CASSINI',
             instrument=str.upper, source=_source_url('IK'), dest='Cassini/IK')
KernelFile.veto(r'cas_([a-z]+)_v(\d\d)\.ti', r'cas_\1_v\d\d\.ti')

# _source_url() returns one URL per archive, so the release subdirectories have to be
# built for each of them.
_ik_bases = _source_url('IK')
_ik_source = list(_ik_bases) + [f'{_url}/release.{i:02d}'
                                for _url in _ik_bases
                                for i in range(1, 14)]

_ik_notes = """\
    This function returns an Instrument Kernel object for one or more instruments.

    The final versions of these files are: CAPS=3, CDA=1, CIRS=10, INMS=2, ISS=10, MAG=1,
    MIMI=11, RADAR=11, RPWS=1, RSS=3, UVIS=7, VIMS=6.
"""

_ik_docstrings = {'instrument': """\
        instrument (str, set, list, tuple, optional): One or more of the Cassini
            instruments: "CAPS", "CDA", "CIRS", "INMS", "ISS", "MAG", "MIMI", "RADAR",
            "RPWS", "RSS", "UVIS", or "VIMS". Use None or an empty set to include all
            instruments.
"""}

ik = _spicefunc('ik',
                title = 'Cassini IK',
                known = _CASSINI_IKS,
                unknown = _rule.pattern,
                source = _ik_source,
                exclude = ('instrument',),
                notes = _ik_notes,
                docstrings = _ik_docstrings)

##########################################################################################
# SCLKs
##########################################################################################

from ._CASSINI_SCLKS import _CASSINI_SCLKS

_rule = Rule(r'cas(NNNNN)\.tsc', mission='CASSINI', source=_source_url('SCLK'),
             dest='Cassini/SCLK')
KernelFile.mutual_veto(_rule.pattern)

_sclk_notes = """\
    The final Cassini clock kernel is version 172. However, it differs from all prior
    clock kernels in that it corrects an error of nearly one second that appeared in
    earlier kernels. Users who which to reconstruct work that they did prior to 2018 might
    consider using version 171 instead.
"""

sclk = _spicefunc('sclk',
                  title = 'Cassini SCLK',
                  known = _CASSINI_SCLKS,
                  unknown = _rule.pattern,
                  source = _source_url('SCLK'),
                  exclude = True,   # never more than one
                  notes = _sclk_notes)

##########################################################################################
# Metakernel...
##########################################################################################

def meta(planet=None, instrument=None, *, irregular=False,
         tmin=None, tmax=None, ids=None,
         ck={}, fk={}, ik={}, sclk={}, spk={}, **keywords):
    """A metakernel object for the Cassini mission or any subset thereof.

    Parameters:
        planet (str, set, list, tuple, optional): One or more of "VENUS", "EARTH",
            "MASURSKY", "JUPITER", or "SATURN", naming the phase or phases of the Cassini
            mission this metakernel should cover. "SATURN" is the default.
        instrument (str, set, list, tuple, optional): One or more of "ISS", "MAG",
            "MIMI", "RADAR", "RPWS", "RSS", "UVIS", or "VIMS". Use None or an empty set
            to include all instruments.
        irregular (bool, optional): True to include Saturn's irregular satellites in the
            SPK.
        tmin (float, str, optional): Lower time limit in seconds TDB or as a date-time
            string; None for all times.
        tmax (float, str, optional): Upper time limit in seconds TDB or as a date-time
            string; None for all times.
        ids (int, set[int], optional): A NAIF ID or set of NAIF IDs; None to ignore NAIF
            IDs.
        ck (dict, optional): Non-default inputs to the ck() function.
        fk (dict, optional): Non-default inputs to the fk() function.
        ik (dict, optional): Non-default inputs to the ik() function.
        sclk (dict, optional): Non-default inputs to the sclk() function.
        spk (dict, optional): Non-default inputs to the spk() function.
        **keywords: Additional inputs applied to every one of the function calls.

    Returns:
        Metakernel: A Metakernel object covering the selected mission phases.
    """

    # _meta avoids the name conflicts between ck the input and ck the function, etc.
    return _meta(planet=planet, instrument=instrument, irregular=irregular,
                 tmin=tmin, tmax=tmax, ids=ids,
                 ck_=ck, fk_=fk, ik_=ik, sclk_=sclk, spk_=spk, **keywords)


def _meta(*, planet, instrument, irregular, tmin, tmax, ids,
          ck_, fk_, ik_, pck_, sclk_, spk_, **keywords):
    """Assemble the Cassini metakernel from one call to each kernel function.

    This exists separately from meta() because there, the names ck, fk, ik, sclk, and spk
    each refer to an input dictionary, shadowing the kernel function of the same name.

    Parameters:
        planet (str, set, list, tuple, optional): The mission phases to include.
        instrument (str, set, list, tuple, optional): The instruments to include.
        irregular (bool): True to include Saturn's irregular satellites.
        tmin (float, str, optional): Lower time limit in seconds TDB or as a date-time
            string.
        tmax (float, str, optional): Upper time limit in seconds TDB or as a date-time
            string.
        ids (int, set[int], optional): A NAIF ID or set of NAIF IDs.
        ck_ (dict): Non-default inputs to the ck() function.
        fk_ (dict): Non-default inputs to the fk() function.
        ik_ (dict): Non-default inputs to the ik() function.
        pck_ (dict): Non-default inputs to the pck() function.
        sclk_ (dict): Non-default inputs to the sclk() function.
        spk_ (dict): Non-default inputs to the spk() function.
        **keywords: Additional inputs merged into each of the dictionaries above.

    Returns:
        Metakernel: A Metakernel object built from one kernel of each ktype.
    """

    for dict_ in (ck_, fk_, ik_, pck_, sclk_, spk_):
        for k, v in keywords.items():
            if k not in dict_:
                dict_[k] = v

    ck_   = ck(planet=planet, instrument=instrument,
               tmin=tmin, tmax=tmax, ids=ids, **ck_)
    fk_   = fk(ids=ids, **fk_)
    ik_   = ik(instrument=instrument, ids=ids, **ik_)
    sclk_ = sclk(**sclk_)
    spk_  = spk(planet=planet, irregular=irregular,
                tmin=tmin, tmax=tmax, ids=ids, **spk_)

    return Metakernel([ck_, fk_, ik_, sclk_, spk_])

##########################################################################################
# CKs and SPKs (delayed import)
##########################################################################################

from .ck  import gapfill_ck, jupiter_ck, ck
from .spk import cruise_spk, small_satellite_spk, irregular_satellite_spk, spk

__all__ = ['ck', 'cruise_spk', 'fk', 'gapfill_ck', 'ik', 'irregular_satellite_spk',
           'jupiter_ck', 'meta', 'sclk', 'small_satellite_spk', 'spk', 'HOST_ID',
           'FRAME_IDS', 'FRAME_NAMES', 'INSTRUMENT_IDS', 'INSTRUMENT_NAMES', 'NAME']

##########################################################################################
