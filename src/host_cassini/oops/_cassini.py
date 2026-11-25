##########################################################################################
# host_cassini/oops/_cassini.py
##########################################################################################

import contextlib
import re

import numpy as np

import julian
import spicedb
import cspyce
from vicar     import VicarError

import oops
from oops.body import Body


# There are two C-matrix conventions here, related by _CMATRIX_ROTATION (a 180-degree spin
# about the boresight):
#   * spice-frame: the pointing straight from SPICE (a CK or cspyce.pxform), i.e.
#     the J2000 -> CASSINI_ISS_<camera> rotation. Axes: z along the line of
#     sight, x to the left, y up.
#   * oops-frame: the oops observation frame that the FOV uses. Axes follow the
#     oops convention: z along the line of sight, x to the right, y down.
# oops-frame = _CMATRIX_ROTATION * spice-frame. The instrument's internal coordinate
# system matches the oops-frame, so a recorded (spice-frame) C-matrix is rotated by
# _CMATRIX_ROTATION to build the observation frame. The generic
# Snapshot.get_spice_cmatrix() uses this matrix as attribute `spice_to_frame`.
_CMATRIX_ROTATION = oops.Matrix3([[-1,0,0],[0,-1,0],[0,0,1]])

# The NAIF ID of the Cassini spacecraft clock
_CASSINI_SCLK = -82

# The subfields that every Cassini observation carries about its pointing: the C kernels
# tabulate it for the spacecraft, -82000 (CASSINI_SC_COORD), against the Cassini clock,
# and every instrument frame is a frame chain from SPICE, never a frozen C-matrix.
_POINTING_SUBFIELDS = {
    'ck_object_id'         : -82000,
    'ck_sclk_id'           : _CASSINI_SCLK,
    'spice_frame_is_frozen': False,
}

# The cruise C kernels, in Cassini/CK-cruise/ under the SPICE resources directory, each
# with the start and stop of its coverage, from ckcov. The SPICE database lists none of
# them, and it is never changed, so without this table no cruise image outside the
# Jupiter flyby has pointing (issue 42). The table goes when the kernel database is
# replaced. The files are listed by start time, so that a later one is furnished after,
# and takes precedence over, an earlier one that it overlaps.
_CRUISE_CK_DIRECTORY = 'Cassini/CK-cruise/'
_CRUISE_CKS = (
    ('97288_98002rc.bc', '1997-10-15T00:01:15.375', '1998-01-02T00:56:56.743'),
    ('98002_98091rc.bc', '1998-01-02T00:56:56.743', '1998-04-01T00:04:10.096'),
    ('98091_98185rc.bc', '1998-04-01T00:04:10.096', '1998-07-04T12:00:32.369'),
    ('98185_98279rc.bc', '1998-07-04T12:00:32.369', '1998-10-06T00:10:58.052'),
    ('98279_99001rc.bc', '1998-10-06T00:10:58.052', '1999-01-01T00:00:02.445'),
    ('99001_99093rc.bc', '1999-01-01T00:00:02.445', '1999-04-03T04:03:13.494'),
    ('99093_99182rc.bc', '1999-04-03T04:03:13.494', '1999-07-01T00:00:27.021'),
    ('99182_99274rc.bc', '1999-07-01T00:00:27.021', '1999-10-01T00:04:36.747'),
    ('99274_00001rc.bc', '1999-10-01T00:04:36.747', '2000-01-01T02:09:34.947'),
    ('00001_00092rc.bc', '2000-01-01T02:09:34.947', '2000-04-01T00:00:03.152'),
    ('00092_00183rc.bc', '2000-04-01T00:00:03.152', '2000-07-01T00:00:03.362'),
    ('00183_00275rc.bc', '2000-07-01T00:00:03.362', '2000-10-01T00:00:01.695'),
    ('00275_01001rc.bc', '2000-10-01T00:00:01.695', '2001-01-01T02:39:40.851'),
    ('01001_01091rc.bc', '2001-01-01T02:39:40.851', '2001-04-01T00:00:03.117'),
    ('01091_01182rc.bc', '2001-04-01T00:00:03.117', '2001-07-01T00:00:04.053'),
    ('01182_01274rc.bc', '2001-07-01T00:00:04.053', '2001-10-01T00:00:03.826'),
    ('01274_02001rc.bc', '2001-10-01T00:00:03.826', '2002-01-01T00:00:03.986'),
    ('02001_02091rc.bc', '2002-01-01T00:00:03.986', '2002-04-01T00:00:01.067'),
    ('02091_02182rc.bc', '2002-04-01T00:00:01.067', '2002-07-01T00:00:01.346'),
    ('02182_02274rc.bc', '2002-07-01T00:00:01.346', '2002-10-01T00:00:01.236'),
    ('02274_03001rc.bc', '2002-10-01T00:00:01.236', '2003-01-01T00:00:01.191'),
    ('03001_03091rc.bc', '2003-01-01T00:00:01.191', '2003-04-01T00:00:02.212'),
    ('03091_03182rc.bc', '2003-04-01T00:00:02.212', '2003-07-01T00:00:01.770'),
    ('03182_03274rc.bc', '2003-07-01T00:00:01.770', '2003-10-01T00:00:01.872'),
    ('03274_04001rc.bc', '2003-10-01T00:00:01.872', '2004-01-01T00:00:02.006'),
    ('03313_03323ra.bc', '2003-11-09T00:00:03.891', '2003-11-18T23:59:58.009'),
    ('03323_03333ra.bc', '2003-11-19T00:00:02.009', '2003-11-28T23:59:56.377'),
    ('03333_03343ra.bc', '2003-11-29T00:00:00.377', '2003-12-07T05:40:19.728'),
    ('03343_03353ra.bc', '2003-12-09T03:46:54.643', '2003-12-18T23:56:45.331'),
    ('03353_03363ra.bc', '2003-12-19T00:14:33.324', '2003-12-28T23:59:59.695'),
    ('03363_04002ra.bc', '2003-12-29T00:00:03.695', '2004-01-01T23:59:57.442'),
)

# The kernel set name and load priority given to the cruise C kernels, below the 100 of
# CAS-CK-RECONSTRUCTED in the SPICE database
_CRUISE_CK_NAME = 'CAS-CK-CRUISE'
_CRUISE_CK_PRIORITY = 90


class _Cassini:
    """An instance-free class to hold Cassini-specific parameters.

    Attributes:
        TOUR (int): A rough time in seconds TDB dividing the Jupiter flyby from the
            Saturn tour; observations before it are treated as Jupiter observations.
        START_TIME (str): The start of the mission as an ISO date.
        STOP_TIME (str): The end of the mission as an ISO date.
        MONTHS (int): The number of equal "months" into which the mission is divided for
            the purpose of loading kernels.
        TDB0 (float): The mission start time in seconds TDB.
        TDB1 (float): The mission stop time in seconds TDB.
        DTDB (float): The duration of one "month" in seconds.
        SLOP (float): The margin, in seconds, by which each month is extended when
            deciding which kernels apply to it.
        CK_LOADED (numpy.ndarray): Boolean array with one flag per month, True if the C
            kernels for that month have been furnished.
        CK_LIST (numpy.ndarray): Object array holding, for each month, the list of
            KernelInfo objects for the C kernels needed within that month.
        CK_DICT (dict[str, KernelInfo]): The furnished C kernels, keyed by filespec.
        SPK_LOADED (numpy.ndarray): Boolean array with one flag per month, True if the SP
            kernels for that month have been furnished.
        SPK_LIST (numpy.ndarray): Object array holding, for each month, the list of
            KernelInfo objects for the SP kernels needed within that month.
        SPK_DICT (dict[str, KernelInfo]): The furnished SP kernels, keyed by filespec.
        loaded_instruments (list[str]): The names of the instruments whose kernels have
            been loaded.
        initialized (bool): True after :meth:`initialize` has been called.
    """

    TOUR = (2003 - 2000) * 365 * 86400      # Rough ET dividing Saturn from Jupiter
    START_TIME = '1997-10-01'
    STOP_TIME  = '2017-10-01'
    MONTHS = 240        # 20 years * 12 months/year

    TDB0 = julian.tdb_from_tai(julian.tai_from_iso(START_TIME))
    TDB1 = julian.tdb_from_tai(julian.tai_from_iso(STOP_TIME))
    DTDB = (TDB1 - TDB0) / MONTHS
    SLOP = 43200.

    # We load CK and SPK files on a very rough month-by-month basis. This is simpler than
    # a more granular approach involving detailed calendar calculations. We divide the
    # period October 1, 1997 to October 1, 2017 up into 240 "months" of equal length.
    # Given any TDB, we quickly determine the month within which it falls. Each month is
    # associated with a list of kernels that should be loaded whenever information is
    # needed about any time within that month +/- 12 hours. The kernels needed for a given
    # month only get loaded when they are needed, and are only loaded once. For any
    # geometry calculation involving Cassini, a quick call to load_ck(time) or
    # load_spk(time) will ensure that the information is available.

    CK_LOADED = np.zeros(MONTHS, dtype='bool')      # True if month was loaded
    CK_LIST   = np.empty(MONTHS, dtype='object')    # Kernels needed by month
    CK_DICT   = {}      # Dictionary keyed by filespec returns kernel info object, but
                        # only if loaded.

    SPK_LOADED  = np.zeros(MONTHS, dtype='bool')
    SPK_LIST    = np.empty(MONTHS, dtype='object')
    SPK_DICT    = {}

    loaded_instruments = []

    initialized = False

    @staticmethod
    def initialize(*, ck='reconstructed', spk='reconstructed', gapfill=True,
                   planets=None, asof=None, mst_pck=True, irregulars=True):
        """Initialize the Cassini mission internals.

        After the first call, later calls to this function are ignored.

        Parameters:
            ck (str, optional): The set of C kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no C kernels automatically,
                leaving their handling to the caller. 'reconstructed' also loads the
                reconstructed C kernels of the Jupiter flyby and the cruise C kernels,
                which the reconstructed kernels take precedence over where both cover a
                time.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            planets (list, optional): A list of planets to pass to
                :meth:`~oops.Body.define_solar_system`. None or 0 means all.
            asof (str, optional): Only use SPICE kernels that existed before this date;
                None to ignore.
            mst_pck (bool, optional): True to include MST PCKs, which update the rotation
                models for some of the small moons.
            irregulars (bool, optional): True to include the irregular satellites; False
                otherwise.
        """

        if _Cassini.initialized:
            return

        # Make sure the leap seconds have been loaded
        oops.spice.load_leap_seconds()

        # Define some important paths and frames
        Body.define_solar_system(_Cassini.START_TIME, _Cassini.STOP_TIME, asof=asof,
                                 planets=planets, mst_pck=mst_pck, irregulars=irregulars)

        _ = oops.path.SpicePath('CASSINI', 'SATURN')

        spicedb.open_db()

        spk = spk.upper()
        if spk == 'NONE':

            # This means no SPK will ever be loaded; handling is manual
            _Cassini.initialize_kernels([], _Cassini.SPK_LIST)
            _Cassini.SPK_LOADED = np.ones(_Cassini.MONTHS, dtype='bool')
        else:
            kernels = spicedb.select_spk(-82, name='CAS-SPK-' + spk,
                                         time=(_Cassini.START_TIME, _Cassini.STOP_TIME),
                                         asof=asof)
            _Cassini.initialize_kernels(kernels, _Cassini.SPK_LIST)

        ck = ck.upper()
        if ck == 'NONE':

            # This means no CK will ever be loaded; handling is manual
            _Cassini.initialize_kernels([], _Cassini.CK_LIST)
            _Cassini.CK_LOADED = np.ones(_Cassini.MONTHS, dtype='bool')
        else:
            kernels = spicedb.select_ck(-82, name='CAS-CK-' + ck,
                                        time=(_Cassini.START_TIME, _Cassini.STOP_TIME),
                                        asof=asof)

            # The reconstructed C kernels of the Jupiter flyby are a set of their own. The
            # cruise C kernels come first, so that reconstructed pointing, furnished after
            # them, takes precedence where both cover a time, as late in 2003.
            if ck == 'RECONSTRUCTED':
                kernels = (_cruise_kernels() + kernels
                           + spicedb.select_ck(-82, name='CAS-CK-JUP',
                                               time=(_Cassini.START_TIME,
                                                     _Cassini.STOP_TIME),
                                               asof=asof))

            _Cassini.initialize_kernels(kernels, _Cassini.CK_LIST)

        # Load extra kernels if necessary
        if gapfill and ck not in ('PREDICTED', 'NONE'):
            _ = spicedb.furnish_ck(-82, name='CAS-CK-GAPFILL')

        spicedb.close_db()

        _Cassini.initialized = True

    @staticmethod
    def reset():
        """Reset the internal parameters.

        Can be useful for debugging.
        """
        _Cassini.loaded_instruments = []

        _Cassini.CK_LOADED = np.zeros(_Cassini.MONTHS, dtype='bool')
        _Cassini.CK_LIST = np.empty(_Cassini.MONTHS, dtype='object')
        _Cassini.CK_DICT = {}

        _Cassini.SPK_LOADED = np.zeros(_Cassini.MONTHS, dtype='bool')
        _Cassini.SPK_LIST = np.empty(_Cassini.MONTHS, dtype='object')
        _Cassini.SPK_DICT = {}

        _Cassini.initialized = False

    @staticmethod
    def load_ck(t):
        """Furnish the C kernels applicable at or near a given time.

        The time can be tai or tdb.

        Parameters:
            t (float): The time in seconds.
        """
        _Cassini.load_kernels(t, t, _Cassini.CK_LOADED, _Cassini.CK_LIST,
                              _Cassini.CK_DICT)

    @staticmethod
    def load_cks(t0, t1):
        """Furnish the C kernels applicable within a time interval.

        Every C kernel applicable near or within the interval `t0` to `t1` is furnished.

        The time can be tai or tdb.

        Parameters:
            t0 (float): The start time in seconds.
            t1 (float): The stop time in seconds.
        """
        _Cassini.load_kernels(t0, t1, _Cassini.CK_LOADED, _Cassini.CK_LIST,
                              _Cassini.CK_DICT)

    @staticmethod
    def load_spk(t):
        """Furnish the SPK kernels applicable at or near a given time.

        The time can be tai or tdb.

        Parameters:
            t (float): The time in seconds.
        """
        _Cassini.load_kernels(t, t, _Cassini.SPK_LOADED, _Cassini.SPK_LIST,
                              _Cassini.SPK_DICT)

    @staticmethod
    def load_spks(t0, t1):
        """Furnish the SPK kernels applicable within a time interval.

        Every SPK kernel applicable near or within the interval `t0` to `t1` is
        furnished.

        The time can be tai or tdb.

        Parameters:
            t0 (float): The start time in seconds.
            t1 (float): The stop time in seconds.
        """
        _Cassini.load_kernels(t0, t1, _Cassini.SPK_LOADED, _Cassini.SPK_LIST,
                              _Cassini.SPK_DICT)

    @staticmethod
    def load_kernels(t0, t1, loaded, lists, kernel_dict):
        """Furnish every kernel in the given lists that applies within a time range.

        The kernels of a month are furnished in the order of its list, so that a later
        kernel takes precedence over an earlier one wherever both cover a time. A kernel
        that an earlier month furnished is furnished again after the first kernel of the
        month that is new, so that the order of the list holds across months too.

        Parameters:
            t0 (float): The start time in seconds TDB.
            t1 (float): The stop time in seconds TDB.
            loaded (numpy.ndarray): Boolean array with one flag per month, True if that
                month's kernels have already been furnished; it is updated in place.
            lists (numpy.ndarray): Object array holding, for each month of the mission,
                the list of KernelInfo objects needed within that month.
            kernel_dict (dict[str, KernelInfo]): The furnished KernelInfo objects, keyed
                by filespec; it is updated in place.
        """

        # Find the range of months needed
        m1 = int((t0 - _Cassini.TDB0) // _Cassini.DTDB)
        m2 = int((t1 - _Cassini.TDB0) // _Cassini.DTDB)

        m1 = max(m1, 0)         # ignore time limits outside mission duration
        m2 = min(m2, _Cassini.MONTHS - 1)

        # Load any months not already loaded. SPICE gives precedence to the kernel
        # furnished last, so a kernel already furnished that follows a new one in the
        # month's list is furnished again after it; otherwise a cruise C kernel new to
        # this month would take precedence over the reconstructed kernel that an earlier
        # month furnished (issue 42).
        for m in range(m1, m2+1):
            if not loaded[m]:
                new = [k for (k, kernel) in enumerate(lists[m])
                       if kernel.filespec not in kernel_dict]
                if new:
                    kernels = lists[m][new[0]:]
                    spicedb.furnish_kernels(kernels, fast=False)
                    kernel_dict.update((kernel.filespec, kernel) for kernel in kernels)
                loaded[m] = True

    ######################################################################################
    # Initialize the kernel lists
    ######################################################################################

    @staticmethod
    def initialize_kernels(kernels, lists):
        """Initialize the monthly lists of KernelInfo objects.

        After initialization, `lists[m]` holds the KernelInfo objects needed within the
        specified month.

        Parameters:
            kernels (list[KernelInfo]): The KernelInfo objects to distribute among the
                months of the mission.
            lists (numpy.ndarray): Object array with one entry per month; each entry is
                replaced by the list of KernelInfo objects that apply within that month,
                extended by :attr:`SLOP` at each end.
        """
        for i in range(_Cassini.MONTHS):
            lists[i] = []

        for kernel in kernels:

            # Find the range of months applicable, extended by 12 hours
            t0 = cspyce.str2et(kernel.start_time) - _Cassini.SLOP
            t1 = cspyce.str2et(kernel.stop_time)  + _Cassini.SLOP

            m1 = int((t0 - _Cassini.TDB0) // _Cassini.DTDB)
            m2 = int((t1 - _Cassini.TDB0) // _Cassini.DTDB)

            m1 = max(m1, 0)     # ignore time limits outside mission duration
            m2 = min(m2, _Cassini.MONTHS - 1)

            # Add this kernel to each month's list
            for m in range(m1, m2+1):
                lists[m] += [kernel]

    ######################################################################################
    # Routines for managing the loading other kernels
    ######################################################################################

    @staticmethod
    def load_instruments(instruments=None, asof=None):
        """Load the SPICE kernels for the Cassini mission.

        This also defines the basic paths and frames.

        It is generally only to be called once.

        Parameters:
            instruments (list[str], optional): The names of the instruments whose kernels
                to load. On the first call, 'ISS', 'VIMS', 'CIRS' and 'UVIS' are always
                included.
            asof (str, optional): If this specifies a date or date-time in ISO format,
                then only kernels that existed before the specified date are used.
                Otherwise, the most recent versions are always loaded.
        """

        # Copy so that the caller's list is never modified
        instruments = [] if instruments is None else list(instruments)

        # Load the default instruments on the first pass
        if _Cassini.loaded_instruments == []:
            instruments += ['ISS', 'VIMS', 'CIRS', 'UVIS']

        # Skip instruments that are already loaded, and duplicates
        instruments = [inst for inst in dict.fromkeys(instruments)
                       if inst not in _Cassini.loaded_instruments]

        # On later calls, return quickly if there's nothing to do
        if instruments == []:
            return

        # Check the formatting of the "as of" date
        if asof is not None:
            (day, sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        # Furnish instruments and frames
        spicedb.open_db()
        _ = spicedb.furnish_inst(-82, inst=instruments, asof=asof)
        spicedb.close_db()

        _Cassini.loaded_instruments += instruments

    ######################################################################################
    # Routines for managing text kernel information
    ######################################################################################

    @staticmethod
    def spice_instrument_kernel(inst, asof=None):
        """A dictionary containing the Instrument Kernel information.

        Also furnishes it for use by the SPICE tools.

        Parameters:
            inst (str | list | tuple): One of "ISS", "UVIS", "VIMS", "CIRS", etc.
            asof (str, optional): An optional date in the past, in ISO date or date-time
                format. If provided, then the information provided will be applicable as
                of that date. Otherwise, the most recent information is always provided.

        Returns:
            tuple[dict, str]: The dictionary generated by :func:`textkernel.from_file`
            and the name of the kernel.
        """
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_info = spicedb.select_inst(-82, types='IK', inst=inst, asof=asof)
        spicedb.furnish_kernels(kernel_info, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_info), spicedb.as_names(kernel_info)[0])

    @staticmethod
    def spice_frames_kernel(asof=None):
        """A dictionary containing the _Cassini Frames Kernel information.

        Also furnishes the kernels for use by the SPICE tools.

        Parameters:
            asof (str, optional): An optional date in the past, in ISO date or date-time
                format. If provided, then the information provided will be applicable as
                of that date. Otherwise, the most recent information is always provided.

        Returns:
            tuple[dict, list[str]]: The dictionary generated by
                :func:`textkernel.from_file` and an ordered list of the names of the
                kernels.
        """
        if asof is not None:
            (day,sec) = julian.day_sec_from_iso(asof)
            asof = julian.ymdhms_format_from_day_sec(day, sec)

        spicedb.open_db()
        kernel_list = spicedb.select_inst(-82, types='FK', asof=asof)
        spicedb.furnish_kernels(kernel_list, fast=True)
        spicedb.close_db()

        return (spicedb.as_dict(kernel_list), spicedb.as_names(kernel_list))

    @staticmethod
    def used_kernels(time, inst, return_all_planets=False, ck=True):
        """The kernels associated with a _Cassini observation.

        The list covers a selected range of times.

        Parameters:
            time (tuple[float, float]): A (start, stop) tuple of times in seconds TDB.
            inst (str | list | tuple): The instrument name, e.g., 'iss'.
            return_all_planets (bool, optional): Include kernels for all planets not just
                Jupiter or Saturn.
            ck (bool, optional): True (default) to include CK (pointing) kernels; False to
                exclude them, e.g. for an observation whose pointing came from a custom
                cmatrix rather than SPICE, where any furnished CK is unrelated to how the
                observation was actually pointed.

        Returns:
            list[str]: The basenames of the furnished kernels that apply to the
            observation.
        """
        if return_all_planets:
            bodies = [1, 199, 2, 299, 3, 399, 4, 499, 5, 599, 6, 699, 7, 799, 8, 899]
            if time[0] >= _Cassini.TOUR:
                bodies += Body.SATURN_MOONS_LOADED
            else:
                bodies += Body.JUPITER_MOONS_LOADED
        else:
            if time[0] >= _Cassini.TOUR:
                bodies = [6, 699] + Body.SATURN_MOONS_LOADED
            else:
                bodies = [5, 599] + Body.JUPITER_MOONS_LOADED

        types = None
        if not ck:
            types = [t for t in spicedb.KERNEL_TYPE_SORT_ORDER if t != 'CK']

        return spicedb.used_basenames(types=types, time=time, inst=inst, sc=-82,
                                      bodies=bodies)


_OBS_CODE_PATTERN = re.compile(r'(?:ISS|UVIS|VIMS)_\d\d\w\w\w_(.*?)\d\d\d_\w+$')


def _cassini_obs_id_code(obs_id):
    """The observation code from a Cassini Observation ID.

    This is the third element in the OBSERVATION_ID without the trailing three digits.
    """

    match = _OBS_CODE_PATTERN.match(obs_id)
    if not match:
        return ''

    return match.group(1)


def _fix_cassini_ring_target(target, dict_, tstart):
    """The target of a ring observation, or the given target otherwise.

    For ISS and VIMS, an observation is identified as a ring observation by the target
    code in its OBSERVATION_ID, which has the form "ISS_<orbit><code>_<activity>_<suffix>"
    (e.g., "ISS_053RI_PHOTOMDRK002_PRIME"); the ring codes are RI and RA through RG.

    Parameters:
        target (str): The target name from the label, after repairs.
        dict_ (dict): The label or index row, from which OBSERVATION_ID is read.
        tstart (float): The observation start time in seconds TDB, which determines
            whether a ring observation is of Jupiter's rings or Saturn's.

    Returns:
        str: "JUPITER_RING_PLANE" for a ring observation before the Saturn tour,
        "SATURN_RING_PLANE" for one during it, and `target` for any other observation
        or if OBSERVATION_ID is absent or not of the expected form.
    """

    parts = dict_.get('OBSERVATION_ID', '').split('_')
    if len(parts) < 3:
        return target
    if parts[1][-2:] in {'RI', 'RA', 'RB', 'RC', 'RD', 'RE', 'RF', 'RG'}:
        if tstart < _Cassini.TOUR:
            return 'JUPITER_RING_PLANE'
        return 'SATURN_RING_PLANE'

    return target


def _pds3_options_with_defaults(options):
    """The PDS3 option dictionary with defaults.

    The default method is "strict", the one method with which the Cassini hosts read
    every label of the current archive, with the corrections of each host's
    ``_correct_label_text``; the "fast" method silently splits a quoted METHOD_DESC that
    holds "=" over two lines in thousands of ISS labels. Specifying sources=False
    prevents the Host module from overriding with sources=True.

    Parameters:
        options (dict[str, Any] | None): The options given by the caller, if any; not
            modified.

    Returns:
        dict[str, Any]: A copy of `options`, with method "strict" and sources False
        unless it gives others.
    """

    options = {} if options is None else options.copy()
    options.setdefault('method', 'strict')
    options.setdefault('sources', False)
    return options


def _metadata_subfields(meta):
    """The subfields of a UVIS or VIMS observation that come from its metadata.

    Parameters:
        meta (dict[str, Any]): The metadata: the label without its null values, or the
            merged index row.

    Returns:
        dict[str, Any]: The subfields `dict`, which is `meta`, `ck_object_id`,
        `ck_sclk_id` and `spice_frame_is_frozen`, and `label_target` and
        `observation_id` when `meta` gives TARGET_NAME and OBSERVATION_ID.
    """

    subfields = {'dict': meta, **_POINTING_SUBFIELDS}
    for (item, name) in (('TARGET_NAME', 'label_target'),
                         ('OBSERVATION_ID', 'observation_id')):
        if item in meta:
            subfields[name] = meta[item]
    return subfields


def _cruise_kernels():
    """The cruise C kernels, as kernel information for the monthly kernel lists.

    Returns:
        list[KernelInfo]: One for each entry of `_CRUISE_CKS`, in table order, named for
        its file in `_CRUISE_CK_DIRECTORY`, which spicedb resolves against the root of
        the SPICE resources, as it does every file named in the SPICE database.
    """

    return [spicedb.KernelInfo([_CRUISE_CK_NAME, None, 'CK', _CRUISE_CK_DIRECTORY + name,
                                start, stop, None, _CASSINI_SCLK, _CRUISE_CK_PRIORITY,
                                None])
            for (name, start, stop) in _CRUISE_CKS]


def _tdb_from_sclk(count):
    """The time in seconds TDB of a Cassini spacecraft clock count.

    Parameters:
        count (str): The clock count, such as "1294562649.000"; the digits after the
            point count 1/256 s ticks.

    Returns:
        float: The time in seconds TDB, from the Cassini spacecraft clock kernel.
    """

    return cspyce.scs2e(_CASSINI_SCLK, count)


def _sclk_from_tdb(tdb):
    """The Cassini spacecraft clock count of a time, without its partition.

    Parameters:
        tdb (float): The time in seconds TDB.

    Returns:
        str: The clock count, such as "1294562649.000", from the Cassini spacecraft clock
        kernel.
    """

    return cspyce.sce2s(_CASSINI_SCLK, tdb).rpartition('/')[2]

##########################################################################################
# Reading labels and image files
##########################################################################################

# The number that the Cassini labels write for a value that was not determined. It is
# missing only in the keywords that a host lists, where it cannot be a real value, such
# as a temperature or a data rate; elsewhere, as in a position vector, -999 can be real.
_FILL_VALUE = -999.


@contextlib.contextmanager
def _file_errors(filepath):
    """A context within which every label and image file of a Cassini product is read
    and interpreted, so that each failure to do so raises a host error naming the file.

    Within it, a failure to read the VICAR content of a file raises VicarHostError, and a
    KeyError, IndexError, TypeError or ValueError, such as a label that lacks a keyword
    that the host needs, a value that cannot be read as its keyword requires, or an image
    file that ends before its array, raises HostFormatError. Each message begins with the
    file. A host error, an exception of oops itself, and an error that SPICE reports,
    whose message begins with its code, such as "SPICE(CKINSUFFDATA)", pass unchanged.

    Parameters:
        filepath (str | pathlib.Path | FCPath | None): The file that the caller passed,
            the label or the image file, named in the error.

    Raises:
        VicarHostError: For a failure to read VICAR content.
        HostFormatError: For any other failure to read or interpret the file.
    """

    try:
        yield
    except (oops.host.HostError, oops.OopsException):
        raise
    except VicarError as err:
        raise oops.host.VicarHostError(f'{filepath}: {err}') from err
    except (KeyError, IndexError, TypeError, ValueError) as err:
        if _is_spice_error(err):
            raise
        raise oops.host.HostFormatError(f'{filepath}: {type(err).__name__}: {err}'
                                        ) from err


def _is_spice_error(err):
    """True if an exception is an error that SPICE reported through cspyce.

    Parameters:
        err (Exception): The exception.

    Returns:
        bool: True if its message begins with a SPICE error code, as in
        "SPICE(NOFRAMECONNECT) -- ...".
    """

    return (len(err.args) > 0 and isinstance(err.args[0], str)
            and err.args[0].startswith('SPICE('))


def _is_null(value, fill=False):
    """True if a value of a label or header states nothing.

    Parameters:
        value (Any): The value of a keyword.
        fill (bool, optional): True if -999 is a fill value in this keyword.

    Returns:
        bool: True if `value` is None or text that PDS3 writes for no value, as
        ``oops.host.is_pds3_null`` tells ("UNK", "UNKNOWN", "N/A", "NULL", "NOT
        APPLICABLE", "---" or the empty string, in any case and with any surrounding
        blanks); -999 if `fill` is True; or a non-empty list, tuple or set of nothing but
        these.
    """

    if value is None or oops.host.is_pds3_null(value):
        return True
    if isinstance(value, (str, bool)):
        return False
    if isinstance(value, (int, float)):
        return fill and value == _FILL_VALUE
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 0:
        return all(_is_null(item, fill) for item in value)
    return False


def _without_nulls(dict_, fill_keys=frozenset()):
    """A copy of the content of a label or header without its missing values.

    Parameters:
        dict_ (dict[str, Any]): The keywords of a label or header.
        fill_keys (collections.abc.Container[str], optional): The keywords in which -999
            is a fill value.

    Returns:
        dict[str, Any]: The keywords whose values are not missing in the sense of
        :func:`_is_null`. In a list some of whose elements are missing, each of them is
        None: every list in a Cassini label is a sequence whose positions carry meaning,
        such as the front and back temperatures of OPTICS_TEMPERATURE or the two filters
        of FILTER_NAME. Each dictionary inside, such as an OBJECT, is treated the same
        way.
    """

    result = {}
    for (key, value) in dict_.items():
        fill = key in fill_keys
        if _is_null(value, fill):
            continue
        if isinstance(value, dict):
            value = _without_nulls(value, fill_keys)
        elif isinstance(value, list):
            value = [None if _is_null(item, fill) else item for item in value]
        result[key] = value
    return result

##########################################################################################
