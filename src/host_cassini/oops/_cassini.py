##########################################################################################
# host_cassini/oops/_cassini.py
##########################################################################################

import datetime
import pathlib
import re

import numpy as np

import julian
import spicedb
import cspyce

from filecache import FCPath

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
                reconstructed C kernels of the Jupiter flyby.
            spk (str, optional): The set of SP kernels to load, 'reconstructed' or
                'predicted' (case-insensitive); 'none' to load no SP kernels
                automatically, leaving their handling to the caller.
            gapfill (bool, optional): True to include gapfill CKs. False otherwise.
            planets (int | Sequence[int], optional): The number of a planet, 1 to 9, or a
                sequence of such numbers, whose systems
                :meth:`~oops.Body.define_solar_system` defines. None or 0 means all
                nine.
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

            # The reconstructed C kernels of the Jupiter flyby are a set of their own
            if ck == 'RECONSTRUCTED':
                kernels = kernels + spicedb.select_ck(
                    -82, name='CAS-CK-JUP',
                    time=(_Cassini.START_TIME, _Cassini.STOP_TIME), asof=asof)

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

        # Load any months not already loaded
        for m in range(m1, m2+1):
            if not loaded[m]:
                for kernel in lists[m]:
                    filespec = kernel.filespec
                    if filespec not in kernel_dict:
                        spicedb.furnish_kernels([kernel])
                        kernel_dict[filespec] = kernel
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

    Parameters:
        options (dict[str, Any] | None): The options given by the caller, if any; not
            modified.

    Returns:
        dict[str, Any]: A copy of `options` with the defaults of oops.Host, whose method
        is "fast" unless it names another, but whose `sources` is always False: the label
        text that the parser can keep for each keyword is never wanted, in `label_dict` or
        anywhere else.
    """

    return oops.Host._pds3_options_with_defaults(options) | {'sources': False}


# The order in which the PDS3 label parsers are tried, from the one requested onward
_PARSER_METHODS = ('fast', 'strict', 'loose')


def _read_pds3_fileinfo(host, fileinfo, *, astrometry, pds3_options, repairs=()):
    """The parsed PDS3 label of a UVIS or VIMS product, read with the parser fallbacks.

    The label is parsed with the method in `pds3_options` and, if that parser rejects it
    or cannot convert one of its values, by each later method in the order "fast",
    "strict", "loose". A method that is not one of the three is tried alone. The repairs
    in `pds3_options`, a list of (pattern, replacement) pairs or a single pair, and then
    `repairs`, are applied in every attempt. Parsed content is not parsed again, but must
    describe a product of this host.

    Parameters:
        host (type): The host class, such as CassiniVIMS.
        fileinfo (str | pathlib.Path | FCPath | Pds3Label): The path to the label or the
            file holding it, or the parsed label.
        astrometry (bool): True to read no data array.
        pds3_options (dict[str, Any]): The options for the PDS3 label parser, including
            its method.
        repairs (tuple[tuple[str, str], ...], optional): The host's own repairs.

    Returns:
        Pds3Label: The parsed label.

    Raises:
        OSError: If the file cannot be read.
        PdsHostError: If no parser can read the label; the message names the file.
        WrongHostError: If the file or the parsed content is not of this host.
    """

    # Parsed content is checked as its file would be, so that another instrument's label
    # raises WrongHostError here rather than a KeyError later
    if not isinstance(fileinfo, (str, pathlib.Path, FCPath)):
        if not host._detect_in_pds3(fileinfo):
            raise oops.Host._no_host_error([host], 'PDS3 label',
                                           getattr(fileinfo, 'filepath', None))
        return fileinfo

    method = pds3_options['method']
    if method in _PARSER_METHODS:
        methods = _PARSER_METHODS[_PARSER_METHODS.index(method):]
    else:
        methods = (method,)

    given_repairs = pds3_options.get('repairs', [])
    if (isinstance(given_repairs, tuple) and len(given_repairs) == 2
            and all(isinstance(item, str) for item in given_repairs)):
        given_repairs = [given_repairs]     # one (pattern, replacement) pair

    error = None
    for method in methods:
        options = pds3_options | {'method': method,
                                  'repairs': list(given_repairs) + list(repairs)}
        try:
            return host._read_fileinfo(fileinfo, formats='P', astrometry=astrometry,
                                       pds3_options=options)
        except (oops.host.PdsHostError, oops.OopsValueError) as err:
            # An OopsValueError is a value that this parser could not convert, which a
            # later parser may still read
            error = err
    raise oops.host.PdsHostError(f'{error} in {fileinfo}') from error


def _index_file_path(volumes, row_dict):
    """The path to the file that one row of an index describes.

    The row's FILE_SPECIFICATION_NAME is relative to the root directory of its volume,
    the directory named by its VOLUME_ID, inside the directory that holds the volumes.
    An index in the "metadata" tree of the holdings, such as
    metadata/COISS_2xxx/COISS_2999/COISS_2999_index.lbl, may list files of any volume of
    its series, so for it that directory is the series in the "volumes" tree beside it,
    volumes/COISS_2xxx; for an index anywhere else it is the index's own directory.
    ``oops.Host._volumes_directory`` gives it.

    Parameters:
        volumes (FCPath): The directory that holds the volume directories.
        row_dict (dict[str, Any]): The row, which gives VOLUME_ID and
            FILE_SPECIFICATION_NAME.

    Returns:
        FCPath: The path or URL of the file.
    """

    volume_id = str(row_dict['VOLUME_ID']).strip()
    return volumes / volume_id / str(row_dict['FILE_SPECIFICATION_NAME']).strip()


# The prefix of the volume names of each Cassini instrument that has a host here
_VOLUME_PREFIXES = ('COISS', 'COUVIS', 'COVIMS')


def _detect_index_by_path(label, prefix):
    """Whether the path of an index label shows that it belongs to one instrument.

    The volumes of the Cassini instruments are named COISS_nnnn, COUVIS_nnnn, and
    COVIMS_nnnn, and the indexes, the volume's own and the metadata index alike, are
    stored under those names.

    Parameters:
        label (Pds3Label | dict[str, Any]): The parsed label of an index; a dictionary
            has no path.
        prefix (str): The volume prefix of the instrument, such as "COUVIS".

    Returns:
        bool | None: True if the label's path contains `prefix`, in any case; False if it
        contains the prefix of another Cassini instrument; otherwise None.
    """

    filepath = getattr(label, 'filepath', None)
    if filepath is None:
        return None
    text = FCPath(filepath).as_posix().upper()
    if prefix in text:
        return True
    if any(other in text for other in _VOLUME_PREFIXES if other != prefix):
        return False
    return None


def _known_path(filepath):
    """The path of parsed content, or None if it came from no file.

    Parameters:
        filepath (str | pathlib.Path | FCPath | None): The `filepath` attribute of a
            parsed label or header.

    Returns:
        FCPath | None: The path, or None if `filepath` is None or empty.
    """

    if filepath is None or str(filepath) in ('', '.'):
        return None
    return FCPath(filepath)


# The classes of exception that cspyce raises for a SPICE error
_SPICE_ERRORS = (LookupError, OSError, RuntimeError, ValueError)


def _tdb_from_sclk(count, filepath):
    """The time in seconds TDB of a Cassini spacecraft clock count.

    Parameters:
        count (str): The clock count, such as "1294562649.000" or "1/1294562649.000".
        filepath (FCPath | str | None): The file, named in the error.

    Returns:
        float: The time in seconds TDB.

    Raises:
        OopsSpiceError: If the spacecraft clock kernel cannot convert the count.
    """

    try:
        return cspyce.scs2e(_CASSINI_SCLK, count)
    except _SPICE_ERRORS as err:
        raise oops.OopsSpiceError.from_spice_error(
            err, description=f'no Cassini clock time for count {count}',
            filepath=filepath) from err


def _sclk_from_tdb(tdb, filepath):
    """The Cassini spacecraft clock count of a time, without its partition.

    Parameters:
        tdb (float): The time in seconds TDB.
        filepath (FCPath | str | None): The file, named in the error.

    Returns:
        str: The clock count, such as "1294562649.000".

    Raises:
        OopsSpiceError: If the spacecraft clock kernel cannot convert the time.
    """

    try:
        count = cspyce.sce2s(_CASSINI_SCLK, tdb)
    except _SPICE_ERRORS as err:
        raise oops.OopsSpiceError.from_spice_error(
            err, description='no Cassini clock count', time=tdb,
            filepath=filepath) from err
    return count.rpartition('/')[2]


def _common_subfields(meta, *, label_dict=None, time_from_sclk=False):
    """The subfields of a UVIS or VIMS observation that come from its metadata.

    Parameters:
        meta (dict[str, Any]): The metadata, without missing values, as the `dict`
            subfield gives it.
        label_dict (dict[str, Any], optional): The content of the detached PDS3 label,
            when it was read.
        time_from_sclk (bool, optional): True if the observation's time came from a
            spacecraft clock count.

    Returns:
        dict[str, Any]: The subfields `dict` and `time_from_sclk`; `label_dict` when it is
        given; and `label_target`, `observation_id`, `spacecraft_clock_start_count` and
        `spacecraft_clock_stop_count`, each when `meta` gives its item, the clock counts
        only as text.
    """

    subfields = {'dict': meta, 'time_from_sclk': time_from_sclk}
    if label_dict is not None:
        subfields['label_dict'] = label_dict
    for (item, name) in (('TARGET_NAME', 'label_target'),
                         ('OBSERVATION_ID', 'observation_id')):
        if item in meta:
            subfields[name] = meta[item]
    for (item, name) in (('SPACECRAFT_CLOCK_START_COUNT', 'spacecraft_clock_start_count'),
                         ('SPACECRAFT_CLOCK_STOP_COUNT', 'spacecraft_clock_stop_count')):
        if isinstance(meta.get(item), str):
            subfields[name] = meta[item]
    return subfields

##########################################################################################
# Metadata in JSON form, without missing values
##########################################################################################

# The PDS3 null constants and the other text that the archive writes for "no value",
# compared without regard to case or surrounding blanks
_NULL_STRINGS = {'UNK', 'UNKNOWN', 'N/A', 'NULL', '', '---'}

# The number that the Cassini labels and indexes write for a value that was not
# determined. It is missing only in the keywords and columns that the host lists, where
# it is a declared null constant or cannot be a real value, such as a temperature, a data
# rate or a compression ratio; elsewhere, as in a position vector, -999 can be real.
_FILL_VALUE = -999.


def _is_null(value, fill=False):
    """True if a value states nothing.

    Parameters:
        value (Any): The value of a keyword or column.
        fill (bool, optional): True if -999 is a fill value in this keyword or column.

    Returns:
        bool: True if `value` is None; one of "UNK", "UNKNOWN", "N/A", "NULL", "---" or
        the empty string, in any case and with any surrounding blanks; -999 if `fill` is
        True; or a non-empty list, tuple or set of nothing but these.
    """

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in _NULL_STRINGS
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return fill and value == _FILL_VALUE
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 0:
        return all(_is_null(item, fill) for item in value)
    return False


def _without_nulls(dict_, fill_keys=frozenset()):
    """A copy of a dictionary without its missing values.

    Parameters:
        dict_ (dict[str, Any]): The keywords of a label or the columns of an index row.
        fill_keys (collections.abc.Container[str], optional): The keywords in which -999
            is a fill value, at any depth.

    Returns:
        dict[str, Any]: The keywords whose values are not missing in the sense of
        :func:`_is_null`. In a list some of whose elements are missing, each of them is
        None: every list in a Cassini label or index row is a sequence whose positions
        carry meaning, such as the front and back temperatures of OPTICS_TEMPERATURE,
        the two filters of FILTER_NAME, or the three components of a vector. Each
        dictionary inside is treated the same way.
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


def _json_form(dict_):
    """A copy of a parsed PDS3 dictionary with its times as text and sets as lists.

    Parameters:
        dict_ (dict[str, Any]): A dictionary from ``Pds3Label.as_dict()``, or one of the
            OBJECT or GROUP dictionaries inside it.

    Returns:
        dict[str, Any]: The dictionary, with each date, time, or date-time value replaced
        by the text of its "_fmt" key, each set replaced by the list of its "_list" key,
        each tuple replaced by a list, and each dictionary inside it converted the same
        way.
    """

    result = {}
    for (key, value) in dict_.items():
        if isinstance(value, dict):
            value = _json_form(value)
        elif isinstance(value, (datetime.date, datetime.time)):     # incl. datetime
            value = dict_.get(key + '_fmt', value.isoformat())
        elif isinstance(value, (set, frozenset)):
            value = dict_.get(key + '_list', sorted(value))
        elif isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def _items(dict_):
    """A copy of a parsed PDS3 dictionary without the keys the parser adds.

    The PDS3 keywords are upper case. The keys that the parser adds are not: those it
    derives from a keyword, named by the keyword plus a lower-case suffix, such as
    "START_TIME_fmt", "DETECTOR_TEMPERATURE_unit" or "^IMAGE_offset", and its "objects"
    and "groups" lists. They state nothing of their own.

    Parameters:
        dict_ (dict[str, Any]): The content of a PDS3 label, or an OBJECT or GROUP
            dictionary inside it.

    Returns:
        dict[str, Any]: The upper-case keywords of `dict_`, without any whose value is
        None, such as END. Each dictionary inside is treated the same way.
    """

    result = {}
    for (key, value) in dict_.items():
        if key != key.upper() or value is None:
            continue
        result[key] = _items(value) if isinstance(value, dict) else value
    return result


def _label_dict(label_dict):
    """The content of a parsed PDS3 label, as the `label_dict` subfield gives it.

    This is every keyword of the label's ``as_dict()``, in label order, with each date or
    time as the text that the label gives it (the parser's "_fmt" form, such as
    "2005-016T13:01:59.864") and each set as the list of its members in label order, so
    that the dictionary can be written as JSON. END, any other key whose value is None,
    and the lower-case keys that the parser adds (see :func:`_items`) are left out.

    Parameters:
        label_dict (dict[str, Any]): The ``as_dict()`` of the parsed PDS3 label.

    Returns:
        dict[str, Any]: The label's content.
    """

    # Consumers write the metadata out as JSON, which cannot hold a datetime or a set
    # (#16); the derived keys are dropped only after the times and sets are read from them
    return _items(_json_form(label_dict))


def _row_items(row_dict, *, masks, fill_values=None):
    """The metadata of one row of an index table, in Python types.

    Parameters:
        row_dict (dict[str, Any]): The row, as the index reader gives it: NumPy values
            and, for each column, a "_mask" entry.
        masks (bool): True to treat each value that the table's label declares a null
            constant (its "_mask" entry) as missing.
        fill_values (dict[str, Any], optional): Further values that are missing, by
            column name.

    Returns:
        dict[str, Any]: The row's columns, each value converted to a Python value or, for
        an array column, a list, and text stripped of its padding. Each missing value,
        declared by `masks` or `fill_values` or in the sense of :func:`_is_null`, is
        removed; such an element of an array is None. The "_mask" entries are left out.
    """

    fill_values = fill_values or {}
    items = {}
    for (key, value) in row_dict.items():
        if key.endswith('_mask') and key[:-5] in row_dict:
            continue
        value = np.asarray(value).tolist()
        mask = np.asarray(row_dict.get(key + '_mask', False)).tolist() if masks else False
        fill = fill_values.get(key)
        if isinstance(value, list):
            if not isinstance(mask, list):
                mask = [mask] * len(value)
            value = [None if masked else _index_value(item, fill)
                     for (item, masked) in zip(value, mask, strict=True)]
        else:
            value = None if mask else _index_value(value, fill)
        items[key] = value
    return _without_nulls(items)


def _index_value(value, fill):
    """A value of an index row, or None if it is a fill value.

    Parameters:
        value (Any): The value, or one element of an array column's value, as a Python
            value.
        fill (Any): The column's own fill value, or None.

    Returns:
        Any: None for 1e32 or -1e32, which are missing in every Cassini index, or for
        `fill`; text stripped of its padding; any other value unchanged.
    """

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and (abs(value) == _INDEX_FILL_BOUND
                                            or (fill is not None and value == fill)):
        return None
    return value


# The fill value that every Cassini index writes, positive or negative, for a quantity
# that was not determined (#28)
_INDEX_FILL_BOUND = 1.e32

##########################################################################################
