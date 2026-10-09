##########################################################################################
# spyceman/hosts/Cassini/_CASSINI_SCI_PCKS.py
##########################################################################################
"""Managed list of the Cassini science PCK files, complete as of 9/2/10.

These are the cpckDDMonYYYY_Sci.tpc kernels, the science subset of the full PCK of the
same date. Each carries the shapes, orientations, rings, atmosphere and magnetic-pole
constants of the full file, and omits only the gravity harmonics: no JCOEF, CCOEF or
SCOEF. They are the complement of _CASSINI_NAV_PCKS, which keeps the gravity model and
discards everything else.

Only three were ever produced, against 199 full files and 187 navigation files, and they
stop in 2010 while the other two lists run to the end of the mission. Each covers the same
33 bodies.

Where a file's internal date and the date in its name disagree, the release
date here is the later of the two. For these three files the two always
agree.
"""

from spyceman.kernelfile import KTuple

_CASSINI_SCI_PCKS = [
    KTuple('cpck09Dec2004_Sci.tpc',
        None, None,
        {3, 5, 6, 10, 299, 301, 399, 501, 502, 503, 504, 506, 599, 601, 602, 603, 604,
         605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 699, 799},
        '2004-12-09'),
    KTuple('cpck16Apr2008_Sci.tpc',
        None, None,
        {3, 5, 6, 10, 299, 301, 399, 501, 502, 503, 504, 506, 599, 601, 602, 603, 604,
         605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 699, 799},
        '2008-04-16'),
    KTuple('cpck02Sep2010_Sci.tpc',
        None, None,
        {3, 5, 6, 10, 299, 301, 399, 501, 502, 503, 504, 506, 599, 601, 602, 603, 604,
         605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 699, 799},
        '2010-09-02'),
]

##########################################################################################
