.. _FHI_aims_interface:

FHI-aims & phonopy calculations
===============================

The script ``phonopy-FHI-aims`` allows to conveniently employ the
infrastructure provided by phonopy in order to calculate phonons with
FHI-aims.  For compatibility reasons, most parameters are set via the
``phonon`` tag in ``control.in`` as documented for the FHI-aims
internal implementation in the FHI-aims manual. But several
additional parameters are also handled via command line options as
listed by ``-h``.

Some examples can be found under ``FHI-aims`` in the ``example``
directory of the phonopy tarball. A subset of them can now also be
found among the FHI-aims "testcases" (only available from the
developers' repositories at the moment). They have been slightly
modified to also serve as a step-by-step guide.

For questions, please make use of the official FHI-aims forums accessible
from here: 

https://aimsclub.fhi-berlin.mpg.de
