.. _abinit_interface:

Abinit & phonopy calculation
=========================================

Supported Abinit variables
---------------------------

Currently Abinit variables that phonopy can read are shown below. More
variables may be supported on request.

::

   acell, natom, ntypat, rprim, typat, xred, znucl

Procedure
----------

A procedure of Abinit-phonopy calculation is as follows:

1) Read an Abinit main input file and create
   supercells with ``--abinit`` option (:ref:`abinit_mode`)::

   % phonopy -d --dim="2 2 2" --abinit=unitcell.in

   In this example, 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-xxx.in`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, only crystal structure related lines are
   generated. Perhaps the supercell files are stored in ``disp-xxx``
   directories, then Abinit calculations are executed in these
   directories.

2) Calculate forces on atoms in the supercells with
   displacements. Calculation specification variables have to be added
   to supercell-xxx.in files. Crystal structure is not allowed to
   relaxed, because atomic forces induced by a small atomic
   displacement are what we need for phonon calculation.

3) Create ``FORCE_SETS`` by

   ::
   
     % phonopy --abinit=unitcell.in -f disp-001/supercell-001.out disp-002/supercell-002.out  ...

   An example is found in ``example/NaCl-abinit``.

4) Run post-process of phonopy with the Abinit main input file for the
   unit cell used in the step 1::

   % phonopy --abinit=unitcell.in --dim="2 2 2" [other-OPTIONS] [setting-file]

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net

|sflogo|
