.. _abinit_interface:

Abinit & phonopy calculation
=========================================

Supported Abinit variables
---------------------------

Currently Abinit variables that phonopy can read are shown below. More
variables may be supported on request.

::

   acell, natom, ntypat, rprim, scalecart, typat, xangst, xcart, xred, znucl

How to run
-----------

A procedure of Abinit-phonopy calculation is as follows:

1) Read an Abinit main input file and create
   supercells with :ref:`abinit_mode` option::

   % phonopy --abinit -d --dim="2 2 2" -c NaCl.in

   In this example, 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-xxx.in`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, lines only related to crystal structures are
   generated. ``disp.yaml`` is also created. This file contains
   information on displacements. Perhaps the supercell files are
   stored in ``disp-xxx`` directories, then Abinit calculations are
   executed in these directories.

2) Calculate forces on atoms in the supercells with
   displacements. Calculation specification variables have to be added
   to ``supercell-xxx.in`` files. Crystal structure is not allowed to
   relaxed in the force calculations, because atomic forces induced by
   a small atomic displacement are what we need for phonon
   calculation.

3) Create ``FORCE_SETS`` by

   ::
   
     % phonopy --abinit -f disp-001/supercell-001.out disp-002/supercell-002.out  ...

   To run this command, ``disp.yaml`` has to be located in the current
   directory because the atomic displacements are written into the
   FORCE_SETS file. See some more detail at
   :ref:`abinit_force_sets_option`. An example is found in
   ``example/NaCl-abinit``.

4) Run post-process of phonopy with the Abinit main input file for the
   unit cell used in the step 1::

   % phonopy --abinit -c NaCl.in -p band.conf

   or::
   
   % phonopy --abinit -c NaCl.in --dim="2 2 2" [other-OPTIONS] [setting-file]

