.. _pwscf_interface:

Pwscf & phonopy calculation
=========================================

Supported Pwscf tags
---------------------------

Currently Pwscf tags that phonopy can read are shown below.  Only
``ibrav = 0`` type representation of crystal structure is supported.
More tags may be supported on request.

::

   nat, ntyp, ATOMIC_SPECIES, ATOMIC_POSITIONS, CELL_PARAMETERS

How to run
----------

A procedure of Pwscf-phonopy calculation is as follows:

1) Read a Pwscf input file and create supercells with
   :ref:`pwscf_mode` option::

   % phonopy --pwscf -d --dim="2 2 2" -c NaCl.in

   In this example, 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-xxx.in`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, lines only relevant to crystal structures are
   generated. ``disp.yaml`` is also created. This file contains
   information on displacements. Perhaps the supercell files are
   stored in ``disp-xxx`` directories, then Pwscf calculations are
   executed in these directories.

2) Calculate forces on atoms in the supercells with
   displacements. Calculation specification tags have to be added to
   ``supercell-xxx.in`` files. Crystal structure is not allowed to relaxed
   in the force calculations, because atomic forces induced by a small
   atomic displacement are what we need for phonon calculation.

3) Create ``FORCE_SETS`` by

   ::
   
     % phonopy --pwscf -f disp-001/supercell-001.out disp-002/supercell-002.out  ...

   Here ``*.out`` files are the saved texts of standard outputs of
   Pwscf calculations. To run this command, ``disp.yaml`` has to be
   located in the current directory because the atomic displacements are
   written into the FORCE_SETS file. See some more detail at
   :ref:`pwscf_force_sets_option`. An example is found in
   ``example/NaCl-pwscf``.

4) Run post-process of phonopy with the Pwscf input file for the
   unit cell used in the step 1::

   % phonopy --pwscf -c NaCl.in -p band.conf

   or::
   
   % phonopy --pwscf -c NaCl.in --dim="2 2 2" [other-OPTIONS] [setting-file]

