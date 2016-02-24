.. _siesta_interface:

Siesta & phonopy calculation
=========================================

Supported Siesta tags
---------------------------

Currently phonopy can read the siesta tags listed below.
More tags may be supported on request.

::

   AtomicCoordinatesFormat, ChemicalSpeciesLabel, AtomicCoordinatesFormat,
   AtomicCoordinatesAndAtomicSpecies, LatticeVectors

How to run
----------

The procedure of a Siesta-phonopy calculation is the following:

1) Read a Siesta input file and create supercells with
   :ref:`siesta_mode` option::

   % phonopy --siesta -d --dim="2 2 2" -c Si.fdf

   In this example, 2x2x2 supercells are created. ``supercell.fdf`` and
   ``supercell-xxx.fdf`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, lines only relevant to crystal structures are
   generated. ``disp.yaml`` is also created. This file contains
   information on displacements. Perhaps the supercell files are
   stored in ``disp-xxx`` directories, then Siesta calculations are
   executed in these directories.

2) Calculate forces on atoms in the supercells with
   displacements. Calculation specification tags have to be added to
   ``supercell-xxx.in`` files. Crystal structure is not allowed to relax
   in the force calculations, because atomic forces induced by a small
   atomic displacement are what we need for the phonon calculation.

3) Create ``FORCE_SETS`` by

   ::
   
     % phonopy --siesta -f disp-001/Si.FA ...

   Here ``*.FA`` files are the forces files created by Siesta.
   To run this command, ``disp.yaml`` has to be
   located in the current directory because the atomic displacements are
   written into the FORCE_SETS file. An example is found in
   ``example/Si-siesta``.

4) Run post-process of phonopy with the Siesta input file for the
   unit cell used in the step 1::

   % phonopy --siesta -c Si.fdf -p band.conf

   or::
   
   % phonopy --siesta -c Si.fdf --dim="2 2 2" [other-OPTIONS] [setting-file]

