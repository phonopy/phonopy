.. _elk_interface:

Elk & phonopy calculation
=========================================

Supported Elk tags
---------------------------

Currently Elk tags that phonopy can read are shown below.
More tags may be supported on request.

::

   atoms, avec, scale, scale1, scale2, scale3

How to run
----------

A procedure of Elk-phonopy calculation is as follows:

1) Read an Elk input file and create supercells with
   :ref:`elk_mode` option::

   % phonopy --elk -d --dim="2 2 2" -c elk-unitcell.in

   In this example, 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-xxx.in`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, lines only relevant to crystal structures are
   generated. ``disp.yaml`` is also created. This file contains
   information on displacements. Perhaps the supercell files of
   ``supercell-xxx.in`` are stored in ``disp-xxx`` directories as
   ``elk.in``, respectively, then Elk calculations are executed in
   these directories.

2) Calculate forces on atoms in the supercells with
   displacements. Calculation specification tags have to be added to
   each ``elk.in`` file. Crystal structure is not allowed to
   relaxed in the force calculations, because atomic forces induced by
   a small atomic displacement are what we need for phonon
   calculation.

3) Create ``FORCE_SETS`` by

   ::
   
     % phonopy --elk -f disp-001/INFO.OUT disp-002/INFO.OUT  ...

   To run this command, ``disp.yaml`` has to be located in the current
   directory because the atomic displacements are written into the
   FORCE_SETS file. See some more detail at
   :ref:`elk_force_sets_option`. An example is found in
   ``example/Si-elk``.

4) Run post-process of phonopy with the Elk input file for the
   unit cell used in the step 1::

   % phonopy --elk -c elk-unitcell.in -p band.conf

   or::
   
   % phonopy --elk -c elk-unitcell.in --dim="2 2 2" [other-OPTIONS] [setting-file]

