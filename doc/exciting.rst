.. _exciting_interface:

exciting & phonopy calculation
=========================================

Supported exciting tags
---------------------------

Phonopy can currently process the geometric section of the
input file. Currently, the following parameters/elements are ignored
when generating input files:

::

    autormt, autormtscaling, cartesian, epslat, primcell, tshift, stretch, fixrmt, rmt, LDAplusU, dfthalfparam, shell

If any of these are required for your calculations,
they must be manually re-added by the user after processing.
Note that although it is possible to process files where atomic
positions are given in Cartesian coordinates, the generated files
will always use reduced (fractional) coordinates. In addition,
``tshift`` will be set to ``false`` for consistency with the input file.
For more information, see: https://exciting-code.org/.

How to run
----------

A procedure of exciting-phonopy calculation is as follows:

1) Read an exciting input file and create supercells with
   :ref:`exciting_mode` option::

   % phonopy-init --exciting -d --dim="2 2 2"

   In this example, 2x2x2 supercells are created. ``supercell.xml`` and
   ``supercell-xxx.xml`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In these
   supercell files, lines only relevant to crystal structures are
   generated. ``phonopy_disp.yaml`` is also created. This file contains
   information on displacements. Put those files in folders ``001 002 ...``
   and create a symbolic link called ``input.xml`` to those. Morever,
   if needed copy the species files,

2) Calculate forces on atoms in the supercells with
   displacements. Calculation block ``groundstate`` have to be added to
   each ``supercell.xml`` file.

3) Create ``FORCE_SETS`` by

   ::

     % phonopy-init --exciting -f 001/INFO.OUT 002/INFO.OUT  ...

   To run this command, ``phonopy_disp.yaml`` has to be located in the current
   directory because the atomic displacements are written into the
   FORCE_SETS file.

4) Run post-process of phonopy.  Crystal structure and calculator
   interface are read from ``phonopy_disp.yaml``::

   % phonopy -p band.conf

   or::

   % phonopy [other-OPTIONS] [setting-file]
