.. _dftbp_interface:

DFTB+ & phonopy calculation
=========================================

How to run
-----------

DFTB+ phonon band structures are created as follows:

1) Create a DFTB+ command file dftb_in.hsd that is set up to perform a single-point calculation for a structure mandatorily named ``geo.gen``. Turn on the force evaluation by ``CalculateForces = Yes``. Then issue the command ::


   % phonopy -d --dim="2 2 2" --dftb+

   This example builds 2 x 2 x 2 supercell files. These are stored in the ``disp-xxx`` directories. Note that you have to increase the supercell dimension until you reach convergence of the band structure.

2) Go to each ``disp-xxx`` directory and perform a DFTB+ calculations. After this operation each of the directories should contain a ``detailed.out`` file.

3) Create ``FORCE_SETS`` by

   ::

     % phonopy -f disp-*/detailed.out --dftb+  ...

   To run this command, ``disp.yaml`` has to be located in the current
   directory because the atomic displacements are written into the
   FORCE_SETS file.

4) Create a ``band.conf`` file to specify the path in the Brillouin zone you are interested in (see phonopy documentation). Then post-process the phonopy data ::

   % phonopy -p band.conf --dftb+

5) Create a band structure in gnuplot format ::

   % phonopy-bandplot --gnuplot band.yaml > band.dat

All major phonopy options should be available in the DFTB+ interface.
