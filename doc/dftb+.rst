.. _dftbp_interface:

DFTB+ & phonopy calculation
=========================================

How to run
-----------

DFTB+ phonon band structures are created as follows:

1) Create a DFTB+ input file dftb_in.hsd that is set up to perform a
   single-point energy and force calculation for a structure which is named
   ``geo.gen``. The dftb_in.hsd file should turn on force evaluation by setting
   ``CalculateForces = Yes`` in its analysis block, and write the tagged results
   by enabling ``WriteResultsTag = Yes`` in its options.

2) Generate the the required set of structures and the ``disp.yaml`` file by
   issuing the command ::

   % phonopy -d --dim="4 4 4" --dftb+

   This example builds 2 x 2 x 2 supercell files. The undistorted supercell is
   stored in ``geo.genS``, while the required displacements are stored in files
   matching the pattern ``geo.genS-*``. Note that you have to increase the
   supercell dimension until you reach convergence of the band structure.

2) For each each ``geo.genS-*`` structure perform a DFTB+ calculations,
   retaining the resulting ``detailed.out`` file.

3) Create the ``FORCE_SETS`` file with the command ::

     % phonopy -f disp-*/results.tag --dftb+  ...

   Where the location of all of the ``results.tag`` files is given on the
   command line. To run this command, the ``disp.yaml`` file has to be located
   in the current directory, because the atomic displacements are written into
   the FORCE_SETS file.

4) Create a ``band.conf`` file to specify the path in the Brillouin zone you are
   interested in (see the phonopy documentation). Then post-process the phonopy
   data, either in the settings file (DIM) or by providing the dimensions of the
   the supercell repeat on the command line ::

   % phonopy -p band.conf --dim="4 4 4" --dftb+


5) Create a band structure in gnuplot format ::

   % phonopy-bandplot --gnuplot band.yaml > band.dat

All major phonopy options should be available for the DFTB+ interface.
