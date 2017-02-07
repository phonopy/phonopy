.. _crystal_interface:

CRYSTAL & phonopy calculation
=========================================

CRYSTAL program package has a robust built-in phonon calculation 
workflow. However, the Phonopy interface enables convenient access
to many phonon-related properties, such as subsequent Phono3py
calculations of lattice thermal conductivity.

Supported features
---------------------------

The CRYSTAL interface reads the unit cell from a CRYSTAL output file
(lattice vectors, conventional atomic numbers, fractional atomic positions)
For optimization outputs, the final geometry in the file is read.

If dielectric tensor and effective Born charges are present, the interface
creates a BORN file for Non-analytical correction (:ref:`nac_tag`).
The recommended strategy is to carry out a Gamma-point frequency calculation 
with INTENS and INTCPHF. This produces all required quantities and also confirms that
the structure optimization has converged to a true local minimum.

If ATOMSPIN keyword is present, magnetic moments are read from it. There
is very little experience on using this feature, so be careful.

How to run
----------

The workflow for a CRYSTAL-Phonopy calculation is outlined here using the
Si example found in ``example/Si-CRYSTAL``. 

In this example, the CRYSTAL output file is ``crystal.o``. 
This is the default for the CRYSTAL interface, so, the ``-c crystal.o`` 
parameter is not needed

1) Create supercells with :ref:`crystal_mode` option::

     % phonopy --crystal -d --dim="4 4 4"

   In this example, 4x4x4 supercells are created. For every supercell file, the
   interface creates a .d12 input file and an .ext structure file. The files 
   ``supercell.d12/.ext`` contain the perfect supercell. The files
   ``supercell-xxx.d12/.ext`` (``xxx`` are numbers) contain the supercells
   with displacements. File ``disp.yaml`` is also generated, containing information 
   about the supercell and the displacements.

   In the case of the Si example, files ``supercell-001.d12`` and 
   ``supercell-001.ext`` will be created.

2) To make valid CRYSTAL input files, there are two possible options:

   a) Manually: modify the generated supercell-xxx.d12 files by replacing 
      the line ``Insert basis sets and parameters here`` with the 
      basis set and computational parameters.

   b) Recommended option: before generating the supercells, include a file named
      ``TEMPLATE`` in the current directory. This file should contain the
      basis sets and computational parameters for CRYSTAL (see the Si example).
      When phonopy finds this file, it automatically generates complete
      CRYSTAL input files in the step 1

   Note that supercells with displacements must not be relaxed in the 
   force calculations, because atomic forces induced by a small atomic 
   displacement are what we need for phonon calculation. To get accurate
   forces, TOLDEE parameter should be 10 or higher. Phonopy includes this
   parameter and the necessary GRADCAL keyword automatically in the inputs.

   Then, CRYSTAL supercell calculations are executed to obtain forces on
   atoms, e.g., as follows::

     % runcry14 supercell-001.d12

3) To create ``FORCE_SETS``, that is used by phonopy, 
   the following phonopy command is executed::

     % phonopy --crystal -f supercell-001.o

   Here ``.o`` files are the CRYSTAL output files from the force
   calculations. saved text files of standard outputs of the
   Pwscf calculations. All ``.o`` files corresponding to the generated
   ``supercell-xxx.d12`` files have to be given in the above command. 
   To run this command, ``disp.yaml`` has to be located in the current 
   directory because the information on atomic displacements stored in 
   ``disp.yaml`` are used to generate ``FORCE_SETS``. See some more detail at
   :ref:`crystal_force_sets_option`.

4) Now, Phonopy post-prcessing commands can be run. ``FORCE_SETS`` is
   automatically read in. Create phonon dispersion plot with:

   ::

     % phonopy --crystal --dim="4 4 4" -p -s band.conf
             _
       _ __ | |__   ___  _ __   ___   _ __  _   _
      | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
      | |_) | | | | (_) | | | | (_) || |_) | |_| |
      | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
      |_|                            |_|    |___/
                                           1.11.6


     Python version 2.7.3
     Spglib version 1.9.9
     Calculator interface: crystal
     Band structure mode
     Settings:
       Supercell: [4 4 4]
     Spacegroup: Fd-3m (227)
     Computing force constants...
     Reciprocal space paths in reduced coordinates:
     [ 0.00  0.00  0.00] --> [ 0.50  0.00  0.50]
     [ 0.50  0.00  0.50] --> [ 0.50  0.25  0.75]
     [ 0.50  0.25  0.75] --> [ 0.37  0.38  0.75]
     [ 0.38  0.38  0.75] --> [ 0.00  0.00  0.00]
     [ 0.00  0.00  0.00] --> [ 0.50  0.50  0.50]
     [ 0.50  0.50  0.50] --> [ 0.63  0.25  0.63]
     [ 0.62  0.25  0.62] --> [ 0.50  0.25  0.75]
     [ 0.50  0.25  0.75] --> [ 0.50  0.50  0.50]
     [ 0.50  0.50  0.50] --> [ 0.37  0.37  0.75]
     [ 0.62  0.25  0.62] --> [ 0.50 -0.00  0.50]
                      _
        ___ _ __   __| |
       / _ \ '_ \ / _` |
      |  __/ | | | (_| |
       \___|_| |_|\__,_|


   |crystal-band|

   .. |crystal-band| image:: Si-crystal-band.png
			   :width: 33%

   For further settings and command options, see the general Phonopy documentation
   :ref:`setting_tags` and :ref:`command_options`, respectively, and
   for examples, see :ref:`examples_link`.

Non-analytical term correction (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The workflow for a CRYSTAL-Phonopy calculation with Non-analytical correction
is outlined here using the NaCl example found in ``example/NaCl-CRYSTAL``.

In this example, the CRYSTAL output file is ``crystal.o``.
This is the default for the CRYSTAL interface, so, the ``-c crystal.o``
parameter is not needed.

To activate non-analytical term correction, :ref:`born_file` is
required. This file contains the Born effective charges
and the dielectric tensor. They can be calculated with CRYSTAL.
The recommended strategy is to carry out a Gamma-point frequency calculation
with INTENS and INTCPHF. This produces all required quantities and also confirms that
the structure optimization has converged to a true local minimum.
(see the FREQCALC-INTENS-INTCPHF block in the beginning of ``crystal.o``)

The workflow is very similar to the Si example below:

1) Create displaced supercells::

     phonopy --crystal --dim="4 4 4" -d

   Note that now the CRYSTAL interface automatically creates the ``BORN``
   file. It should look like this::

     default
     1.8126 0.0000 0.0000 0.0000 1.8126 0.0000 0.0000 0.0000 1.8126
     1.0238 -0.0000 -0.0000 -0.0000 1.0238 0.0000 -0.0000 0.0000 1.0238
     -1.0238 0.0000 0.0000 0.0000 -1.0238 0.0000 0.0000 0.0000 -1.0238

   However, if you don't want to run a FREQCALC-INTENS-INTCPHF calculation,
   but have the necessary data from some other source, you can create
   the ``BORN`` file manually following the ``BORN`` format
   (:ref:`born_file`).

2) Run the supercell inputs with CRYSTAL

3) Collect forces::

     phonopy --crystal -f supercell-*o

4) Calculate phonon dispersion data into band.yaml and save band.pdf,
   using Non-analytical correction --nac::

     phonopy --crystal --dim="4 4 4" -p -s --nac band.conf

   |crystal-band-nac|

   .. |crystal-band-nac| image:: NaCl-crystal-band-NAC.png
                               :width: 33%

