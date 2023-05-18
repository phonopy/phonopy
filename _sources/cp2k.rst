.. _cp2k_interface:

CP2K & phonopy calculation
==========================

Supported CP2K configuration options
------------------------------------

All CP2K input file features like unit prefixes  (``[rad]`` or ``[Bohr]``) or preprocessor variables are supported,
except for the functionality to load cell information or coordinates from external/referenced files (CIF, XYZ).

The cell information has to be provided using one of the following options.

Either by specifying the ``A``, ``B`` and ``C`` cell vectors directly::

    &FORCE_EVAL
       &SUBSYS
          &CELL
             A 11.452705822048308 0.0 0.0
             B 0.0 11.452705822048308 0.0
             C 0.0 0.0 11.452705822048308
             PERIODIC XYZ
          &END CELL
       &END SUBSYS
    &END FORCE_EVAL

or by using the ``ABC`` scalings and ``ALPHA_BETA_GAMMA`` angles (the latter is optional, the cell is assumed to be orthorombic if no angles are specified)::

    &FORCE_EVAL
       &SUBSYS
          &CELL
             ALPHA_BETA_GAMMA 11.452705822048308 11.452705822048308 11.452705822048308
             ! ALPHA_BETA_GAMMA 90. 90. 90
             PERIODIC XYZ
          &END CELL
       &END SUBSYS
    &END FORCE_EVAL

The coordinates have to be specified in a coordinate section and can be either specified as scaled coordinates
(in cell vector coordinates) or as absolute coordinates (Angstrom if not otherwise specified)::

    &FORCE_EVAL
       &SUBSYS
          &COORD
             SCALED
             Na  0   0   0
             Na  0  1/2 1/2
             Na 1/2  0  1/2
             Na 1/2 1/2  0
             Cl 1/2 1/2 1/2
             Cl 1/2  0   0
             Cl  0  1/2  0
             Cl  0   0  1/2
          &END COORD
       &END SUBSYS
    &END FORCE_EVAL

Please note:

* The cell information in the generated (supercell) configuration files will always be using the ``A``, ``B``, ``C`` configuration options.
* All other configuration options will be forwarded to the generated supercell configurations and all preprocessor options will be resolved prior to it.
* Multiple ``&FORCE_EVAL`` sections are not allowed.
* A ``&CELL_REF`` section and the ``MULTIPLE_UNIT_CELL`` will not be read or modified but simply forwarded to the generated configurations.


How to run
-----------

A procedure of CP2K-phonopy calculation is as follows:

1) Read a CP2K input file and create supercells with the :ref:`cp2k_mode` option::

       % phonopy --cp2k -d --dim="2 2 2" -c NaCl.inp

   In this example, 2x2x2 supercells are created. ``NaCl-supercell.inp`` and
   ``NaCl-supercell-xxx.inp`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively.
   Almost all other options will be copied from the original input ``NaCl.inp``, except for the following:

   * the ``PROJECT_NAME`` will be suffixed to avoid name clashes of output files
   * additional ``&PRINT`` sections will be added to generate separate output files containing the forces
   * the ``RUN_TYPE`` is set to ``ENERGY_FORCE``

   The ``phonopy_disp.yaml`` will also be created containing the Phonopy configuration.

2) Calculate forces on atoms in the supercells with displacements.

   In this example there are two supercells with displacements to run.
   Using ``mpirun`` and an MPI-enabled ``CP2K`` this would simply be::

       % mpirun -np 16 cp2k.popt NaCl-supercell-001.inp
       % mpirun -np 16 cp2k.popt NaCl-supercell-002.inp

   Please do not change any options in the ``NaCl-supercell-002.inp`` which would
   require a re-relaxation of the cell parameters/structure.
   Also do not relax the generated supercells with displacements,
   because atomic forces induced by a small atomic displacement are
   what is need for phonon calculation.

3) Create ``FORCE_SETS`` by running::

       % phonopy --cp2k -f NaCl-supercell-001-forces-1_0.xyz NaCl-supercell-002-forces-1_0.xyz  ...

   To run this command, ``phonopy_disp.yaml`` has to be located in the current
   directory. More information about the configuration options can be found in :ref:`cp2k_force_sets_option`.
   The example outputs are located in ``example/NaCl-CP2K``.

4) Run post-process of phonopy with the original CP2K main input file for the
   unit cell used in step 1::

   % phonopy --cp2k -c NaCl.inp -p band.conf

   or::

   % phonopy --cp2k -c NaCl.inp --dim="2 2 2" [other-OPTIONS] [setting-file]
