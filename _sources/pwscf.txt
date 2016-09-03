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

The procedure of Pwscf-phonopy calculation is shown below using the
NaCl example found in ``example/NaCl-pwscf`` directory.

1) Read a Pwscf input file and create supercells with
   :ref:`pwscf_mode` option::

     % phonopy --pwscf -d --dim="2 2 2" -c NaCl.in

   In this example, 2x2x2 supercells are created. ``supercell.in`` and
   ``supercell-xxx.in`` (``xxx`` are numbers) give the perfect
   supercell and supercells with displacements, respectively. In the
   case of the NaCl example, two files ``supercell-001.in`` and
   ``supercell-002.in`` are created. In these supercell files, lines
   only relevant to crystal structures are given. ``disp.yaml`` is
   also generated, which contains information about supercell and
   displacements.

2) To make Pwscf input files, necessary setting information is added to
   ``supercell-xxx.in`` files, e.g., by::

     % for i in {001,002};do cat header.in supercell-$i.in >| NaCl-$i.in; done

   where ``header.in`` is specially made for this NaCl example and
   this file is found in ``example/NaCl-pwscf`` directory. This
   setting is of course dependent on systems and has to be written for
   each interested system. Note that supercells with displacements
   must not be relaxed in the force calculations, because atomic
   forces induced by a small atomic displacement are what we need for
   phonon calculation.

   Then Pwscf supercell calculations are executed to obtain force on
   atoms, e.g., as follows::

     % mpirun pw.x -i NaCl-001.in |& tee NaCl-001.out
     % mpirun pw.x -i NaCl-002.in |& tee NaCl-002.out

3) To create ``FORCE_SETS``, that is used by phonopy, 
   the following phonopy command is executed::

     % phonopy --pwscf -f NaCl-001.out NaCl-002.out

   Here ``.out`` files are the saved text files of standard outputs of the
   Pwscf calculations. If more supercells with displacements were
   created in the step 1, all ``.out`` files are given in the above
   command. To run this command, ``disp.yaml`` has to be located in
   the current directory because the information on atomic
   displacements stored in ``disp.yaml`` are used to generate
   ``FORCE_SETS``. See some more detail at
   :ref:`pwscf_force_sets_option`.

4) Now post-process of phonopy is ready to run. The unit cell file
   used in the step 1 has to be specified but ``FORCE_SETS`` is
   automatically read. Examples of post-process are shown below.

   ::

     % phonopy --pwscf -c NaCl.in -p band.conf
             _
       _ __ | |__   ___  _ __   ___   _ __  _   _
      | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
      | |_) | | | | (_) | | | | (_) || |_) | |_| |
      | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
      |_|                            |_|    |___/
                                           1.11.0
     
     Python version 2.7.12
     Spglib version 1.9.2
     Calculator interface: pwscf
     Band structure mode
     Settings:
       Supercell: [2 2 2]
       Primitive axis:
         [ 0.   0.5  0.5]
         [ 0.5  0.   0.5]
         [ 0.5  0.5  0. ]
     Spacegroup: Fm-3m (225)
     Computing force constants...
     Reciprocal space paths in reduced coordinates:
     [ 0.00  0.00  0.00] --> [ 0.50  0.00  0.00]
     [ 0.50  0.00  0.00] --> [ 0.50  0.50  0.00]
     [ 0.50  0.50  0.00] --> [-0.00 -0.00  0.00]
     [ 0.00  0.00  0.00] --> [ 0.50  0.50  0.50]
                      _
        ___ _ __   __| |
       / _ \ '_ \ / _` |
      |  __/ | | | (_| |
       \___|_| |_|\__,_|


   |pwscf-band|

   .. |pwscf-band| image:: NaCl-pwscf-band.png
			   :width: 50%

   ``--pwscf -c NaCl.in`` is specific for the Pwscf-phonopy
   calculation but the other settings are totally common among calculator
   interfaces such as

   ::

     % phonopy --pwscf -c NaCl.in --dim="2 2 2" [other-OPTIONS] [setting-file]

   For settings and command options, see
   :ref:`setting_tags` and :ref:`command_options`, respectively, and
   for examples, see :ref:`examples_link`.

