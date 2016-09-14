.. _pwscf_interface:

Pwscf & phonopy calculation
=========================================

Quantum espresso package itself has a set of the phonon calculation
system. But the document here explains how to calculate phonons using
phonopy, i.e., using the finite displacement and supercell approach.

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

Non-analytical term correction (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate non-analytical term correction, :ref:`born_file` is
required. This file contains the information of Born effective charge
and dielectric constant. These physical values are also obtained from
the pwscf (``pw.x``) & phonon (``ph.x``) codes in quantum-espresso
package. There are two steps. The first step is usual self-consistent
field (SCF) calculation
by and the second step is running its response function calculations
under DFPT.

For the SCF calculation, the input file ``NaCl.in`` looks like::

    &control
       calculation = 'scf'
       tprnfor = .true.
       tstress = .true.
       pseudo_dir = '/home/togo/espresso/pseudo/'
    /
    &system
       ibrav = 0
       nat = 8
       ntyp = 2
       ecutwfc = 70.0
    /
    &electrons
       diagonalization = 'david'
       conv_thr = 1.0d-9
    /
   ATOMIC_SPECIES
    Na  22.98976928 Na.pbe-spn-kjpaw_psl.0.2.UPF
    Cl  35.453      Cl.pbe-n-kjpaw_psl.0.1.UPF
   ATOMIC_POSITIONS crystal
    Na   0.0000000000000000  0.0000000000000000  0.0000000000000000
    Na   0.0000000000000000  0.5000000000000000  0.5000000000000000
    Na   0.5000000000000000  0.0000000000000000  0.5000000000000000
    Na   0.5000000000000000  0.5000000000000000  0.0000000000000000
    Cl   0.5000000000000000  0.5000000000000000  0.5000000000000000
    Cl   0.5000000000000000  0.0000000000000000  0.0000000000000000
    Cl   0.0000000000000000  0.5000000000000000  0.0000000000000000
    Cl   0.0000000000000000  0.0000000000000000  0.5000000000000000
   CELL_PARAMETERS angstrom
    5.6903014761756712 0 0
    0 5.6903014761756712 0
    0 0 5.6903014761756712
   K_POINTS automatic
    8 8 8 1 1 1

where more the k-point mesh numbers are specified. This may be exectued as::

   mpirun ~/espresso/bin/pw.x -i NaCl.in |& tee NaCl.out

Many files whose names stating with ``pwscf`` should be created. These
are used for the next calculation. The input file for the response
function calculations, ``NaCl.ph.in``, is
created as follows::

    &inputph
     tr2_ph = 1.0d-14,
     epsil = .true.
    /
   0 0 0

Similary ``ph.x`` is executed::

   % mpirun ~/espresso/bin/ph.x -i NaCl.ph.in |& tee NaCl.ph.out

Finally the Born effective charges and dielectric constant are
obtained in the output file ``NaCl.ph.out``. The ``BORN`` file has to
be created manually following the ``BORN`` format
(:ref:`born_file`). The ``BORN`` file for this NaCl calculation would
be something like below::

   default value
   2.472958201 0 0 0 2.472958201 0 0 0 2.472958201
   1.105385 0 0 0 1.105385 0 0 0 1.105385
   -1.105385 0 0 0 -1.105385 0 0 0 -1.105385

Once this is made, the non-analytical term correction is included 
just adding the ``--nac`` option as follows::

     % phonopy --pwscf --nac -c NaCl.in -p band.conf


|pwscf-band-nac|

.. |pwscf-band-nac| image:: NaCl-pwscf-band-NAC.png
   			    :width: 50%
