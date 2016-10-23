.. _tutorial:

VASP & phonopy calculation
==================================

Pre-process
~~~~~~~~~~~~

The input stureture of ``POSCAR`` (:ref:`this <example_POSCAR1>`) is
used as an example here. Most files are found at `SiO2-HP example <https://github.com/atztogo/phonopy/tree/master/example/SiO2-HP/>`_.

In the pre-process, supercell structures with (or without)
displacements are created from a unit cell fully consiering crystal
symmetry.

To obtain supercells (:math:`2\times 2\times 3`) with displacements,
run phonopy::

   phonopy -d --dim="2 2 3"

You should find the files, ``SPOSCAR``, ``disp.yaml``, and ``POSCAR-{number}`` as
follows::

   % ls
   disp.yaml  POSCAR  POSCAR-001  POSCAR-002  POSCAR-003  SPOSCAR

``SPOSCAR`` is the perfect supercell structure, ``disp.yaml`` contains
the information on displacements, and ``POSCAR-{number}`` are the
supercells with atomic displacements. ``POSCAR-{number}`` corresponds
to the different atomic displacements written in ``disp.yaml``.

Calculation of sets of forces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Force constants are calculated using the structure files ``POSCAR-{number}``
(from forces on atoms) or using the ``SPOSCAR`` file (direct calculation of force
constants) by your favorite calculator. See the
:ref:`details<details_of_procedure>`.

In the case of VASP, the calculations for the finite displacement method can be proceeded just using the ``POSCAR-{number}`` files as ``POSCAR`` of VASP calculations. An example of the ``INCAR`` is as follows::

      PREC = Accurate
    IBRION = -1
     ENCUT = 500
     EDIFF = 1.0e-08
    ISMEAR = 0; SIGMA = 0.01
     IALGO = 38
     LREAL = .FALSE.
     LWAVE = .FALSE.
    LCHARG = .FALSE.

Be careful not to relax the structures. Then create ``FORCE_SETS``
file using :ref:`vasp_force_sets_option`::

   % phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml disp-003/vasprun.xml

or

::

   % phonopy -f disp-{001..003}/vasprun.xml

If you want to calculate force constants by VASP-DFPT directory, see :ref:`vasp_dfpt_interface`.

Post-process
~~~~~~~~~~~~~

In the post-process,

1. Force constants are calculated from the sets of forces
2. A part of dynamical matrix is built from the force constants
3. Phonon frequencies and eigenvectors are calculated from the
   dynamical matrices with the specified *q*-points.

For mesh sampling calculation, prepare the following setting file named, e.g., 
``mesh.conf``::

   ATOM_NAME = Si O
   DIM = 2 2 3
   MP = 8 8 8

The density of states (DOS) is plotted by::

   % phonopy -p mesh.conf

Thermal properties are calculated with the sampling mesh by::

   % phonopy -t mesh.conf

You should check the convergence with respect to the mesh numbers.
Thermal properties can be plotted by::

   % phonopy -t -p mesh.conf

Projected DOS is calculated by the following setting file named, e.g., ``pdos.conf``::

   ATOM_NAME = Si O
   DIM = 2 2 3
   MP = 8 8 8
   PDOS = 1 2, 3 4 5 6

and plotted by::

   % phonopy -p pdos.conf

Band structure is calculated with the following setting file named, e.g., `band.conf` by::

   ATOM_NAME = Si O
   DIM =  2 2 3
   BAND = 0.5 0.5 0.5  0.0 0.0 0.0  0.5 0.5 0.0  0.0 0.5 0.0

The band structure is plotted by::

   % phonopy -p band.conf

In either case, by setting the ``-s`` option, the plot is going to be saved in the PDF
format. If you don't need to plot DOS, the (partial) DOS
is just calculated using the ``--dos`` option.

Non-analytical term correction (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To activate non-analytical term correction, :ref:`born_file` is
required. This file contains the information of Born effective charge
and dielectric constant. These physical values are also obtained from
the first-principles calculations, e.g., by using VASP, pwscf, etc. In
the case of VASP, an example of ``INCAR`` will be as shown below::
 
       PREC = Accurate
     IBRION = -1
     NELMIN = 5
      ENCUT = 500
      EDIFF = 1.000000e-08
     ISMEAR = 0
      SIGMA = 1.000000e-02
      IALGO = 38
      LREAL = .FALSE.
      LWAVE = .FALSE.
     LCHARG = .FALSE.
   LEPSILON = .TRUE.

In addition, it is recommended to increase the number of k-points to
be sampled. Twice the number for each axis may be a choice. After
running this VASP calculation, ``BORN`` file has to be created
following the ``BORN`` format (:ref:`born_file`). However for VASP, an
auxiliary tool is prepared, which is ``outcar-born``. There is an
option ``--pa`` for this command to set a transformation matrix from
supercell or unit cell with centring to the primitive cell. Since this
rutile-type SiO2 has the primitive lattice, it is unnecessary to set
this option. Running ``outcar-born`` in the directory containing
``OUTCAR`` of this VASP calculation::

   % outcar-born
   # epsilon and Z* of atoms 1 3
      3.2605670   0.0000000   0.0000000   0.0000000   3.2605670   0.0000000   0.0000000   0.0000000   3.4421330
      3.7558600   0.3020100   0.0000000   0.3020100   3.7558600   0.0000000   0.0000000   0.0000000   3.9965200
     -1.8783900  -0.5270900   0.0000000  -0.5270900  -1.8783900   0.0000000   0.0000100   0.0000100  -1.9987900

To employ symmetry constraints, ``--st`` option may used as follows::

   % outcar-born --st
   # epsilon and Z* of atoms 1 3
      3.2605670   0.0000000   0.0000000   0.0000000   3.2605670   0.0000000   0.0000000   0.0000000   3.4421330
      3.7561900   0.3020100   0.0000000   0.3020100   3.7561900   0.0000000   0.0000000   0.0000000   3.9968733
     -1.8780950  -0.5270900   0.0000000  -0.5270900  -1.8780950   0.0000000   0.0000000   0.0000000  -1.9984367

The values are slightly modified by symmetry, but we can see the
original values obtained directly from VASP was already very good.

To put ``BORN`` file in the current directly, and running phonopy with
``--nac`` option, non-analytical term correction is activated, such as::

   % phonopy -p --nac band.conf

Please watch the example of NaCl with and without ``--nac`` option shown in
:ref:`examples_link`.

.. _details_of_procedure:

Some more details
------------------

The structure file names are different in different force calculators,
see :ref:`calculator_interfaces`. To notify this, (*) is put
after the structure file name. By refering
:ref:`calculator_interfaces`, please read the file name according to
those for the other calculators.

Following files are required in your working directory.

- ``POSCAR`` (*), and ``FORCE_SETS`` or ``FORCE_CONSTANTS``
- ``disp.yaml`` is required to create ``FORCE_SETS``.

In the case of finite difference approach, there are three steps.

1. Create supercells and introduce atomic displacements. Each
   supercell contains one atomic displacement. It is done by using
   ``-d`` option with ``--dim`` option that specifies supercell
   dimension.  The files of supercells with atomic displacements like
   as ``POSCAR-001``, ``POSCAR-002``, ..., (*) are created in current
   directory by running phonopy. The files ``disp.yaml`` and
   ``SPOSCAR`` are also created. The file ``SPOSCAR`` is the perfect
   supercell that contains no atomic displacement. This file is not
   usually used.

2. Calculate forces on atoms of the supercells with atomic
   displacements. After obtaining forces on atoms that calculated by
   some calculator (it's out of phonopy), the forces are summarized in
   ``FORCE_SETS`` file following the :ref:`format <file_forces>`.

3. Calculate phonon related properties. See :ref:`analyze_phonon`.

If you already have force constants, the first and second steps can be
omitted. However your force constants have to be converted to the
:ref:`format <file_force_constants>` that phonopy can read. The
:ref:`VASP interface <vasp_force_constants>` to convert force
constants is prepared in phonopy.
