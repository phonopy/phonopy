.. _tutorial:

Short tutorials
================

Tutorials for different force calculators, VASP, WIEN2k, PWscf (QE),
ABINIT, SIESTA, Elk, and, CRYSTAL, are given at
:ref:`interfaces_to_force_calculators`.

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

3. Calculate phonon related properties.

If you already have force constants, the first and second steps can be
omitted. However your force constants have to be converted to the
:ref:`format <file_force_constants>` that phonopy can read. The
:ref:`VASP interface <vasp_force_constants>` to convert force
constants is prepared in phonopy.
