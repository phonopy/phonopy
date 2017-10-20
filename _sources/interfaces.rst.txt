.. _calculator_interfaces:

Interfaces to calculators
==========================

The interfaces for VASP, Wien2k, Pwscf, Abinit, Elk and CRYSTAL are built in
to the usual phonopy command. See the command options and how to
invoke each of them at :ref:`force_calculators`.

Physical unit system for calculator
------------------------------------

Physical unit systems used for the calculators are as follows::

           | Distance   Atomic mass   Force         Force constants
   -----------------------------------------------------------------
   VASP    | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
   Wien2k  | au (bohr)  AMU           mRy/au        mRy/au^2
   Pwscf   | au (bohr)  AMU           Ry/au         Ry/au^2
   Abinit  | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   Siesta  | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   elk     | au (bohr)  AMU           hartree/au    hartree/au^2
   CRYSTAL | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2

For these sets of physical properties, phonon frequency is calculated
in THz.

Default file name, value, and conversion factor
---------------------------------------------------

Default unit cell file name for calculator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without specifying ``-c`` option, default file name for unit cell is
used as shown below::

   VASP    | POSCAR
   Wien2k  | case.struct
   Pwscf   | unitcell.in
   Abinit  | unitcell.in
   Siesta  | input.fdf
   Elk     | elk.in
   CRYSTAL | crystal.o

Default displacement distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without specifying ``DISPLACEMENT_DISTANCE`` tag or ``--amplitude``
option, default displacement distance is used when creating supercells
with displacements ``CREATE_DISPLACEMENTS = .TRUE.`` or ``-d``
option. The default value is dependent on calculator, and the list is
shown below::

   VASP    | 0.01 Angstrom
   Wien2k  | 0.02 au (bohr)
   Pwscf   | 0.02 au (bohr)
   Abinit  | 0.02 au (bohr)
   Siesta  | 0.02 au (bohr)
   Elk     | 0.02 au (bohr)
   CRYSTAL | 0.01 Angstrom

.. _nac_default_value_interfaces:

Default unit conversion factor for non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   VASP    | 14.399652
   Wien2k  | 2000
   Pwscf   | 2
   Abinit  | 51.422090
   Siesta  | 51.422090
   Elk     | 1
   CRYSTAL | 14.399652


.. _tutorials_for_calculators:

Tutorials for calculators
--------------------------

Force calculators
^^^^^^^^^^^^^^^^^^^

Short tutorials for force calculators are found in the following pages.

.. toctree::
   :maxdepth: 1

   vasp
   wien2k
   pwscf
   abinit
   siesta
   elk
   crystal

VASP DFPT force constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using VASP DFPT feature, force constants are directly
calculated. Phonopy VASP DFPT interface reads ``vasprun.xml`` and
creates ``FORCE_CONSTANTS`` file.

.. toctree::
   :maxdepth: 1

   vasp-dfpt

For FHI-aims
^^^^^^^^^^^^^

For FHI-aims, there is a special command, ``phonopy-FHI-aims``. This
tool is maintained by FHI-aims community and questions may be sent to the
FHI-aims mailing list.

.. toctree::
   :maxdepth: 1

   FHI-aims
