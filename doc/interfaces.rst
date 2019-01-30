.. _calculator_interfaces:

Interfaces to calculators
==========================

The interfaces for VASP, WIEN2k, Quantum ESPRESSO (QE), ABINIT, Elk,
SIESTA, and CRYSTAL are built in to the usual phonopy command. See the
command options and how to invoke each of them at
:ref:`force_calculators`. :ref:`LAMMPS interface
<external_tools_phonolammps>` is provided as an external tool by Abel
Carreras.

Physical unit system for calculator
------------------------------------

Physical unit systems used for the calculators are as follows::

           | Distance   Atomic mass   Force         Force constants
   -----------------------------------------------------------------
   VASP    | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
   WIEN2k  | au (bohr)  AMU           mRy/au        mRy/au^2
   QE      | au (bohr)  AMU           Ry/au         Ry/au^2
   ABINIT  | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   SIESTA  | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   Elk     | au (bohr)  AMU           hartree/au    hartree/au^2
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
   WIEN2k  | case.struct
   QE      | unitcell.in
   ABINIT  | unitcell.in
   SIESTA  | input.fdf
   Elk     | elk.in
   CRYSTAL | crystal.o
   DFTB+   | geo.gen

Default displacement distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without specifying ``DISPLACEMENT_DISTANCE`` tag or ``--amplitude``
option, default displacement distance is used when creating supercells
with displacements ``CREATE_DISPLACEMENTS = .TRUE.`` or ``-d``
option. The default value is dependent on calculator, and the list is
shown below::

   VASP    | 0.01 Angstrom
   WIEN2k  | 0.02 au (bohr)
   QE      | 0.02 au (bohr)
   ABINIT  | 0.02 au (bohr)
   SIESTA  | 0.02 au (bohr)
   Elk     | 0.02 au (bohr)
   CRYSTAL | 0.01 Angstrom
   DFTB+   | 0.01 au (bohr)

.. _frequency_default_value_interfaces:

Default unit conversion factor of phonon frequency to THz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   VASP    | 15.633302
   WIEN2k  | 3.44595837
   QE      | 108.97077
   ABINIT  | 21.49068
   SIESTA  | 21.49068
   Elk     | 154.10794
   CRYSTAL | 15.633302
   DFTB+   | 154.10794

.. _nac_default_value_interfaces:

Default unit conversion factor for non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   VASP    | 14.399652
   WIEN2k  | 2000
   QE      | 2
   ABINIT  | 51.422090
   SIESTA  | 51.422090
   Elk     | 1
   CRYSTAL | 14.399652
   DFTB+   | 14.399652


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
   qe
   abinit
   siesta
   elk
   crystal
   dftb+

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
