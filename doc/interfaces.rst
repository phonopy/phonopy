.. _calculator_interfaces:

Interfaces to calculators
==========================

.. contents::
   :depth: 3
   :local:

The interfaces for VASP, WIEN2k, Quantum ESPRESSO (QE), ABINIT, Elk,
SIESTA, CRYSTAL, DFTB+, TURBOMOLE, and FHI-AIMS are built in to the usual phonopy command. See the
command options and how to invoke each of them at
:ref:`force_calculators`. :ref:`LAMMPS interface
<external_tools_phonolammps>` is provided as an external tool by Abel
Carreras.

.. _interfaces_to_force_calculators:

List of force calculators
-------------------------

Short tutorials for the force calculators are found in the following
pages.

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
   turbomole
   cp2k
   aims

The VASP DFPT
interface reads ``vasprun.xml`` and creates ``FORCE_CONSTANTS`` file.

.. toctree::
   :maxdepth: 1

   vasp-dfpt

Physical unit system for calculator
------------------------------------

Physical unit systems used for the calculators are as follows::

             | Distance   Atomic mass   Force         Force constants
   -----------------------------------------------------------------
   VASP      | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
   WIEN2k    | au (bohr)  AMU           mRy/au        mRy/au^2
   QE        | au (bohr)  AMU           Ry/au         Ry/au^2
   ABINIT    | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   SIESTA    | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   Elk       | au (bohr)  AMU           hartree/au    hartree/au^2
   CRYSTAL   | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
   TURBOMOLE | au (bohr)  AMU           hartree/au    hartree/au^2
   CP2K      | Angstrom   AMU           hartree/au    hartree/Angstrom.au
   FHI-AIMS  | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2

For these sets of physical properties, phonon frequency is calculated
in THz.

Default file name, value, and conversion factor
---------------------------------------------------

Default unit cell file name for calculator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without specifying ``-c`` option, default file name for unit cell is
used as shown below::

   VASP      | POSCAR
   WIEN2k    | case.struct
   QE        | unitcell.in
   ABINIT    | unitcell.in
   SIESTA    | input.fdf
   Elk       | elk.in
   CRYSTAL   | crystal.o
   DFTB+     | geo.gen
   TURBOMOLE | control
   CP2K      | unitcell.inp
   FHI-AIMS  | geometry.in

Default displacement distances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Without specifying ``DISPLACEMENT_DISTANCE`` tag or ``--amplitude``
option, default displacement distance is used when creating supercells
with displacements ``CREATE_DISPLACEMENTS = .TRUE.`` or ``-d``
option. The default value is dependent on calculator, and the list is
shown below::

   VASP      | 0.01 Angstrom
   WIEN2k    | 0.02 au (bohr)
   QE        | 0.02 au (bohr)
   ABINIT    | 0.02 au (bohr)
   SIESTA    | 0.02 au (bohr)
   Elk       | 0.02 au (bohr)
   CRYSTAL   | 0.01 Angstrom
   DFTB+     | 0.01 au (bohr)
   TURBOMOLE | 0.02 au (bohr)
   CP2K      | 0.01 Angstrom
   FHI-AIMS  | 0.01 Angstrom

.. _frequency_default_value_interfaces:

Default unit conversion factor of phonon frequency to THz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   VASP      | 15.633302
   WIEN2k    | 3.44595837
   QE        | 108.97077
   ABINIT    | 21.49068
   SIESTA    | 21.49068
   Elk       | 154.10794
   CRYSTAL   | 15.633302
   DFTB+     | 154.10794
   TURBOMOLE | 154.10794
   CP2K      | 112.10516
   FHI-AIMS  | 15.633302

.. _nac_default_value_interfaces:

Default unit conversion factor for non-analytical term correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   VASP      | 14.399652
   WIEN2k    | 2000
   QE        | 2
   ABINIT    | 51.422090
   SIESTA    | 51.422090
   Elk       | 1
   CRYSTAL   | 14.399652
   DFTB+     | 14.399652
   TURBOMOLE | 1
   CP2K      | None (N/A)
   FHI-AIMS  | 14.399652
