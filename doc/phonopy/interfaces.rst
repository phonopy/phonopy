Interfaces to calculators and phonopy API
==========================================

.. _calculator_interfaces:

Interfaces to force calculators
--------------------------------

The interfaces for VASP, Wien2k, Pwscf, Abinit, and Elk are built in
to the usual phonopy command. See the command options and how to
invoke each of them at :ref:`force_calculators`. 

For each calculator, each physical unit system is used. The physical
unit systems used for the calculators are summarized below.

::

           | Distance   Atomic mass   Force         Force constants
   -----------------------------------------------------------------
   VASP    | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
   Wien2k  | au (bohr)  AMU           mRy/au        mRy/au^2
   Pwscf   | au (bohr)  AMU           Ry/au         Ry/au^2
   Abinit  | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
   elk     | au (bohr)  AMU           hartree/au    hartree/au^2

Default unit cell file names are also changed according to the calculators::
    
   VASP    | POSCAR     
   Wien2k  | case.struct
   Abinit  | unitcell.in
   Pwscf   | unitcell.in
   Elk     | elk.in

Short tutorials for there calculators are found in the following pages.

.. toctree::
   :maxdepth: 2

   procedure
   wien2k
   pwscf
   abinit
   elk

Interface to  VASP DFPT force constants
---------------------------------------

Using VASP DFPT feature, force constants are directly
calculated. Phonopy VASP DFPT interface reads ``vasprun.xml`` and
creates ``FORCE_CONSTANTS`` file.

.. toctree::
   :maxdepth: 2

   vasp

Interface to FHI-aims forces
-----------------------------

For FHI-aims, there is a special command, ``phonopy-FHI-aims``. This
tool is maintained by FHI-aims community and questions may be sent to the
FHI-aims mailing list.

.. toctree::
   :maxdepth: 2

   FHI-aims

Phonopy API
------------

Phonopy can be used as a python module. Phonopy API is explained in
the following page.

.. toctree::
   :maxdepth: 2

   phonopy-module


|sflogo|

.. |sflogo| image:: http://sflogo.sourceforge.net/sflogo.php?group_id=161614&type=1
            :target: http://sourceforge.net


