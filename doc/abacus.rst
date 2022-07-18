.. _abacus_interface:

ABACUS & phonopy calculation
=========================================

How to run
-----------

A procedure of ABACUS-phonopy calculation is as follows:

1) Prepare a 'setting.conf' with following tags
    ::
    DIM = 2 2 2         
    ATOM_NAME = Al     

    * when three integers are specified after ``DIM =``, a supercell elongated along axes of unit cell is created
    * chemical symbols are specified after ``ATOM_NAME =``, and number of symbols should match ``ntype`` in ABACUS INPUT file.
        The interface reads ABACUS STRU file according to the settings of this tag.

2) To obtain supercells with displacements, run phonopy:
    ::
        % phonopy setting.conf --abacus -d

    In this example, 2x2x2 supercells are created. ``STRU.in`` and
    ``STRU-{number}`` give the perfect supercell and supercells 
    with displacements, respectively. ``phonopy_disp.yaml`` is also created. 
    This file contains information on displacements. Perhaps the supercell files are
    stored in ``disp-{number}`` directories, then ABACUS calculations are
    executed in these directories.

3) Calculate forces on atoms in the supercells with displacements. For each SCF calculation, 
    you should specify ``stru_file`` with ``STRU-{number}`` and ``cal_force=1`` in INPUT in order to calculate force using ABACUS. 
    Be careful not to relax the structures

4) Then create ``FORCE_SETS`` file using ABACUS inteface
    ::
        % phonopy -f ./disp-001//OUT*/running*.log ./disp-002//OUT*/running*.log ...

    Two examples with different settings of basis sets are found in ``example/Al-ABACUS``.