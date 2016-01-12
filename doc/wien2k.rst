.. _wien2k_interface:

Wien2k & phonopy calculation
=========================================

How to run
-----------

The Wien2k-phonopy calculation works as follows:

1) Read a Wien2k struct file with the P lattice format and create
   supercells with the Wien2k struct format of P lattice using
   ``--wien2k`` option (:ref:`wien2k_mode`)::

   % phonopy --wien2k -d --dim="2 2 2" -c case.struct

   In this example, 2x2x2 supercells are created. ``case.structS`` and
   ``case.structS-xxx`` (``xxx`` are numbers) are the perfect
   supercell and the supercells with displacements,
   respectively. Perhaps these are renamed to ``case-xxx.struct`` and
   stored in ``case-xxx`` directories, then to be calculated using
   Wien2k.

2) Calculate forces on atoms in the supercells with
   displacements. Select to use ``case.struct_nn`` file when running
   ``init_lapw``. In the Wien2k calculations, the force convergence
   option of ``-fc`` has to be specified to obtain ``total forces``. A
   first attempt of the force convergence criterion may be 0.1
   (mRy/a.u.). It is recommended to try more strict convergence
   criteria with saving one by one using ``save_lapw``.

3) Create ``FORCE_SETS``
   
   * Use ``scf2forces`` that is found at
     http://www.wien2k.at/reg_user/unsupported/.
   * Or try experimetal support of ``-f`` option::

     % phonopy --wien2k -f case-001.scf case-002.scf ...

     where ``case-xxx.scf`` are the Wien2k results for the
     supercells. ``case-xxx.scf`` has to contain ``FGLxxx`` lines with
     ``total forces``. When calculating supercells, the number of
     non-equivalent atoms determined by ``nn`` has to match with the
     number of non-equivalent atoms determined by ``phonopy``. The
     former is found to watch ``case-xxx.struct`` after ``nn`` (it is
     supposed that ``case-xxx.struct`` created by ``nn`` is used to
     calculate forces), and the later is displayed in the step 1. An
     example is found in ``example/NaCl-wien2k``.

     The above procedure with ``-f`` option may fail. In this case,
     Wien2k calculations of ``case-xxx.scf`` with P1 symmetry may be
     used for phonopy testing purpose though it computationally
     demands a lot. If phonopy finds that ``case-xxx.scf`` are
     calculated with P1 symmetry, phonopy handles this as a special
     case. An example is found in ``example/NaCl-wien2k-P1``.


4) Run post-process of phonopy with the Wien2k unit cell struct file
   used in the step 1::

   % phonopy --wien2k -c case.struct --dim="2 2 2" [other-OPTIONS] [setting-file]

Phonopy can read only the **P lattice format**. Therefore you have to
convert your struct file to that with the P lattice format. This may
be done using ``supercell`` script in the Wien2k package by making
1x1x1 supercell.

