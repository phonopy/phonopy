.. _vasp_fd_interface:

VASP & phonopy calculation
===========================

Please follow the page :ref:`tutorial` and :ref:`examples_link`.

.. _vasp_dfpt_interface:

VASP-DFPT & phonopy calculation
===========================================

How to run
-----------

VASP can calculate force constants in real space using DFPT. The
procedure to calculate phonon properties may be as follows:

1) Prepare unit cell structure named, e.g., ``POSCAR-unitcell``. The
   following structure is a conventional unit cell of NaCl.

   ::

       Na Cl                         
          1.00000000000000     
            5.6903014761756712    0.0000000000000000    0.0000000000000000
            0.0000000000000000    5.6903014761756712    0.0000000000000000
            0.0000000000000000    0.0000000000000000    5.6903014761756712
          4   4
       Direct
         0.0000000000000000  0.0000000000000000  0.0000000000000000
         0.0000000000000000  0.5000000000000000  0.5000000000000000
         0.5000000000000000  0.0000000000000000  0.5000000000000000
         0.5000000000000000  0.5000000000000000  0.0000000000000000
         0.5000000000000000  0.5000000000000000  0.5000000000000000
         0.5000000000000000  0.0000000000000000  0.0000000000000000
         0.0000000000000000  0.5000000000000000  0.0000000000000000
         0.0000000000000000  0.0000000000000000  0.5000000000000000


2) Prepare a perfect supercell structure from ``POSCAR-unitcell``,
   e.g.,

   ::

      % phonopy -d --dim="2 2 2" -c POSCAR-unitcell

3) Rename ``SPOSCAR`` created in (2) to
   ``POSCAR`` (``POSCAR-{number}`` and ``disp.yaml`` files will never be used.)

   ::

      % mv SPOSCAR POSCAR

4) Calculate force constants of the perfect supercell by running VASP
   with ``IBRION = 8`` and ``NSW = 1``. An example of ``INCAR`` for
   insulator may be such like (**just an example!**)::

        PREC = Accurate
       ENCUT = 500
      IBRION = 8
       EDIFF = 1.0e-08
       IALGO = 38
      ISMEAR = 0; SIGMA = 0.1
       LREAL = .FALSE.
     ADDGRID = .TRUE.
       LWAVE = .FALSE.
      LCHARG = .FALSE.

5) After finishing the VASP calculation, confirm ``vasprun.xml``
   contains ``hessian`` elements, and then create ``FORCE_CONSTANTS``::

   % phonopy --fc vasprun.xml
  
6) Run phonopy with the original unit cell ``POSCAR-unitcell`` and
   setting tag ``FORCE_CONSTANTS = READ`` or ``--readfc`` option,
   e.g., as found in ``example/NaCl-VASPdfpt``

   ::

      % phonopy --dim="2 2 2" -c POSCAR-unitcell band.conf
              _                                    
        _ __ | |__   ___  _ __   ___   _ __  _   _ 
       | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
       | |_) | | | | (_) | | | | (_) || |_) | |_| |
       | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
       |_|                            |_|    |___/
      
                                           1.1
      
      Band structure mode
      Settings:
        Force constants: read
        Supercell:  [2 2 2]
        Primitive axis:
           [ 0.   0.5  0.5]
           [ 0.5  0.   0.5]
           [ 0.5  0.5  0. ]
      Spacegroup:  Fm-3m (225)
      Paths in reciprocal reduced coordinates:
      [ 0.00  0.00  0.00] --> [ 0.50  0.00  0.00]
      [ 0.50  0.00  0.00] --> [ 0.50  0.50  0.00]
      [ 0.50  0.50  0.00] --> [-0.00 -0.00  0.00]
      [ 0.00  0.00  0.00] --> [ 0.50  0.50  0.50]

.. |NaCl-VASPdfpt| image:: NaCl-VASPdfpt.png
                   :scale: 50

|NaCl-VASPdfpt|


