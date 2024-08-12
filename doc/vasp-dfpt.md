(vasp_dfpt_interface)=
# VASP-DFPT & phonopy calculation

## How to run

VASP can calculate force constants in real space using DFPT. The
procedure to calculate phonon properties may be as follows:

1) Prepare unit cell structure named, e.g., `POSCAR-unitcell`. The
   following structure is a conventional unit cell of NaCl.

   ```
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
   ```


2) Prepare a perfect supercell structure from `POSCAR-unitcell`,

   ```bash
   % phonopy -d --dim 2 2 2 -c POSCAR-unitcell
   ```

3) For later convenience, it is recommended to generate `phonopy_disp.yaml`
   using `SPOSCAR` file,

   ```bash
   % phonopy -d --dim 1 1 1 --pa auto -c SPOSCAR
   ```

4) Rename `SPOSCAR` created in (3) to `POSCAR` to be used in the VASP
   calculation.

   ```bash
   % mv SPOSCAR POSCAR
   ```

   `POSCAR-{number}` files will never be used.

4) Calculate force constants of the perfect supercell by running VASP
   with `IBRION = 8`. An example of `INCAR` for
   insulator may be such like (**just an example!**)

   ```
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
   ```

5) After finishing the VASP calculation, confirm `vasprun.xml`
   contains `hessian` elements, and then create `FORCE_CONSTANTS` by

   ```bash
   % phonopy --fc vasprun.xml
   ```

6) Run phonopy

   ~~~bash
   % phonopy-load --readfc --band "0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5" -p

           _
     _ __ | |__   ___  _ __   ___   _ __  _   _
    | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
    | |_) | | | | (_) | | | | (_) || |_) | |_| |
    | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
    |_|                            |_|    |___/
                                         2.26.6

   Compiled with OpenMP support (max 10 threads).
   Running in phonopy.load mode.
   Python version 3.12.4
   Spglib version 2.4.0

   Crystal structure was read from "phonopy_disp.yaml".
   Unit of length: angstrom
   Band structure mode
   Settings:
     Supercell: [1 1 1]
     Primitive matrix:
       [0.   0.25 0.25]
       [0.25 0.   0.25]
       [0.25 0.25 0.  ]
   Spacegroup: Fm-3m (225)
   Number of symmetry operations in supercell: 1536
   Use -v option to watch primitive cell, unit cell, and supercell structures.

   Force constants are read from "FORCE_CONSTANTS".
   Force constants format was transformed to compact format.
   Array shape of force constants: (2, 64, 3, 3)
   Max drift after symmetrization by translation: -0.000000 (zz) -0.000000 (zz)

   Reciprocal space paths in reduced coordinates:
   [ 0.000  0.000  0.000] --> [ 0.500  0.000  0.000]
   [ 0.500  0.000  0.000] --> [ 0.500  0.500  0.000]
   [ 0.500  0.500  0.000] --> [ 0.000  0.000  0.000]
   [ 0.000  0.000  0.000] --> [ 0.500  0.500  0.500]

   Summary of calculation was written in "phonopy.yaml".
                    _
      ___ _ __   __| |
     / _ \ '_ \ / _` |
    |  __/ | | | (_| |
     \___|_| |_|\__,_|
   ~~~

   ```{image} NaCl-VASPdfpt.png
   :scale: 50
   ```

   When running with `phonopy` command, `--readfc` option is necessary:

   ```bash
   % phonopy --readfc band.conf
   ```
