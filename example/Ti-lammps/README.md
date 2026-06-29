# HCP Ti example using LAMMPS interface

The `pair_style` of `polymlp` is a LAMMPS module of the polynomial machine
learning potentials provided at https://sekocha.github.io/lammps/index-e.html.
For the HCP Ti calculation found in the [example
directory](https://github.com/phonopy/phonopy/tree/develop/example), `mlp.lammp`
of gtinv-294 was obtained from [Polynomial Machine Learning Potential Repository
at Kyoto
University](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html).

1. Read a lammps input structure file and create supercells with

   ```
   % phonopy-init --lammps -c lammps_structure_Ti -d --dim 4 4 3
           _
     _ __ | |__   ___  _ __   ___   _ __  _   _
   | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
   | |_) | | | | (_) | | | | (_) || |_) | |_| |
   | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
   |_|                            |_|    |___/
                                         4.3.1

   -------------------------[time 2026-06-29 15:11:48]-------------------------
   Rust backend (phonors) using rayon (10 threads).
   Python version 3.13.11
   Spglib version 2.7.0

   Calculator interface: lammps
   Crystal structure was read from "lammps_structure_Ti".
   Unit of length: angstrom
   Displacements creation mode
     Systematic displacements
   Settings:
     Supercell: [4 4 3]
     Primitive matrix:
       [ 1.00000000e+00 -1.30752844e-17  0.00000000e+00]
       [0. 1. 0.]
       [0. 0. 1.]
   Spacegroup: P6_3/mmc (194)
   Number of symmetry operations in supercell: 1152
   Use -v option to watch primitive cell, unit cell, and supercell structures.

   Generated number of supercells: 1
   "phonopy_disp.yaml" and supercells have been created.

   Summary of calculation was written in "phonopy_disp.yaml".
   -------------------------[time 2026-06-29 15:11:48]-------------------------
                    _
      ___ _ __   __| |
     / _ \ '_ \ / _` |
    |  __/ | | | (_| |
     \___|_| |_|\__,_|
   ```

2. Run LAMMPS

   ```bash
   % lmp_serial -in in.polymlp
   ```

   Suppose that the LAMMPS output file name is removed to `lammps_forces_Ti.0`
   after the LAMMPS calculation.

3. Make `FORCE_SETS`

   ```
   % phonopy-init -f lammps_forces_Ti.0
           _
     _ __ | |__   ___  _ __   ___   _ __  _   _
    | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
    | |_) | | | | (_) | | | | (_) || |_) | |_| |
    | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
    |_|                            |_|    |___/
                                          4.3.1

   -------------------------[time 2026-06-29 15:14:24]-------------------------
   Rust backend (phonors) using rayon (10 threads).
   Python version 3.13.11
   Spglib version 2.7.0

   Calculator interface: lammps
   Displacements were read from "phonopy_disp.yaml".
     Number of supercell files: 1
     Number of displacements in "phonopy_disp.yaml": 1
   1. Drift force of "lammps_forces_Ti.0" to be subtracted
    -0.00000000  -0.00000000   0.00000000
   Forces parsed from LAMMPS output were rotated by F=R.F(lammps) with R:
     1.00000 0.00000 0.00000
     0.00000 1.00000 0.00000
     0.00000 0.00000 1.00000
   "FORCE_SETS" has been created.
   -------------------------[time 2026-06-29 15:14:24]-------------------------
                    _
      ___ _ __   __| |
     / _ \ '_ \ / _` |
    |  __/ | | | (_| |
     \___|_| |_|\__,_|
   ```
