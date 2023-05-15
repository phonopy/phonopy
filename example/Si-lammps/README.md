# Si example using LAMMPS interface

The `pair_style` of
`polymlp` is a LAMMPS module of the polynomial machine learning potentials
provided at https://sekocha.github.io/lammps/index-e.html. For the silicon
calculation found in the [example
directory](https://github.com/phonopy/phonopy/tree/develop/example), `mlp.lammp`
of gtinv-289 was obtained from [Polynomial Machine Learning Potential Repository
at Kyoto
University](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html).

This example shows how to generate supercells from non-LAMMPS structure but the
structure defined in yaml style format. This is not supported by usual phonopy
command line interface. So a python script that runs this task using phonopy-API
is provided in this example directory. See also the documentation of the phonopy
LAMMPS interface.

1. Generate supercells using a python script

   ```
   % python generate_displacements.py
   ```

   The `supercell-001` LAMMPS structure file follows the LAMMPS structure input
   format. The basis vectors are rotated from that defined in
   `phonopy_disp.yaml`. Therefore forces as obtained LAMMPS calculation have to
   be rotated back to the original coordinate system, which is performed
   automatically in the step 3.

2. Run LAMMPS

   ```bash
   % lmp_serial -in in.polymlp
   ```

   Suppose that the LAMMPS output file name is removed to `lammps_forces_Si.0`
   after the LAMMPS calculation.

3. Make `FORCE_SETS`

   ```
   % phonopy -f lammps_forces_Si.0
           _
     _ __ | |__   ___  _ __   ___   _ __  _   _
    | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
    | |_) | | | | (_) | | | | (_) || |_) | |_| |
    | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
    |_|                            |_|    |___/
                                         2.18.0

   Compiled with OpenMP support (max 10 threads).
   Python version 3.11.0
   Spglib version 2.0.2

   Calculator interface: lammps
   Displacements were read from "phonopy_disp.yaml".
   1. Drift force of "lammps_forces_Si.0" to be subtracted
    -0.00000000  -0.00000000  -0.00000000
   Forces parsed from LAMMPS output were rotated by F=R.F(lammps) with R:
     0.00000 0.57735 0.57735
     0.70711 0.57735 0.57735
     0.70711 -0.57735 -0.57735
   "FORCE_SETS" has been created.
                    _
      ___ _ __   __| |
     / _ \ '_ \ / _` |
    |  __/ | | | (_| |
     \___|_| |_|\__,_|
   ```

   The rotation of forces is performed and the rotation matrix is found in last
   lines of the output.
