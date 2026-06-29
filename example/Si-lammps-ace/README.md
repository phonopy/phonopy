# Si example using LAMMPS interface (ACE potential)

This example computes the phonon band structure of diamond silicon using the
LAMMPS `pair_style pace` (atomic cluster expansion, ACE). It is a self-running
alternative to the `polymlp`-based `Si-lammps` example: any LAMMPS build that
includes the `ML-PACE` package can run it (for instance the conda-forge `lammps`
package).

## Potential file and license

The silicon ACE potential file `Si_npj_CompMat2021.ace` is taken from the
dataset accompanying

> Y. Lysogorskiy, C. van der Oord, A. Bochkarev, S. Menon, M. Rinaldi, T.
> Hammerschmidt, M. Mrovec, A. Thompson, G. Csanyi, C. Ortner, and R. Drautz,
> "Performant implementation of the atomic cluster expansion (PACE) and
> application to copper and silicon", npj Comput. Mater. 7, 97 (2021).

It is distributed on Zenodo under the **CC-BY-4.0** license:

> DOI: 10.5281/zenodo.4734036 (https://doi.org/10.5281/zenodo.4734036)

Download `Si_npj_CompMat2021.ace` from that record into this directory before
running LAMMPS. The file is not redistributed with phonopy; please observe the
CC-BY-4.0 attribution terms (cite the reference above).

## Lattice constant

`lammps_structure_Si` uses the experimental room-temperature (~300 K) lattice
constant of silicon, a = 5.431 A (a / 2 = 2.71550 A). The ACE equilibrium
lattice constant differs slightly from this value, so a small residual stress
may show up as near-zero or slightly imaginary acoustic frequencies around
Gamma. To remove it, relax the cell with the potential first (see the
structure-optimization appendix in the phonopy LAMMPS documentation) and use the
relaxed constant instead.

## Steps

1. Generate supercells with displacements. Two equivalent routes are provided.

   (a) From the LAMMPS structure file `lammps_structure_Si`:

   ```
   % phonopy-init --lammps -c lammps_structure_Si -d --dim 2 2 2
   ```

   (b) From the unit cell defined in `phonopy_unitcell.yaml`:

   ```
   % python generate_displacements.py
   ```

   Route (b) keeps the cell in its original (unrotated) orientation in
   `phonopy_disp.yaml`, whereas route (a) reads the cell in the rotated LAMMPS
   triclinic convention. Either way, `supercell-001` follows the LAMMPS structure
   input format with rotated basis vectors, and the forces obtained from LAMMPS
   are rotated back to the original coordinate system automatically in step 3.

2. Run LAMMPS

   ```bash
   % lmp -in in.force
   ```

   Rename the LAMMPS output file `force.0` to `lammps_forces_Si.0`.

3. Make `FORCE_SETS`

   ```
   % phonopy-init -f lammps_forces_Si.0
   ```

   The drift force (the net force on the whole supercell) is subtracted and the
   forces are rotated back to the original orientation; the rotation matrix is
   printed in the output.

4. Plot the phonon band structure

   ```
   % phonopy --band auto -p
   ```
