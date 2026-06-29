(lammps_interface)=

# LAMMPS & phonopy calculation

Phonopy drives a LAMMPS force calculation in the same way as an ab-initio
calculator: it builds supercells with atomic displacements, LAMMPS evaluates the
forces on every atom, and phonopy collects them into `FORCE_SETS`.

## Workflow

1. (Optional) Relax the unit cell with the potential
   ({ref}`lammps_structure_optimization`).
2. Generate supercells with displacements
   ({ref}`lammps_supercell_generation`).
3. Compute forces with LAMMPS for each supercell
   ({ref}`lammps_force_calculation`).
4. Create `FORCE_SETS` with `phonopy-init -f` ({ref}`lammps_force_calculation`).
5. Run the phonon calculation (band structure, DOS, ...) with `phonopy-load`.

Assumptions:

- The LAMMPS calculation uses `units metal` and `atom_style atomic`.
- LAMMPS version 15Sep2022 or later is assumed.
- Forces are read from a LAMMPS dump written in a specific format
  ({ref}`lammps_force_calculation`).

Worked examples are in the [example
directory](https://github.com/phonopy/phonopy/tree/develop/example): `Ti-lammps`
and `Si-lammps` (`polymlp` potential) and `Si-lammps-ace` (ACE `pace`
potential).

(lammps_structure_input_format)=

## LAMMPS structure format

Phonopy reads and writes crystal structures in the LAMMPS
[read_data](https://docs.lammps.org/read_data.html) format, with two points to
keep in mind:

- The structure must be described like a `read_data` file (see the supported
  keywords below).
- Basis vectors are rotated to the LAMMPS [triclinic simulation
  box](https://docs.lammps.org/Howto_triclinic.html) convention,

  ```
  a = (a_x 0   0  )
  b = (b_x b_y 0  )
  c = (c_x c_y c_z)
  ```

  so a structure written for LAMMPS is a rotated copy of the phonopy cell. The
  forces from LAMMPS are rotated back automatically when `FORCE_SETS` is created.

### Supported `read_data` keywords

Header:

```
atoms
atom types
xlo xhi
ylo yhi
zlo zhi
xy xz yz
```

Body:

```
Atom Type Labels
Masses
Atoms
```

`Atom Type Labels` is new in LAMMPS 15Sep2022; see [Type
labels](https://docs.lammps.org/Howto_type_labels.html). Phonopy writes a
`Masses` section (after `Atom Type Labels`, since LAMMPS requires the labels to
be read first) and reads it back. Because the masses are in the structure file,
the LAMMPS input script for the force calculation does not need a `mass` command.

### Example

An HCP structure in the LAMMPS triclinic box format:

```
#

2 atoms
1 atom types

0.0 2.923479689273095 xlo xhi   # xx
0.0 2.531807678358337 ylo yhi   # yy
0.0 4.624022835916574 zlo zhi   # zz

-1.461739844636547 0.000000000000000 0.000000000000000 xy xz yz

Atom Type Labels

1 Ti

Masses

1 47.867 # Ti

Atoms

1 Ti 0.000000000000001 1.687871785572226 3.468017126937431
2 Ti 1.461739844636549 0.843935892786111 1.156005708979144
```

The `Masses` section is optional in an input structure file: when `Atom Type
Labels` are present, phonopy assigns the standard atomic mass of each element, so
`Masses` can be omitted. Phonopy still writes a `Masses` section into the
supercells it generates, so the force calculation needs no `mass` command.

(lammps_supercell_generation)=

## Generating supercells with displacements

There are two routes, depending on how the unit cell is provided. After either
route, `phonopy_disp.yaml` and the supercell files (`supercell`,
`supercell-001`, `supercell-002`, ...) are created. Symmetry reduces the number
of inequivalent displacements, so a high-symmetry crystal may have only
`supercell-001`.

### Route (a): from a LAMMPS structure file

When the unit cell is already in the LAMMPS format (e.g. `lammps_structure_Si`),
generate the supercells directly:

```
% phonopy-init --lammps -c lammps_structure_Si -d --dim 2 2 2
```

### Route (b): from a unit cell defined in yaml

A LAMMPS structure file is expressed in the rotated triclinic convention. To keep
the cell in its original (for example, the symmetric, unrotated primitive)
orientation, define it in yaml and generate the supercells with a short script. A
silicon primitive cell:

```yaml
lattice:
- [0.000000000000000, 2.733099421887393, 2.733099421887393] # a
- [2.733099421887393, 0.000000000000000, 2.733099421887393] # b
- [2.733099421887393, 2.733099421887393, 0.000000000000000] # c
points:
- symbol: Si # 1
  coordinates: [0.875000000000000, 0.875000000000000, 0.875000000000000]
- symbol: Si # 2
  coordinates: [0.125000000000000, 0.125000000000000, 0.125000000000000]
```

Saved as `phonopy_unitcell.yaml`, generate a 2x2x2 supercell with:

```python
import phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.calculator import write_supercells_with_displacements

cell = read_cell_yaml("phonopy_unitcell.yaml")
ph = phonopy.load(unitcell=cell, supercell_matrix=[2, 2, 2], calculator='lammps')
ph.generate_displacements()
ph.save("phonopy_disp.yaml")
write_supercells_with_displacements("lammps", ph.supercell, ph.supercells_with_displacements)
```

With route (b), `phonopy_disp.yaml` stores the cell in the original (unrotated)
orientation, whereas route (a) stores the rotated triclinic cell. In both cases
`supercell-001` follows the LAMMPS structure file format. The `Si-lammps` example
uses route (b); `Ti-lammps` and `Si-lammps-ace` use route (a).

(lammps_force_calculation)=

## Force calculation and FORCE_SETS

Phonopy reads forces from a LAMMPS dump written in a fixed text format. For each
`supercell-xxx`, evaluate the forces once with `run 0` (no time integration):

```
units metal

read_data supercell-001

pair_style  <potential>
pair_coeff  <...>

dump phonopy all custom 1 force.* id type x y z fx fy fz
dump_modify phonopy format line "%d %d %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f"
run 0
```

Only the `pair_style`/`pair_coeff` lines change between potentials. Keep the
`dump` and `dump_modify` lines verbatim so that phonopy can parse the output;
`force.*` expands to `force.0` for `run 0`.

Rename each output (e.g. to `lammps_forces_Si.0`, `lammps_forces_Si.1`, ...) and
create `FORCE_SETS`:

```
% phonopy-init -f lammps_forces_Si.0 lammps_forces_Si.1 ...
```

List the force files in the same order as the supercells. Phonopy subtracts the
drift force (the net force on the supercell) and rotates the forces back from the
LAMMPS frame to the phonopy cell, printing the rotation matrix `R`:

```
Forces parsed from LAMMPS output were rotated by F=R.F(lammps) with R:
  1.00000 0.00000 0.00000
  0.00000 0.00000 0.00000
  0.00000 1.00000 1.00000
```

## Running the phonon calculation

Once `FORCE_SETS` exists, run phonopy with `phonopy-load`, which reads
`phonopy_disp.yaml` and `FORCE_SETS` automatically, e.g. to plot the band
structure:

```
% phonopy-load phonopy_disp.yaml --config band.conf -p
```

## Examples

### Ti-lammps and Si-lammps (polymlp)

These use the `polymlp` polynomial machine-learning potential, a LAMMPS module
provided at <https://sekocha.github.io/lammps/index-e.html>. The potential files
(`mlp.lammps`, gtinv-294 for Ti and gtinv-289 for Si) are obtained from the
[Polynomial Machine Learning Potential Repository at Kyoto
University](http://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html). The
`pair_coeff` line uses a placeholder element label:

```
pair_style  polymlp
pair_coeff * * mlp.lammps dummy
```

`Ti-lammps` uses route (a) with `lammps_structure_Ti`; `Si-lammps` uses route
(b) with `phonopy_unitcell.yaml`. After generating supercells, run `lmp -in
in.polymlp` and create `FORCE_SETS` as above.

### Si-lammps-ace (ACE)

This uses the LAMMPS `pair_style pace` (atomic cluster expansion). Any LAMMPS
build with the `ML-PACE` package can run it, including the conda-forge `lammps`
package:

```
pair_style pace
pair_coeff * * Si_npj_CompMat2021.ace Si
```

The potential file `Si_npj_CompMat2021.ace` is from the dataset accompanying Y.
Lysogorskiy *et al.*, "Performant implementation of the atomic cluster expansion
(PACE) and application to copper and silicon", npj Comput. Mater. **7**, 97
(2021), distributed on Zenodo (<https://doi.org/10.5281/zenodo.4734036>) under
the **CC-BY-4.0** license. Download it into the working directory; it is not
redistributed with phonopy.

The unit cell uses the experimental room-temperature lattice constant of
silicon, a = 5.431 A. Because the ACE equilibrium constant differs slightly, a
small residual stress may appear as near-zero or slightly imaginary acoustic
frequencies around Gamma; relax the cell with the potential to remove it (see
the appendix below).

(lammps_structure_optimization)=

## Appendix: structure optimization using LAMMPS

Relax the crystal structure with the potential before the phonon calculation so
that the residual forces, and the residual stress on the lattice, vanish. The
following relaxes both the cell and the internal coordinates:

```
units metal

read_data unitcell

pair_style  polymlp
pair_coeff * * mlp.lammps dummy

variable etol equal 0.0
variable ftol equal 1e-8
variable maxiter equal 1000
variable maxeval equal 100000

fix relax all box/relax iso 0.0 vmax 0.001
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

write_data dump.unitcell
```

Drop the `fix box/relax` line to relax only the internal coordinates. More
instruction is found at
<https://gist.github.com/lan496/e9dff8449cd7489f6722b276282e66a0>.
