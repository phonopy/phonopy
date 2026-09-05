---
orphan: true
---

# Anisotropic QHA

This page computes the anisotropic thermal expansion of a crystal in the
quasi-harmonic approximation (QHA): one expansion coefficient per crystal axis,
rather than one for the volume. The ordinary QHA minimizes the free energy
along a single path in volume. This one samples the lattice parameters on a
grid and minimizes over them directly, so the axes are free to expand by
different amounts.

**Steps 0 to 4 are the whole calculation.** The phonons come from displaced
supercells computed with the calculator, and no machine-learning potential
(MLP) is involved.

**Step 5 is a variant**, for a crystal whose anharmonicity the harmonic
approximation cannot carry. An MLP is trained at each grid point and gives
force constants that change with temperature. The free energies from those
force constants enter the analysis directly, in place of the force sets. Most
calculations do not need this step.

{math}`a`, {math}`b` and {math}`c` on this page are the lattice parameters of
the **standardized conventional unit cell**, never of the primitive cell.
Step 0 explains why the recipe is built on that cell.

The free lattice degrees of freedom (DOF) are detected from the symmetry: one
for cubic ({math}`a`), two for hexagonal, tetragonal and rhombohedral
({math}`a, c`), and three for orthorhombic ({math}`a, b, c`). Cell angles are
held fixed, so monoclinic and triclinic crystals are out of scope. This page
uses {math}`(a, c)` throughout as a concrete example; substitute the free DOF
of your system. The lattice parameters and axial thermal expansions are
produced for any of the supported systems. Contour maps of the free energy
{math}`F(a, c)` at a fixed temperature -- the surface whose minimum gives those
lattice parameters -- are drawn only when there are exactly two free DOF.

```{warning}
**This workflow is experimental.** Everything on this page works and is
tested, but the interfaces are not settled. The command-line options, the
`aniso_qha_dataset.hdf5` layout, and the `phonopy.qha.anisotropic` and
`phonopy.qha.anisotropic_dataset` APIs may change in a backward-incompatible
way between releases, without a deprecation period; options have already been
added and removed as the recipe was used on real systems.

So rebuild the dataset from the calculator outputs rather than relying on an
old file being readable, keep beside every result the commands that produced
it, and pin the phonopy version whenever a campaign has to stay reproducible
across releases. The page is not yet part of the documentation navigation for
the same
reason.
```

Four commands do the work, one per step:

- `phonopy-init` prepares the equilibrium reference of step 0, and collects
  the forces of the phonon grid in step 2.
- `phonopy-strain-cells` samples the strained cells of step 1.
- `phonopy-anisotropic-qha-dataset` gathers the calculator outputs into the
  intermediate dataset of step 3.
- `phonopy-anisotropic-qha` runs the analysis of step 4.

Step 4 runs the analysis in one command. The API script printed under it does
the same thing with more control.

Prerequisites: `h5py`, `symfc`, a VASP setup, and `pypolymlp` for the variant
of step 5.

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

```{note}
**VASP is the only calculator this workflow runs with.** The commands and the
helper scripts assume VASP inputs and outputs (`POSCAR`, `vasprun.xml`,
`vaspout.h5`).

The binding constraint is the static grid. The internal energy
{math}`U(a_i, c_i)` and the electronic states are read from VASP outputs, and
phonopy has no interface yet for reading the static single-point energy of the
other calculators. So `phonopy-anisotropic-qha-dataset` stops early on a
reference naming one of them, rather than building a dataset with
{math}`U(a_i, c_i)` missing. Nothing else stands in the way: the readers for
the other calculators are simply not written yet.
```

## The free energy

The free energy minimized at each temperature is

```{math}
F(a, c; T) = U(a, c) + F_\mathrm{ph}(a, c; T) + F_\mathrm{el}(a, c; T),
```

where the electronic term {math}`F_\mathrm{el}` is optional.
{math}`U(a, c)` is the static internal energy, computed by the calculator on
the static grid of step 1. {math}`F_\mathrm{ph}(a, c; T)` is the phonon free
energy of the harmonic crystal,

```{math}
F_\mathrm{ph}(a, c; T) = \sum_{\mathbf{q}\nu} \left[
\frac{\hbar \omega_{\mathbf{q}\nu}}{2}
+ k_\mathrm{B} T \ln \left(
1 - e^{-\hbar \omega_{\mathbf{q}\nu} / k_\mathrm{B} T} \right) \right],
```

where the frequencies {math}`\omega_{\mathbf{q}\nu}` come from that grid
point's force constants of step 2, and the sum runs over the modes at the
q points of the sampling mesh. Steps 0 to 4 use this expression as written.
Step 5 replaces it with the SSCHA free energy.

{math}`a` and {math}`c` are continuous above: {math}`F` is written for any
lattice. The calculator returns values only at the sampled cells
{math}`(a_i, c_i)` of step 1. Step 4 therefore fits a surface through those
values and minimizes the surface, and that is what makes {math}`a(T)` and
{math}`c(T)` continuous functions of temperature.

## Overview

The boxes are jobs run by phonopy tools or the API, the hexagons are calculator
runs, and the rounded nodes are input and intermediate data. This is the
recipe of steps 0 to 4, with the phonons from the calculator:

```{mermaid}
flowchart TD
    EQ(["Equilibrium cell<br/>(phonopy_disp.yaml)"])
    EQ --> SC["phonopy-strain-cells<br/>(a, c grid)"]
    SC --> RELAX{{"calculator relax + static"}}
    RELAX --> SGRID(["static-grid/grid-NNN<br/>U, F_el"])

    SGRID --> PD["generate displacements<br/>per relaxed cell"]
    PD --> CALCF{{"calculator forces"}}
    CALCF --> PGRID(["phonon-grid/grid-NNN<br/>disp-*"])

    SGRID --> BUILD["phonopy-anisotropic-qha-dataset<br/>--static ... --phonon ..."]
    PGRID --> BUILD
    BUILD --> DS(["aniso_qha_dataset.hdf5<br/>cells, U, F_el,<br/>displacements, forces"])

    DS --> ANA["phonopy-anisotropic-qha"]
    ANA --> RES(["a(T), c(T),<br/>alpha_a, alpha_c,<br/>F(a,c) maps"])
```

## 0. The equilibrium reference (`phonopy_disp.yaml`)

Step 0 is done once, before the grid exists; steps 1 to 4 are the calculation
itself. Start from one relaxed equilibrium cell and turn it into the reference
`phonopy_disp.yaml` with `phonopy-init`. Every later step reads this file:

```bash
% phonopy-init -c REFERENCE_UNITCELL -d --dim 4 4 4
```

`REFERENCE_UNITCELL` must be the standardized conventional cell, whose lattice
vectors are the crystal axes a, b and c in that row order.

The conventional cell is what makes the grid small. The number of grid points
is set by the number of free lattice DOF, and the symmetry of the conventional
cell is what reduces that number: two for hexagonal instead of three, one for
cubic instead of three. The primitive cell of a centred lattice hides that
symmetry in its rows, and the same crystal would then be sampled over more DOF
than it has.

The free lattice DOF are taken per row, so a primitive cell of a centred
lattice cannot be used at all: its rows are centring vectors rather than
crystal axes. For body-centred tetragonal, for example, all three primitive
rows have the same length, and scaling them would only change the volume, never
{math}`c/a`. A rhombohedral cell must likewise be given in the hexagonal
setting. `phonopy-strain-cells` rejects such a cell; if in doubt, take the
`BPOSCAR` written by {ref}`phonopy-init --symmetry <symmetry_option>`. That
file is the conventional cell. A conventional cell that is merely rotated in
Cartesian space is fine.

Everything the recipe writes follows from this choice. `phonopy-strain-cells`
strains the unit cell of `phonopy_disp.yaml` and writes unit cells; the
displaced supercells are built on them; and the lattice parameters reported at
the end are theirs. The primitive cell enters only as the normalization of the
energies, which "Run the anisotropic QHA" covers.

`--dim` fixes the supercell matrix, which `phonopy-init` records together with
the unit cell, the primitive matrix and the calculator. `phonopy-strain-cells`
reads the equilibrium cell and calculator from it;
`phonopy-anisotropic-qha-dataset` reads the
calculator, and takes the free lattice DOF from the symmetry of that cell
(which lengths are independent --
{math}`a, c` with {math}`b = a` for hexagonal). Keep `--dim` consistent with
the phonon-grid
supercell in step 2.

## 1. Build the static grid (internal energy U)

Sample strained unit cells over the free lattice DOF, then relax and run a
static single point for each with the calculator. The static grid supplies
{math}`U(a_i, c_i)` and, optionally, the electronic states for
{math}`F_\mathrm{el}`.

```bash
# Inspect the free lattice DOF first (no ranges -> DOF report):
% phonopy-strain-cells phonopy_disp.yaml

# --grid N is the number of grid points per free axis (5 -> 5 x 5 = 25 cells);
# one N per free DOF gives a rectangular grid, e.g. --grid 5 6 -> 30 cells.
% phonopy-strain-cells phonopy_disp.yaml --a 3.168 3.232 --c 5.148 5.252 --grid 5
Wrote 25 strained unit cell(s) as unitcell-001 .. unitcell-025 in vasp format.
Grid sampling: 5 x 5 over (a, c).
  Main diagonal (5 cells), the --compare-eos volume path:
    a  c   c/a
    3.1680  5.1480   1.6250
    3.1840  5.1740   1.6250
    3.2000  5.2000   1.6250
    3.2160  5.2260   1.6250
    3.2320  5.2520   1.6250
Provenance written to strain_cells.yaml
```

The main diagonal of the grid, printed above with each cell's {math}`c/a`, is
the volume path that `phonopy-anisotropic-qha --compare-eos` fits. Equal
fractional ranges and equal counts keep {math}`c/a` constant along it, which is
the cleanest input to the cross-check. Unequal ranges or counts still give a
path, but {math}`c/a` then varies along it, and the fit mixes a change of shape
into what it reads as a change of volume.

The cells are written as strained **unit cells** `unitcell-NNN`. When
`primitive_matrix` is not the identity, the primitive cell is a different cell,
and the same run writes it as `primcell-NNN` too:

```text
Wrote 25 strained unit cell(s) as unitcell-001 .. unitcell-025 in vasp format.
Also wrote the primitive cell of each as primcell-001 .. primcell-025
(primitive 2 atoms, unit cell 8 atoms). The static single point may be run on
either cell.
```

The test is on the matrix, not on the atom counts, so a primitive cell taken in
an unusual setting is written as well even though it holds the same atoms. The
hexagonal example above is the other case: `auto` resolves to the identity
there, the two cells are one cell, and only the unit cells are written.

The two files are one grid point. Run the static single point on whichever
suits the calculation and pass that one to the builder; "Run the anisotropic
QHA" says how {math}`U` is put on one normalization afterwards.

The primitive cells take their atoms from `phonopy_disp.yaml` rather than from
a fresh symmetry search. The strain scales the crystal axes and leaves the
fractional coordinates alone, so only the lattice is rebuilt. The atom order is
then the same at every grid point, and does not move with the symmetry
tolerance.

Each run also writes `strain_cells.yaml`, recording the ranges, the grid shape
and the free-DOF lengths of every cell, with the `primcell-NNN` name beside
each `unitcell-NNN` when both were written. Nothing in the run is random, so
the same command reproduces the same cells.

For each grid point, on whichever of the two cells you chose:

1. Relax the internal coordinates if the structure has free internal parameters
   (e.g. the wurtzite `u`). A crystal with no internal DOF (e.g. HCP) skips
   this step -- the strained cell is already the relaxed cell.
2. Run a static single point. Three settings matter.

   - `ISIF >= 2`, if you also want the stress.
   - Write `vaspout.h5`, if you want the electronic states for
     {math}`F_\mathrm{el}`. A run that writes only `vasprun.xml` carries no
     eigenvalues, and the dataset is then built without {math}`F_\mathrm{el}`.
   - Sample the Brillouin zone on a Gamma-centred regular mesh. The k points
     and the mesh are then stored with the states, and {math}`F_\mathrm{el}`
     is integrated by the linear tetrahedron method rather than summed over k
     points.

   The mesh a static single point already needs is dense enough for the
   tetrahedron method, so no extra one has to be chosen. Where a denser mesh is
   wanted for the electronic states alone, add a KPOINTS_OPT block: the states
   are read from that block in preference to the SCF mesh.
3. Place the output in `static-grid/grid-NNN/` (one directory per grid point,
   containing `vaspout.h5` or `vasprun.xml`). Any layout works -- the builder
   is given the paths explicitly -- but a name that sorts in the sampling order
   keeps the two grids easy to pass together, and no index file is needed.
   Sorting in that order also matters for `--compare-eos`: the builder
   recognizes a tensor grid only when the cells reach it in the order they
   were generated, and records its shape for the main-diagonal volume path.

Script 1 below lays out those directories. Edit the paths at the top of it and
run it. It writes the POSCARs only; the rest of the VASP input -- `INCAR`,
`POTCAR`, `KPOINTS` -- has to be distributed separately.

```{code-block} python
:caption: Script 1 -- the static-grid input POSCARs, from the strained cells above

import glob
from pathlib import Path
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure

# Strained cells from phonopy-strain-cells. Put "primcell-*" here instead to
# run the static grid on the primitive cells.
CELLS = "unitcell-*"
STATIC_GRID = "static-grid"

for path in sorted(glob.glob(CELLS)):
    idx = int(Path(path).stem.split("-")[-1])
    cell, _ = read_crystal_structure(path, interface_mode="vasp")
    static_dir = Path(STATIC_GRID) / f"grid-{idx:03d}"
    static_dir.mkdir(parents=True, exist_ok=True)
    write_crystal_structure(static_dir / "POSCAR", cell, interface_mode="vasp")
    print(f"grid-{idx:03d}: static POSCAR")
```

Then relax (if the crystal has internal DOF) and run the static single point in
each `static-grid/grid-NNN/`.

## 2. Compute the phonons (phonon grid)

For each relaxed static-grid cell, generate displaced supercells and compute
their forces with the calculator.

Place the results as `phonon-grid/grid-NNN/` each containing the
`phonopy_disp.yaml` and the per-displacement subdirectories `disp-001/`,
`disp-002/`, ... (each with `vaspout.h5` or `vasprun.xml`). The names need not
match those under `static-grid/`, since the two grids are paired by the order
they are passed in, but matching names make that order easy to get right.

Script 2 below reads each `static-grid/grid-NNN/CONTCAR`, the relaxed
structure, which equals the input POSCAR when there is no internal DOF. So run
it only after the static grid is done. Edit the paths at the top of it, and
distribute the rest of the VASP input separately, as in step 1.

`SUPERCELL_MATRIX` and `DISTANCE` must both be the same at every grid point,
for one reason. The analysis fits a surface through the {math}`F_i(T)` of the
grid and differentiates it. Either setting biases {math}`F_\mathrm{ph}` a
little -- the supercell size through how far the force constants reach, the
displacement distance through the anharmonic error of the finite difference. A
bias that is equal everywhere shifts the surface without tilting it, and the
axial expansions do not see it. A bias that changes from one grid point to the
next tilts the surface, and the axial expansions follow the tilt.

So keep `SUPERCELL_MATRIX` at the `--dim` of step 0, which is also the value
stored when a dataset is built from the static grid alone. `DISTANCE` is the
displacement distance in Angstrom, 0.03 here against phonopy's own default of
0.01. Nothing checks either of them: the builder reads the supercell matrix
from each phonon grid point separately, and never compares the grid points with
one another or with step 0.

```{code-block} python
:caption: Script 2 -- the phonon grid, from the relaxed static-grid cells

from pathlib import Path
import phonopy
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure

STATIC_GRID = "static-grid"  # relaxed cells at static-grid/grid-NNN/CONTCAR
PHONON_GRID = "phonon-grid"
N_GRID = 25  # cells written by phonopy-strain-cells in step 1
SUPERCELL_MATRIX = [4, 4, 4]
DISTANCE = 0.03

for idx in range(1, N_GRID + 1):
    contcar = Path(STATIC_GRID) / f"grid-{idx:03d}" / "CONTCAR"
    cell, _ = read_crystal_structure(str(contcar), interface_mode="vasp")
    ph = phonopy.Phonopy(cell, supercell_matrix=SUPERCELL_MATRIX, calculator="vasp")
    ph.generate_displacements(distance=DISTANCE)
    phonon_dir = Path(PHONON_GRID) / f"grid-{idx:03d}"
    phonon_dir.mkdir(parents=True, exist_ok=True)
    ph.save(phonon_dir / "phonopy_disp.yaml")

    for k, sc in enumerate(ph.supercells_with_displacements, 1):
        disp_dir = phonon_dir / f"disp-{k:03d}"
        disp_dir.mkdir(parents=True, exist_ok=True)
        write_crystal_structure(disp_dir / "POSCAR", sc, interface_mode="vasp")
    print(f"grid-{idx:03d}: {len(ph.supercells_with_displacements)} disp")
```

Then build the intermediate dataset, which is step 3.

(anisotropic-qha-build)=
## 3. Build the intermediate dataset

The analysis of step 4 reads `aniso_qha_dataset.hdf5`, and so does step 5. The
grid points are given as two path lists, `--static` and `--phonon`, and paired
**by position** after shell expansion:

```{note}
The builder reads VASP outputs only, and stops early on a reference naming
another calculator. The opening note of this page says why.
```

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static static-grid/grid-{001..025}/ \
    --phonon phonon-grid/grid-{001..025}/ \
    -o aniso_qha_dataset.hdf5
Reading pre-computed forces for 25 grid point(s)
  grid 1 U=... eV n_disp=...
  ...
  grid 25 U=... eV n_disp=...
Grid shape [5, 5] recorded for the main-diagonal path.
Wrote 25 grid point(s) to aniso_qha_dataset.hdf5
```

The entries may be files rather than directories. Use that form when the
calculations were laid out by something other than Scripts 1 and 2:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static runs/*/static/vaspout.h5 \
    --phonon runs/*/phonons/phonopy_params.yaml \
    -o aniso_qha_dataset.hdf5
```

Any names work, and one list can mix directories and files. The `--static` and
`--phonon` lists must have the same number of entries.

(anisotropic-qha-builder-reads)=
### What the builder reads

For each grid point the builder reads:

- the static single point, giving the internal energy {math}`U(a_i, c_i)`. It
  may have been run on the grid point's unit cell or on its primitive cell, and
  the builder accepts either; anything else is reported as a mis-pairing. With
  `--phonon` given, the grid-point cell is the phonon entry's cell rather than
  this one, so a relaxation carried into the phonon grid is honored. Without
  `--phonon` this cell becomes the grid point itself, and it then has to be the
  conventional unit cell, since the free lattice DOF are read from its rows.
  The electronic states for
  {math}`F_\mathrm{el}` are read automatically from the same `vaspout.h5` when
  it carries the eigenvalues (a static point written with only `vasprun.xml` is
  built without {math}`F_\mathrm{el}`; pass `--no-electronic` to skip them
  deliberately). A directory entry is resolved to the VASP output it holds, and
  `vaspout.h5` is used in preference to `vasprun.xml`.
- the phonon grid point, in one of two forms. A **directory** holding
  `phonopy_disp.yaml` and the per-displacement `disp-*` subdirectories: the
  builder reads each `disp-*` calculator output itself, so no `FORCE_SETS` or
  `phonopy_params.yaml` is needed. Or a **phonopy.yaml-like file** carrying
  forces that {ref}`phonopy-init -f <f_force_sets_option>` has already
  collected. This second form is the simpler route when the calculations were
  not laid out by Script 2:

  ```bash
  % phonopy-init --sp -f disp-*/vasprun.xml   # -> phonopy_params.yaml
  % phonopy-init -f disp-*/vasprun.xml        # -> FORCE_SETS, beside phonopy_disp.yaml
  ```

  Pass the resulting `phonopy_params.yaml`. A `phonopy_disp.yaml` with its
  `FORCE_SETS` beside it works too ({ref}`--sp <save_params_option>` merges the
  two into one file). A file with no forces and no neighboring `FORCE_SETS` is
  rejected rather than silently producing an empty grid point. Either form
  supplies the per-grid-point supercell / primitive matrices.

The positional `phonopy_disp.yaml` is the equilibrium reference; it supplies
the free lattice DOF metadata and the calculator. The grid-point index recorded
in the dataset is the position in the list, a label only, since the analysis
reads the lattice parameters from each stored cell.

(anisotropic-qha-ordering)=
### How the ordering is checked

The builder pairs `--static` and `--phonon` by position after shell expansion,
and takes the `disp-*` subdirectories of one grid point in sorted order. It
checks both against the structures in the calculator outputs.

**Across grid points**, the lattice of each static single point must match the
cell of the phonon grid point it is paired with. A mismatch stops the command
and names both paths. Two mistakes are caught this way.

- A grid point missing from each list. The list lengths still match, so
  nothing else notices, and the {math}`U` of one lattice would be combined
  with the forces of another.
- A static single point run on a supercell. Its {math}`U` would then be on the
  wrong normalization.

**Within one grid point**, sorted order is only a guess at the displacement
order: `disp-1, disp-10, disp-2` sorts differently from how it counts. Each
calculator output carries the structure it was run on, so the builder compares
it against the displaced supercell of its position and names the directory and
the displacement on a mismatch.

Zero-padded names (`grid-001`, `disp-001`) get both orderings right in the
first place, whether the shell expands a glob lexicographically or a brace
range in order. Scripts 1 and 2 write `grid-{idx:03d}` and `disp-{k:03d}`,
and `phonopy -d` does the same. Padding is a convenience
rather than a requirement: a wrong order is reported, not turned into force
constants from mismatched forces.

The examples write the range out as `grid-{001..025}` rather than `grid-*`.
The count is then visible in the command, and a missing grid point stops the
builder on a path that does not exist instead of quietly passing a shorter
list. `grid-*` also works, and is the shorter thing to type once the grid is
known to be complete.

(anisotropic-qha-file-contents)=
### What the file holds

`aniso_qha_dataset.hdf5` is self-contained. Per grid point it stores the
relaxed cell, the supercell and primitive matrices, the raw displacements and
forces, the static internal energy {math}`U`, and optionally the electronic
states. The displacements and forces are kept in phonopy's native
displacement-force dataset form -- type-1 (one displaced atom per supercell)
or type-2 (dense/random) -- and tagged with which one, so the reader picks the
force-constant solver from the type instead of inferring it. Storing them raw
rather than as force constants keeps the file independent of the
force-constant method, and makes it an archive that outlives the calculator
scratch.

The electronic states always carry the eigenvalues, the k-point weights and the
electron count. They carry the k points and the mesh as well, but only when the
static calculation sampled a Gamma-centred regular mesh. Those last two are
what the linear tetrahedron method needs, so a file without them is integrated
only by the k-point sum. The two readers then part company: the
`run_anisotropic_qha` API falls back on the k-point sum, while the
`phonopy-anisotropic-qha` command stops instead. Step 4 gives the reason.

The layout, as written by `phonopy-anisotropic-qha-dataset` and read back by
`phonopy.qha.anisotropic_dataset.read_aniso_qha_dataset`:

```text
/                    attrs: creator, phonopy_version, calculator, length_unit,
                            crystal_system, free_dof, tie_description,
                            n_grid_points, grid_shape
/grid/NNN            attrs: index, internal_energy, displacement_type
    lattice                          (3, 3)       the relaxed grid-point cell
    lattice_lengths                  (3,)         a, b, c
    scaled_positions                 (natom, 3)
    numbers                          (natom,)
    masses                           (natom,)
    magnetic_moments                 (natom,)     only when the cell has them
    supercell_matrix                 (3, 3)
    primitive_matrix                 (3, 3)
    displaced_atoms                  (ndisp,)     type-1 only
    displacements                    (ndisp, 3)   type-1
                                     (ndisp, natom_super, 3)   type-2
    forces                           (ndisp, natom_super, 3)
    electronic_states/eigenvalues    (nspin, nkpt, nband)   optional
    electronic_states/weights        (nkpt,)
    electronic_states/n_electrons    scalar
    electronic_states/fermi_energy   scalar
    electronic_states/spin_degeneracy  scalar     only when set
    electronic_states/kpoints        (nkpt, 3)    only with a regular mesh
    electronic_states/mesh           (3,)         only with a regular mesh
```

`grid_shape` is present only when the cells reached the builder as a tensor
grid; `--compare-eos` needs it. `NNN` is the position in the `--static`
list, and the analysis reads the lattice parameters from `lattice` rather than
from that number.

## 4. Run the anisotropic QHA

Run the analysis directly on the intermediate dataset:

```bash
% phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
    --contour-temp 0 500 1000 --compare-eos
```

`--tmax` and `--dt` are in K and set the temperature grid, here 0 to 1000 K in
steps of 10 K. The thermal expansions are central differences on that grid, so
the top temperature is consumed and the results stop one step below `--tmax`:
990 K for the command above.

`--mesh` sets the phonon sampling mesh and defaults to 200, denser than
`run_qha`'s 100. The axial thermal expansions need the denser mesh, while the
volumetric expansion is already converged at 100.

The command rebuilds one Phonopy per grid point, with force constants from the
stored displacements and forces, runs `run_anisotropic_qha`, and writes
`lattice_parameters-temperature.dat`, `axial_thermal_expansion.dat`,
`volume-temperature.dat` and `anisotropic_qha.png`.

With exactly two free lattice DOF it also writes the `F(a, c)` contour maps.
`--decompose-contours` adds the {math}`U` / {math}`F_\mathrm{ph}` /
{math}`F_\mathrm{el}` / total panels.

`--compare-eos` adds a volume-path cross-check along the main diagonal of
the grid. The diagonal comes from the grid shape the builder recorded, so the
cross-check needs a `--grid` run in step 1 whose cells were gathered in order.
Without a recorded shape the command skips the cross-check and lists the cells
by volume with their {math}`c/a`, so that `--eos-index` can name a path by
hand. Cells that share a {math}`c/a` differ in volume alone, and differing in
volume alone is what a volume-path equation-of-state fit assumes. Any five or
more such cells are printed as a ready-made `--eos-index` line.

The electronic free energy {math}`F_\mathrm{el}` is added whenever the dataset
carries the electronic states, and `--no-electronic` leaves it out. The
integration is the linear tetrahedron method. That method needs the k-point
grid the states were computed on, and without that grid the command stops. It
does not fall back on the k-point sum. The sum integrates a delta-function
density of states, and so needs far more irreducible k points than a mesh
chosen for the
total energy has. The `run_anisotropic_qha` API is looser and does fall back on
the sum, one grid point at a time, so a script calling the API has to check the
mesh for itself. The run prints the energy window and the grid spacing it
integrated over.
The result is written to `fel.hdf5`, which `--electronic-free-energies` takes
back on a later run, since the integration is the same every time.

Converge the mesh on the thermal expansion rather than on
{math}`F_\mathrm{el}`. Two meshes can agree closely on {math}`F_\mathrm{el}`
and still differ severalfold in {math}`\alpha_c`, because the expansion is a
derivative of the free-energy surface and the error varies across the lattice
grid.

The same analysis, driven from the API:

```{code-block} python
:caption: Script 3 -- the analysis through `run_anisotropic_qha`

import numpy as np
from phonopy import run_anisotropic_qha
from phonopy.qha import anisotropic_output, anisotropic_plot
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")

phonopys = []
internal_energies = []
electronic_structures = []
for point in dataset.grid_points:
    # to_phonopy() rebuilds the Phonopy and force constants from the stored
    # dataset, picking the site-symmetry or symfc solver by dataset type.
    phonopys.append(point.to_phonopy())
    internal_energies.append(point.internal_energy)
    electronic_structures.append(point.electronic_states)

has_electronic = all(e is not None for e in electronic_structures)
temperatures = np.arange(0, 1001, 10.0)  # one extra point for finite diff
result = run_anisotropic_qha(
    phonopys,
    temperatures,
    internal_energies=internal_energies,
    electronic_structures=electronic_structures if has_electronic else None,
    # mesh defaults to 200 here, denser than run_qha's 100: the axial split
    # needs it, while the volumetric expansion is converged at 100.
)

anisotropic_output.write_lattice_parameters_temperature(result)
anisotropic_output.write_axial_thermal_expansion(result)
anisotropic_output.write_volume_temperature(result)

fig = anisotropic_plot.plot_anisotropic_qha(result)
fig.savefig("anisotropic_qha.png")

# The F(a, c) diagnostics, one file per temperature. plot_component_contours
# splits the surface into its U / F_ph / F_el / total parts, which is what
# shows where the valley comes from; both need exactly two free lattice DOF
# and return the names they wrote.
contour_temperatures = [0.0, 300.0, 600.0, 1000.0]
anisotropic_plot.plot_F_contours(result, contour_temperatures)
anisotropic_plot.plot_component_contours(
    result,
    internal_energies,
    electronic_structures if has_electronic else None,
    contour_temperatures,
)
```

Both forms run the same analysis. `run_anisotropic_qha` detects the free
lattice DOF from the input cells, and then, at each temperature:

1. Add the terms at every sampled cell,
   {math}`F_i(T) = U(a_i, c_i) + F_\mathrm{ph}(a_i, c_i; T)`, with
   {math}`F_\mathrm{el}(a_i, c_i; T)` when it is included and a
   {math}`pV_i` term when a pressure is given.
2. Fit a polynomial of total degree {math}`n` in the free lattice
   parameters to those {math}`F_i(T)`, by least squares. `--surface-degree`
   sets {math}`n`, 3 by default. For {math}`d` free DOF the polynomial has
   {math}`\binom{n + d}{n}` terms -- 10 for two free DOF at degree 3 -- and
   the grid needs at least that many cells, or the fit is rank deficient
   and says so.
3. Minimize that polynomial, from the centroid of the sampled cells and
   using its analytic gradient. The minimizer is {math}`a(T)`,
   {math}`c(T)`. A minimum outside the sampled range is reported as an
   extrapolation.

The axial thermal expansions then follow by central differences over the
temperature grid,

```{math}
\alpha_a = \frac{1}{a}\frac{da}{dT}, \qquad
\alpha_c = \frac{1}{c}\frac{dc}{dT},
```

The fit is over the lattice only: each temperature is minimized on its own,
and the raw {math}`a(T)`, {math}`c(T)` are differenced. The calculator's
harmonic free energies carry no sampling scatter, so differencing them
directly is the right thing to do here. A free energy that does carry scatter
needs the lattice parameters smoothed along temperature first, which is
`--smooth-lattice`; {ref}`step 5 <anisotropic-qha-temperature-dependent>`
covers it, since that is where such free energies come from.

The **energies** are normalized **per primitive cell**, in eV: the internal
energies, the phonon free energies and the electronic free energies. The
volumes the analysis works in are that cell's too, which is what a pressure
run builds its {math}`pV` term on. Which cell it is comes from
`primitive_matrix` in `phonopy_disp.yaml`, which `phonopy-init` sets to `auto`
unless `--pa` says otherwise.

Nothing else on the page follows the primitive cell. The grid, the lattice
parameters and the axial thermal expansions are all the conventional cell's,
as step 0 says.

The static single point need not be run on that cell. Either the conventional
unit cell or the primitive cell will do, whichever suits the calculation -- a
long rhombohedral cell is easier to handle in its hexagonal setting, while a
large conventional cell may be cheaper to run as its primitive cell. The
builder settles the normalization: it reads the number of atoms in the cell
{math}`U` was computed on, and multiplies {math}`U` by the primitive cell's
share of it. A static point already on the primitive cell is left alone. One
run on a cell holding four of them is divided by four, and the builder says so:

```text
The static single point was run on a cell holding 4 primitive cells, so U is
stored as 1/4 of the calculator energy, matching the phonon free energy.
```

Which cell it was is read off the atom counts rather than assumed, because
strain changes the volumes but not them.

Two restrictions come with this freedom. The static cell has to be the
conventional unit cell when `--phonon` is omitted, since that cell then becomes
the grid point and the free lattice DOF are read from its rows; the builder
stops and says so. And with `--phonon` given, the static cell must be a cell of
the grid point it is paired with, either of the two. A cell from another grid
point matches neither, so a mis-pairing is still caught.

{math}`F_\mathrm{el}` is normalized the same way, where the analysis
integrates it from the stored electronic states. The states record the cell
they were computed on, so states already on the primitive cell are not scaled a
second time. Arrays handed to `run_anisotropic_qha` directly are the exception:
`internal_energies`, `electronic_free_energies` and `phonon_free_energies` are
taken as they are, and have to be per primitive cell already.

### Supplying {math}`F_\mathrm{el}` ready-made

The tetrahedron integration costs of order a minute per grid point on a dense
mesh, and the analysis works through the grid points one after another.
Computing {math}`F_\mathrm{el}` outside lets one result be reused across runs,
and lets the grid points be spread over processes or jobs. Pass it as
`electronic_free_energies`, the counterpart of `phonon_free_energies` for the
electronic term.

```{code-block} python
:caption: Script 4 -- {math}`F_\mathrm{el}` computed apart from the analysis

import numpy as np

from phonopy import run_anisotropic_qha
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.electron import compute_free_energy_by_tetrahedron

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")
temperatures = np.arange(0, 1001, 10.0)  # one extra point for finite diff

fe_el = np.column_stack(
    [
        # F_el(T) - F_el(0) in eV per primitive cell, one grid point at a time.
        compute_free_energy_by_tetrahedron(point.electronic_states, temperatures)[0]
        for point in dataset.grid_points
    ]
)

result = run_anisotropic_qha(
    [point.to_phonopy() for point in dataset.grid_points],
    temperatures,
    internal_energies=[point.internal_energy for point in dataset.grid_points],
    electronic_free_energies=fe_el,
)
```

The grid points are independent, so this loop splits across processes or jobs
unchanged; each one writes its own column of `fe_el`. The integration itself
is already threaded through BLAS, so cap the threads per process when several
run on one node.

The values are anchored at T = 0 and normalized per primitive cell,
consistently with `internal_energies`. `electronic_structures` and
`electronic_free_energies` are two ways of giving the same term, so pass one
or the other.

`phonopy-anisotropic-qha` takes the same thing from a file. Write it with the
temperatures it was computed on:

```python
from phonopy.qha.electron import write_electronic_free_energies_hdf5

write_electronic_free_energies_hdf5(temperatures, fe_el, "fel.hdf5")
```

```bash
% phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
    --electronic-free-energies fel.hdf5
```

The file carries the temperatures it was computed on. The command compares
them with the grid `--tmax` and `--dt` ask for and stops if they differ,
instead of pairing the rows as they come. `--electronic-free-energies`
replaces `--electronic`; passing both stops the command. It also skips
`--compare-eos`, since the volume-path driver takes the electronic term as
states and the two paths would then carry different physics.

(anisotropic-qha-temperature-dependent)=
## 5. Variant: temperature-dependent force constants from per-grid-point MLPs

The displacements of step 2 are small, and the forces they give are almost
harmonic. Anharmonic effects appear only in supercells displaced further than
that. Here such supercells are made by moving every atom at once, by amounts
drawn at random from the thermal distribution of the harmonic crystal at a
chosen temperature.

The force constants of step 2 define that harmonic crystal. It separates into
independent normal modes. Each mode is a harmonic oscillator, so its own normal
coordinate is Gaussian about zero, and the width of that Gaussian is set by the
mode's frequency and by the temperature. One snapshot draws every mode from its
own Gaussian, and the displacement of each atom follows from the sum over the
modes. "The thermal distribution" below writes the width down.

The calculator then computes forces on supercells displaced that way at that
temperature. These displacements are usually not small, so the forces carry
more of the anharmonic contribution than those of step 2 do, and are enough to
train an MLP on.

The displacements, the forces and the supercell energies together are used to
build the MLP at every grid point. Each MLP is fitted to that grid point's own
training set.

The MLP is then used, within the temperature range its training set covers. An
MLP is an intermediate representation, convenient because it interpolates
between the temperatures it was trained at. The self-consistent harmonic
approximation ({ref}`SSCHA <mlp-sscha>`) runs with it at every grid point and
every temperature, and returns force constants that change with temperature.
The anharmonic free energies follow from those. The analysis takes them and
minimizes {math}`F(a, c; T)` over the lattice at each
temperature. What comes out is {math}`a(T)`, {math}`c(T)` and the axial
thermal expansions.

### Where this step fits

Steps 0 to 4 displace one atom at a time and need a few supercells per grid
point. They give the quasi-harmonic answer.

Step 5 needs a training set at every grid point instead. In the script below
that is 50 structures at each of four temperatures, so 200 calculator runs per
grid point. What it gives back is force constants that change with
temperature. Use it when anharmonicity is large enough to matter for the
property being computed.

Steps 0 to 3 are as in the Overview, up to and including the dataset. What
differs comes after it: the MLPs and the free energies,

```{mermaid}
flowchart TD
    DS(["aniso_qha_dataset.hdf5<br/>cells, U, F_el,<br/>harmonic force constants"])
    DS --> DISP["thermal displacements<br/>one shared draw of<br/>standard normals"]
    DISP --> CALCT{{"calculator forces"}}
    CALCT --> DEV["train one MLP<br/>per grid point"]
    DEV --> MLP(["polymlp.yaml<br/>per grid point"])
    MLP --> SSCHA["SSCHA<br/>per grid point and temperature"]
    SSCHA --> FE(["F_ph(T) per<br/>grid point"])
```

and the analysis they meet in:

```{mermaid}
flowchart TD
    DS2(["aniso_qha_dataset.hdf5"])
    FE2(["F_ph(T) per<br/>grid point"])
    DS2 --> AQ["run_anisotropic_qha"]
    FE2 -->|"phonon_free_energies"| AQ
    AQ --> RES(["a(T), c(T),<br/>alpha_a, alpha_c"])
```

### One MLP per grid point

The MLP gives the temperature-dependent force constants.
{ref}`The SSCHA step <anisotropic-qha-sscha>` below computes the anharmonic
phonon free energy {math}`F_\mathrm{ph}(a, c; T)` from them.

{math}`U(a, c)` comes from the calculator on the static grid of step 1, not
from the MLP. The analysis minimizes {math}`F` over the lattice at each
temperature. The shape of {math}`U` near the minimum therefore decides where
the minimum lies. The axial expansions measure how that minimum moves as the
temperature rises. A small error in the shape of {math}`U` moves the minimum,
and the axial expansions follow that error. That is why {math}`U` is taken
from the calculator rather than from the MLP. The same sensitivity is behind
the choice of the linear tetrahedron method for {math}`F_\mathrm{el}` in
step 4: both terms shape the surface that the analysis differentiates.

Each MLP is evaluated only at its own grid point. It has to reproduce the
forces and energies of displaced supercells there and nowhere else. The forces
give the temperature-dependent force constants. The energies enter the SSCHA
free energy as differences from the undisplaced supercell of that grid point.
The grid as a whole carries the lattice dependence of
{math}`F_\mathrm{ph}`, so in this procedure the MLPs are never asked to
interpolate in the lattice parameters.

The MLPs are fitted independently, so their errors can jump from one grid
point to the next. Drawing the displacements from one set of standard normals
is expected to smooth those errors out. See {ref}`anisotropic-qha-normals`.

### The thermal distribution

The training structures are supercells displaced along the thermal
distribution of the harmonic crystal. Drawing them costs nothing beyond the
force constants of step 2, and the amplitudes are the ones the crystal
actually visits at the temperature asked for.

The distribution is drawn once, from the harmonic force constants. SSCHA
iterates its own distribution to self-consistency with the anharmonic force
constants; this draw does not. In practice that has been enough for a training
set.

Each mode {math}`(\mathbf{q}, \nu)` of the supercell is a harmonic oscillator
in equilibrium at {math}`T`, so its normal coordinate is normally distributed
about zero with variance

```{math}
\sigma_{\mathbf{q}\nu}^2 = \langle |Q_{\mathbf{q}\nu}|^2 \rangle
= \frac{\hbar}{2\omega_{\mathbf{q}\nu}}
  \coth \frac{\hbar \omega_{\mathbf{q}\nu}}{2 k_\mathrm{B} T}.
```

A **snapshot** is one supercell with every atom displaced at once, unlike
the displacements of step 2, which move one atom at a time. Each snapshot
draws one {math}`\xi_{\mathbf{q}\nu}` per mode from the **standard normal
distribution** -- mean 0, variance 1 -- so that
{math}`Q_{\mathbf{q}\nu} = \sigma_{\mathbf{q}\nu} \xi_{\mathbf{q}\nu}`, and
the displacements follow from the eigenvectors,

```{math}
\mathbf{u}_{lj} = \frac{1}{\sqrt{N m_j}} \sum_{\mathbf{q}\nu}
\sigma_{\mathbf{q}\nu}\, \xi_{\mathbf{q}\nu}\,
\mathbf{e}^{j}_{\mathbf{q}\nu}\,
e^{i \mathbf{q} \cdot \mathbf{r}_l},
\qquad
\xi_{\mathbf{q}\nu} \sim \mathcal{N}(0, 1).
```

A grid point's frequencies set the amplitudes {math}`\sigma_{\mathbf{q}\nu}`,
and its eigenvectors set the pattern of atomic motion each mode displaces
along. Each {math}`\xi_{\mathbf{q}\nu}` fixes how far its own mode is
displaced in this snapshot, and the sum turns those into the displacement
of every atom.

### The training displacements

Four steps, once the dataset of step 3 exists. The harmonic force constants
it carries are what set the widths of the draw.

1. Pick temperatures covering the range you want, and add one point above
   that range. An MLP tends to be poor outside the range it was trained on.
   Script 5 below trains at 0, 100, 250 and 400 K. Script 7 then analyses up
   to 400 K, and the finite difference consumes that top point, so the
   reported results stop at 390 K and stay inside the trained range.
2. Generate the displacements with the script below. It reads the harmonic
   force constants from the dataset of step 3, and uses one draw of standard
   normals at every grid point; see {ref}`anisotropic-qha-normals`.
3. Run the calculator on every supercell, and collect the forces of each set
   with {ref}`phonopy-init -f <f_force_sets_option>`. This writes one
   `phonopy_params.yaml` per set, holding its displacements, forces and
   supercell energies.
4. Merge the `phonopy_params.yaml` files of each grid point, one per
   temperature, into a single training set, and train that grid point's MLP
   on it.

```{code-block} python
:caption: Script 5 -- the thermal training displacements

from pathlib import Path

import numpy as np

from phonopy.interface.vasp import write_vasp
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

TRAIN = Path("train")
TEMPERATURES = (0.0, 100.0, 250.0, 400.0)  # K
SNAPSHOTS = 50  # structures per grid point and temperature
SEED = 20260815

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")
TRAIN.mkdir(parents=True, exist_ok=True)

for temperature in TEMPERATURES:
    normals = None
    for point in dataset.grid_points:
        phonon = point.to_phonopy()
        phonon.init_random_displacements()
        rd = phonon.random_displacements
        if normals is None:
            # Drawn once and reused at every grid point; the shapes are set by
            # the supercell matrix and the primitive cell, which they share.
            normals = rd.draw_standard_normals(
                SNAPSHOTS, random_seed=SEED + int(temperature)
            )
            # Keep the draw itself. The displacements below record the
            # ensemble, but only these reproduce it, and extending it with
            # first_snapshot needs them.
            np.savez_compressed(
                TRAIN / f"normals-{int(temperature)}K.npz",
                ii=normals[0],
                ij=normals[1],
                seed=SEED + int(temperature),
                temperature=temperature,
            )
        rd.run(temperature, standard_normals=normals)
        phonon.dataset = {"displacements": rd.u.copy()}

        # The stored index is 0-origin; the directories are numbered from 001.
        set_dir = TRAIN / f"grid-{point.index + 1:03d}-{int(temperature)}K"
        set_dir.mkdir(parents=True, exist_ok=True)
        # The displacements have to be saved beside the supercells: it is what
        # phonopy-init -f attaches the forces to below.
        phonon.save(
            set_dir / "phonopy_disp.yaml",
            settings={"force_constants": False, "displacements": True},
        )
        for i, cell in enumerate(phonon.supercells_with_displacements, 1):
            disp_dir = set_dir / f"disp-{i:03d}"
            disp_dir.mkdir(exist_ok=True)
            write_vasp(disp_dir / "POSCAR", cell)
```

Each set directory then holds `phonopy_disp.yaml` and
`disp-001/POSCAR .. disp-050/POSCAR`, the same layout as the phonon grid of
step 2, and `train/` holds one `normals-*.npz` per temperature beside them.
Run the calculator in every `disp-*`, then collect the forces of each set:

```bash
% phonopy-init -f disp-*/vaspout.h5 --save-params
# -> phonopy_params.yaml, with the displacements, forces and supercell energies
```

A grid point has one such set per temperature, and its MLP is trained on all
of them at once. The sets are merged by interleaving, so that the temperatures
alternate through the merged list instead of following one another in blocks.

The reason is how `ntrain` and `ntest` cut the merged list. `ntrain` takes that
many structures from its **head** and `ntest` takes that many from its
**tail**; neither looks at what is in them. In blocks the head would be the
coldest temperatures and the tail the hottest, so the MLP would be fitted to
one part of the range and tested on another. Interleaved, any head and any
tail hold the temperatures in equal parts:

```{code-block} python
:caption: Script 6 -- one training set per grid point, from its temperatures

from pathlib import Path

import numpy as np

import phonopy

TRAIN = Path("train")
TEMPERATURES = (0, 100, 250, 400)
N_GRID = 25

for index in range(1, N_GRID + 1):
    sets = [
        phonopy.load(
            TRAIN / f"grid-{index:03d}-{t}K" / "phonopy_params.yaml",
            produce_fc=False,
            log_level=0,
        )
        for t in TEMPERATURES
    ]

    # Interleave the temperatures, so that any prefix and any suffix of the
    # merged set holds them in equal parts.
    merged = {}
    for key in ("displacements", "forces", "supercell_energies"):
        stacked = np.array([s.dataset[key] for s in sets])
        merged[key] = stacked.swapaxes(0, 1).reshape(-1, *stacked.shape[2:])

    mlp_dir = TRAIN / f"grid-{index:03d}"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    phonon = sets[0]
    phonon.dataset = merged
    phonon.save(
        mlp_dir / "merged.yaml",
        settings={"force_sets": True, "displacements": True},
    )
```

To check the sets before starting the calculator, see
{ref}`checking the training displacements <anisotropic-qha-check-displacements>`.

Then train one MLP per grid point on its merged set. `phonopy --pypolymlp`
always writes the MLP as `polymlp.yaml` in the current directory, and the name
cannot be changed from the command line. So run the training inside that grid
point's own directory. Run it in `train/` instead and the 25 grid points write
over one file, leaving only the last one:

```bash
% cd train/grid-013
% phonopy merged.yaml --pypolymlp --mlp-params="ntrain=..., ntest=..." -v
# -> train/grid-013/polymlp.yaml
```

Script 7 reads the MLPs back from `train/grid-NNN/polymlp.yaml`, which is where
this puts them.

The displacements are drawn at random, so a training set of a given size is
one draw among many. A second draw of the same size would give a different
training set, and with it a different MLP and different results downstream.

That difference can be measured. Pick one grid point. Draw a second set of the
same size there with a different seed, and train a second MLP on it. Comparing
the phonon frequencies the two MLPs give is the cheap check, but it may not be
enough. The frequencies can agree closely while the axial expansions still
differ, so compare the quantity you intend to report.

### What the draw leaves at its defaults

Script 5 leaves the other parameters of `init_random_displacements` at their
defaults. Three of these parameters change the displacements:
`cutoff_frequency`, `dist_func` and `max_distance`.

`cutoff_frequency` is 0.01 THz by default. A mode's amplitude grows without
bound as its {math}`|\omega|` goes to zero, so the draw leaves out every mode
below the cutoff. The acoustic modes at {math}`\Gamma` fall below it in any
calculation, and a grid point close to an instability can have others that do.

The draw takes {math}`|\omega|`, so an imaginary mode is drawn as a real mode
of the same magnitude. A quasi-harmonic grid can reach grid points that are
dynamically unstable, so look at their frequencies before training on them.
`RandomDisplacements.treat_imaginary_modes` is the explicit treatment. It takes
{math}`|\omega|` at the commensurate points, shifts the modes between
`freq_from` and `freq_to` up by `freq_shift`, and rebuilds the force constants
from the shifted modes.

`dist_func` chooses the occupation the draw uses, quantum by default or
classical. `max_distance` shortens any displacement longer than the length
given, which caps the tail of the distribution.

(anisotropic-qha-normals)=
### Sharing one draw of standard normals

Drawing the {math}`\xi` once and using the same values at every grid point
gives one realization of the randomness over the whole grid. The surface fit
then carries it as a smooth function of the lattice parameters. The analysis
differentiates that surface, so smoothness matters.

`draw_standard_normals` returns the drawn {math}`\xi`, not displacements, and
`run(standard_normals=...)` takes a set back. The script above does exactly
this.

The same {math}`\xi` still gives different displacements at each grid point.
Each grid point scales them by its own frequencies and eigenvectors, which
makes the draw thermal there. The displacement fields of neighboring grid
points then resemble one another.

Snapshot *i* is drawn from `SeedSequence([random_seed, i])`, so it depends on
its index and on nothing else. Asking for snapshots 0 to 99 and later for 100
to 199 gives the same 200 as asking for 0 to 199 at once. An ensemble can
therefore be extended, or generated in blocks, with
`draw_standard_normals(..., first_snapshot=N)`.

A seed alone does not reproduce an ensemble after a NumPy upgrade. NumPy does
not promise that `Generator` distribution methods give the same stream across
its own versions (NEP 19). **Save the {math}`\xi` themselves.** Script 5 does
that, writing one `normals-*.npz` per temperature beside the training sets.

Read one back with `np.load` and hand `(ii, ij)` to
`run(standard_normals=...)`, or to
`draw_standard_normals(..., first_snapshot=N)` as the block already drawn.

The displacements in each `phonopy_disp.yaml` record the ensemble that was
run, and the forces attach to those, so a lost `npz` costs the extension
rather than the training set.

(anisotropic-qha-validate)=
### The descriptor and the amount of training data

There are two things to choose here: how big a descriptor to use, and how
many structures to train it on. The ridge penalty is not a third choice. It is
what the fit falls back on when the descriptor is too large for the training
set, so the penalty pypolymlp selects indicates whether the two are matched.

`--mlp-params` also sets the descriptor. Example feature counts for a
one-element system, with pypolymlp 0.20.5. The last column is the time to
evaluate the descriptor once, relative to the first row, measured in one
execution:

| features | model parameters added to `--mlp-params` | relative time |
|---|---|---|
| 781 | nothing; phonopy's defaults | 1.0 |
| 1,176 | `gaussian_params2 = 0 7 15` | 1.2 |
| 2,600 | `gaussian_params2 = 0 7 15, gtinv_maxl = 12 12` | 5.6 |
| 3,848 | `gaussian_params2 = 0 7 15, gtinv_order = 4, gtinv_maxl = 16 12 4` | 7.1 |
| 6,820 | `model_type = 4` | 1.4 |
| 13,920 | `model_type = 4, gaussian_params2 = 0 7 15` | 1.4 |
| 22,495 | `model_type = 4, gtinv_order = 6, gtinv_maxl = 16 12 4 1 1` | 6.4 |
| 27,664 | `model_type = 4, gaussian_params2 = 0 7 15, gtinv_maxl = 12 12` | 5.9 |
| 45,680 | `model_type = 4, gaussian_params2 = 0 7 15, gtinv_order = 6, gtinv_maxl = 16 12 4 1 1` | 7.7 |

Phonopy's defaults are `model_type = 3`, `max_p = 2`, `gtinv_order = 3`,
`gtinv_maxl = 8 8`, `gaussian_params2 = 0 7 10` and `cutoff = 8.0`. Phonopy
passes them to pypolymlp itself. The other rows change one or two of them.

The SSCHA of this step evaluates the descriptor once per snapshot per
iteration, so the evaluation time sets the cost of the whole step. The last
column is that time in one execution example, relative to the first row.

The evaluation time is dominated by `gtinv_maxl` rather than by the feature
count. The table holds two pairs that differ in `gtinv_maxl` alone, 1,176
against 2,600 and 13,920 against 27,664, and both cost about four times more
at `12 12` than at `8 8`. Raising `model_type` or the number of gaussians
instead multiplies the feature count while adding a few tens of per cent to
the time: 781 to 13,920 is eighteen times the features for 1.4 times the
time. So a descriptor that gains its features that way can carry several
times more of them for a fraction of the time.

More evaluation time still usually buys a lower force RMSE, once there is
enough training data to determine the extra features. The rule is only rough:
a fast descriptor with many features can beat a slow one with few. A lower
force RMSE also does not have to reach the thermal expansion, and the thermal
expansion is what this step is for. The next section says what the MLPs are
judged by instead.

How much training data a descriptor needs is a separate question. Fit with the
default `reg_alpha_params`, which scans five penalties from 1e-3 to 1e1, and
see which one pypolymlp keeps. With too few structures the penalty with the
smallest test RMSE sits at the large-penalty end of the range. As structures
are added, that penalty moves towards smaller values, and it stops moving once
there are enough structures. If that penalty is still at the large end, the
training set is what limits the accuracy, not the descriptor. Adding features
will not help until more structures are added.

Repeat that fit at several training-set sizes. Plot the selected penalty
against the size, with one line per descriptor. A line that is still falling
at the largest size has not converged. A line that has flattened has
converged, and more structures will not improve that descriptor. Each point
costs one fit. Running the SSCHA at every grid point and temperature, which is
the next step, costs far more than that. Make this plot first.

These two things are independent: how long a descriptor takes to evaluate, and
how much training data it needs. A descriptor that is slow to evaluate may
need only a few dozen structures, and a fast one may need many more. Use the
last column to judge what a descriptor costs and what force RMSE it can reach.
Use the selected penalty to judge whether the training set is large enough for
it.

### Pinning the ridge penalty across the grid

pypolymlp fits every penalty in `reg_alpha_params` and keeps the one with the
smallest test RMSE, chosen separately at every grid point. The analysis
differentiates across the grid, so a penalty that changes from one grid point
to the next puts a step into the quantity being differentiated. Collapse the
range to one point to pin it:

```bash
% phonopy grid-NNN-merged.yaml --pypolymlp \
    --mlp-params="ntrain=..., ntest=..., reg_alpha_params = -3.0 -3.0 1" -v
```

The three numbers are `linspace(p0, p1, p2)` of the base-10 logarithm, so
`-3.0 -3.0 1` is alpha = 1e-3 alone, against the default `-3.0 1.0 5` of 1e-3
to 1e1 in five steps.

### Validate the MLP

The MLPs are judged by their phonons rather than by the force RMSE. The RMSE
includes large-amplitude structures that the harmonic and quasi-harmonic
quantities never visit, while the frequencies are what enter
{math}`F_\mathrm{ph}`. The comparison below is still made on the forces,
because it is cheap and it is what shows where an MLP is weak; the frequencies
are what decides whether it is good enough.

The thermal supercells of this step already carry calculator forces. Evaluate
the same supercells with the MLP, and compare the two sets of forces. Nothing
new has to be run with the calculator. Use the structures held out as the test
set, since the MLP was fitted to the others.

Compare one temperature at a time. Each temperature has its own displacement
amplitudes, and one MLP is trained on all of them together, so its accuracy
can differ from one temperature to the next.

The phonon grid of step 2 can be compared in the same way, and there the force
constants and the frequencies can be compared as well. Its displacements are
one fixed distance, 0.03 Angstrom in the script there, which is usually
smaller than the amplitudes of the 0 K draw. An MLP trained across a
temperature range tends to be hard to make accurate at displacements that
small. A difference there need not mean the temperature-dependent run is
wrong.

(anisotropic-qha-sscha)=
### Computing the free energies with SSCHA

{math}`F_\mathrm{ph}` is the SSCHA free energy defined in
{ref}`SSCHA <mlp-sscha>`, computed from force constants that change with
temperature. It is no longer the
harmonic expression of "The free energy" above.

The `aniso_qha_dataset.hdf5` of step 3 is used as it is. Script 7, listed in
{ref}`the appendix <anisotropic-qha-sweep-script>` at the end of this page,
makes one SSCHA run at every grid point and every temperature. Save it as
`script7.py`. Each run starts from the harmonic force constants the dataset
carries. With no options it runs the whole grid and writes `sscha.hdf5`, and
`-a` turns that into the `fph.hdf5` the analysis reads:

```bash
% python script7.py -v
% python script7.py -a
```

`-v` lists the iterations of every run. The transient is chosen from that
list, so the first sweep is worth running with it; `-vv` adds the
force-constant fit.

Sampling and averaging are separate steps whichever way the runs are made.
The averaging is where the transient is chosen, and doing it apart means
choosing again costs a second of arithmetic rather than the whole sweep.

The constants at the top set the run and what its averaging leaves out.

`SNAPSHOTS` is how many supercells each SSCHA iteration draws, 2000 in the
script against phonopy's own 1000. Each one is an MLP evaluation, which makes
it the main cost of the step.

`ITERATIONS` is how many iterations each run makes, 16 against phonopy's 10.
The early ones drive the force constants to self-consistency, and they are the
run's transient.

After the transient the iterations do not settle on a value. Each one refits
the force constants from a **fresh sample**, so the step between iterations
stops shrinking, and the free energies scatter about a fixed point instead of
approaching one. Every iteration past the transient is therefore an
independent sample of the free energy, and averaging them is what improves the
estimate.

`MESH` is the mesh the harmonic part of the SSCHA free energy is sampled on,
and matching it to the `--mesh` of the analysis keeps one sampling through the
calculation. `SEED` fixes the whole run: iteration *i* draws from
`SeedSequence([SEED, i])`, so the run is reproducible while the iterations stay
independent.

A run stores every iteration and averages none of them. `-a` takes the mean
over the iterations after the transient, one of them by default, and
`--transient` sets how many. Choosing another is a second gather and costs no
sampling, and `--transient` on a run marks its log alone.

The error of one iteration is the standard error of the mean of the anharmonic
term over that iteration's snapshots,

```{math}
e_i = \frac{\sigma_i}{\sqrt{N}}, \qquad
\sigma_i^2 = \frac{1}{N - 1} \sum_{k=1}^{N}
\left( E^\mathrm{anh}_k - \bar{E}^\mathrm{anh} \right)^2,
```

where {math}`E^\mathrm{anh}_k` is the anharmonic energy of the {math}`k`-th
displaced supercell of that iteration, per primitive cell, written out in
Eq. {eq}`eq_sscha_anharmonic` of {ref}`SSCHA <mlp-sscha>`.
{math}`\bar{E}^\mathrm{anh}` is the mean over the {math}`N` snapshots, and
{math}`N` is `SNAPSHOTS`.

The iterations are independent draws, so the error of their mean is

```{math}
e = \frac{1}{m} \sqrt{\sum_{i=1}^{m} e_i^2} = \frac{\sigma}{\sqrt{mN}},
```

where {math}`m` = `ITERATIONS` - `transient`. The second form holds when the
{math}`\sigma_i` are alike, with {math}`\sigma` their common value.
{math}`mN` is how many supercells the run evaluates.

The error depends on that product ({math}`mN`) alone, so it says nothing about
how to split the cost between `ITERATIONS` and `SNAPSHOTS`. Two other things
settle that split. `ITERATIONS` has to exceed the transient with samples left
to average, and the log shows how long the transient is. Each iteration fits
its force constants from its own `SNAPSHOTS` supercells, so a small
`SNAPSHOTS` leaves every iteration's force constants noisy however many
iterations follow.

It is recommended to give the budget to `SNAPSHOTS`, and to raise
`ITERATIONS` only until the transient is cleared with samples to spare. The
two buy the same error, and only `SNAPSHOTS` improves the force constants that
everything other than the free energy is computed from.

Each {math}`e_i` holds its own iteration's force constants fixed, so {math}`e`
counts the sampling of the snapshots alone. The force constants were fitted
from a sample as well, and that variation appears as scatter of the kept
iterations about their mean, which is the column `-v` prints. Iterations that
scatter by about their own {math}`e_i` say that {math}`e` is the whole of it.

### Choosing the transient

`--transient` sets how many iterations are left out of the averages. The
default is `DEFAULT_TRANSIENT`, 1. Iteration 1 uses the harmonic force
constants of step 2, so its free energy is that of those force constants and
not of self-consistent ones.

How long the transient is depends on the system. It lasts until the force
constants reach self-consistency, and that takes longer the further the
harmonic force constants start from them. The default of 1 is a floor, not a
measurement.

`-v` prints each iteration's distance from the mean of the kept ones, divided
by that iteration's own error. An iteration past the transient gives about 1,
since it scatters about the fixed point by its own error. A larger value means
the iteration was still approaching that point.

```
  iter       F [meV]   error [meV]   (F - mean)/error
     1*      98.5931        0.0121              +46.9
     2       98.0729        0.0113               +4.1
     3       98.0141        0.0108               -1.1
     4       98.0166        0.0110               -0.9
     5       98.0118        0.0112               -1.3
     6       98.0153        0.0109               -1.0
  * left out as the transient. Of the kept iterations the furthest from the
    mean is 2, at 4.1 sigma.
```

Iteration 2 above gives 4.1, so `--transient 2` drops it. Raise `--transient`
until every kept iteration is within a few units. Check the highest
temperature as well as the lowest, since the distance between the harmonic and
the self-consistent force constants grows with temperature.

### Splitting the runs across jobs

One SSCHA run is minutes with the lightest descriptor and longer with a heavy
one. Multiplied by the grid points and the temperatures, the step comes to
hours or days. Spreading the runs over processes, nodes or jobs is therefore
worth the trouble.

Nothing constrains how they are spread. A run's randomness comes from `SEED`
and the iteration number alone, so a run gives the same numbers wherever it
sits in the two loops and whenever it is made. Either loop can be sliced, or
both.

`-g` and `-t` run one grid point at one temperature, and write that one value
to its own file. The two go together, and one without the other stops the
script. The grid point is numbered from 1, and the temperature is in K, taken
to the nearest of `TEMPERATURES`:

```bash
% python script7.py -g 13 -t 250
(013, 250.0)
Wrote sscha-g013-t250K.hdf5
```

Sampling writes `sscha*.hdf5`, one file per run here and one for the whole
grid when the grid is run in one process. Averaging writes `fph.hdf5`, which
is the file the analysis reads.

The two are different types, not two spellings of one. A `sscha*.hdf5` file
holds `SSCHAIterations`, every iteration and no average, and is what another
transient is taken from. `fph.hdf5` holds `SSCHAFreeEnergies`, the averages
and the transient they were taken with. Handing the analysis a `sscha*.hdf5`
file stops with a message about its type, rather than averaging it over
iterations nobody chose.

One such call is one job. `-a` gathers the `sscha*.hdf5` files into
`fph.hdf5`:

```bash
% python script7.py -a
Wrote fph.hdf5 from 1025 file(s)
```

Each file is placed by the lattice lengths and the temperature it carries
rather than by its name, so the order they are gathered in does not matter. A
missing value stops the write and is named, since the analysis would otherwise
read a gap as a zero.

`-a` averages the iterations each file carries rather than reading the
averages the runs wrote, so `--transient` applies to the whole grid at once:

```bash
% python script7.py -a --transient 2
```

That costs no sampling, so gathering the same files again with another value
is how one transient is compared with another. The runs gathered into one file
have to have made the same number of iterations, since one array cannot hold
runs of two lengths, and one grid point and temperature covered by two files
stops the write.


### Supplying the free energies to the analysis

In steps 0 to 4 each `Phonopy` instance of `phonopys` carries one
force-constant array, and it serves every temperature. The analysis computes
{math}`F_\mathrm{ph}` from them:

```python
run_anisotropic_qha(phonopys, temperatures, internal_energies=internal_energies)
```

Here they are computed elsewhere and passed to `run_anisotropic_qha` instead:

```python
run_anisotropic_qha(
    phonopys,
    temperatures,
    internal_energies=internal_energies,
    phonon_free_energies=free_energies,  # shape (temperatures, grid points)
)
```

The `Phonopy` instances then supply only the cells and volumes. They can be
built without force constants, which are never read. Without
`phonon_free_energies`, `run_anisotropic_qha` computes the phonon free energy
itself, and the force constants have to be there.

The values are per primitive cell. They are normalized the same way as
`internal_energies`, and they do not include the static energy, which
`internal_energies` already carries.

An MLP evaluates the undisplaced supercell, and its free energies are measured
from that energy. `SSCHAFreeEnergies` records it as `reference_energies`, and
`phonopy-anisotropic-qha --use-mlp-internal-energies` adds it back and takes
`U = 0`, which puts the whole surface on the potential's own energy scale
rather than the calculator's.

Since no force constants are read, the analysis also runs on a dataset built
from the static grid alone. That is all a method has to work with when it
never computed calculator phonons:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static static-grid/grid-{001..025}/ -o aniso_qha_dataset.hdf5
No --phonon given: building from the static grid alone, 25 grid point(s), with
no displacements or forces. Such a dataset is for use with the
phonon_free_energies argument of run_anisotropic_qha.
  grid 1 U=... eV n_disp=0
  ...
  grid 25 U=... eV n_disp=0
```

Its grid points carry the cells, {math}`U` and the electronic states, but no
harmonic force constants. Script 7 therefore cannot use such a dataset, having
nothing to start the SSCHA runs
from.

### Running the analysis

The file is given to the analysis with `--phonon-free-energies`:

```bash
% phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --phonon-free-energies fph.hdf5
```

The file carries the temperatures it was computed on, and the command takes its
temperature grid from them unless `--tmax` or `--dt` says otherwise. It also
carries the lattice lengths of the grid points, when they were written, and
those are checked against the dataset. A file computed on another machine
cannot then be paired with the wrong grid.

A free energy from a sampled method carries the scatter of its sampling. The
lattice parameters that minimize the free-energy surface at one temperature
move with that scatter, and they move independently of those at the next
temperature. Without smoothing the analysis takes each of {math}`da/dT`,
{math}`db/dT` and {math}`dc/dT` as a central difference between neighboring
temperatures, which amplifies that scatter. The command above therefore smooths
the lattice parameters before differentiating them, and says so as it runs.

The form fitted to each lattice parameter is

```{math}
a(T) = a_0 + \sum_i A_i \frac{\theta_i}{e^{\theta_i / T} - 1},
```

with amplitudes {math}`A_i` of opposite sign and Einstein temperatures
{math}`\theta_i`, and the same for {math}`b(T)` and {math}`c(T)`. Each term is
zero at {math}`T = 0` with zero slope, so the fitted expansion vanishes at 0 K,
as the third law requires. A general-purpose smoother assumes only smoothness,
and a spline fitted to the same data returns a finite expansion at 0 K instead.
Other forms share the property; this one is also the usual model for a lattice
parameter that contracts at low temperature and expands at high temperature.

Differentiating it term by term gives

```{math}
\frac{da}{dT} = \sum_i A_i
\frac{(\theta_i / 2T)^2}{\sinh^2(\theta_i / 2T)},
```

which is the derivative the axial expansions are built from once the fit is
made.

The smoothing is `--smooth-lattice einstein`, the default whenever
`--phonon-free-energies` is given. `--smooth-terms` sets how many terms, 2 by
default. More terms follow a curve more closely, and follow its scatter more
closely too. `--smooth-lattice none` turns the smoothing off.

The free energies of steps 0 to 4 carry no sampling scatter, so the default
there is `none` and the central differences are used.
`--smooth-lattice einstein` works there as well, for the analytic derivative in
place of them.

The same result comes from the API, with `phonon_free_energies` and
`lattice_smoothing`:

```python
result = run_anisotropic_qha(
    phonopys,
    temperatures,
    internal_energies=internal_energies,
    phonon_free_energies=free_energies,
    lattice_smoothing="einstein",
)
```

With `phonon_free_energies` given, `run_anisotropic_qha` skips the mesh
sampling and ignores `mesh`.

Fitting a sum of Einstein terms is not a linear least squares, and different
starting values converge to different curves. A converged curve can be wrong in
shape: monotone where the data contracts, or dipping several times deeper than
the data does. The fit therefore starts from a set of starting values, drops
the curves whose shape disagrees with the data in those ways, and keeps the
closest of what is left. If nothing is left, the command stops rather than
returning a curve of the wrong shape.

(anisotropic-qha-sweep-script)=
## Appendix: the SSCHA sweep script

The script {ref}`the SSCHA step <anisotropic-qha-sscha>` calls. With no options it
runs every grid point and every temperature and writes `sscha.hdf5`. `-g` and
`-t` run one of them and write it to its own file, which is how the sweep is
spread over jobs. `-a` gathers what the runs wrote into `fph.hdf5`.
`--transient` says how many iterations at the start of each run to leave out
of the averages, and applies to a gather as well as to a run. `-v` lists each
iteration with its departure from the mean of the kept ones, which is what
the transient is chosen from; `-vv` adds the force-constant fit.

```{code-block} python
:caption: Script 7 -- SSCHA at every grid point and temperature

import argparse
import glob

import numpy as np
from phonopy.interface.mlp import PhonopyMLP
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.free_energy_io import (
    SSCHAFreeEnergies,
    SSCHAIterations,
    read_sscha_iterations_hdf5,
    write_free_energies_hdf5,
    write_sscha_iterations_hdf5,
)
from phonopy.sscha.core import MLPSSCHA

DATASET = "aniso_qha_dataset.hdf5"
MLP = "train/grid-{:03d}/polymlp.yaml"  # grid point, numbered from 1
TEMPERATURES = np.arange(0, 410, 10.0)  # one extra point for finite diff
SNAPSHOTS = 2000
ITERATIONS = 16
MESH = 200.0
SEED = 1000
DEFAULT_TRANSIENT = 1


def report_iterations(history, transient):
    """List the iterations, and how far each sits from the mean of the kept ones.

    The departure is measured in the iteration's own error, so it reads the
    same whatever the system: an iteration standing well outside the scatter
    of the others has not reached self-consistency and belongs to the
    transient.

    """
    kept = history[transient:]
    mean = np.mean([h.free_energy for h in kept])
    print("  iter       F [meV]   error [meV]   (F - mean)/error", flush=True)
    for position, h in enumerate(history):
        mark = "*" if position < transient else " "
        print(
            f"  {h.iteration:4d}{mark} {h.free_energy * 1e3:12.4f} "
            f"{h.free_energy_error * 1e3:13.4f} "
            f"{(h.free_energy - mean) / h.free_energy_error:+18.1f}",
            flush=True,
        )
    worst = max(kept, key=lambda h: abs(h.free_energy - mean) / h.free_energy_error)
    sigma = abs(worst.free_energy - mean) / worst.free_energy_error
    print(
        f"  * left out as the transient. Of the kept iterations the furthest "
        f"from the mean is {worst.iteration}, at {sigma:.1f} sigma.",
        flush=True,
    )
    print(
        "  A kept iteration far outside the scatter of the rest is still in "
        "the transient: raise --transient and look again.",
        flush=True,
    )


def average(iterations, transient):
    """Return the mean over the iterations after the transient."""
    return iterations[..., transient:].mean(axis=-1)


def combine(errors, transient):
    """Return the error of that mean, over the iterations after the transient.

    The iterations are independent draws, so their errors add in quadrature
    rather than being averaged.

    """
    kept = errors[..., transient:]
    return np.sqrt(np.square(kept).sum(axis=-1)) / kept.shape[-1]


def write_sampled(temperatures, lattice_lengths, terms, reference, filename):
    """Write what the runs sampled, one value per iteration."""
    free_energies, errors, potential, harmonic_potential = terms
    write_sscha_iterations_hdf5(
        SSCHAIterations(
            temperatures,
            free_energies,
            errors,
            potential,
            harmonic_potential,
            reference,
            lattice_lengths=lattice_lengths,
        ),
        filename,
    )


def sscha_free_energy(
    ph,
    mlp,
    temperature,
    force_constants=None,
    transient=DEFAULT_TRANSIENT,
    log_level=0,
):
    """Return the iterations of one SSCHA run, in eV per primitive cell.

    The free energy, its error, and the two ensemble averages the anharmonic
    correction is the difference of, one value per iteration; then the energy
    of the supercell without displacements the whole is measured from. That
    last one is returned twice: as the static energy to put back, and as the
    origin the free energy is measured from. They are the same here and need
    not be.

    Every iteration is returned and no averaging is done here, so that
    another transient can be taken later without sampling again. ``transient``
    is used for the log alone.

    ``force_constants`` starts the iteration from a set of one's own. Without
    it, it starts from the set ``ph`` carries. ``log_level`` is passed to
    MLPSSCHA, and from 1 the iterations are listed here as well.

    """
    if force_constants is not None:
        ph = ph.replicate()
        ph.force_constants = force_constants
    sscha = MLPSSCHA(
        ph,
        mlp,
        temperature=temperature,
        number_of_snapshots=SNAPSHOTS,
        max_iterations=ITERATIONS,
        mesh=MESH,
        random_seed=SEED,
        log_level=log_level,
    ).run()
    history = sscha.history
    if log_level:
        report_iterations(history, transient)
    return (
        np.array([h.free_energy for h in history]),
        np.array([h.free_energy_error for h in history]),
        np.array([h.potential_energy for h in history]),
        np.array([h.harmonic_potential_energy for h in history]),
        sscha.supercell_energy / sscha.n_cell,
    )


def run_all(dataset, transient=DEFAULT_TRANSIENT, log_level=0):
    """Run all and write it to file."""
    points = dataset.grid_points
    shape = (len(TEMPERATURES), len(points), ITERATIONS)
    free_energies = np.zeros(shape)
    errors = np.zeros(shape)
    potential = np.zeros(shape)
    harmonic_potential = np.zeros(shape)
    reference = np.zeros(len(points))
    for column, point in enumerate(points):
        ph = point.to_phonopy()
        mlp = PhonopyMLP().load(MLP.format(point.index + 1))
        for row, temperature in enumerate(TEMPERATURES):
            print(f"({point.index + 1:03d}, {temperature})", flush=True)
            (
                free_energies[row, column],
                errors[row, column],
                potential[row, column],
                harmonic_potential[row, column],
                reference[column],
            ) = sscha_free_energy(
                ph, mlp, float(temperature), transient=transient, log_level=log_level
            )

    write_sampled(
        TEMPERATURES,
        np.array([np.linalg.norm(p.cell.cell, axis=1) for p in points]),
        (free_energies, errors, potential, harmonic_potential),
        reference,
        "sscha.hdf5",
    )


def run_one(dataset, grid_point, temperature, transient=DEFAULT_TRANSIENT, log_level=0):
    """Run one grid point at one temperature and write it to its own file.

    ``grid_point`` is numbered from 1, as the directories are. ``temperature``
    is in K, and the nearest of TEMPERATURES is the one run.

    """
    if not 1 <= grid_point <= len(dataset.grid_points):
        raise SystemExit(f"-g is 1 to {len(dataset.grid_points)}.")
    point = dataset.grid_points[grid_point - 1]
    row = int(np.argmin(np.abs(TEMPERATURES - temperature)))
    t = TEMPERATURES[row]

    ph = point.to_phonopy()
    mlp = PhonopyMLP().load(MLP.format(grid_point))
    print(f"({grid_point:03d}, {t})", flush=True)
    *terms, reference = sscha_free_energy(
        ph, mlp, float(t), transient=transient, log_level=log_level
    )

    filename = f"sscha-g{grid_point:03d}-t{t:g}K.hdf5"
    write_sampled(
        TEMPERATURES[[row]],
        np.linalg.norm(point.cell.cell, axis=1)[None, :],
        tuple(term[None, None, :] for term in terms),
        np.array([reference]),
        filename,
    )
    print(f"Wrote {filename}", flush=True)


def assemble(
    dataset,
    transient=DEFAULT_TRANSIENT,
    pattern="sscha*.hdf5",
    filename="fph.hdf5",
):
    """Gather what the runs wrote, and average it over one transient.

    The pattern takes the whole-grid sscha.hdf5 of run_all and the per-run
    files of run_one alike. Each is placed by the lattice lengths and the
    temperature it carries, not by its name; a gap stops the write, and so
    does a grid point and temperature covered twice. The averages are taken
    here from the iterations the files carry, so --transient covers the whole
    grid at once and costs no sampling.

    """
    points = dataset.grid_points
    lattice_lengths = np.array([np.linalg.norm(p.cell.cell, axis=1) for p in points])
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No file matches {pattern}.")
    parts = [read_sscha_iterations_hdf5(path) for path in paths]
    counts = {part.n_iterations for part in parts}
    if len(counts) > 1:
        raise SystemExit(
            f"The files hold {sorted(counts)} iterations. One array cannot "
            "carry runs of different lengths, so gather them apart."
        )
    shape = (len(TEMPERATURES), len(points), counts.pop())
    free_energies = np.full(shape, np.nan)
    errors = np.full(shape, np.nan)
    potential = np.full(shape, np.nan)
    harmonic_potential = np.full(shape, np.nan)
    reference = np.full(len(points), np.nan)

    seen = set()
    for path, part in zip(paths, parts):
        for i, t in enumerate(part.temperatures):
            row = int(np.argmin(np.abs(TEMPERATURES - t)))
            for j, lengths in enumerate(part.lattice_lengths):
                column = int(np.argmin(np.abs(lattice_lengths - lengths).sum(axis=1)))
                if (row, column) in seen:
                    raise SystemExit(
                        f"{path} covers grid point {column + 1} at "
                        f"{TEMPERATURES[row]:g} K, which another file covers "
                        "as well. Gather one sweep at a time."
                    )
                seen.add((row, column))
                free_energies[row, column] = part.free_energies[i, j]
                errors[row, column] = part.errors[i, j]
                potential[row, column] = part.potential_energies[i, j]
                harmonic_potential[row, column] = part.harmonic_potential_energies[i, j]
                reference[column] = part.reference_energies[j]

    located = free_energies[:, :, 0]
    missing = np.argwhere(np.isnan(located))
    if len(missing) > 0:
        first = ", ".join(
            f"(grid {c + 1}, {TEMPERATURES[r]:g} K)" for r, c in missing[:5]
        )
        raise SystemExit(
            f"{len(paths)} file(s) read, {len(missing)} of "
            f"{located.size} values missing: {first} ..."
        )

    write_free_energies_hdf5(
        SSCHAFreeEnergies(
            TEMPERATURES,
            average(free_energies, transient),
            errors=combine(errors, transient),
            lattice_lengths=lattice_lengths,
            reference_energies=reference,
            potential_energies=average(potential, transient),
            harmonic_potential_energies=average(harmonic_potential, transient),
            transient_iterations=transient,
        ),
        filename,
    )
    print(f"Wrote {filename} from {len(paths)} file(s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-point",
        "-g",
        type=int,
        default=None,
        help="grid point, numbered from 1 (default: all of them)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="temperature in K; the nearest of TEMPERATURES is run "
        "(default: all of them)",
    )
    parser.add_argument(
        "--assemble",
        "-a",
        action="store_true",
        help="gather the sscha*.hdf5 the runs wrote into fph.hdf5",
    )
    parser.add_argument(
        "--transient",
        type=int,
        default=DEFAULT_TRANSIENT,
        help="how many iterations at the start of a run are its transient "
        "and are left out of the averages (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log each SSCHA iteration; -vv adds the force-constant fit",
    )
    args = parser.parse_args()

    dataset = read_aniso_qha_dataset(DATASET)
    if args.assemble:
        assemble(dataset, args.transient)
    elif args.grid_point is None and args.temperature is None:
        run_all(dataset, args.transient, args.verbose)
    elif args.grid_point is not None and args.temperature is not None:
        run_one(
            dataset, args.grid_point, args.temperature, args.transient, args.verbose
        )
    else:
        raise SystemExit("-g and -t go together.")


if __name__ == "__main__":
    main()
```

(anisotropic-qha-check-displacements)=
## Appendix: checking the training displacements

Script 5 writes one set of displaced supercells per grid point and temperature,
into `train/grid-NNN-TK/`. Script 6 merges the sets of one grid point into
`train/grid-NNN/merged.yaml`. The calculator is run on the POSCAR of each
`disp-*`, and `phonopy --pypolymlp` trains on `merged.yaml`.

The checks in this appendix compare what script 5 and script 6 wrote against
the distribution the supercells were drawn from. They take seconds. The checks
on the displacements read `phonopy_disp.yaml` and the dataset of step 3, so
run them before the calculator. Script 6 merges the forces with the
displacements, so the check on the merged sets has something to read only
after the calculator, and before then it reports every grid point as having no
`merged.yaml`.

The draw and the reference use the same force constants {math}`\Phi`. The
check therefore tests the temperature a set was drawn at, and tests nothing
about {math}`\Phi` itself. Force constants that are wrong at a grid point
change the draw and the reference by the same amount, and the ratio still
comes out 1. Wrong force constants have to be caught at step 2, from the
frequencies.

### What each check compares

One check compares the amplitude of a set against the temperature its
directory is named after. Script 5 draws the supercells of a set from the
harmonic density matrix {math}`\tilde{\rho}_\Phi(T)` of that grid point's
force constants {math}`\Phi`, the distribution the SSCHA free energy of
{ref}`SSCHA <mlp-sscha>` averages over. `run_correlation_matrix(T)` fills
`RandomDisplacements.uu` with its second moment,

```{math}
\langle u_{l\kappa j} u_{l'\kappa' j'} \rangle_{\tilde{\rho}_\Phi(T)},
```

in Angstrom squared, at the same commensurate points and with the same cutoff
as the draw. Summing the diagonal over the supercell,

```{math}
\langle u^2 \rangle_{\tilde{\rho}_\Phi(T)} = \sum_{l\kappa j}
\langle u_{l\kappa j} u_{l\kappa j} \rangle_{\tilde{\rho}_\Phi(T)},
```

is `np.einsum("iiaa->", rd.uu)` in `reference_u2`.

A set of {math}`N` snapshots is a sample of {math}`\tilde{\rho}_\Phi(T')`,
where {math}`T'` is the temperature it was drawn at. The same sum over the
set,

```{math}
\overline{u^2} = \frac{1}{N} \sum_{n=1}^{N} \sum_{l\kappa j}
\bigl( u_{l\kappa j}^{(n)} \bigr)^2,
```

is computed by `sample_u2` and estimates {math}`\langle u^2
\rangle_{\tilde{\rho}_\Phi(T')}`. The ratio {math}`\overline{u^2} / \langle
u^2 \rangle_{\tilde{\rho}_\Phi(T)}` is 1 when {math}`T'` equals {math}`T`, the
temperature the directory is named after. `check_amplitudes` prints it for
every set, and a set with the wrong label gives a ratio far from 1.

The next check compares neighbouring grid points. Grid points {math}`g` and
{math}`g'` draw from {math}`\tilde{\rho}_{\Phi^{(g)}}(T)` and
{math}`\tilde{\rho}_{\Phi^{(g')}}(T)`, whose force constants differ only by
the strain between the two grid points. The grid points also share one draw of
standard normals ({ref}`anisotropic-qha-normals`), so at a fixed temperature
and snapshot their displacements differ only through {math}`\Phi^{(g)}` and
{math}`\Phi^{(g')}`. `correlation` measures how close the two sets are as

```{math}
\rho(g, g') = \frac{\sum_{n} \sum_{l\kappa j}
u_{l\kappa j}^{(g,n)} u_{l\kappa j}^{(g',n)}}
{\Bigl[ \sum_{n} \sum_{l\kappa j} \bigl( u_{l\kappa j}^{(g,n)} \bigr)^2
\sum_{n} \sum_{l\kappa j} \bigl( u_{l\kappa j}^{(g',n)} \bigr)^2
\Bigr]^{1/2}},
```

where {math}`u_{l\kappa j}^{(g,n)}` is snapshot {math}`n` of the set at grid
point {math}`g`, and every sum runs over the {math}`N` snapshots and the
supercell. Shared normals put {math}`\rho` near 1, and the spacing of the grid
sets how far below 1 it falls. `check_neighbours` prints {math}`1 - \rho`.

Two draws that do not share their normals are uncorrelated, so {math}`\rho` is
0. A value of {math}`1 - \rho` near 1 means the normals were not shared.

`read_set` compares the basis vectors of each set against those of the grid
point its directory is named after. The set carries its own unit cell in
`phonopy_disp.yaml`, the cell script 5 built the supercells from, and the
dataset of step 3 carries the relaxed cell of each grid point as `point.cell`.
Both comparands are `cell`, the 3x3 matrix whose rows are the basis vectors,
and they agree element by element to 1e-8 Angstrom when the directory holds
the grid point it names. `read_set` also counts the snapshots of the set.

`check_merged` reads the merged set of script 6. It counts the structures and
checks that the structures of each temperature are at the offsets
`k::len(TEMPERATURES)` the merge writes them to. It closes with the number of
merged sets it read, and names the grid points that have no `merged.yaml`.

### The script

```{code-block} python
:caption: Script 8 -- checking the training displacements

from pathlib import Path

import numpy as np

import phonopy
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

DATASET = "aniso_qha_dataset.hdf5"
TRAIN = Path("train")
TEMPERATURES = (0, 100, 250, 400)
SNAPSHOTS = 50


def reference_u2(point, temperatures):
    """Return <u^2> per temperature, from the force constants of one point.

    uu[i, j] is the 3x3 block of atoms i and j, in Angstrom^2.

    """
    phonon = point.to_phonopy()
    phonon.init_random_displacements()
    rd = phonon.random_displacements
    reference = {}
    for temperature in temperatures:
        rd.run_correlation_matrix(float(temperature))
        reference[temperature] = np.einsum("iiaa->", rd.uu)
    return reference


def sample_u2(displacements):
    """Return <u^2> over the snapshots of one set."""
    return (displacements**2).sum() / len(displacements)


def correlation(u_a, u_b):
    """Return the correlation of two sets of displacements."""
    a, b = u_a.ravel(), u_b.ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def neighbour_pairs(shape):
    """Yield the grid points adjacent along each axis, numbered from 1."""
    strides = [int(np.prod(shape[axis + 1 :])) for axis in range(len(shape))]
    for flat in range(int(np.prod(shape))):
        position = np.unravel_index(flat, shape)
        for axis, stride in enumerate(strides):
            if position[axis] + 1 < shape[axis]:
                yield flat + 1, flat + 1 + stride


def read_set(set_dir, point):
    """Return the displacements of one set, with its cell and count checked."""
    ph = phonopy.load(set_dir / "phonopy_disp.yaml", produce_fc=False, log_level=0)
    if not np.allclose(ph.unitcell.cell, point.cell.cell, atol=1e-8):
        print(f"  {set_dir}: not the basis vectors of grid point {point.index + 1}")
    displacements = np.array(ph.dataset["displacements"])
    if len(displacements) != SNAPSHOTS:
        print(f"  {set_dir}: {len(displacements)} snapshots, not {SNAPSHOTS}")
    return displacements


def check_amplitudes(dataset, temperatures):
    """Compare each set with the reference at the temperature it is named after."""
    u = {}  # (grid point numbered from 1, temperature) -> displacements
    print("grid    T        <u^2> / uu")
    for point in dataset.grid_points:
        index = point.index + 1
        reference = reference_u2(point, temperatures)
        for temperature in temperatures:
            set_dir = TRAIN / f"grid-{index:03d}-{temperature}K"
            u[index, temperature] = read_set(set_dir, point)
            ratio = sample_u2(u[index, temperature]) / reference[temperature]
            print(f"{index:4d} {temperature:5d} K   {ratio:10.3f}")
    return u


def check_neighbours(u, temperatures, pairs):
    """Compare the displacement field of neighbouring grid points."""
    print("\n   T    neighbouring grid points, 1 - correlation")
    for temperature in temperatures:
        values = [
            1.0 - correlation(u[a, temperature], u[b, temperature]) for a, b in pairs
        ]
        print(
            f"{temperature:5d} K   median {np.median(values):.1e}   "
            f"max {max(values):.1e}"
        )


def check_merged(dataset, u, temperatures):
    """Check the size and the interleaving of each merged set."""
    print("\nmerged sets")
    expected = len(temperatures) * SNAPSHOTS
    checked = 0
    missing = []
    for point in dataset.grid_points:
        index = point.index + 1
        merged = TRAIN / f"grid-{index:03d}" / "merged.yaml"
        if not merged.exists():
            missing.append(index)
            continue
        ph = phonopy.load(merged, produce_fc=False, log_level=0)
        d = np.array(ph.dataset["displacements"])
        if len(d) != expected:
            print(f"  {merged}: {len(d)} structures, not {expected}")
        for k, temperature in enumerate(temperatures):
            if not np.allclose(d[k :: len(temperatures)], u[index, temperature]):
                print(
                    f"  {merged}: the {temperature} K structures are not at "
                    f"{k}::{len(temperatures)}"
                )
        checked += 1
    print(
        f"  {checked} of {len(dataset.grid_points)} grid points, "
        f"{expected} structures each"
    )
    if missing:
        print(f"  no merged.yaml at grid points {', '.join(map(str, missing))}")


def main():
    """Check the sets of script 5 and the merged sets of script 6."""
    dataset = read_aniso_qha_dataset(DATASET)
    u = check_amplitudes(dataset, TEMPERATURES)
    check_neighbours(u, TEMPERATURES, list(neighbour_pairs(dataset.grid_shape)))
    check_merged(dataset, u, TEMPERATURES)


if __name__ == "__main__":
    main()
```

The grid points are ordered row-major over `dataset.grid_shape`, so
`neighbour_pairs` adds the stride of each axis to the flat index of a grid
point. A dataset that is not a tensor grid has `grid_shape` None, and its
pairs have to be given to `check_neighbours` explicitly.
