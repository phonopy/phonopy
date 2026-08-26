---
orphan: true
---

# Anisotropic QHA

This recipe computes the anisotropic (axis-resolved) thermal expansion of a
crystal in the quasi-harmonic approximation (QHA), by directly optimizing the
lattice parameters on a grid rather than along the 1D volume path. Steps 0 to
4 are the recipe: the phonons come from displaced supercells computed with the
calculator, and no machine-learning potential (MLP) is involved.

Step 5 is a variant for the case where the harmonic approximation is what
limits the answer. An MLP is trained at each grid point and used for
temperature-dependent force constants, whose free energies enter the analysis
directly instead of as force sets. It is the exception; most calculations do
not need it.

The free lattice degrees of freedom (DOF) are detected from the symmetry: one
for cubic ({math}`a`), two for hexagonal, tetragonal and rhombohedral
({math}`a, c`), and three for orthorhombic ({math}`a, b, c`). Cell angles are held fixed, so monoclinic
and triclinic crystals are out of scope. This page uses {math}`(a, c)` throughout as a
concrete example; substitute the free DOF of your system. The lattice
parameters and axial thermal expansions are produced for any of the supported
systems. Contour maps of the free energy {math}`F(a, c)` at a fixed
temperature -- the surface whose minimum gives those lattice parameters -- are
drawn only when there are exactly two free DOF.

```{warning}
**This workflow is experimental.** Everything on this page works and is
tested, but the interfaces are not settled. The command-line options, the
`aniso_qha_dataset.hdf5` layout, and the `phonopy.qha.anisotropic` and
`phonopy.qha.anisotropic_dataset` APIs may change in a backward-incompatible
way between releases, without a deprecation period; options have already been
added and removed as the recipe was used on real systems.

So rebuild the dataset from the calculator outputs rather than relying on an
old file being readable, keep the commands that produced a result alongside
it, and pin the phonopy version if a campaign has to stay reproducible across
it. The page is not yet part of the documentation navigation for the same
reason.
```

The commands are `phonopy-strain-cells`, the dataset builder
`phonopy-anisotropic-qha-dataset` and the analysis command
`phonopy-anisotropic-qha`. Step 4 runs the analysis in one command; the API
script beneath it does the same thing with more control.

Prerequisites: `h5py`, `symfc`, a VASP setup (VASP is the supported
calculator), and, for the variant of step 5, `pypolymlp`.

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

```{note}
This page is written with VASP in mind, the only calculator interface this
workflow has been exercised with. The commands and helper scripts assume VASP
inputs and outputs (`POSCAR`, `vasprun.xml`, `vaspout.h5`); other calculators are
not tested here.
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
q points of the sampling mesh. Step 5 replaces this term; the rest of the
page up to it uses it as written.

{math}`a` and {math}`c` are continuous above: {math}`F` is written for any
lattice. The calculator returns values only at the sampled cells
{math}`(a_i, c_i)` of step 1, and step 4 fits a surface through those values
before minimizing it, which is what makes {math}`a(T)` and {math}`c(T)`
continuous functions of temperature.

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
vectors are the crystal axes a, b and c in that row order. The free lattice DOF
are taken per row, so a primitive cell of a centred lattice cannot be used: its
rows are centring vectors rather than crystal axes. For body-centred tetragonal,
for example, all three primitive rows have the same length, and scaling them
would only change the volume, never {math}`c/a`. A rhombohedral cell must likewise be
given in the hexagonal setting. `phonopy-strain-cells` rejects such a cell; if
in doubt, take the `BPOSCAR` written by
{ref}`phonopy-init --symmetry <symmetry_option>`,
which is the conventional cell. A conventional cell that is merely rotated in
Cartesian space is fine.

`--dim` fixes the supercell matrix, which `phonopy-init` records together with
the unit cell, the primitive matrix and the calculator. `phonopy-strain-cells`
reads the equilibrium cell and calculator from it;
`phonopy-anisotropic-qha-dataset` reads the
calculator and the free lattice DOF it implies (which lengths are independent --
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
the cleanest input to the cross-check; unequal ranges or counts still give a
path, but its shape varies along it.

The cells are written as strained **unit cells** `unitcell-NNN`, which is what
the static grid needs. Each run also writes `strain_cells.yaml`, recording the
ranges, the grid shape and the free-DOF lengths of every cell. Nothing in the
run is random, so the same command reproduces the same cells.

For each `unitcell-*`:

1. Relax the internal coordinates if the structure has free internal parameters
   (e.g. the wurtzite `u`). A crystal with no internal DOF (e.g. HCP) skips this
   -- the strained cell is already the relaxed cell.
2. Run a static single point. Use `ISIF >= 2` if you also want stress; write
   `vaspout.h5` if you want the electronic states for {math}`F_\mathrm{el}`.
   Sample the Brillouin zone on a Gamma-centred regular mesh: the k points and
   the mesh are then stored with the states, and {math}`F_\mathrm{el}` is
   integrated by the linear tetrahedron method rather than summed over k
   points. The mesh a static single point already needs is dense enough for
   the tetrahedron method; a KPOINTS_OPT block, which is read in preference to
   the SCF mesh, gives the electronic states a denser one where that is wanted.
3. Place the output in `static-grid/grid-NNN/` (one directory per grid point,
   containing `vaspout.h5` or `vasprun.xml`). Any layout works -- the builder
   is given the paths explicitly -- but a name that sorts in the sampling order
   keeps the two grids easy to pass together, and no index file is needed.
   Sorting in that order also matters for `--compare-eos`: the builder
   recognises a tensor grid only when the cells reach it in the order they
   were generated, and records its shape for the main-diagonal volume path.

Edit the paths at the top and run it; distribute the VASP inputs separately.

```{code-block} python
:caption: Script 1 -- the static-grid input POSCARs, from the `unitcell-*` above

import glob
from pathlib import Path
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure

UNITCELLS = "unitcell-*"  # strained cells from phonopy-strain-cells
STATIC_GRID = "static-grid"

for path in sorted(glob.glob(UNITCELLS)):
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

The script below reads each `static-grid/grid-NNN/CONTCAR` (the relaxed
structure, equal to the input POSCAR when there is no internal DOF), so run
it only after the static grid is done. Edit the paths at the top and run it;
distribute the VASP inputs separately.

`SUPERCELL_MATRIX` must be the `--dim` of step 0, since the dataset builder
takes the free lattice DOF and the calculator from the same
`phonopy_disp.yaml`. `DISTANCE` is the displacement distance in Angstrom
(phonopy's own default is 0.01); the same value has to be used at every grid
point, because the force constants of the grid points are differentiated
against one another.

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
`phonopy-anisotropic-qha-dataset` works with VASP only for now, simply because
the readers for the other calculators are not implemented yet. The binding
constraint is the static grid: the internal energy {math}`U(a_i, c_i)` and the
electronic states are read from VASP outputs (`vaspout.h5` / `vasprun.xml`), and
phonopy has no interface yet to read the static single-point energy of the other
calculators. A reference naming one of them therefore stops the command early,
rather than producing a dataset with a missing {math}`U(a_i, c_i)`.
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

- the static single point, giving the internal energy {math}`U(a_i, c_i)`; its
  relaxed cell becomes the grid-point cell. The electronic states for
  {math}`F_\mathrm{el}` are read automatically from the same `vaspout.h5` when
  it carries the eigenvalues (a static point written with only `vasprun.xml` is
  built without {math}`F_\mathrm{el}`; pass `--no-electronic` to skip them
  deliberately). A directory entry is resolved to the VASP output it holds, and
  `vaspout.h5` is used in preference to `vasprun.xml`.
- the phonon grid point, in one of two forms. A **directory** holding
  `phonopy_disp.yaml` and the per-displacement `disp-*` subdirectories: the
  builder reads each `disp-*` calculator output itself, so no `FORCE_SETS` or
  `phonopy_params.yaml` is needed. Or a **phonopy.yaml-like file** whose forces
  {ref}`phonopy-init -f <f_force_sets_option>` has already collected, which is
  the simpler route when the calculations were not laid out by Script 2:

  ```bash
  % phonopy-init --sp -f disp-*/vasprun.xml   # -> phonopy_params.yaml
  % phonopy-init -f disp-*/vasprun.xml        # -> FORCE_SETS, beside phonopy_disp.yaml
  ```

  Pass the resulting `phonopy_params.yaml`, or the `phonopy_disp.yaml` whose
  `FORCE_SETS` sits beside it ({ref}`--sp <save_params_option>` merges the two
  into one file). A file with no forces and no neighboring `FORCE_SETS` is
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
and names both paths. This catches a grid point missing from each list, which
would otherwise combine the {math}`U` of one lattice with the forces of
another while the list lengths still matched, and a static single point run on
a supercell rather than the unit cell, which would put {math}`U` on the wrong
normalization.

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

The electronic states carry the eigenvalues, the k-point weights and the
electron count, and, when the static calculation sampled a Gamma-centred
regular mesh, the k points and the mesh themselves. That is what lets a reader
of the file integrate {math}`F_\mathrm{el}` by the linear tetrahedron method;
a file written without them is integrated by the k-point sum.

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
the last temperature point is consumed by them and the results stop one step
below `--tmax`. `--mesh` sets the phonon sampling mesh and defaults to 200,
denser than `run_qha`'s 100: the axial split needs it, while the volumetric
expansion is already converged at 100.

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
hand. Cells sharing a {math}`c/a` have one shape, which is what a volume-path
equation-of-state fit assumes; five or more of them are printed as a
ready-made `--eos-index` line.

The electronic free energy {math}`F_\mathrm{el}` is added whenever the dataset
carries the electronic states, and `--no-electronic` leaves it out. The
integration is the linear tetrahedron method, which needs the k-point grid the
states were computed on. Without it the command stops rather than falling back
on the k-point sum: that sum integrates a delta-function density of states and
needs far more irreducible k points than a mesh chosen for the total energy
has. The run prints the energy window and the grid spacing it integrated over.
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

which is what consumes one temperature point: with `--tmax 1000 --dt 10`
the results stop at 990 K.

The fit is over the lattice only: each temperature is minimized on its own,
and the raw {math}`a(T)`, {math}`c(T)` are differenced. With the
calculator's harmonic free energies that is what is wanted, since they
carry no sampling scatter. `--smooth-lattice` fits the lattice parameters
along temperature first and differentiates the fit instead, which step 5
needs. It defaults to `none` here, and to `einstein` when
`--phonon-free-energies` is given.

The internal energies are expected in eV per primitive cell, which they are when
the static-grid single point is the primitive (unit) cell.

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

Anharmonic effects appear only in structures that the small displacements of
step 2 never reach. One way to get at them is to displace the atoms with
the temperature folded in: every atom is moved at once, by an amount drawn
from the canonical distribution of the harmonic crystal at that
temperature, which the force constants of step 2 define. That distribution
is Gaussian in the normal coordinates, with the width the equation below
gives. The forces the calculator returns for such supercells are a
first-principles sample of the potential energy surface in the region a
crystal at that temperature actually visits.

Training an MLP on that sample is what this page does with it, and the
trained MLP then supplies the
temperature-dependent force constants whose free energies the analysis
takes directly. The displacements come first.

The distribution is fixed once, from the harmonic force constants: a
one-shot draw, not iterated to self-consistency with the anharmonic force
constants as SSCHA does. For a training set that is enough, since it has
to cover the region rather than reproduce the ensemble. Where
anharmonicity softens the modes, the true distribution is the wider of the
two, which is one reason to draw at a temperature above the production
range.

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

Steps 0 to 4 need one or a few displaced supercells per grid point, and
give the quasi-harmonic answer. Step 5 needs a training set at every grid
point instead -- 50 structures at each of four temperatures in the script
below, so 200 calculator runs per grid point -- and gives force constants
that change with temperature. Use it when anharmonicity is large enough to
matter for the property being computed.

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

The MLP enters one term of {math}`F(a, c; T)`: the anharmonic phonon free
energy {math}`F_\mathrm{ph}(a, c; T)`. The MLP gives the
temperature-dependent force constants, and the free energy that goes with
them is computed as "Computing and supplying the free energies" below
describes.

{math}`U(a, c)` comes from the calculator on the static grid of step 1, not
from the MLP. The analysis finds {math}`a(T)` and {math}`c(T)` by
minimizing {math}`F` over the lattice, so the shape of {math}`U` near that
minimum decides where the minimum sits and how it moves with temperature.
The axial expansions are derivatives of that motion, so an error in the
shape of {math}`U` reaches them amplified: an error that is small on the
scale of {math}`F` can be a large fraction of a small axial expansion, and
can change whether it comes out positive or negative.

One MLP is trained per grid point, on that grid point's own training set, and
evaluated only there. Each MLP then has to reproduce the forces
and energies of displaced supercells at one grid point. The forces give the
temperature-dependent force constants, and the energies enter the SSCHA free
energy as differences from the undisplaced supercell of that grid point. The
grid as a whole carries the lattice dependence of {math}`F_\mathrm{ph}`, so
the MLPs never interpolate in the lattice parameters.

The MLPs are fitted independently, so their errors can jump from one grid
point to the next. Drawing the displacements from one set of
standard normals smooths them; see {ref}`anisotropic-qha-normals`.

### The training displacements

Four steps, once the phonon grid of step 2 exists.

1. Pick the temperatures. Cover the production range and add one point above
   it, since an MLP is poor outside the range it was trained on.
2. Generate the displacements with the script below. It uses one draw of
   standard normals at every grid point; see
   {ref}`anisotropic-qha-normals`.
3. Run the calculator on every supercell, and collect the forces of each set
   with {ref}`phonopy-init -f <f_force_sets_option>`.
4. Merge each grid point's temperatures into one training set, and train its
   MLP.

Step 2 is the script below, which reads the harmonic force constants from
the dataset of step 3:

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
of them at once. Concatenating them in temperature order would put one
temperature at the front of the list and another at the back, and pypolymlp
takes its training structures from the front and its test structures from the
back, so the sets are interleaved instead:

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

    phonon = sets[0]
    phonon.dataset = merged
    phonon.save(
        TRAIN / f"grid-{index:03d}-merged.yaml",
        settings={"force_sets": True, "displacements": True},
    )
```

Then train one MLP per grid point on its merged set:

```bash
% phonopy grid-NNN-merged.yaml --pypolymlp --mlp-params="ntrain=..., ntest=..." -v
# -> polymlp.yaml, beside that grid point's training set
```

The structures behind that fit are the ones Script 5 drew: `SNAPSHOTS` per grid
point and temperature, a finite sample of the harmonic crystal's canonical
distribution rather than the distribution itself. A grid point's MLP therefore
depends on which structures its own draw happened to produce, and so does
everything computed from it. To see by how much, draw a second set of the same
size at one grid point, train a second MLP on it, and compare the frequencies
the two give there.

### The descriptor and the ridge penalty

Three choices interact here: how many features the descriptor has, how many
structures the training set holds, and how hard the ridge penalty damps the
fit. The penalty absorbs a mismatch between the first two, so it also tells
whether they are matched.

`--mlp-params` also sets the descriptor. Example feature counts for a
one-element system, with pypolymlp 0.20.5:

| features | added to `--mlp-params` |
|---|---|
| 781 | nothing; the default |
| 1,176 | `gaussian_params2 = 0 7 15` |
| 2,600 | `gaussian_params2 = 0 7 15, gtinv_maxl = 12 12` |
| 3,848 | `gaussian_params2 = 0 7 15, gtinv_order = 4, gtinv_maxl = 16 12 4` |
| 6,820 | `model_type = 4` |
| 13,920 | `model_type = 4, gaussian_params2 = 0 7 15` |
| 22,495 | `model_type = 4, gtinv_order = 6, gtinv_maxl = 16 12 4 1 1` |
| 27,664 | `model_type = 4, gaussian_params2 = 0 7 15, gtinv_maxl = 12 12` |
| 45,680 | `model_type = 4, gaussian_params2 = 0 7 15, gtinv_order = 6, gtinv_maxl = 16 12 4 1 1` |

A heavier descriptor costs more to fit and more to evaluate, and the SSCHA of
this step evaluates it once per snapshot per iteration, so the choice sets the
cost of the whole step.

A heavier descriptor also needs more training data to converge, and where the
ridge penalty lands -- the alpha that `reg_alpha_params` scans -- is a cheap
way to watch that. Fit the default range and see which penalty pypolymlp keeps: with
too few structures the smallest test RMSE sits at the large-penalty end, and
the optimum moves towards smaller penalties as structures are added, settling
once there are enough of them. A descriptor whose optimum has not moved off
the large end is held up by the training set rather than by the descriptor,
and adding features will not help until adding structures does.

**Pin the ridge penalty across the grid.** pypolymlp fits every penalty in
`reg_alpha_params` and keeps the one with the smallest test RMSE, chosen
separately at every grid point. The analysis differentiates across the grid,
so a penalty that changes from one grid point to the next puts a step into the
quantity being differentiated. Collapse the range to one point to pin it:

```bash
% phonopy grid-NNN-merged.yaml --pypolymlp \
    --mlp-params="ntrain=..., ntest=..., reg_alpha_params = -3.0 -3.0 1" -v
```

The three numbers are `linspace(p0, p1, p2)` of the base-10 logarithm, so
`-3.0 -3.0 1` is alpha = 1e-3 alone, against the default `-3.0 1.0 5` of 1e-3
to 1e1 in five steps.

The MLPs are judged by their phonons rather than by the force RMSE. The RMSE
includes large-amplitude structures that the harmonic and quasi-harmonic
quantities never visit, while the frequencies are what enter
{math}`F_\mathrm{ph}`.

### What the draw leaves at its defaults

`init_random_displacements` takes settings the script above does not pass.
Three of them decide what the ensemble looks like.

`cutoff_frequency` is 0.01 THz by default. A mode's amplitude grows without
bound as its {math}`|\omega|` goes to zero, so the draw leaves out every mode
below the cutoff. The acoustic modes at {math}`\Gamma` fall below it in any
calculation, and a grid point close to an instability can have others that do.

The draw takes {math}`|\omega|`, so an imaginary mode is drawn as a real mode
of the same magnitude. A quasi-harmonic grid can reach grid points that are
dynamically unstable, so look at their frequencies before training on them. `RandomDisplacements.treat_imaginary_modes` is the explicit
treatment. It takes {math}`|\omega|` at the commensurate points, shifts the
modes between `freq_from` and `freq_to` up by `freq_shift`, and rebuilds the
force constants from that.

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
makes the draw thermal there. The displacement fields of neighbouring grid
points then resemble one another.

Snapshot *i* is drawn from `SeedSequence([random_seed, i])`, so it depends on
its index and on nothing else. Asking for snapshots 0 to 99 and later for 100
to 199 gives the same 200 as asking for 0 to 199 at once. An ensemble can
therefore be extended, or generated in blocks, with
`draw_standard_normals(..., first_snapshot=N)`.

A seed alone does not reproduce an ensemble after a NumPy upgrade. NumPy does
not promise that `Generator` distribution methods give the same stream across
its own versions (NEP 19). **Save the {math}`\xi` themselves**, which Script 5
does, one `normals-*.npz` per temperature beside the training sets:

```python
np.savez_compressed(
    f"normals-{int(temperature)}K.npz", ii=normals[0], ij=normals[1]
)
```

Read one back with `np.load` and hand `(ii, ij)` to `run(standard_normals=)`,
or to `draw_standard_normals(..., first_snapshot=N)` as the block already
drawn. The displacements in each `phonopy_disp.yaml` record the ensemble that
was run, and that is what the forces attach to, so a lost `npz` costs the
extension rather than the training set.

(anisotropic-qha-validate)=
### Validate the MLP

The thermal supercells of this step already carry calculator forces. The same
supercells evaluated with the MLP give a second set of forces to compare them
with, and nothing new has to be run with the calculator. It is recommended to
use the structures held out as the test set, because the MLP was fitted to the
rest.

The comparison is made one temperature at a time. Each temperature has its own
amplitudes,
and one MLP is trained on all the temperatures together. Its accuracy can
differ from one temperature to the next.

The phonon grid of step 2 can be compared in the same way, and there the force
constants and the frequencies can be compared as well. Its displacements are
one fixed distance, 0.03 Angstrom in the script there, which is usually
smaller than the amplitudes of the 0 K draw. An MLP trained across a
temperature range is empirically hard to make accurate at displacements that
small, so a difference there need not mean the temperature-dependent run is
wrong.

(anisotropic-qha-sscha)=
### Computing the free energies with SSCHA

The MLPs are for the temperature-dependent route ({ref}`mlp-sscha`), not for a
harmonic force-constant set. They are trained on thermal displacements, and a
harmonic set would ask them for the small fixed displacements of step 2
instead. Steps 2 to 4 give the harmonic answer from the calculator directly.

{math}`F_\mathrm{ph}` is then no longer the harmonic expression of "The free
energy" above. With SSCHA it is the SSCHA free energy of {ref}`mlp-sscha`,
computed from force constants that change with temperature.

The `aniso_qha_dataset.hdf5` of step 3 is used as it is. Script 7, listed in
{ref}`anisotropic-qha-sweep-script` at the end of this page, makes one SSCHA run at
every grid point and every temperature. Each run starts from the harmonic force
constants the dataset carries. With no options it runs the whole grid and
writes `fph.hdf5`:

```bash
% python script7.py
```

`MESH` is the mesh the harmonic part of the SSCHA free energy is sampled on,
and matching it to the `--mesh` of the analysis keeps one sampling through
the calculation. `SEED` fixes the whole run: iteration *i* draws from
`SeedSequence([SEED, i])`, so the run is reproducible while the iterations stay
independent.

### Splitting the runs across jobs

One SSCHA run is minutes with the lightest descriptor and longer with a heavy
one. Multiplied by the grid points and the temperatures, the step comes to
hours or days. Spreading the runs over processes, nodes or jobs is therefore
worth the trouble. A run's randomness comes from `SEED` and the iteration
number alone, since iteration *i* draws from `SeedSequence([SEED, i])`. It
depends on neither its position in the two loops nor the order the runs are
made in. Either loop can be sliced, or both.

`-g` and `-t` run one grid point at one temperature, and write that one value
to its own file. The grid point is numbered from 1, and the temperature is in
K, taken to the nearest of `TEMPERATURES`:

```bash
% python script7.py -g 13 -t 250
(013, 250.0)
Wrote fph-g013-t250K.hdf5
```

One such call is one job. `-c` gathers what they wrote into `fph.hdf5`:

```bash
% python script7.py -c
Wrote fph.hdf5 from 1025 file(s)
```

Each file is placed by the lattice lengths and the temperature it carries
rather than by its name, so the order they are gathered in does not matter. A
missing value stops the write and is named, since a gap would otherwise reach
the analysis as a zero.

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

Since no force constants are read, the analysis also runs on a dataset built
from the static grid alone, which is what a method that never computed
calculator phonons has to work with:

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

Its grid points carry the cells, {math}`U` and the electronic states. Script 7
cannot use it, since it has no harmonic force constants to start the SSCHA runs
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
{math}`db/dT` and {math}`dc/dT` as a central difference between neighbouring
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

The script the section {ref}`anisotropic-qha-sscha` calls. With no options it
runs every grid point and every temperature and writes `fph.hdf5`. `-g` and
`-t` run one of them and write it to its own file, which is how the sweep is
spread over jobs, and `-c` gathers those files back into `fph.hdf5`. `-v` logs
each SSCHA iteration and lists them at the end, which is worth watching on the
first few runs; `-vv` adds the force-constant fit.

```{code-block} python
:caption: Script 7 -- SSCHA at every grid point and temperature

import argparse
import glob

import numpy as np
from phonopy.interface.mlp import PhonopyMLP
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.free_energy_io import (
    read_free_energies_hdf5,
    write_free_energies_hdf5,
)
from phonopy.sscha.core import MLPSSCHA

DATASET = "aniso_qha_dataset.hdf5"
MLP = "train/grid-{:03d}/polymlp.yaml"  # grid point, numbered from 1
TEMPERATURES = np.arange(0, 410, 10.0)  # one extra point for finite diff
SNAPSHOTS = 2000
ITERATIONS = 16
MESH = 200.0
SEED = 1000


def sscha_free_energy(ph, mlp, temperature, force_constants=None, log_level=0):
    """Return the SSCHA free energy and its error in eV per primitive cell.

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
    history = sscha.history[1:]
    if log_level:
        print("  iter        F [eV]      error", flush=True)
        for h in sscha.history:
            mark = " " if h in history else "*"
            print(
                f"  {h.iteration:4d}{mark} {h.free_energy:12.6f} "
                f"{h.free_energy_error:10.6f}",
                flush=True,
            )
        print("  * the starting ensemble, left out of the mean", flush=True)
    return (
        np.mean([h.free_energy for h in history]),
        np.mean([h.free_energy_error for h in history]),
    )


def run_all(dataset, log_level=0):
    """Run all and write it to file."""
    points = dataset.grid_points
    free_energies = np.zeros((len(TEMPERATURES), len(points)))
    errors = np.zeros_like(free_energies)
    for column, point in enumerate(points):
        ph = point.to_phonopy()
        mlp = PhonopyMLP().load(MLP.format(point.index + 1))
        for row, temperature in enumerate(TEMPERATURES):
            print(f"({point.index + 1:03d}, {temperature})", flush=True)
            free_energies[row, column], errors[row, column] = sscha_free_energy(
                ph, mlp, float(temperature), log_level=log_level
            )

    write_free_energies_hdf5(
        TEMPERATURES,
        free_energies,
        "fph.hdf5",
        kind="phonon",
        errors=errors,
        lattice_lengths=np.array(
            [np.linalg.norm(p.cell.cell, axis=1) for p in points]
        ),
    )


def run_one(dataset, grid_point, temperature, log_level=0):
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
    free_energy, error = sscha_free_energy(ph, mlp, float(t), log_level=log_level)

    filename = f"fph-g{grid_point:03d}-t{t:g}K.hdf5"
    write_free_energies_hdf5(
        TEMPERATURES[[row]],
        [[free_energy]],
        filename,
        kind="phonon",
        errors=[[error]],
        lattice_lengths=np.linalg.norm(point.cell.cell, axis=1)[None, :],
    )
    print(f"Wrote {filename}", flush=True)


def collect(dataset, pattern="fph-g*K.hdf5", filename="fph.hdf5"):
    """Gather the files run_one wrote into one file over the whole grid.

    Each file is placed by the lattice lengths and the temperature it carries,
    not by its name, and a gap stops the write.

    """
    points = dataset.grid_points
    lattice_lengths = np.array([np.linalg.norm(p.cell.cell, axis=1) for p in points])
    free_energies = np.full((len(TEMPERATURES), len(points)), np.nan)
    errors = np.full_like(free_energies, np.nan)

    paths = sorted(glob.glob(pattern))
    for path in paths:
        part = read_free_energies_hdf5(path)
        for i, t in enumerate(part.temperatures):
            row = int(np.argmin(np.abs(TEMPERATURES - t)))
            for j, lengths in enumerate(part.lattice_lengths):
                column = int(np.argmin(np.abs(lattice_lengths - lengths).sum(axis=1)))
                free_energies[row, column] = part.free_energies[i, j]
                errors[row, column] = part.errors[i, j]

    missing = np.argwhere(np.isnan(free_energies))
    if len(missing) > 0:
        first = ", ".join(
            f"(grid {c + 1}, {TEMPERATURES[r]:g} K)" for r, c in missing[:5]
        )
        raise SystemExit(
            f"{len(paths)} file(s) read, {len(missing)} of "
            f"{free_energies.size} values missing: {first} ..."
        )

    write_free_energies_hdf5(
        TEMPERATURES,
        free_energies,
        filename,
        kind="phonon",
        errors=errors,
        lattice_lengths=lattice_lengths,
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
        "--collect",
        "-c",
        action="store_true",
        help="gather the fph-g*K.hdf5 files written by -g and -t into fph.hdf5",
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
    if args.collect:
        collect(dataset)
    elif args.grid_point is None and args.temperature is None:
        run_all(dataset, args.verbose)
    elif args.grid_point is not None and args.temperature is not None:
        run_one(dataset, args.grid_point, args.temperature, args.verbose)
    else:
        raise SystemExit("-g and -t go together.")


if __name__ == "__main__":
    main()
```
