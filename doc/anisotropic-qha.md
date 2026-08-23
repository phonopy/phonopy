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

# --grid N is the number of points per free axis (5 -> 5 x 5 = 25 cells); one
# N per free DOF gives a rectangular grid, e.g. --grid 5 6 -> 30 cells.
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
# F_el is stored automatically when the static vaspout.h5 carries the electron
#   eigenvalues; pass --no-electronic to skip it
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
  supplies the per-point supercell / primitive matrices.

The positional `phonopy_disp.yaml` is the equilibrium reference; it supplies
the free lattice DOF metadata and the calculator. The grid-point index recorded
in the dataset is the position in the list, a label only, since the analysis
reads the lattice parameters from each stored cell.

```{admonition} How the ordering is checked
:class: note

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
```

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
    --contour-temp 0 500 1000 --compare-eos --electronic
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

`--electronic` adds the electronic free energy {math}`F_\mathrm{el}`, and only
when the dataset carries the electronic states; by default it is ignored. The
integration is the linear tetrahedron method when the states carry the k-point
grid, and the k-point sum otherwise. The k-point sum integrates a
delta-function density of states and converges slowly, so a mesh chosen for
the total energy is not dense enough for it. The run prints which of the two
it used, and at how many grid points.

Check that convergence on the thermal expansion, not on {math}`F_\mathrm{el}`.
The two integrations can agree closely on {math}`F_\mathrm{el}` at a mesh
where they differ severalfold in {math}`\alpha_c`: the expansion is a
derivative of the free-energy surface, and the k-point sum's error varies
across the lattice grid.

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
needs; the default is `none`.

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
    DS --> DISP["thermal displacements<br/>shared normals"]
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

Train one MLP per grid point, on that grid point's own training set, and
evaluate it only at that grid point. Each MLP then has to reproduce the
forces and energies of displaced supercells at one fixed cell: the forces
give the temperature-dependent force constants, and the energies enter the
SSCHA free energy as differences from the undisplaced supercell of that
same cell. No single MLP has to interpolate in the lattice parameters: the
lattice dependence of {math}`F_\mathrm{ph}` is carried by the grid, one MLP
per point, and made continuous by the surface fit the analysis performs.

### The training displacements

Four steps, once the phonon grid of step 2 exists.

1. Pick the temperatures. Cover the production range and add one point above
   it; an MLP used outside its training range is much the worse for it.
2. Generate the displacements with the script below. It draws the
   {math}`\xi` once and uses those same values at every grid point, which
   is what makes the errors of the MLPs vary smoothly across the grid
   rather than independently.
3. Run the calculator on every supercell, and collect the forces of each set
   with {ref}`phonopy-init -f <f_force_sets_option>`.
4. Merge each grid point's temperatures into one training set, and train its
   MLP.

Step 2 is the script below, which reads the harmonic force constants from
the dataset of step 3:

```{code-block} python
:caption: Script 5 -- the thermal training displacements

from pathlib import Path

from phonopy.interface.vasp import write_vasp
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

TRAIN = "train"
TEMPERATURES = (0.0, 100.0, 250.0, 400.0)  # K
SNAPSHOTS = 50  # structures per grid point and temperature
SEED = 20260815

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")

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
        rd.run(temperature, standard_normals=normals)
        phonon.dataset = {"displacements": rd.u.copy()}

        set_dir = Path(TRAIN) / f"grid-{point.index:03d}-{int(temperature)}K"
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
step 2. Run the calculator in every `disp-*`, then collect the forces of each
set:

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
# -> polymlp.yaml, beside that grid point's cell
```

### The descriptor and the ridge penalty

`--mlp-params` also sets the descriptor. Feature counts for one element,
measured with pypolymlp 0.20.5:

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

The snapshots are a sample of the harmonic crystal's canonical
distribution, not the distribution itself. Drawing a second set of the
same size gives different structures, so an MLP trained on one set, and
everything computed from that MLP, carries a scatter that falls
as `SNAPSHOTS` grows.

Each structure comes from its own seeded generator, so the run can be stopped
at any snapshot with every grid point and temperature equally represented, and
`draw_standard_normals(..., first_snapshot=N)` continues it later without
disturbing what was already computed.

```{note}
Two choices here reach {math}`\alpha_c`, not only the force error.
Displacements drawn uniformly at a fixed maximum amplitude under-sample the
soft modes that set the low-temperature {math}`\alpha_c`, which is why the
draw is thermal. Drawn independently at each grid point, the errors are white
across the lattice grid and the surface fit cannot absorb them, which is why
the normals are shared.
```

Judge the MLPs by their phonons rather than by the force RMSE. The RMSE
includes large-amplitude structures that the harmonic and quasi-harmonic
quantities never visit, while the frequencies are what enter
{math}`F_\mathrm{ph}`.

### What the draw leaves at its defaults

`init_random_displacements` takes settings the script above does not pass,
and two of them are worth knowing.

`cutoff_frequency`, 0.01 THz by default, drops modes below it from the draw.
That is how an imaginary mode is handled: the grid point is displaced without
that mode's contribution rather than failing. A quasi-harmonic grid can reach
cells that are dynamically unstable, so look at the frequencies of the grid
points before training on them -- a set built from a cell with an imaginary
mode carries no information about the direction that mode would have moved.

`dist_func` chooses the quantum occupation, the default, or the classical
one; `max_distance` shortens any displacement longer than the length given,
which caps the tail of the distribution.

### The shared normals

Every grid point draws {math}`\xi` from the same distribution in any case, so
sharing the distribution would change nothing. What is shared here is one set
of **drawn values**: the same numbers are used at every grid point, which is
what correlates the grid points instead of leaving them independent.

`draw_standard_normals` returns those values on their own, before any
frequency or eigenvector is applied. There are `3 * len(supercell)` of them
per snapshot, in two arrays: one for the {math}`\mathbf{q}` with
{math}`\mathbf{q} = -\mathbf{q} + \mathbf{G}`, one for the pairs of the rest,
whose real and imaginary parts each take their own.

Two grid points given the same values still get different displacements, since
each scales them by its own frequencies and eigenvectors, which is what makes
the draw thermal at that cell. Neighbouring cells then get displacement fields
that resemble one another rather than being drawn afresh.

Snapshot *i* is drawn from `SeedSequence([random_seed, i])`, so it depends on
its index and on nothing else. Asking for snapshots 0 to 99 and later for 100
to 199 gives the same 200 as asking for 0 to 199 at once, which is what lets
an ensemble be extended or generated in blocks.

Two caveats, both about reproducing an ensemble later:

- The displacements come back only to roundoff, because the summation order
  downstream of the random numbers depends on the array shapes and on the
  BLAS. Compare a regenerated structure by displacement, not byte for byte.
- NumPy does not promise that `Generator` distribution methods give the same
  stream across its own versions (NEP 19), so a seed alone does not reproduce
  an ensemble across a NumPy upgrade.

Keep the normals themselves beside the training sets if the ensemble has to be
reproducible:

```python
import numpy as np

np.savez_compressed("normals.npz", ii=normals[0], ij=normals[1])
```

```{note}
The errors of the per-grid-point MLPs are independent of one another, which
is a different failure mode from a single MLP's. Shared normals
make that error smooth in (a, c) rather than white, but not small.
{math}`\alpha_c` is a derivative of the surface, so it is what exposes the
rest. Validate as below before using a number.
```

(anisotropic-qha-validate)=
### Validate the MLP

An MLP that gives smooth phonons can still give wrong ones. Compare it
with the calculator at a few grid points before using its result.

**Swap the force evaluator on the same displacements.** The phonon grid of
step 2 already holds displaced supercells and their calculator forces.
Evaluating those same supercells with the MLP, and building the force
constants the same way, leaves the force evaluator as the only difference
between the two: the displacements, the solver and the q-points are identical,
so any difference in the frequencies, or in the anisotropic Gruneisen
parameters, belongs to the MLP.

**Compare stresses, and optionally elastic constants.** The stress is the
gradient of the free energy the analysis minimizes and the elastic constants
are its curvature, so both reach {math}`\alpha_c` through the surface fit.

**Validate at the amplitude the production run uses.** An MLP's relative
force error is largest where the displacements are smallest, because the
forces are smallest there. A comparison at the 0.03 Angstrom of the harmonic
phonon grid therefore measures the MLP at its worst, while the
temperature-dependent run it is used for samples the amplitudes the atoms
visit, where the training data is dense. The same MLP can look poor in
the first test and be accurate in the second, so match the test to the use.

Read the relative force error, not the absolute one. Near equilibrium the
forces themselves are small, and it is the relative error that maps to the
frequencies, and through them to {math}`F_\mathrm{ph}` and its lattice
derivatives.

### Computing and supplying the free energies

Use the MLPs for the temperature-dependent route ({ref}`mlp-sscha`)
rather than for a harmonic force-constant set. A harmonic set is built from
small fixed displacements, where an MLP's *relative* force error is
largest, while a temperature-dependent calculation samples the amplitudes the
atoms visit, where the training data is dense. For harmonic accuracy, steps 2
to 4 and the calculator give it directly.

{math}`F_\mathrm{ph}` is then no longer the harmonic expression of "The free
energy" above. The force constants change with temperature, and the free
energy that goes with them is the one the method itself defines -- for
SSCHA, the free energy of {ref}`mlp-sscha`.

One force-constant set per grid point cannot represent force constants that
differ at every temperature, so `phonopy-anisotropic-qha` and the default
form of `run_anisotropic_qha` do not apply here: both compute
{math}`F_\mathrm{ph}` themselves from the force constants they are given.

Build the intermediate dataset with `--phonon`, as in step 3: its harmonic
force constants are what the thermal displacements were drawn from, and
`run_anisotropic_qha` reads no force constants at all once
`phonon_free_energies` is given.

A dataset built from the static grid alone -- no `--phonon` -- also serves
here, and is what a method that never computed calculator phonons produces:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static static-grid/grid-{001..025}/ -o aniso_qha_dataset.hdf5
```

Its grid points carry the cells, {math}`U` and the electronic states, and no
displacements or forces. The builder reports this as it writes, a grid point
reports `n_displacements == 0`, and `to_phonopy()` on it raises instead of
returning force constants. `phonopy-anisotropic-qha` refuses such a dataset
with an explanation: the phonon free energy is the one thing it cannot compute
from it.

Compute the free energies outside instead, one value per grid point and
temperature in eV per primitive cell, and write them to a file where they are
computed. With SSCHA that is one run per grid point and temperature:

```{code-block} python
:caption: Script 7 -- SSCHA at every grid point and temperature

import numpy as np

from phonopy.interface.mlp import PhonopyMLP
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.free_energy_io import write_free_energies_hdf5
from phonopy.sscha.core import MLPSSCHA

TEMPERATURES = np.arange(0, 410, 10.0)  # one extra point for finite diff
SNAPSHOTS = 2000
ITERATIONS = 16
MESH = 200.0
SEED = 1000

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")
points = dataset.grid_points

free_energies = np.zeros((len(TEMPERATURES), len(points)))
errors = np.zeros_like(free_energies)

# The grid points come back in the order they were given to the builder,
# which is the order the training sets were made in.
for column, point in enumerate(points):
    mlp = PhonopyMLP().load(f"train/grid-{column + 1:03d}/polymlp.yaml")
    for row, temperature in enumerate(TEMPERATURES):
        sscha = MLPSSCHA(
            point.to_phonopy(),
            mlp,
            temperature=float(temperature),
            number_of_snapshots=SNAPSHOTS,
            max_iterations=ITERATIONS,
            mesh=MESH,
            random_seed=SEED,
        ).run()
        # The first entry is the ensemble of the harmonic force constants the
        # dataset carries, which is where the iteration starts rather than a
        # SSCHA solution. The rest are independent samples of the same one.
        history = sscha.history[1:]
        free_energies[row, column] = np.mean([h.free_energy for h in history])
        errors[row, column] = np.mean([h.free_energy_error for h in history])

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
```

`MESH` is the mesh the harmonic part of the SSCHA free energy is sampled on,
and matching it to the `--mesh` of the analysis keeps one sampling through
the calculation. `SEED` fixes the whole run: each iteration derives its own
seed from it, so the run is reproducible while the iterations stay
independent.

**One run per grid point and temperature is the cost of this step**, and at
production settings it is hours to days. The runs are independent, so slice
either loop and give each slice its own process, node or job. A slice over
grid points writes the columns it was given:

```python
columns = range(0, 5)          # this job's grid points
points = [dataset.grid_points[j] for j in columns]
...
write_free_energies_hdf5(TEMPERATURES, free_energies, "fph-000-004.hdf5", ...)
```

and the pieces are concatenated once they are all in:

```python
import glob

import numpy as np

from phonopy.qha.free_energy_io import (
    read_free_energies_hdf5,
    write_free_energies_hdf5,
)

parts = [read_free_energies_hdf5(f) for f in sorted(glob.glob("fph-*.hdf5"))]
temperatures = parts[0].temperatures
assert all(np.allclose(p.temperatures, temperatures) for p in parts)
write_free_energies_hdf5(
    temperatures,
    np.column_stack([p.free_energies for p in parts]),
    "fph.hdf5",
    kind="phonon",
    errors=np.column_stack([p.errors for p in parts]),
    lattice_lengths=np.vstack([p.lattice_lengths for p in parts]),
)
```

Then read the file in the analysis:

```bash
% phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 400 --dt 10 \
    --phonon-free-energies fph.hdf5 --smooth-lattice einstein
```

The file carries the temperatures it was computed on and, when written, the
lattice lengths of the grid points. The command checks both against the
dataset and stops on a mismatch, which is what keeps a file computed on
another machine from being paired with the wrong grid. `--phonon-free-energies`
also accepts a dataset built without `--phonon`, since no force constants are
read.

**Smooth the lattice parameters along temperature.** A free energy from a
sampled method carries the scatter of its sampling, the minimum at each
temperature moves with it independently of its neighbours, and the axial
expansions are derivatives of those minima, so the scatter reaches them
amplified. `--smooth-lattice einstein` fits each of {math}`a(T)`,
{math}`b(T)`, {math}`c(T)` to a sum of Einstein terms of opposite sign
before differentiating, and differentiates the fit rather than the points.
`--smooth-terms` sets how many terms, 2 by default; more terms follow a
curve more closely, at the cost of more freedom to follow its scatter. The
default is `none`, which is right for the force-constant route of steps 0
to 4, where the free energies carry no sampling scatter.

Each Einstein term is zero at {math}`T = 0` with zero slope, so the fitted
expansion vanishes at 0 K as the third law requires. The fit runs every
starting guess and keeps the best of those that pass its checks; when none
passes it raises rather than falling back on something else, so a
qualitatively wrong curve is not returned silently.

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
sampling and ignores `mesh`. The `Phonopy` instances supply only the cells and
volumes, and their force constants are never read, so they can be built
without any, as above. Normalize the values per primitive cell, consistently
with `internal_energies`, and exclude the static energy from them:
`internal_energies` already carries it.
