---
orphan: true
---

# Anisotropic QHA

This recipe computes the anisotropic (axis-resolved) quasi-harmonic thermal
expansion of a crystal by directly optimizing the lattice parameters on a grid,
rather than the 1D volume-path QHA. Steps 0 to 4 are the recipe: the phonons
come from displaced supercells computed with the calculator, and no
machine-learning potential is involved.

{ref}`Step 6 <anisotropic-qha-temperature-dependent>` is a variant for the case
where the harmonic approximation is what limits the answer. A potential is
trained at each grid point and used for temperature-dependent force constants,
whose free energies enter the analysis directly instead of as force sets. It is
the exception; most calculations do not need it.

The free lattice degrees of freedom are detected from the symmetry: one for
cubic (`a`), two for hexagonal, tetragonal and rhombohedral (`a, c`), and three
for orthorhombic (`a, b, c`). Cell angles are held fixed, so monoclinic and
triclinic crystals are out of scope. This page uses `(a, c)` throughout as a
concrete example; substitute the free DOF of your system. The lattice parameters
and axial thermal expansions are produced for any of the supported systems, but
the `F` contour maps are drawn only when there are exactly two free DOF.

```{warning}
**This workflow is experimental.** Everything on this page works and is
tested, but nothing about it is settled: the command-line options, the
`aniso_qha_dataset.hdf5` layout, and the `phonopy.qha.anisotropic` and
`phonopy.qha.anisotropic_dataset` APIs may all change in a
backward-incompatible way between releases, without a deprecation period.
Options have already been added and removed as the recipe was exercised on
real systems.

In practice: rebuild the dataset from the calculator outputs rather than
relying on an old file being readable, keep the commands that produced a
result alongside it, and pin the phonopy version if a campaign has to stay
reproducible across it. The page is not yet part of the documentation
navigation for the same reason.
```

All tools referenced here are implemented -- `phonopy-strain-cells`, the
dataset builder `phonopy-anisotropic-qha-dataset` and the analysis command
`phonopy-anisotropic-qha`. Step 4 gives the one-command analysis; the API
script beneath it is an equivalent alternative for finer control.

Prerequisites: `h5py`, `symfc`, a VASP setup (VASP is the supported
calculator), and, for the variant of step 6, `pypolymlp`.

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

```{note}
This page is written with VASP in mind, the only calculator interface this
workflow has been exercised with. The commands and helper scripts assume VASP
inputs and outputs (`POSCAR`, `vasprun.xml`, `vaspout.h5`); other calculators are
not tested here.
```

## Design principle: U is always from the calculator, the MLP is phonons-only

The free energy minimized per temperature is

```{math}
F(a, c; T) = U(a, c) + F_\mathrm{ph}(a, c; T) + F_\mathrm{el}(a, c; T),
```

where the electronic term {math}`F_\mathrm{el}` is optional. The static internal
energy {math}`U(a, c)` (and any elastic response) sets the valley *shape* and is
sensitive; it is **always taken from the calculator** on the static grid. The
machine-learning potential, when used, supplies **only the phonon force
constants** {math}`F_\mathrm{ph}(a, c; T)`, where the quantity is smooth and
cheap, and it is trained per grid point at a fixed lattice, so it never carries
the lattice dependence itself. Never take {math}`U` from the MLP. This keeps everything single-functional
and avoids the static-surface error that can flip {math}`\alpha_c` negative.

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

The recipe starts from one relaxed equilibrium cell. Turn it into the reference
`phonopy_disp.yaml` once with `phonopy-init`; every later step reads this file:

```bash
% phonopy-init -c REFERENCE_UNITCELL -d --dim 4 4 4
```

`REFERENCE_UNITCELL` must be the standardized conventional cell, whose lattice
vectors are the crystal axes a, b and c in that row order. The free lattice DOF
are taken per row, so a primitive cell of a centred lattice cannot be used: its
rows are centring vectors rather than crystal axes. For body-centred tetragonal,
for example, all three primitive rows have the same length, and scaling them
would only change the volume, never c/a. A rhombohedral cell must likewise be
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
`a, c` with `b = a` for hexagonal). Keep `--dim` consistent with the phonon-grid
supercell in step 2.

## 1. Build the static grid (internal energy U)

Sample strained unit cells over the free lattice DOF, then relax and run a
static single point for each with the calculator. The static grid supplies
{math}`U(a, c)` and, optionally, the electronic states for {math}`F_\mathrm{el}`.

```bash
# Inspect the free lattice DOF first (no ranges -> DOF report):
% phonopy-strain-cells phonopy_disp.yaml

# Random sampling: -n N cells over the (a, c) box (--grid is not used here):
% phonopy-strain-cells phonopy_disp.yaml --a 3.168 3.232 --c 5.148 5.252 \
    -n 25 --random-seed 2
# -> unitcell-00001 .. unitcell-00025

# Regular grid instead (--grid replaces -n): --grid N is the number of points
# per free axis (5 -> 5 x 5 = 25 cells); one N per free DOF gives a rectangular
# grid, e.g. --grid 5 6 -> 30 cells.
% phonopy-strain-cells phonopy_disp.yaml --a 3.168 3.232 --c 5.148 5.252 --grid 5
Wrote 25 strained unit cell(s) as unitcell-00001 .. unitcell-00025 in vasp format.
Grid sampling: 5 x 5 over (a, c).
  Main diagonal (5 cells), the --compare-vinet volume path:
    a  c   c/a
    3.1680  5.1480   1.6250
    3.1840  5.1740   1.6250
    3.2000  5.2000   1.6250
    3.2160  5.2260   1.6250
    3.2320  5.2520   1.6250
Provenance written to strain_cells.yaml
```

Prefer a grid when you want the Vinet cross-check: its main diagonal (printed
above, with each cell's c/a) is the volume path `phonopy-anisotropic-qha
--compare-vinet` fits, which random sampling does not provide. With equal
fractional ranges and equal counts the c/a is constant (isotropic scaling) --
the cleanest input to the cross-check; unequal ranges or counts still yield a
path, but a varying-shape one.

What each option writes: `-n N` and `--grid N [N ...]` set how many strained
cells are made, and they are written as strained **unit cells**
`unitcell-NNNNN`, which is what this step (the static grid) needs. A
`strain_cells.yaml` provenance manifest is written alongside, recording the
sampling, the resolved seed and the free-DOF lengths of every cell.

For each `unitcell-*`:

1. Relax the internal coordinates if the structure has free internal parameters
   (e.g. the wurtzite `u`). A crystal with no internal DOF (e.g. HCP) skips this
   -- the strained cell is already the relaxed cell.
2. Run a static single point. Use `ISIF >= 2` if you also want stress; write
   `vaspout.h5` if you want the electronic states for {math}`F_\mathrm{el}`.
3. Place the output in `static-grid/grid-NNN/` (one directory per grid point,
   containing `vaspout.h5` or `vasprun.xml`). Any layout works -- the builder
   is given the paths explicitly -- but a name that sorts in the sampling order
   keeps the two grids easy to pass together, and no index file is needed.

To scaffold the static-grid input POSCARs from the `unitcell-*` of step 1, edit
the paths at the top and run (distribute the VASP inputs separately):

```python
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

To scaffold the whole phonon grid from the relaxed static-grid cells, edit the
paths at the top and run. It reads each `static-grid/grid-NNN/CONTCAR` (the
relaxed structure; equal to the input POSCAR when there is no internal DOF), so
run it only after the static grid is done. Distribute the VASP inputs
separately.

```python
import glob
from pathlib import Path
import phonopy
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure

STATIC_GRID = "static-grid"  # relaxed cells at static-grid/grid-NNN/CONTCAR
PHONON_GRID = "phonon-grid"
SUPERCELL_MATRIX = [4, 4, 4]
DISTANCE = 0.03

for contcar in sorted(glob.glob(f"{STATIC_GRID}/grid-*/CONTCAR")):
    idx = int(Path(contcar).parent.name.split("-")[-1])
    cell, _ = read_crystal_structure(contcar, interface_mode="vasp")
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

The analysis reads `aniso_qha_dataset.hdf5`, and so does the variant of
step 6. The grid points are given as two path lists, usually expanded by the
shell, and paired **by position**:

```{note}
`phonopy-anisotropic-qha-dataset` works with VASP only for now, simply because
the readers for the other calculators are not implemented yet. The binding
constraint is the static grid: the internal energy {math}`U(a, c)` and the
electronic states are read from VASP outputs (`vaspout.h5` / `vasprun.xml`), and
phonopy has no interface yet to read the static single-point energy of the other
calculators. A reference naming one of them therefore stops the command early,
rather than producing a dataset with a missing {math}`U(a, c)`.
```

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static static-grid/grid-*/ --phonon phonon-grid/grid-*/ \
    -o aniso_qha_dataset.hdf5
# F_el is stored automatically when the static vaspout.h5 carries the electron
#   eigenvalues; pass --no-electronic to skip it
```

{ref}`Step 6 <anisotropic-qha-temperature-dependent>` omits `--phonon`, having
no calculator phonons to read; the dataset it gets then carries no forces.

Each entry may equally be a file rather than a directory, which is what a
layout the scaffolding scripts did not produce usually needs:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static runs/*/static/vaspout.h5 \
    --phonon runs/*/phonons/phonopy_params.yaml \
    -o aniso_qha_dataset.hdf5
```

No naming convention applies in either form: the paths may be laid out and
named however the calculations already are, and the two forms may be mixed
within one list. The two lists must have equal length.

For each grid point the builder reads:

- the static single point, giving the internal energy {math}`U(a, c)`; its
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
  the simpler route when the calculations were not laid out by the scaffolding
  script:

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

### The pairing is verified, not assumed

Two orderings decide which numbers end up together, and neither is trusted.

**Across grid points**, the two lists are paired by position, so the lattice of
each static single point must match the cell of the phonon grid point it is
paired with; a mismatch stops the command naming both paths. This catches a
point missing from each list, which would otherwise combine the {math}`U` of
one lattice with the forces of another while the list lengths still matched. It
also catches a static single point that was run on a supercell rather than the
unit cell, which would put {math}`U` on the wrong normalization.

**Within one grid point**, in the directory form, the `disp-*` subdirectories
are taken in sorted order, which is only a guess at the displacement order:
`disp-1, disp-10, disp-2` sorts differently from how it counts. Each calculator
output carries the structure it was run on, so it is compared against the
displaced supercell of its position, and a mismatch names the directory and the
displacement.

Nothing therefore has to be padded for correctness -- a wrong order is reported
rather than silently producing force constants from mismatched forces -- but
zero-padded names (`grid-001`, `disp-001`) keep both orderings right in the
first place, and the shell expands a glob lexicographically. The scaffolding
scripts above write `grid-{idx:03d}` and `disp-{k:03d}`, and `phonopy -d` does
the same.

### What the file holds

`aniso_qha_dataset.hdf5` is self-contained: per grid point it stores the relaxed
cell, supercell / primitive matrices, the raw displacements and forces, the
static internal energy {math}`U`, and optionally the electronic states. The
displacements and forces are kept in phonopy's native displacement-force dataset
form -- type-1 (one displaced atom per supercell) or type-2 (dense/random) --
tagged so the force-constant solver is chosen from the dataset type, not guessed.
Because the displacements and forces are stored raw (not force constants), the
file is independent of the force-constant method and can serve as an archive
after the calculator scratch is discarded.

## 4. Run the anisotropic QHA

Run the analysis directly on the intermediate dataset:

```bash
% phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
    --contour-temp 0 500 1000 --compare-vinet --electronic
```

This rebuilds one Phonopy per grid point (force constants from the stored
displacements and forces), runs `run_anisotropic_qha`, and writes
`lattice_parameters-temperature.dat`, `axial_thermal_expansion.dat`,
`volume-temperature.dat` and `anisotropic_qha.png`. With exactly two free
lattice DOF it also writes the `F(a, c)` contour maps; `--decompose-contours`
adds the U / F_ph / F_el / total panels and `--compare-vinet` adds a
volume-path cross-check (it needs the grid main diagonal from a `--grid` run in
step 1, and is skipped when no such diagonal is found). The electronic free
energy {math}`F_\mathrm{el}` is added only with `--electronic` (and only when
the dataset carries the electronic states); by default it is ignored.

Equivalently, drive `run_anisotropic_qha` from the API:

```python
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
    # The axial split needs a denser mesh than the volumetric expansion; 100
    # can leave alpha_c off by ~20% where beta is already converged.
    mesh=200.0,
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

`run_anisotropic_qha` detects the free lattice DOF from the input cells, fits the
free energy surface {math}`F(a, c; T)` and minimizes it per temperature, giving
{math}`a(T)`, {math}`c(T)` and the axial thermal expansions

```{math}
\alpha_a = \frac{1}{a}\frac{da}{dT}, \qquad
\alpha_c = \frac{1}{c}\frac{dc}{dT}.
```

The internal energies are expected in eV per primitive cell, which they are when
the static-grid single point is the primitive (unit) cell.

(anisotropic-qha-validate)=
## 5. Validate a machine-learning potential before trusting it

A smooth MLP is not automatically a correct one. Before trusting an MLP-phonon
result, validate against the calculator at a few points:

- Compare the MLP vs calculator phonon anisotropy directly. With a calculator
  phonon grid available, a same-displacement force-swap comparison isolates any
  anisotropic Gruneisen error: the displacements, the force-constant calculator
  and the q-points are then identical between the two, so only the force
  evaluator differs.
- Compare MLP and calculator stresses at a few cells (the stress is the
  free-energy gradient the QHA minimizes), and optionally elastic constants (the
  surface curvature).

**Validate at the amplitude the production run uses.** The relative force error
of a potential is largest where the displacements are smallest, because the
forces are smallest there, and a comparison made at a small fixed displacement
therefore measures the potential at its worst. A quasi-harmonic run with 0.03
Angstrom displacements is exactly that case, whereas a temperature-dependent run
samples the amplitudes the atoms actually visit, where the training data is
dense. A potential that looks poor in the first test can be accurate in the
second, so the test has to match the use.

Do not read a small absolute force error near equilibrium as reassurance. The
forces are small there too; it is the relative error that maps to frequencies,
and through them to {math}`F_\mathrm{ph}` and its lattice derivatives.

If the MLP phonons and equilibrium shape agree with the calculator within
tolerance, the dense (a, c) grid can be trusted.

(anisotropic-qha-temperature-dependent)=
## 6. Variant: temperature-dependent force constants from per-grid-point MLPs

Everything up to the static grid of step 1 is unchanged. What differs is where
the vibrational free energy comes from: instead of the phonon grid of step 2,
a machine-learning potential is trained at each grid point and used for
temperature-dependent force constants, whose free energies are handed to the
analysis directly.

This is the exception rather than the rule. Reach for it when the harmonic
approximation is what limits the answer and a calculator cannot afford the
sampling that replaces it; otherwise steps 0 to 4 are the recipe.

Two things have to be produced and meet at the end: a dataset that carries
no forces, and the free energies themselves.

```{mermaid}
flowchart TD
    EQ(["Equilibrium cell<br/>(phonopy_disp.yaml)"])
    EQ --> SC["phonopy-strain-cells<br/>(a, c grid)"]
    SC --> RELAX{{"calculator relax + static"}}
    RELAX --> SGRID(["static-grid/grid-NNN<br/>U, F_el"])

    SGRID --> DISP["displacements per<br/>grid-point cell"]
    DISP --> CALCT{{"calculator forces"}}
    CALCT --> DEV["train one MLP<br/>per grid point"]
    DEV --> MLP(["polymlp.yaml<br/>per grid point"])
    MLP --> SSCHA["SSCHA, TDEP, ...<br/>per grid point and temperature"]
    SSCHA --> FE(["F_ph(T) per<br/>grid point"])

    SGRID --> BUILD["phonopy-anisotropic-qha-dataset<br/>--static ... (no --phonon)"]
    BUILD --> DS(["aniso_qha_dataset.hdf5<br/>cells, U, F_el,<br/>no forces"])

    DS --> AQ["run_anisotropic_qha"]
    FE -->|"phonon_free_energies"| AQ
    AQ --> RES(["a(T), c(T),<br/>alpha_a, alpha_c"])
```

### One potential per grid point


Train one machine-learning potential **per grid point, at that point's fixed
lattice**. The potential then never has to represent the lattice dependence:
each one is used only at the cell it was trained on, there is no box to cover
and no extrapolation at the edges of one.

This is a deliberate change from an earlier design in which a single potential
was trained once over the whole {math}`(a, c)` box. That design asked one
descriptor set to carry the lattice dependence and the displacement dependence
at the same time, and it did not reach the accuracy the axial split needs. The
`phonopy-strain-cells` options that existed to build such a training set
(`--rd`, `--amplitude`, `--amax`, `--amax-per-atom`) have been removed; the
command writes strained unit cells only.

The training structures of each grid point are produced by the ordinary phonopy
workflow on that grid point's relaxed cell, exactly as for a single-cell MLP,
so there is nothing specific to the anisotropic workflow here:

```bash
# For each grid point, in static-grid/grid-NNN (the relaxed cell of step 1):
% phonopy --pa auto -c CONTCAR --dim 4 4 4 --rd 200 --amax 0.2 -v
# -> run the calculator on the supercells, collect, then train:
% phonopy phonopy_params.yaml --pypolymlp --mlp-params="ntrain=..., ntest=..." -v
# -> polymlp.yaml, beside that grid point's cell
```

Judge the potentials by phonons, not by the force RMSE: the RMSE mixes in
large-amplitude structures the harmonic and quasi-harmonic quantities never
visit, while the frequencies are what enter {math}`F_\mathrm{ph}`.

```{note}
The error of a per-grid-point potential is independent from one grid point to
the next, which is a different failure mode from a single potential's. It does
not cancel in the surface fit, and {math}`\alpha_c` is a derivative of that
surface, so it is the quantity that exposes it. Validate as in
{ref}`anisotropic-qha-validate` before trusting a number.
```

### Computing and supplying the free energies

What the potentials are then good for is the temperature-dependent routes
({ref}`mlp-sscha`), not a harmonic force-constant set. A harmonic set would be
built from small fixed displacements, which is where a potential's *relative*
force error is largest, whereas a temperature-dependent calculation samples the
amplitudes the atoms actually visit, where the training data is dense. If
harmonic accuracy is what is wanted, steps 2 to 4 and the calculator give it
directly.

The force constants then differ at every temperature, which one force-constant
set per grid point cannot represent, so `phonopy-anisotropic-qha` and the
default form of `run_anisotropic_qha` do not apply: they compute
{math}`F_\mathrm{ph}` themselves from the force constants they are given.

Build the intermediate dataset of step 3 without `--phonon`, since there are
no calculator phonons to read:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static static-grid/grid-*/ -o aniso_qha_dataset.hdf5
```

The grid points then carry the cells, {math}`U` and the electronic states, and
no displacements or forces. The builder says so as it writes, a grid point
reports `n_displacements == 0`, and `to_phonopy()` on it raises rather than
returning force constants it cannot produce. `phonopy-anisotropic-qha` refuses
such a dataset with an explanation, since the phonon free energy is exactly
what it cannot compute from it.

Compute the free energies outside instead -- one value per grid point and
temperature, in eV per primitive cell -- and hand them over:

```python
import numpy as np
from phonopy import Phonopy, run_anisotropic_qha
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")
temperatures = np.arange(0, 310, 10.0)  # one extra point for finite diff

# free_energies[i, j]: temperature i, grid point j, eV per primitive cell.
free_energies = ...  # from SSCHA, TDEP, or any other method

phonopys = [
    Phonopy(
        point.cell,
        supercell_matrix=point.supercell_matrix,
        primitive_matrix=point.primitive_matrix,
        log_level=0,
    )
    for point in dataset.grid_points
]
result = run_anisotropic_qha(
    phonopys,
    temperatures,
    internal_energies=[point.internal_energy for point in dataset.grid_points],
    electronic_structures=[point.electronic_states for point in dataset.grid_points],
    phonon_free_energies=free_energies,
)
```

Given `phonon_free_energies`, the mesh sampling is skipped and `mesh` is
unused. The `Phonopy` instances then supply only the cells and volumes, so
their force constants are neither required nor read -- they may be built
without any, as above. The values must be normalized per primitive cell,
consistently with `internal_energies`, and the vibrational free energy must
exclude the static energy, which `internal_energies` already carries.
