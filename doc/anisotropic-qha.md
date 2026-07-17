---
orphan: true
---

# Anisotropic QHA

This recipe computes the anisotropic (axis-resolved) quasi-harmonic thermal
expansion of a crystal by directly optimizing the lattice parameters on a grid,
rather than the 1D volume-path QHA. It covers two interchangeable phonon
sources that share the same downstream pipeline:

- **full-DFT phonons**: force sets from displaced supercells (a machine-learning
  potential is not required); and
- **MLP phonons**: force sets from a pypolymlp trained on first-principles
  energies, forces and stresses (cheap dense sampling).

The free lattice degrees of freedom are detected from the symmetry: one for
cubic (`a`), two for hexagonal, tetragonal and rhombohedral (`a, c`), and three
for orthorhombic (`a, b, c`). Cell angles are held fixed, so monoclinic and
triclinic crystals are out of scope. This page uses `(a, c)` throughout as a
concrete example; substitute the free DOF of your system. The lattice parameters
and axial thermal expansions are produced for any of the supported systems, but
the `F` contour maps are drawn only when there are exactly two free DOF.

Status: working recipe. All tools referenced here are implemented --
`phonopy-strain-cells`, `phonopy-vasp-mlp-dataset`, the dataset builder
`phonopy-anisotropic-qha-dataset` and the analysis command `phonopy-anisotropic-qha`.
Step 4 gives the one-command analysis; the API script beneath it is an
equivalent alternative for finer control.

Prerequisites: `h5py`, `symfc`, a VASP setup (VASP is the supported
first-principles source), and, for the MLP route, `pypolymlp`.

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

```{note}
This page is written with VASP in mind, the only first-principles interface this
workflow has been exercised with. The commands and helper scripts assume VASP
inputs and outputs (`POSCAR`, `vasprun.xml`, `vaspout.h5`); other calculators are
not tested here.
```

## Design principle: U is always DFT, the MLP is phonons-only

The free energy minimized per temperature is

```{math}
F(a, c; T) = U(a, c) + F_\mathrm{ph}(a, c; T) + F_\mathrm{el}(a, c; T),
```

where the electronic term {math}`F_\mathrm{el}` is optional. The static internal
energy {math}`U(a, c)` (and any elastic response) sets the valley *shape* and is
sensitive; it is **always taken from DFT** on the static grid. The
machine-learning potential, when used, supplies **only the phonon force
constants** {math}`F_\mathrm{ph}(a, c; T)`, where the quantity is smooth and
cheap. Never take {math}`U` from the MLP. This keeps everything single-functional
and avoids the static-surface error that can flip {math}`\alpha_c` negative.

## Overview

The boxes are jobs run by phonopy tools or the API, the hexagons are DFT
calculations, and the rounded nodes are input and intermediate data. Both phonon
routes converge on the same intermediate dataset and analysis.

```{mermaid}
flowchart TD
    EQ(["Equilibrium cell<br/>(phonopy_disp.yaml)"])

    EQ --> SC["phonopy-strain-cells<br/>(a, c grid)"]
    SC --> RELAX{{"DFT relax + static"}}
    RELAX --> SGRID(["static-grid/grid-NNN<br/>U, F_el"])

    SGRID -->|"route A"| PD["generate displacements<br/>per relaxed cell"]
    PD --> DFTF{{"DFT forces"}}
    DFTF --> PGRID(["phonon-grid/grid-NNN<br/>disp-*"])

    EQ -->|"route B"| SCT["phonopy-strain-cells --rd"]
    SCT --> DFTT{{"DFT E / F / stress"}}
    DFTT --> MLPDS["phonopy-vasp-mlp-dataset"]
    MLPDS --> DEV["develop_pypolymlp"]
    DEV --> MLP(["polymlp.yaml"])

    SGRID --> BUILD["phonopy-anisotropic-qha-dataset"]
    PGRID -->|"default"| BUILD
    MLP -->|"--from-mlp"| BUILD
    BUILD --> DS(["aniso_qha_dataset.hdf5"])

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
in doubt, take the `BPOSCAR` written by {ref}`phonopy --symmetry <symmetry_option>`,
which is the conventional cell. A conventional cell that is merely rotated in
Cartesian space is fine.

`--dim` fixes the supercell matrix, which `phonopy-init` records together with
the unit cell, the primitive matrix and the calculator. `phonopy-strain-cells`
reads the equilibrium cell and calculator from it (plus the supercell matrix when
`--rd` builds MLP training supercells); `phonopy-anisotropic-qha-dataset` reads the
calculator and the free lattice DOF it implies (which lengths are independent --
`a, c` with `b = a` for hexagonal), and, on the MLP route, the supercell and
primitive matrices. Keep `--dim` consistent with the phonon-grid supercell in
step 2A.

## 1. Build the static grid (internal energy U, both routes)

Sample strained unit cells over the free lattice DOF, then relax and run a
static single point for each with DFT. The static grid supplies {math}`U(a, c)`
and, optionally, the electronic states for {math}`F_\mathrm{el}`.

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
cells are made and, without `--rd`, write them as strained **unit cells**
`unitcell-NNNNN` -- what this step (the static grid) needs. Adding `--rd M`
instead writes M random-displacement **supercells** per strained cell,
`supercell-NNNNN` (that is the route-B, step 2B use, not this step);
`--amplitude` sets their displacement distance, or `--amin` / `--amax` draw it
from a range (see step 2B). Either way a `strain_cells.yaml` provenance
manifest is written.

For each `unitcell-*`:

1. Relax the internal coordinates if the structure has free internal parameters
   (e.g. the wurtzite `u`). A crystal with no internal DOF (e.g. HCP) skips this
   -- the strained cell is already the relaxed cell.
2. Run a static single point. Use `ISIF >= 2` if you also want stress; write
   `vaspout.h5` if you want the electronic states for {math}`F_\mathrm{el}`.
3. Place the output in `static-grid/grid-NNN/` (one directory per grid point,
   containing `vaspout.h5` or `vasprun.xml`). The builder discovers grid points
   from these `grid-NNN` directories; no index file is needed.

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

## 2A. Route A -- full-DFT phonons (phonon grid)

For each relaxed static-grid cell, generate displaced supercells and compute
their forces with DFT.

Place the results as `phonon-grid/grid-NNN/` each containing the
`phonopy_disp.yaml` and the per-displacement subdirectories `disp-001/`,
`disp-002/`, ... (each with `vaspout.h5` or `vasprun.xml`). The grid indices
must match those under `static-grid/`.

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

```{note}
`phonopy-anisotropic-qha-dataset` supports only the VASP calculator. The static
internal energy {math}`U(a, c)`, the forces, and the electronic states are read
from VASP outputs (`vaspout.h5` / `vasprun.xml`); a reference specifying any
other calculator is rejected, because phonopy has no interface yet to read the
static single-point energy of other calculators.
```

Then build the intermediate dataset. The builder expects the two grids below.
Grid indices are zero-padded to three digits (`grid-000`, `grid-001`, ...) and
must match between `static-grid` and `phonon-grid`:

```text
static-grid/                 # --static-grid (default: static-grid)
  grid-000/
    vaspout.h5               # or vasprun.xml; static single point -> U(a, c)
  grid-001/
    vaspout.h5
  ...
phonon-grid/                 # --phonon-grid (default: phonon-grid)
  grid-000/
    phonopy_disp.yaml        # relaxed displaced cell + supercell/primitive
    disp-001/
      vaspout.h5             # or vasprun.xml; forces for displacement 1
    disp-002/
      vaspout.h5
    ...
  grid-001/
    ...
```

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml \
    --static-grid static-grid --phonon-grid phonon-grid \
    -o aniso_qha_dataset.hdf5
# F_el is stored automatically when static-grid/grid-NNN/vaspout.h5 carries the
#   electron eigenvalues; pass --no-electronic to skip it
```

For each grid point the builder reads:

- `static-grid/grid-NNN/` -- the static single point, giving the internal energy
  {math}`U(a, c)`; its relaxed cell becomes the grid-point cell. The electronic
  states for {math}`F_\mathrm{el}` are read automatically from the same
  `vaspout.h5` when it carries the eigenvalues (a static grid written with only
  `vasprun.xml` is built without {math}`F_\mathrm{el}`; pass `--no-electronic` to
  skip them deliberately).
- `phonon-grid/grid-NNN/phonopy_disp.yaml` -- the displaced cell with its
  supercell / primitive matrices and the displacement dataset (type-1 or
  type-2).
- `phonon-grid/grid-NNN/disp-*/` -- the per-displacement forces, read in sorted
  `disp-*` order; their count must equal the number of displacements in
  `phonopy_disp.yaml`.

`vaspout.h5` is used when present (full numerical precision), `vasprun.xml`
otherwise. Grid points are discovered from the `static-grid/grid-NNN`
directories, so a `grid-NNN` that exists only under `phonon-grid` is ignored. The
positional `phonopy_disp.yaml` is the equilibrium reference; it supplies the free
lattice DOF metadata and the calculator (the per-point supercell / primitive
matrices come from each point's `phonon-grid` yaml).

## 2B. Route B -- MLP phonons (train once, then evaluate)

Train a pypolymlp on strained random-displacement supercells (this is a
*different* strain-cells use than step 1: random-displacement supercells over a
box that must **cover the static grid with margin**, since an MLP extrapolates
badly at its domain edges).

```bash
# Training structures: strained supercells with random displacements (--rd):
# --rd N gives N displaced supercells per cell; --amplitude (= --amin) sets the
# distance.
% phonopy-strain-cells phonopy_disp.yaml --a 3.15 3.25 --c 5.10 5.30 \
    -n 100 --random-seed 1 --rd 1 --amplitude 0.03
# -> supercell-00001 .. supercell-00100 ; run a single-point VASP (ISIF >= 2)
#    for each, then assemble the dataset (stress included):
% phonopy-vasp-mlp-dataset disp-*/vaspout.h5 -o polymlp_dataset.hdf5
```

`phonopy-vasp-mlp-dataset` reads `vaspout.h5` or `vasprun.xml`. Prefer
`vaspout.h5`: it carries the full precision of the calculation, whereas
`vasprun.xml` is written with six digits.

Adding `--amax` makes the displacement distance random instead of fixed (the
distance is drawn per supercell; within one supercell every atom moves by that
same distance). The draw is uniform over `[0, --amax)` and is raised to
`--amin` when smaller, so the distances are uniform over `[--amin, --amax)`
except for the weight `--amin / --amax` piled up exactly at `--amin`. A fixed
small distance is what harmonic force constants need and is the default here.
The range form matters when the MLP is to be used beyond the harmonic regime,
e.g. for the temperature-dependent force constants of {ref}`mlp-sscha`, whose
supercells at temperature reach far larger displacements than 0.03 Angstrom;
an MLP trained only near equilibrium would extrapolate there. Then train over
both the lattice box and the amplitude range at once:

```bash
% phonopy-strain-cells phonopy_disp.yaml --a 3.15 3.25 --c 5.10 5.30 \
    -n 100 --random-seed 1 --rd 1 --amin 0.03 --amax 1.5
```

Train the MLP with energies, forces and stresses (structure-based training, so
the varying lattices and the stress/virial are used):

```python
from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    develop_pypolymlp,
    read_pypolymlp_structure_dataset,
    save_pypolymlp,
)

data = read_pypolymlp_structure_dataset("polymlp_dataset.hdf5")
polymlp = develop_pypolymlp(
    data, params=PypolymlpParams(), test_size=0.1, verbose=True
)
save_pypolymlp(polymlp, "polymlp.yaml")
```

`develop_pypolymlp` picks the training mode from the dataset type: a
`PypolymlpStructureData` like this one trains on independent structures, so the
lattices may differ and the stress is used; a `PypolymlpData` trains on
displacements of one supercell instead.

`test_size` splits the dataset without shuffling: the first 90 percent trains
and the rest tests. Pass `test_data` explicitly when it must stay fixed while
the training dataset varies, as when measuring how many structures the MLP
needs. Datasets slice like sequences, so a training-set size series needs no
helper:

```python
from phonopy.interface.pypolymlp import split_pypolymlp_dataset

train, test = split_pypolymlp_dataset(data, test_size=0.1)
for n in (20, 40, 70, len(train)):
    polymlp = develop_pypolymlp(train[:n], test, params=PypolymlpParams())
```

Then build the intermediate dataset. `--from-mlp` needs only the `static-grid`
layout of step 1 (no `phonon-grid`): the displacements are generated and their
forces evaluated on the fly from the MLP -- `--distance` sets the displacement
amplitude and `--snapshots` the number of random-displacement supercells per grid
point (`--seed` for reproducibility). The MLP forces are stored raw, so the
analysis is blind to their MLP origin; {math}`U` still comes from the static
grid, and the per-point supercell / primitive matrices come from the reference
`phonopy_disp.yaml`:

```bash
% phonopy-anisotropic-qha-dataset phonopy_disp.yaml --from-mlp polymlp.yaml \
    --static-grid static-grid --distance 0.03 --snapshots 20 \
    -o aniso_qha_dataset.hdf5
# F_el is stored automatically when static-grid/grid-NNN/vaspout.h5 carries the
#   electron eigenvalues; pass --no-electronic to skip it
```

## 3. The intermediate dataset

```{warning}
The anisotropic QHA workflow is under active development. The
`aniso_qha_dataset.hdf5` layout and the `phonopy.qha.anisotropic_dataset` API
are not yet stable and may change in a backward-incompatible way between
releases. Rebuild the dataset from the calculator outputs rather than relying on
an old file being readable.
```

`aniso_qha_dataset.hdf5` is self-contained: per grid point it stores the relaxed
cell, supercell / primitive matrices, the raw displacements and forces, the
static internal energy {math}`U`, and optionally the electronic states. The
displacements and forces are kept in phonopy's native displacement-force dataset
form -- type-1 (one displaced atom per supercell) or type-2 (dense/random) --
tagged so the force-constant solver is chosen from the dataset type, not guessed.
Because the displacements and forces are stored raw (not force constants), the
file is independent of the force-constant method and can serve as an archive
after the calculator scratch is discarded. The same file feeds the analysis
whether the forces came from DFT or the MLP.

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
    mesh=100.0,
)

anisotropic_output.write_lattice_parameters_temperature(result)
anisotropic_output.write_axial_thermal_expansion(result)
anisotropic_output.write_volume_temperature(result)
plt = anisotropic_plot.plot_anisotropic_qha(result)
plt.savefig("anisotropic_qha.png")
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

## 5. Validate the MLP equilibrium shape (MLP route, recommended)

A smooth MLP is not automatically a correct one. Before trusting an MLP-phonon
result, validate against DFT at a few points:

- Compare the MLP vs DFT phonon anisotropy directly. With a full-DFT phonon grid
  available, a same-displacement force-swap comparison isolates any anisotropic
  Gruneisen error.
- Compare MLP and DFT stresses at a few cells (the stress is the free-energy
  gradient the QHA minimizes), and optionally elastic constants (the surface
  curvature).

If the MLP phonons and equilibrium shape agree with DFT within tolerance, the
dense (a, c) grid can be trusted.
