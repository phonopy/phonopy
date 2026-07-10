---
orphan: true
---

# Anisotropic QHA -- end-to-end recipe

This recipe computes the anisotropic (axis-resolved) quasi-harmonic thermal
expansion of a crystal by directly optimizing the lattice parameters on a grid,
rather than the 1D volume-path QHA. It covers two interchangeable phonon
sources that share the same downstream pipeline:

- **full-DFT phonons**: force sets from displaced supercells (a machine-learning
  potential is not required); and
- **MLP phonons**: force sets from a pypolymlp trained on first-principles
  energies, forces and stresses (cheap dense sampling).

Status: working recipe for an in-progress feature. The tools referenced here
exist except the one-command analysis CLI (`phonopy-anisotropic-qha`), which is
forthcoming; until it lands, the analysis is a short script shown in step 4.

Prerequisites: `h5py`, `symfc`, a VASP setup (VASP is the supported
first-principles source), and, for the MLP route, `pypolymlp`.

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

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

    SGRID -->|"route A"| PD["phonopy -d<br/>per relaxed cell"]
    PD --> DFTF{{"DFT forces"}}
    DFTF --> PGRID(["phonon-grid/grid-NNN<br/>disp-*"])

    EQ -->|"route B"| SCT["phonopy-strain-cells --rd"]
    SCT --> DFTT{{"DFT E / F / stress"}}
    DFTT --> MLPDS["phonopy-vasp-mlp-dataset"]
    MLPDS --> DEV["develop_pypolymlp"]
    DEV --> MLP(["polymlp.yaml"])

    SGRID --> BUILD["phonopy-aniso-qha-dataset"]
    PGRID -->|"--from-dft"| BUILD
    MLP -->|"--from-mlp"| BUILD
    BUILD --> DS(["aniso_qha_dataset.hdf5"])

    DS --> ANA["run_anisotropic_qha"]
    ANA --> RES(["a(T), c(T),<br/>alpha_a, alpha_c,<br/>F(a,c) maps"])
```

## 1. Build the static grid (internal energy U, both routes)

Sample strained unit cells over the free lattice DOF, then relax and run a
static single point for each with DFT. The static grid supplies {math}`U(a, c)`
and, optionally, the electronic states for {math}`F_\mathrm{el}`.

```bash
# Inspect the free lattice DOF first (no ranges -> DOF report):
phonopy-strain-cells phonopy_disp.yaml

# Sample e.g. 25 strained unit cells spanning the (a, c) grid (no --rd):
phonopy-strain-cells phonopy_disp.yaml --a 3.17 3.23 --c 5.14 5.26 \
    -n 25 --seed 2
# -> unitcell-00001 .. unitcell-00025
```

For each `unitcell-*`:

1. Relax the internal coordinates if the structure has free internal parameters
   (e.g. the wurtzite `u`). A crystal with no internal DOF (e.g. HCP) skips this
   -- the strained cell is already the relaxed cell.
2. Run a static single point. Use `ISIF >= 2` if you also want stress; write
   `vaspout.h5` if you want the electronic states for {math}`F_\mathrm{el}`.
3. Place the output in `static-grid/grid-NNN/` (one directory per grid point,
   containing `vaspout.h5` or `vasprun.xml`). The builder discovers grid points
   from these `grid-NNN` directories; no index file is needed.

## 2A. Route A -- full-DFT phonons (phonon grid)

For each relaxed static-grid cell, generate displaced supercells and compute
their forces with DFT.

```bash
# Per relaxed unit cell (its CONTCAR), build displaced supercells:
phonopy -d --dim "4 4 4" -c CONTCAR
# -> phonopy_disp.yaml + supercells; run VASP forces for each displacement.
```

Place the results as `phonon-grid/grid-NNN/` each containing the
`phonopy_disp.yaml` and the per-displacement subdirectories `disp-001/`,
`disp-002/`, ... (each with `vaspout.h5` or `vasprun.xml`). The grid indices
must match those under `static-grid/`.

Then build the intermediate dataset:

```bash
phonopy-aniso-qha-dataset phonopy_disp.yaml --from-dft \
    --static-grid static-grid --phonon-grid phonon-grid \
    -o aniso_qha_dataset.hdf5
# add --electronic to include F_el from static-grid vaspout.h5
```

The positional `phonopy_disp.yaml` is the equilibrium reference; it supplies the
free lattice DOF metadata (the supercell / primitive matrices for each point come
from that point's `phonon-grid` yaml).

## 2B. Route B -- MLP phonons (train once, then evaluate)

Train a pypolymlp on strained random-displacement supercells (this is a
*different* strain-cells use than step 1: random-displacement supercells over a
box that must **cover the static grid with margin**, since an MLP extrapolates
badly at its domain edges).

```bash
# Training structures: strained supercells with random displacements (--rd):
phonopy-strain-cells phonopy_disp.yaml --a 3.15 3.25 --c 5.10 5.30 \
    -n 100 --seed 1 --rd 0.03
# -> supercell-00001 .. supercell-00100 ; run a single-point VASP (ISIF >= 2)
#    for each, then assemble the dataset (stress included):
phonopy-vasp-mlp-dataset vasprun-*.xml -o polymlp_dataset.hdf5
```

Train the MLP with energies, forces and stresses (structure-based training, so
the varying lattices and the stress/virial are used):

```python
from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    develop_pypolymlp_from_structures,
    read_pypolymlp_structure_dataset,
    save_pypolymlp,
)

data = read_pypolymlp_structure_dataset("polymlp_dataset.hdf5")
n_test = max(1, len(data.structures) // 10)
train, test = _split(data, len(data.structures) - n_test)  # your split helper
polymlp = develop_pypolymlp_from_structures(
    train, test, params=PypolymlpParams(), verbose=True
)
save_pypolymlp(polymlp, "polymlp.yaml")
```

Then build the intermediate dataset. The MLP forces are evaluated at build time
and stored raw, so the analysis is blind to their MLP origin; {math}`U` still
comes from the DFT static grid:

```bash
phonopy-aniso-qha-dataset phonopy_disp.yaml --from-mlp polymlp.yaml \
    --static-grid static-grid --distance 0.03 --snapshots 20 \
    -o aniso_qha_dataset.hdf5
# add --electronic to include F_el from static-grid vaspout.h5
```

## 3. The intermediate dataset

`aniso_qha_dataset.hdf5` is self-contained: per grid point it stores the relaxed
cell, supercell / primitive matrices, the raw displacements and forces, the
static internal energy {math}`U`, and optionally the electronic states. Because
the
displacements and forces are stored raw (not force constants), the file is
independent of the force-constant method and can serve as an archive after the
calculator scratch is discarded. The same file feeds the analysis whether the
forces came from DFT or the MLP.

## 4. Run the anisotropic QHA

Run the analysis directly on the intermediate dataset:

```bash
phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
    --contour-temp 0 500 1000 --compare-vinet
```

This rebuilds one Phonopy per grid point (force constants from the stored
displacements and forces), runs `run_anisotropic_qha`, and writes
`lattice_parameters-temperature.dat`, `axial_thermal_expansion.dat`,
`volume-temperature.dat` and `anisotropic_qha.png`. With exactly two free
lattice DOF it also writes the `F(a, c)` contour maps; `--decompose-contours`
adds the U / F_ph / F_el / total panels and `--compare-vinet` adds a
volume-path cross-check. The electronic free energy is used automatically when
the dataset carries the electronic states (`--no-electronic` to ignore it).

Equivalently, drive `run_anisotropic_qha` from the API:

```python
import numpy as np
from phonopy import Phonopy, run_anisotropic_qha
from phonopy.qha import anisotropic_output, anisotropic_plot
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

dataset = read_aniso_qha_dataset("aniso_qha_dataset.hdf5")

phonopys = []
internal_energies = []
electronic_structures = []
for point in dataset.grid_points:
    ph = Phonopy(
        point.cell,
        supercell_matrix=point.supercell_matrix,
        primitive_matrix=point.primitive_matrix,
        log_level=0,
    )
    ph.dataset = {"displacements": point.displacements, "forces": point.forces}
    ph.produce_force_constants(fc_calculator="symfc")
    phonopys.append(ph)
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
