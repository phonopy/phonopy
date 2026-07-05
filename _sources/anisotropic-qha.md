---
orphan: true
---

# Anisotropic QHA with a stress-trained pypolymlp -- end-to-end recipe

This recipe ties together the pieces that support anisotropic (axis-resolved)
quasi-harmonic thermal expansion driven by a machine-learning potential
(pypolymlp) trained on first-principles energies, forces and **stresses**.

Status: working recipe for an in-progress feature. Commands and functions
referenced here exist in the codebase; the exact per-cell settings
(internal-coordinate relaxation, number of snapshots, symfc options, energy
normalization) are case dependent and are flagged below.

Prerequisites: `pypolymlp`, `symfc`, `h5py`, and a VASP setup (VASP is the
supported first-principles source for the dataset builder).

All lengths are in the native length unit of the input cell (Angstrom for
VASP); no unit conversion is applied by the tools.

## Overview

```
                 phonopy-strain-cells --rd            VASP
 equilibrium cell -----------------------> training supercells ----> vasprun.xml
                                                                        |
                                        phonopy-vasp-mlp-dataset        |
 polymlp_dataset.hdf5  <-------------------------------------------------
        |
        | develop_pypolymlp_from_structures  (energies + forces + STRESS)
        v
 polymlp.yaml (trained MLP)
        |
        |            phonopy-strain-cells --a --c
 equilibrium cell -----------------------> (a, c) grid unit cells
        |                                          |
        |  per grid cell: load_mlp -> random disp -> evaluate_mlp -> force constants
        v                                          v
 internal_energies (U per primitive cell)     phonopys (one per grid point, FC set)
        \\                                        /
         \\        run_anisotropic_qha           /
          -------------------------------------->  AnisotropicQHAResult
                                                     |
                              anisotropic_output / anisotropic_plot
```

## 1. Generate the MLP training set (strained supercells)

Starting from an equilibrium `phonopy_disp.yaml` (which carries the unit cell,
supercell matrix and calculator), sample strained supercells with random
atomic displacements. Training on strained cells is what lets the MLP span the
(a, c) region used later; random displacements make relaxation unnecessary for
these training structures.

```bash
# Inspect the free lattice DOF first (no ranges -> DOF report):
phonopy-strain-cells phonopy_disp.yaml

# Then sample e.g. 100 random-displacement supercells spanning the region:
phonopy-strain-cells phonopy_disp.yaml --a 3.15 3.25 --c 5.10 5.30 \
    -n 100 --seed 1 --rd 0.03
# -> supercell-00001 .. supercell-00100 in the calculator-native format
```

## 2. Run the first-principles calculations

Run a single-point VASP calculation for each `supercell-*` (compute stress:
`ISIF >= 2` so `<varray name="stress">` is written). Collect the resulting
`vasprun.xml` files.

## 3. Build the training dataset (with stress)

```bash
phonopy-vasp-mlp-dataset vasprun-*.xml -o polymlp_dataset.hdf5
# Wrote N structures to polymlp_dataset.hdf5 (stress: yes).
```

The dataset stores, per structure, the cell, total energy (eV), forces
(eV/Angstrom) and stress (GPa). Stress is included only when every vasprun
provides it.

## 4. Train the MLP using energies, forces and stresses

```python
from phonopy.interface.pypolymlp import (
    PypolymlpParams,
    develop_pypolymlp_from_structures,
    read_pypolymlp_structure_dataset,
    save_pypolymlp,
)

data = read_pypolymlp_structure_dataset("polymlp_dataset.hdf5")

# Simple train/test split (see PypolymlpParams for model settings).
n_test = max(1, len(data.structures) // 10)
train, test = _split(data, len(data.structures) - n_test)  # your split helper

polymlp = develop_pypolymlp_from_structures(
    train, test, params=PypolymlpParams(), verbose=True
)
save_pypolymlp(polymlp, "polymlp.yaml")
```

`develop_pypolymlp_from_structures` uses pypolymlp's structure-based training
(`set_datasets_structures`), so the training structures may have different
lattices and the stress (virial) data are used in addition to energies and
forces. This is the key difference from the displacement-based
`develop_pypolymlp` / `Phonopy.develop_mlp`, which assume a single reference
lattice and no stress.

## 5. Build the (a, c) grid of unit cells

```bash
phonopy-strain-cells phonopy_disp.yaml --a 3.17 3.23 --c 5.14 5.26 \
    -n 25 --seed 2
# -> unitcell-00001 .. unitcell-00025 (the anisotropic QHA grid)
```

Relax the internal coordinates of each grid cell if the structure has free
internal parameters (e.g. the wurtzite `u`): this is the user's
responsibility. With a smooth MLP this relaxation is cheap.

## 6. Per grid cell: force constants from the MLP

For each grid unit cell, build a `Phonopy` instance whose force constants come
from the trained MLP, following phonopy's standard pypolymlp usage:

```python
import phonopy
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

# Take the supercell and primitive matrices from the equilibrium disp yaml so
# that every grid cell uses the SAME cell conventions (needed for consistent
# lattice-parameter and per-primitive-cell energy normalization).
eq = phonopy.load("phonopy_disp.yaml", produce_fc=False, is_nac=False, log_level=0)
SUPERCELL_MATRIX = eq.supercell_matrix
PRIMITIVE_MATRIX = eq.primitive_matrix

phonopys = []
internal_energies = []
for filename in grid_unitcell_files:
    cell = read_crystal_structure(filename)[0]
    ph = Phonopy(
        cell,
        supercell_matrix=SUPERCELL_MATRIX,
        primitive_matrix=PRIMITIVE_MATRIX,
    )
    ph.load_mlp("polymlp.yaml")

    # Random-displacement supercells; a small distance is numerically stable
    # for MLP-based force constants.
    ph.generate_displacements(distance=0.03, number_of_snapshots=NSNAP)
    ph.evaluate_mlp()                                    # forces from the MLP
    ph.produce_force_constants(fc_calculator="symfc")    # random-disp -> FC

    phonopys.append(ph)

    # Static energy U per primitive cell for run_anisotropic_qha. Evaluate the
    # perfect (undisplaced) supercell with the MLP and normalize to the
    # primitive cell (consistently with the phonon thermal-property
    # normalization).
    e_supercell = ph.mlp.evaluate([ph.supercell])[0]
    n_ratio = len(ph.supercell) / len(ph.primitive)
    internal_energies.append(e_supercell[0] / n_ratio)
```

Notes:
- `SUPERCELL_MATRIX` and `NSNAP` are user choices; the supercell matrix should
  match the one used for training-set generation. Pass the SAME
  `PRIMITIVE_MATRIX` to every grid cell; otherwise `run_anisotropic_qha`
  (which reads `ph.unitcell` lengths and `ph.primitive.volume`) and the
  per-primitive-cell energy normalization become inconsistent across the grid.
- Random-displacement force constants require the fitting fc calculator, so
  pass `fc_calculator="symfc"`.
- `internal_energies` and the phonon thermal properties must use the same
  per-primitive-cell normalization; `run_anisotropic_qha` expects energies in
  eV per primitive cell.
- This whole chain (train -> save -> load_mlp -> generate_displacements ->
  evaluate_mlp -> produce_force_constants -> run_anisotropic_qha) has been
  run end-to-end; a physically meaningful result of course requires an MLP
  trained on enough data over the relevant (a, c) region.

## 7. Run the anisotropic QHA

```python
import numpy as np
from phonopy import run_anisotropic_qha
from phonopy.qha import anisotropic_output, anisotropic_plot

temperatures = np.arange(0, 1001, 10.0)  # one extra point for finite diff
result = run_anisotropic_qha(
    phonopys,
    temperatures,
    internal_energies=internal_energies,
    mesh=100.0,
)

anisotropic_output.write_lattice_parameters_temperature(result)
anisotropic_output.write_axial_thermal_expansion(result)
plt = anisotropic_plot.plot_anisotropic_qha(result)
plt.savefig("anisotropic_qha.png")
```

`run_anisotropic_qha` detects the free lattice DOF from the input cells, fits
the free energy surface F(a, c; T) and minimizes it per temperature, giving
a(T), c(T) and the axial thermal expansions alpha_a, alpha_c.

## 8. Validate the MLP equilibrium shape (recommended)

A smooth MLP is not automatically a correct one. Before trusting the anisotropic
result, validate the MLP's equilibrium shape against DFT at a few points:

- Compare MLP-relaxed vs DFT-relaxed equilibrium `a`, `c` and especially the
  `c/a` ratio (the anisotropy is encoded there).
- Compare MLP and DFT stresses at a few cells (the stress is the free-energy
  gradient the QHA minimizes).
- Optionally compare elastic constants (the surface curvature).

If the MLP equilibrium shape agrees with DFT within tolerance, the dense (a, c)
grid can be trusted.
