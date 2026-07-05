# Anisotropic QHA for HCP Ti with a stress-trained pypolymlp

Axis-resolved quasi-harmonic thermal expansion (`alpha_a`, `alpha_c`) of
hexagonal-close-packed titanium, obtained by directly optimizing the lattice
parameters `a` and `c` at each temperature instead of the 1D volume path.

The energy/force/stress source is a machine-learning potential (pypolymlp)
trained on VASP energies, forces and **stresses** of strained supercells. A
smooth analytic potential regularizes the first-principles noise and makes a
dense `(a, c)` grid affordable.

For the general design and the reference recipe see
`doc/anisotropic-qha.md`. This directory is a concrete HCP Ti walk-through.

## System

- HCP Ti, space group `P6_3/mmc` (No. 194), 2 atoms per primitive cell.
- Equilibrium `a = 2.8949`, `c = 4.5774` angstrom (`c/a = 1.581`).
- Supercell matrix `diag(4, 4, 2)`; primitive matrix identity.
- No free internal coordinate (atoms on the fixed 2c site), so grid cells
  need no internal-coordinate relaxation.

All lengths are in angstrom (VASP native unit); no unit conversion is applied.

## Pipeline

```
                 phonopy-strain-cells --rd          VASP (ISIF >= 2)
 phonopy_disp.yaml ----------------------> supercell-00001.. ----> vasprun.xml
                       (strain_cells.yaml)                             |
                                        phonopy-vasp-mlp-dataset       |
 polymlp_dataset.hdf5 <------------------------------------------------
        |
        |  train_mlp.py   (energies + forces + STRESS)
        v
 polymlp.yaml
        |
        |  run_ti_anisotropic_qha.py
        |    (a, c) grid -> load_mlp -> random disp -> symfc FC + U
        v
 lattice_parameters-temperature.dat, axial_thermal_expansion.dat,
 volume-temperature.dat, anisotropic_qha.png
```

## Files

- `train_mlp.py` -- train a stress-enabled pypolymlp from
  `polymlp_dataset.hdf5` and write `polymlp.yaml`.
- `run_ti_anisotropic_qha.py` -- build the `(a, c)` grid, compute force
  constants per grid cell from the MLP, and run the anisotropic QHA.

## Step 1 -- generate the MLP training supercells

Start from an equilibrium `phonopy_disp.yaml` (unit cell + supercell matrix +
calculator). Sample strained supercells with random atomic displacements;
training on strained cells is what lets the MLP span the `(a, c)` region, and
random displacements make relaxation unnecessary for these structures.

```bash
# Inspect the free lattice DOF (no ranges -> DOF report):
phonopy-strain-cells phonopy_disp.yaml

# Sample 200 random-displacement supercells over a +/-1% (a, c) box:
phonopy-strain-cells phonopy_disp.yaml \
    --a 2.880405 2.938595 --c 4.55280 4.64477 -n 200 --rd 0.03
# -> supercell-00001 .. supercell-00200 (VASP format)
# -> strain_cells.yaml (provenance: resolved seed, ranges, per-cell a, c)
```

The command writes `strain_cells.yaml`, which records the **resolved random
seed** and the sampled `a, c` of every cell. Replay the run reproducibly with
`--seed <seed from strain_cells.yaml>`.

Do not over-tighten the box, especially in the `c/a` direction: the anisotropic
deviation from the volume path is the physics of interest. An optional
volume-path QHA (`phonopy-qha` / `run_qha`) beforehand gives a bearing on the
`(a, c)` region to sample.

## Step 2 -- first-principles calculations

Run a single-point VASP calculation for each `supercell-*`. Compute stress
(`ISIF >= 2`, so `<varray name="stress">` is written); the stress is the
free-energy gradient the anisotropic method minimizes, so it directly
constrains the equilibrium `a`, `c` and `c/a`. Collect the `vasprun.xml` files.

## Step 3 -- build the training dataset (with stress)

```bash
phonopy-vasp-mlp-dataset vasprun-*.xml -o polymlp_dataset.hdf5
# Wrote N structures to polymlp_dataset.hdf5 (stress: yes).
```

Each structure stores the cell, total energy (eV), forces (eV/angstrom) and
stress (GPa). Stress is included only when every vasprun provides it.

## Step 4 -- train the MLP

```bash
python train_mlp.py polymlp_dataset.hdf5 --test-ratio 0.1 --seed 0
# -> polymlp.yaml
```

`train_mlp.py` splits the dataset into train/test and calls
`develop_pypolymlp_from_structures`, which uses pypolymlp's structure-based
training so the structures may have different lattices and the stress (virial)
data are used in addition to energies and forces. Adjust the MLP settings in
`PypolymlpParams` inside the script as needed.

## Steps 5-7 -- run the anisotropic QHA

```bash
python run_ti_anisotropic_qha.py phonopy_disp.yaml --mlp polymlp.yaml \
    --num 5 --margin 1.0 --distance 0.03 --snapshots 4 --tmax 1000 --dt 10
```

This builds a regular `(a, c)` grid around the equilibrium cell (default 5x5 =
25 cells, `a`, `c` each within +/-`--margin` percent), and for each grid cell:
loads the MLP, generates random-displacement supercells, evaluates forces, and
fits force constants with symfc. It then runs `run_anisotropic_qha`, which
fits the free energy `F(a, c; T)` per temperature and minimizes it.

Outputs:

- `lattice_parameters-temperature.dat` -- `a(T)`, `b(T)`, `c(T)`
- `axial_thermal_expansion.dat` -- `alpha_a`, `alpha_b`, `alpha_c` and their sum
- `volume-temperature.dat` -- `V(T)`
- `anisotropic_qha.png` -- summary plot

Enough grid points are needed to fit the surface polynomial (default total
degree 3: at least `C(2 + 3, 3) = 10` points for 2 DOF); 25 gives margin.
One extra temperature point is consumed by the finite differences, so the
grid runs to `tmax + dt`.

## Step 8 -- validate the MLP equilibrium shape (recommended)

A smooth MLP is not automatically a correct one. Before trusting the result,
validate against DFT at a few points:

- Compare MLP-relaxed vs DFT-relaxed `a`, `c` and especially `c/a`.
- Compare MLP and DFT stresses at a few cells.
- Optionally compare elastic constants (the surface curvature).

If the MLP equilibrium shape agrees with DFT within tolerance, the dense
`(a, c)` grid can be trusted.

## Notes

- `polymlp.yaml` is provided in this example; the VASP training set is large
  and is hosted separately (see the data repository link, when available).
- Reproducibility: `phonopy-strain-cells` records its resolved seed in
  `strain_cells.yaml`; `train_mlp.py` and `run_ti_anisotropic_qha.py` take a
  `--seed`.
