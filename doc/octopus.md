(octopus_interface)=
# Octopus & phonopy calculation

[Octopus](https://octopus-code.org/) is a real-space (grid based) TDDFT /
DFT code using pseudopotentials. This page explains how to calculate phonons
with phonopy using Octopus as the force calculator, i.e., using the finite
displacement and supercell approach.

## How the interface works

The Octopus interface differs from the other calculator interfaces in the way
the crystal structure is provided:

- The **unit cell** is read from a VASP-style `POSCAR` file (the default cell
  file name for `--octopus`). Lattice vectors in `POSCAR` are given in
  Angstrom, as usual; phonopy converts them internally to atomic units.
- The **supercells** are written by phonopy as Octopus *geometry include
  files* named `geometry-000`, `geometry-001`, `geometry-002`, ... These
  contain `%LatticeParameters`, `%LatticeVectors` and `%ReducedCoordinates`
  blocks in atomic units (bohr), ready to be pulled into an Octopus input file
  with the `include` directive.

Octopus uses atomic units, so within this interface the physical units are:

```
          | Distance   Atomic mass   Force         Force constants
-----------------------------------------------------------------
Octopus   | au (bohr)  AMU           hartree/au    hartree/au^2
```

The default displacement distance is `0.01` bohr. Phonon frequencies are
reported in THz, as for the other interfaces.

## Pre-process

The `POSCAR` of a bulk silicon primitive cell is used as an example here:

```
Si
1.0
  0.0000000000   2.7150000000   2.7150000000
  2.7150000000   0.0000000000   2.7150000000
  2.7150000000   2.7150000000   0.0000000000
Si
2
Direct
  0.0000000000   0.0000000000   0.0000000000
  0.2500000000   0.2500000000   0.2500000000
```

### Obtaining the `POSCAR` from an Octopus calculation

Often the unit cell is not available as a `POSCAR` but as an Octopus input
file, for instance after a geometry optimization with Octopus. In that case
Octopus can write the structure in `POSCAR` format directly: add the `geometry`
output in `POSCAR` format to the Octopus input file,

```
%Output
 geometry
%
OutputFormat = poscar
```

and run a ground-state calculation (`CalculationMode = gs`) on the (relaxed)
structure. Octopus writes the cell to `static/POSCAR`, with lattice vectors in
Angstrom and fractional atomic positions. Copy it to the working directory as
`POSCAR` and use it as the phonopy unit cell:

```bash
% cp static/POSCAR POSCAR
```

### Creating supercells with displacements

In the pre-process, supercell structures with (or without) displacements are
created from the unit cell, fully considering crystal symmetry.

To obtain supercells ({math}`2\times 2\times 2`) with displacements, run
`phonopy-init` with the {ref}`--octopus <force_calculators>` option:

```bash
% phonopy-init --octopus -d --dim 2 2 2 --pa auto
```

You should find the files `geometry-000`, `geometry-{number}` and
`phonopy_disp.yaml`:

```bash
% ls
geometry-000  geometry-001  phonopy_disp.yaml  POSCAR
```

`geometry-000` is the perfect supercell structure, `phonopy_disp.yaml`
contains the information on displacements, and `geometry-{number}` are the
supercells with atomic displacements. Each `geometry-{number}` corresponds to
one of the displacements written in `phonopy_disp.yaml`. Because of the high
symmetry of the diamond structure, only a single displacement
(`geometry-001`) is generated in this example.

A generated `geometry-001` looks like:

```
%LatticeParameters
 14.511546484 | 14.511546484 | 14.511546484
%

%LatticeVectors
  0.000000000 | 0.707106781 | 0.707106781
  0.707106781 | 0.000000000 | 0.707106781
  0.707106781 | 0.707106781 | 0.000000000
%

%ReducedCoordinates
  "Si" | 0.000689106 | 0.000000000 | -0.000000000
  "Si" | 0.500000000 | 0.000000000 | -0.000000000
  ...
%
```

## Calculation of sets of forces

For each `geometry-{number}` file an Octopus ground-state calculation
(`CalculationMode = gs`) is run to obtain the forces on the atoms of the
displaced supercell. The `geometry-{number}` file is pulled into the Octopus
input file `inp` with the `include` directive, together with the calculation
settings that are appropriate for your system. It is convenient to run each
displacement in its own directory:

```bash
% mkdir disp-001
% cd disp-001
```

An example `inp` for the silicon supercell (`disp-001/inp`) is:

```
CalculationMode = gs
PeriodicDimensions = 3
Spacing = 0.5*angstrom
BoxShape = parallelepiped

include ../geometry-001

%KPointsGrid
 2 | 2 | 2
%

ExtraStates = 4
```

Then run Octopus:

```bash
% octopus | tee out.log
```

Octopus writes the forces into `static/info`, in the "Forces on the ions"
block (in Hartree/bohr by default):

```
Forces on the ions [H/b]
 Ion                        x              y              z
   1        Si  -1.00521691E-05  -1.11410816E-03  -1.11410816E-03
   2        Si   1.34774669E-03   4.66640181E-05   4.66640168E-05
   ...
```

```{note}
Be careful not to relax the structures. The atomic forces induced by the
small displacement written in `geometry-{number}` are exactly what is needed
for the phonon calculation, so the supercells with displacements must not be
relaxed. Use a ground-state calculation (`CalculationMode = gs`,
no geometry optimization).
```

Since the calculation is a supercell calculation, the convergence parameters
(grid `Spacing`, `%KPointsGrid`, `ExtraStates`, exchange-correlation
functional, pseudopotentials, ...) have to be chosen for your system. The
settings above are only a minimal, fast example.

After the Octopus calculations of all displacements have finished, create the
`FORCE_SETS` file with the {ref}`-f <vasp_force_sets_option>` option, passing
the `static/info` files of the displacement calculations:

```bash
% phonopy-init -f disp-001/static/info
```

or, for several displacements,

```bash
% phonopy-init -f disp-{001..003}/static/info
```

The calculator (`octopus`) is read from `phonopy_disp.yaml`, so the
`--octopus` option is not needed again here.

## Post-process

The post-processing is identical to the other calculators: it reads
`phonopy_disp.yaml` and `FORCE_SETS`, so the `--octopus` option is not
required (the calculator is stored in `phonopy_disp.yaml`).

In the post-process,

1. Force constants are calculated from the sets of forces,
2. A part of the dynamical matrix is built from the force constants,
3. Phonon frequencies and eigenvectors are calculated from the dynamical
   matrices at the specified *q*-points.

The density of states (DOS) is plotted by

```bash
% phonopy --mesh 20 20 20 -p
```

Thermal properties are calculated with the sampling mesh by

```bash
% phonopy --mesh 20 20 20 -t
```

You should check the convergence with respect to the mesh numbers. Thermal
properties can be plotted by

```bash
% phonopy --mesh 20 20 20 -t -p
```

Projected DOS is calculated and plotted by

```bash
% phonopy --mesh 20 20 20 --pdos "1 2, 3 4 5 6" -p
```

Band structure is plotted by

```bash
% phonopy --band "0.5 0.5 0.5  0.0 0.0 0.0  0.5 0.5 0.0  0.0 0.5 0.0" -p
```

In either case, by setting the `-s` option, the plot is going to be saved in
the PDF format. If you don't need to plot the DOS, the (partial) DOS is just
calculated using the `--dos` option.

## Octopus phonon eigenmodes (Optional)

For use inside Octopus (e.g., to displace the atoms along a phonon mode), the
`phonopy-octopus-eigenmodes` command writes the phonon eigenmodes at the
commensurate *q*-points of the supercell into a file that Octopus can read:

```bash
% phonopy-octopus-eigenmodes --filename phonon_modes.txt
```

This reads `phonopy_disp.yaml` and `FORCE_SETS` from the current directory and
writes the frequencies (in Hartree) and eigenvectors to `phonon_modes.txt`.

## Non-analytical term correction (Optional)

To activate the non-analytical term correction, a {ref}`BORN <born_file>` file
is required. It contains the macroscopic dielectric constant and the Born
effective charges of the atoms in the primitive cell. Both quantities can be
obtained from Octopus by a linear-response (Sternheimer) calculation of the
electromagnetic response.

Unlike VASP (which has the `phonopy-vasp-born` helper), there is no automatic
`BORN`-file generator for Octopus, so the `BORN` file has to be assembled by
hand following the {ref}`BORN format <born_file>`. The Octopus output already
uses the units expected by phonopy (Born charges in units of the elementary
charge, dielectric tensor dimensionless), so the values can be transcribed
directly.

```{note}
The non-analytical term correction is only relevant for polar (ionic) crystals;
for a non-polar crystal such as the silicon used in the rest of this page the
Born effective charges vanish by symmetry and the correction has no effect. The
example below therefore uses rock-salt **NaCl**, independently of the silicon
force-calculation workflow above.
```

### Computing Born charges and the dielectric tensor with Octopus

The Born charges and the dielectric tensor are properties of the **unit cell**
(not the supercell), so this calculation is run on the unit-cell `POSCAR`. Here
we use the rock-salt NaCl primitive cell:

```
NaCl
1.0
  0.0000000000   2.8200000000   2.8200000000
  2.8200000000   0.0000000000   2.8200000000
  2.8200000000   2.8200000000   0.0000000000
Na Cl
1 1
Direct
  0.0000000000   0.0000000000   0.0000000000
  0.5000000000   0.5000000000   0.5000000000
```

For a periodic system the electric response is computed with the
{math}`\vec{k}\cdot\vec{p}` perturbation, so **three runs are performed in the
same directory**: a ground state, a `kdotp` calculation, and finally the
electromagnetic response. Each run restarts from the previous one.

The interface only writes *supercell* geometry files, so first convert the
`POSCAR` unit cell to an Octopus geometry include file with
`phonopy-calc-convert`:

```bash
% phonopy-calc-convert -i POSCAR -o geometry-unitcell --calcin vasp --calcout octopus
```

The three runs share a common structure/settings block, which we put in a file
`common.inp`:

```
PeriodicDimensions = 3
Spacing = 0.3*angstrom
BoxShape = parallelepiped
PseudopotentialSet = hgh_lda
include geometry-unitcell
%KPointsGrid
 4 | 4 | 4
%
KPointsUseSymmetries = no
ExperimentalFeatures = yes
ExtraStates = 8
```

1. Ground state:

   ```
   CalculationMode = gs
   include common.inp
   ```

2. {math}`\vec{k}\cdot\vec{p}` perturbation (required for the periodic electric
   response):

   ```
   CalculationMode = kdotp
   KdotPCalcSecondOrder = yes
   include common.inp
   ```

3. Electromagnetic response with Born charges:

   ```
   CalculationMode = em_resp
   RestartFixedOccupations = no
   include common.inp
   # Static (zero-frequency) response
   %EMFreqs
    1 | 0.0
   %
   EMCalcBornCharges = yes
   ```

Run Octopus once per step (replacing the `inp` file between runs).

```{note}
Several constraints apply to this (experimental) calculation, hence
`ExperimentalFeatures = yes`:
- **`kdotp` first.** For a periodic system the electric response reads the
  {math}`\vec{k}\cdot\vec{p}` wavefunctions, so the `kdotp` run must precede
  `em_resp`.
- **`RestartFixedOccupations = no`** in the `em_resp` run, so that occupations
  are recomputed with semiconducting smearing (a gap is required for the
  {math}`\vec{k}\cdot\vec{p}` electric response).
- **LDA only.** The Sternheimer linear response used by `em_resp` evaluates the
  XC kernel without its gradient terms, so it currently supports only LDA
  functionals; a GGA functional stops with "GGA functionals are not allowed for
  now in XCKernel". Use an LDA functional and matching pseudopotentials. (This
  restricts the `em_resp` path only; Octopus' Casida TDDFT does handle GGA
  kernels.)
- **No nonlinear core corrections.** The Born-charge force derivatives are not
  implemented for pseudopotentials with NLCC.

  The `hgh_lda` (Hartwigsen–Goedecker–Hutter LDA) set used above satisfies both
  the LDA and no-NLCC requirements. As with any response calculation, the grid
  spacing, k-point mesh and solver convergence have to be checked.
```

After the `em_resp` run, the results are written under `em_resp/freq_0.0000/`.

The macroscopic dielectric tensor {math}`\epsilon` is in
`em_resp/freq_0.0000/epsilon`:

```
# Real part of dielectric constant
        2.588022        0.000000       -0.000000
        0.000000        2.588022       -0.000000
       -0.000000       -0.000000        2.588022
Isotropic average        2.588022
...
```

The Born effective-charge tensors {math}`Z^*` are in
`em_resp/freq_0.0000/born_charges`, one {math}`3\times3` tensor per atom:

```
# (Frequency-dependent) Born effective charge tensors
Index:     1   Label:    Na   Ionic charge:     1.0000
        1.141794        0.000000       -0.000000
        0.000000        1.141794        0.000000
        0.000000       -0.000000        1.141794
Isotropic average        1.141794

Index:     2   Label:    Cl   Ionic charge:     7.0000
       -1.141794        0.000000        0.000000
        0.000000       -1.141794       -0.000000
       -0.000000        0.000000       -1.141794
Isotropic average       -1.141794

# Discrepancy of Born effective charges from acoustic sum rule before correction, per atom
       -0.025770        0.000000       -0.000000
        0.000000       -0.025770        0.000000
        0.000000       -0.000000       -0.025770
Isotropic average       -0.025770
```

Each tensor is printed as three rows of the matrix ({math}`xx\,xy\,xz` /
{math}`yx\,yy\,yz` / {math}`zx\,zy\,zz`). The `Isotropic average` lines and the
acoustic-sum-rule discrepancy block are not part of the `BORN` file.

The values are physically sensible: {math}`Z^*(\mathrm{Na}) \approx +1.14`,
{math}`Z^*(\mathrm{Cl}) \approx -1.14` and {math}`\epsilon_\infty \approx 2.59`,
close to the experimental NaCl values ({math}`Z^* \approx \pm 1.1`,
{math}`\epsilon_\infty \approx 2.3`). The settings above are only a small
example; check convergence for production use.

### Assembling the `BORN` file

Following the {ref}`BORN format <born_file>`, write:

1. the unit conversion factor on the first line (use the default by giving a
   non-numeric placeholder such as `default`, or the Octopus factor from
   {ref}`nac_default_value_interfaces`),
2. the nine dielectric-tensor components ({math}`xx\,xy\,xz\,yx\,yy\,yz\,zx\,zy\,zz`)
   from `epsilon` on the second line,
3. from the third line on, the nine {math}`Z^*` components for each
   symmetry-independent atom of the primitive cell, taken from the
   corresponding `Index:` block of `born_charges`. The symmetry-independent
   atoms can be identified from the `atom_mapping` section printed by
   `phonopy-init --octopus --symmetry -c POSCAR`; for NaCl the two atoms (Na and
   Cl) are inequivalent.

For the NaCl output above, the resulting `BORN` file is:

```
default
2.588022 0.0 0.0 0.0 2.588022 0.0 0.0 0.0 2.588022
1.141794 0.0 0.0 0.0 1.141794 0.0 0.0 0.0 1.141794
-1.141794 0.0 0.0 0.0 -1.141794 0.0 0.0 0.0 -1.141794
```

Once a `BORN` file is present in the current directory, the non-analytical term
correction is activated automatically in the post-process (e.g., with the
`--nac` option).

```{note}
Octopus can alternatively compute Born effective charges and infrared
intensities directly through its own linear-response phonon calculation
(`CalculationMode = vib_modes` with `CalcInfrared = yes`), which writes them to
`vib_modes/infrared`. That route is independent of the phonopy finite-
displacement workflow described here.
```
