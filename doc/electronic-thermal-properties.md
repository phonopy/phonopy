(electronic_thermal_properties)=

# Electronic thermal properties

In a metal, the electrons near the Fermi level are excited thermally, and
they add to the free energy, the entropy and the heat capacity of the
crystal. Phonopy computes these electronic contributions from the
eigenvalues of an electronic structure calculation. They can be added to
the phonon thermal properties in a quasi-harmonic calculation (see
{ref}`phonopy_qha`) or used on their own.

The free energy is that of Mermin's finite-temperature theory of the
electrons, evaluated in the fixed density-of-states approximation. The
eigenvalues are computed once, at the static lattice, and are kept the same
at every temperature. Only the occupation of the states changes with
temperature. The approximation is intended for metals. In an insulator the
chemical potential is in the band gap, and the electronic contributions are
negligible.

Mermin's theory is described in the following publications.

- N. D. Mermin, Phys. Rev. **137**, A1441 (1965).
- W. Kohn and L. J. Sham, Phys. Rev. **140**, A1133 (1965).
- R. G. Parr and W. Yang, *Density-Functional Theory of Atoms and Molecules*
  (Oxford University Press, 1989).

## Overview

The boxes are phonopy commands and API functions, the hexagon is the
calculator run, and the rounded nodes are input and intermediate data.

```{mermaid}
flowchart TD
    RUN{{"VASP static run<br/>one per crystal structure"}}
    RUN --> XML(["vasprun.xml-NN<br/>eigenvalues, k-point mesh, U"])

    XML --> EFE["phonopy-vasp-efe"]
    EFE --> FEV(["fe-v.dat, e-v.dat<br/>F_el(T) only"])
    FEV --> QHA["phonopy-qha --efe"]
    TP(["thermal_properties.yaml-NN"]) --> QHA

    XML --> ES["phonopy-vasp-efe --es"]
    ES --> H5(["electronic_states.hdf5"])
    H5 --> RQ["run_qha<br/>(electronic_structures=...)"]
    H5 --> API["compute_thermal_properties_by_tetrahedron"]
    API --> PROP(["F_el, S_el, C_el, mu, D(E_F)"])
```

The calculator is run once for each crystal structure, without atomic
displacements. The crystal structures have to be the same as those of the
phonon calculations. `vasprun.xml` of this run holds the eigenvalues, the k-point
mesh and the static total energy {math}`U`. For a calculator other than
VASP, build `ElectronicStates` from its output; see
{ref}`electronic_thermal_properties_input`.

`phonopy-vasp-efe` computes {math}`F_\mathrm{el}(T) - F_\mathrm{el}(0)` of
every crystal structure and writes it to `fe-v.dat`. It also writes the
volumes and {math}`U` to `e-v.dat`. `phonopy-qha --efe` adds these free
energies to the phonon free energies in `thermal_properties.yaml`.
`fe-v.dat` has only the free energy. For this reason,
`Cp-temperature_polyfit.dat` and `gruneisen-temperature.dat` of
`phonopy-qha` are computed without the electronic entropy and heat
capacity. See {ref}`phonopy_qha_efe_option`.

`phonopy-vasp-efe --es` writes the electronic states of every crystal
structure to `electronic_states.hdf5` instead. `run_qha` reads them as
`electronic_structures` and adds the free energy, the entropy and the heat
capacity of the electrons to those of the phonons. See
{ref}`phonopy_qha_electronic_structures`.

The same file can be used without a quasi-harmonic calculation.
`compute_thermal_properties_by_tetrahedron` computes
{math}`F_\mathrm{el}`, {math}`S_\mathrm{el}`, {math}`C_\mathrm{el}` and
{math}`\mu` of one crystal structure; see
{ref}`electronic_thermal_properties_api`.

`phonopy-anisotropic-qha` takes the electronic states from its own dataset,
`aniso_qha_dataset.hdf5`. See {ref}`aniso-thermal-expansion` for how the
dataset is made.

(electronic_thermal_properties_equations)=

## Equations

The states are the eigenvalues {math}`\epsilon_{\mathbf{k}i}` at the
k-points {math}`\mathbf{k}`, where {math}`i` runs over the bands. Each
k-point has a weight {math}`w_\mathbf{k}`, and the weights are normalized so
that {math}`\sum_\mathbf{k} w_\mathbf{k} = 1`. The occupation of a state is
the Fermi-Dirac distribution,

```{math}
f_{\mathbf{k}i} = \left\{ 1 + \exp\left[ \frac{\epsilon_{\mathbf{k}i} - \mu}{k_\mathrm{B} T} \right] \right\}^{-1},
```

where {math}`\mu` is the chemical potential and {math}`T` is the temperature.
The electrons in these states do not interact with each other. Their grand
potential is

```{math}
\Omega(T, \mu) = -g k_\mathrm{B} T \sum_\mathbf{k} w_\mathbf{k} \sum_i \ln \left\{ 1 + \exp\left[ -\frac{\epsilon_{\mathbf{k}i} - \mu}{k_\mathrm{B} T} \right] \right\}.
```

{math}`g` is the number of electrons that one eigenvalue holds; see
{ref}`electronic_thermal_properties_spin`.
The number of electrons in the cell is the derivative of the grand potential
with respect to the chemical potential,

```{math}
N = -\left( \frac{\partial \Omega}{\partial \mu} \right)_T = g \sum_\mathbf{k} w_\mathbf{k} \sum_i f_{\mathbf{k}i}.
```

The chemical potential depends on temperature. At each temperature it is
determined so that {math}`N` stays the same; see
{ref}`electronic_thermal_properties_chemical_potential`.
The entropy of the electrons is the derivative of the grand potential with
respect to temperature,

```{math}
S_\mathrm{el}(T) = -\left( \frac{\partial \Omega}{\partial T} \right)_\mu = -g k_\mathrm{B} \sum_\mathbf{k} w_\mathbf{k} \sum_i \left[ f_{\mathbf{k}i} \ln f_{\mathbf{k}i} + (1 - f_{\mathbf{k}i}) \ln (1 - f_{\mathbf{k}i}) \right].
```

The free energy at a fixed number of electrons is

```{math}
F_\mathrm{el}(T) = \Omega + \mu N.
```

Because {math}`\partial \Omega / \partial \mu = -N`, the entropy is also
{math}`S_\mathrm{el} = -(\partial F_\mathrm{el} / \partial T)_N`. The energy
of the electrons is {math}`E_\mathrm{el} = F_\mathrm{el} + T S_\mathrm{el}`,
and inserting {math}`\Omega`, {math}`N` and {math}`S_\mathrm{el}` gives

```{math}
E_\mathrm{el}(T) = g \sum_\mathbf{k} w_\mathbf{k} \sum_i f_{\mathbf{k}i} \epsilon_{\mathbf{k}i}.
```

The expressions for {math}`E_\mathrm{el}` and {math}`S_\mathrm{el}` are
Eqs. (11) and (12) of C. Wolverton and A. Zunger, Phys. Rev. B **52**, 8813
(1995). Phonopy evaluates the sums for {math}`E_\mathrm{el}` and
{math}`S_\mathrm{el}`, and obtains the free energy as {math}`F_\mathrm{el} =
E_\mathrm{el} - T S_\mathrm{el}`. This free energy is equal to
{math}`\Omega + \mu N`.

The heat capacity at constant volume is the temperature derivative of the
energy,

```{math}
C_\mathrm{el}(T) = \left( \frac{\partial E_\mathrm{el}}{\partial T} \right)_V.
```

Differentiating {math}`E_\mathrm{el}` gives two terms. One term comes from
the change of the occupations at a fixed chemical potential. The other term
comes from the change of the chemical potential with temperature, which is
fixed by the condition that {math}`N` does not change. Together they give

```{math}
C_\mathrm{el}(T) = \frac{1}{k_\mathrm{B} T^2} \left( A_2 - \frac{A_1^2}{A_0} \right),
\qquad
A_n = g \sum_\mathbf{k} w_\mathbf{k} \sum_i f_{\mathbf{k}i} (1 - f_{\mathbf{k}i}) (\epsilon_{\mathbf{k}i} - \mu)^n.
```

The {math}`A_1^2/A_0` term is the contribution of the change of the chemical
potential. It is small when the density of states is nearly constant near
the Fermi level, and it grows with the slope of the density of states there.
Phonopy computes the heat capacity from this expression.

Every sum above has the form {math}`g \sum_\mathbf{k} w_\mathbf{k} \sum_i
h(\epsilon_{\mathbf{k}i})` with a function {math}`h` of energy. Such a sum
is an integral over the electronic density of states per cell,

```{math}
D(E) = g \sum_\mathbf{k} w_\mathbf{k} \sum_i \delta(E - \epsilon_{\mathbf{k}i}),
\qquad
g \sum_\mathbf{k} w_\mathbf{k} \sum_i h(\epsilon_{\mathbf{k}i}) = \int D(E) h(E) \, dE.
```

The two ways of computing these sums, which are described in
{ref}`electronic_thermal_properties_integration`, start from the left-hand
side and from the right-hand side of this equation, respectively.

At low temperature, the entropy and the heat capacity are both linear in
{math}`T`,

```{math}
S_\mathrm{el} \approx C_\mathrm{el} \approx \gamma T = \frac{\pi^2}{3} k_\mathrm{B}^2 D(E_\mathrm{F}) T,
```

where {math}`D(E_\mathrm{F})` is the density of states at the Fermi level
and {math}`\gamma` is the Sommerfeld coefficient. The tetrahedron method
returns {math}`D(E_\mathrm{F})` as `dos_at_fermi_level`, evaluated at the
chemical potential at 0 K; see {ref}`electronic_thermal_properties_api`.

### The reference at 0 K

The free energy is reported as {math}`F_\mathrm{el}(T) - F_\mathrm{el}(0)`.
The total energy of the electronic structure calculation already contains
the energy of the electrons at 0 K. Adding {math}`F_\mathrm{el}(T) -
F_\mathrm{el}(0)` to that total energy adds only the part that depends on
temperature. In a quasi-harmonic calculation, the static energy
{math}`U(V)` is the total energy. The free energy at volume {math}`V`,
before the phonon free energy is added, is

```{math}
U(V) + F_\mathrm{el}(T; V) - F_\mathrm{el}(0; V).
```

Use the total energy extrapolated to zero smearing as {math}`U(V)`, such
as `energy(sigma->0)` of VASP.

(electronic_thermal_properties_integration)=

## Integration over the Brillouin zone

The sums over k-points above can be computed in two ways.

The linear tetrahedron method builds the electronic density of states from
the eigenvalues on the regular k-point mesh, and integrates the expressions
above over energy. This is the default when the states carry the k-point
mesh, the k-points and the crystal structure.

The k-point sum evaluates the sums as they are written, over the
irreducible k-points. It needs only the eigenvalues, the weights and the
number of electrons, so it also works for an explicit list of k-points. It
converges much more slowly with the number of k-points than the tetrahedron
method. Only the states within a few {math}`k_\mathrm{B} T` of the Fermi
level contribute to the entropy and the heat capacity, and on a coarse mesh
few eigenvalues fall in that range. The heat capacity at low temperature is
the most affected. Use the tetrahedron method whenever the k-point mesh is
available.

`phonopy-vasp-efe` uses the tetrahedron method for each `vasprun.xml` that
describes a regular k-point mesh. It uses the k-point sum for a file with an
explicit list of k-points, and for a file whose k-points cannot be mapped
onto the mesh. The `--k-point-sum` option makes it use the k-point sum for
every file. The command prints the method used for each file, and the
`# integration:` line in the header of `fe-v.dat` records it. `run_qha` uses
the k-point sum for the electronic states that do not carry `kpoints`,
`mesh` and `cell`. `phonopy-anisotropic-qha` stops with an error for such
states.

The tetrahedron method integrates only over an energy window from
{math}`E_\mathrm{F} - W` to {math}`E_\mathrm{F} + W`, where
{math}`E_\mathrm{F}` is the Fermi energy and {math}`W` is the half-width of
the window. The states below the window are fully occupied at every
temperature and add a constant to the energy, which cancels in
{math}`F_\mathrm{el}(T) - F_\mathrm{el}(0)`. The states above the window are
empty. By default, the half-width is

```{math}
W = \max(0.5\ \mathrm{eV}, 16 k_\mathrm{B} T_\mathrm{max}),
```

where {math}`T_\mathrm{max}` is the highest temperature. The width is set by
the heat capacity. Its integrand contains {math}`f(1-f)(E-\mu)^2`, which
decreases most slowly away from the Fermi level. The density of states is
sampled on an energy grid of 0.5 meV spacing in the window. The `window` and
`energy_spacing` parameters change {math}`W` and the spacing. The
`--electronic-window` and `--electronic-spacing` options of
`phonopy-vasp-efe` and `phonopy-anisotropic-qha` do the same.

At low temperature, the Fermi-Dirac distribution changes over an energy
range of a few {math}`k_\mathrm{B} T`. When {math}`k_\mathrm{B} T` is
smaller than the spacing, the energy grid does not resolve this change, and
the entropy and the heat capacity become inaccurate. The default spacing of
0.5 meV is equal to {math}`k_\mathrm{B} T` at about 6 K. For temperatures
below that, set the spacing to {math}`k_\mathrm{B} T_\mathrm{min}` or
smaller, where {math}`T_\mathrm{min}` is the lowest temperature above 0 K.

```{figure} electron-window.png

The energy window of the tetrahedron method, drawn for the case
{math}`W = 16 k_\mathrm{B} T_\mathrm{max}`. The black curve is the density
of states. The red curve is {math}`f(1-f)(E-\mu)^2` at the highest
temperature, drawn without the factor {math}`D(E)`. It decreases to nearly
zero at the edges of the window. The states below the window are fully
occupied at every temperature, and the states above the window are empty.
```

The `symmetrize_tetrahedra` parameter averages the tetrahedron weights over
the point group. The `--symmetrize-tetrahedra` option of `phonopy-vasp-efe`
does the same. See {ref}`migration_v5` for this option and its default in
the next major version.

(electronic_thermal_properties_chemical_potential)=

## Chemical potential

The chemical potential {math}`\mu` at temperature {math}`T` is the root of
the equation

```{math}
g \sum_\mathbf{k} w_\mathbf{k} \sum_i f_{\mathbf{k}i}(\mu, T) = N,
```

where {math}`N` is `n_electrons` of `ElectronicStates`. The left-hand side
increases with {math}`\mu`, so the equation has one root. Phonopy finds the
root by Brent's method at every temperature.

The k-point sum evaluates the left-hand side as it is written, over the
irreducible k-points. The root is searched for between the lowest and the
highest eigenvalue.

The tetrahedron method replaces the sum by an integral over the density of
states {math}`D(E)`,

```{math}
\int D(E) f(E; \mu, T) \, dE = N,
```

and solves it in two steps.

At 0 K, the chemical potential {math}`\mu_0` is the energy at which the
integrated density of states is equal to {math}`N`,

```{math}
\int_{-\infty}^{\mu_0} D(E) \, dE = N.
```

The tetrahedron method gives the integrated density of states as a
continuous function of energy, so {math}`\mu_0` does not depend on the
energy grid.

At a finite temperature, the integral is evaluated on the energy grid in the
window. The states below the window hold {math}`N_\mathrm{below}` electrons
at every temperature. This number is fixed by the condition at 0 K,

```{math}
N_\mathrm{below} = N - \int_{E_\mathrm{F} - W}^{\mu_0} D(E) \, dE.
```

The chemical potential at temperature {math}`T` is the root of

```{math}
N_\mathrm{below} + \int_{E_\mathrm{F} - W}^{E_\mathrm{F} + W} D(E) f(E; \mu, T) \, dE = N.
```

The window is centered at the Fermi energy reported by the electronic
structure calculation, `fermi_energy` of `ElectronicStates`. When it is not
given, the center is the energy up to which the eigenvalues hold
{math}`N` electrons. When the chemical potential at 0 K is outside the
window, phonopy stops with an error that asks to widen the window. In that
case, increase `window`.

(electronic_thermal_properties_spin)=

## Spin degeneracy

{math}`g` in the equations above is the number of electrons that one
eigenvalue holds.

- In a calculation without spin polarization, the eigenvalues have one spin
  channel and each eigenvalue holds two electrons, {math}`g = 2`.
- In a collinear spin-polarized calculation, the eigenvalues have two spin
  channels and each eigenvalue holds one electron, {math}`g = 1`.
- In a non-collinear calculation, the eigenvalues have one spin channel, but
  each spinor state holds one electron, {math}`g = 1`.

Phonopy infers {math}`g` from the number of spin channels. This inference is
wrong for a non-collinear calculation. For such a calculation, set
`spin_degeneracy=1` in `ElectronicStates`.

(electronic_thermal_properties_input)=

## Input: electronic states

The input is `ElectronicStates` of `phonopy.electron.states`, one per
crystal structure:

| Field | Content |
|---|---|
| `eigenvalues` | Eigenvalues in eV, with shape (spin, k-points, bands). |
| `weights` | Weights of the k-points, with shape (k-points,). They need not be normalized. |
| `n_electrons` | Number of electrons in the cell. |
| `fermi_energy` | Fermi energy in eV reported by the calculation. Optional. |
| `spin_degeneracy` | {math}`g`. Optional; inferred from the number of spin channels when not given. |
| `kpoints`, `mesh`, `cell` | The irreducible k-points in fractional coordinates, the k-point mesh, and the crystal structure. Optional; all three are needed for the tetrahedron method. |
| `volume`, `internal_energy` | The cell volume and the static total energy. Optional; used by the quasi-harmonic calculation. |

The results are per cell for which the eigenvalues were computed. The phonon
thermal properties of phonopy are per primitive cell. When the eigenvalues
are computed for a larger cell, such as the conventional cell of a centred
lattice, the results are scaled by the volume of the primitive cell divided
by the volume of that cell before they are added to the phonon thermal
properties. `run_qha` does not scale them. It checks that the volumes of the
electronic states are equal to the volumes of the primitive cells of the
phonons, so compute the eigenvalues for the primitive cell.
`phonopy-anisotropic-qha` takes the electronic states of the conventional
cell and scales them.

For VASP, `phonopy-vasp-efe` reads `vasprun.xml` files and writes the
electronic states of all of them to `electronic_states.hdf5`:

```
% phonopy-vasp-efe --es vasprun.xml-{00..10}
```

The k-points, the mesh and the crystal structure are stored when the
`vasprun.xml` describes a regular k-point mesh. Read the file with
`read_electronic_states_hdf5`, which returns a list of `ElectronicStates`.

(electronic_thermal_properties_api)=

## Python API

`compute_thermal_properties_by_tetrahedron` of `phonopy.electron.tetrahedron`
computes the thermal properties by the tetrahedron method:

```python
import numpy as np

from phonopy.electron.states import read_electronic_states_hdf5
from phonopy.electron.tetrahedron import compute_thermal_properties_by_tetrahedron

states = read_electronic_states_hdf5("electronic_states.hdf5")
temperatures = np.arange(0, 1001, 10.0)
properties = compute_thermal_properties_by_tetrahedron(states[0], temperatures)

properties.free_energy  # F_el(T) - F_el(0) in eV
properties.entropy  # S_el in eV/K
properties.heat_capacity  # C_el in eV/K
properties.chemical_potential  # mu in eV
properties.dos_at_fermi_level  # D(E_F) in states/eV
```

`dos_at_fermi_level` is one value, the density of states at the chemical
potential at 0 K. Each of the other fields is an array with one value per
temperature. The temperatures do not have to include 0 K; the value at 0 K
is computed anyway as the reference of the free energy.

The function takes three optional parameters.

- `window` is the half-width {math}`W` of the energy window in eV. The
  default is {math}`\max(0.5\ \mathrm{eV}, 16 k_\mathrm{B} T_\mathrm{max})`;
  see {ref}`electronic_thermal_properties_integration`.
- `energy_spacing` is the spacing of the energy grid in the window in eV.
  The default is 0.0005, that is, 0.5 meV.
- `symmetrize_tetrahedra` averages the tetrahedron weights over the point
  group. The default is `False`.

Choose `energy_spacing` from the lowest temperature above 0 K,
{math}`T_\mathrm{min}`. The entropy and the heat capacity are accurate when
the spacing is {math}`k_\mathrm{B} T_\mathrm{min}` or smaller. The default
of 0.5 meV is {math}`k_\mathrm{B} T` at about 6 K, so it is enough when
{math}`T_\mathrm{min}` is 6 K or higher. For temperatures from 1 K,
{math}`k_\mathrm{B} T_\mathrm{min}` is 0.086 meV, and a spacing of 0.05 meV
is enough:

```python
temperatures = np.arange(0, 301, 1.0)
properties = compute_thermal_properties_by_tetrahedron(
    states[0], temperatures, energy_spacing=0.00005
)
```

The tetrahedron method computes the density of states at every energy of
this grid. The window of width {math}`2W` holds {math}`2W` divided by the
spacing energies, so a finer spacing makes the computation take longer.

The values are per cell for which the eigenvalues were computed. The phonon
thermal properties of phonopy are in kJ/mol and J/K/mol per primitive cell.
To compare with them, first scale the values to the primitive cell as
described in {ref}`electronic_thermal_properties_input`. Then multiply the
free energy by `get_physical_units().EvTokJmol`, and the entropy and the
heat capacity by `get_physical_units().EvTokJmol * 1000`.
`get_physical_units` is in `phonopy.physical_units`.

`compute_thermal_properties_by_kpoint_sum` of `phonopy.electron.kpoint_sum`
takes the electronic states and the temperatures, and computes the thermal
properties by the k-point sum. It has no energy grid, so it takes none of
the three optional parameters. Both functions return `ElectronicThermalProperties`. The k-point sum
gives no density of states, and its `dos_at_fermi_level` is `None`.
