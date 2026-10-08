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
at every temperature. Only the occupation
of the states changes with temperature. The approximation is intended for
metals. In an insulator the chemical potential is in the band gap, and the
electronic contributions are negligible.

(electronic_thermal_properties_equations)=

## Equations

The states are the eigenvalues {math}`\epsilon_{\mathbf{k}i}` at the
k-points {math}`\mathbf{k}`, where {math}`i` runs over the bands. Each
k-point has a weight {math}`w_\mathbf{k}`, and the weights are normalized so
that {math}`\sum_\mathbf{k} w_\mathbf{k} = 1`. The occupation of a state is
the Fermi-Dirac distribution,

```{math}
f_{\mathbf{k}i} = \left\{ 1 + \exp\left[
\frac{\epsilon_{\mathbf{k}i} - \mu}{k_\mathrm{B} T} \right] \right\}^{-1},
```

where {math}`\mu` is the chemical potential and {math}`T` is the
temperature.

The chemical potential depends on temperature. At each temperature it is
determined so that the number of electrons in the cell, {math}`N`, stays the
same:

```{math}
N = g \sum_\mathbf{k} w_\mathbf{k} \sum_i f_{\mathbf{k}i}.
```

{math}`g` is the number of electrons that one eigenvalue holds; see
{ref}`electronic_thermal_properties_spin`.

The energy and the entropy of the electrons are

```{math}
E_\mathrm{el}(T) = g \sum_\mathbf{k} w_\mathbf{k} \sum_i
f_{\mathbf{k}i} \epsilon_{\mathbf{k}i},
```

```{math}
S_\mathrm{el}(T) = -g k_\mathrm{B} \sum_\mathbf{k} w_\mathbf{k} \sum_i
\left[ f_{\mathbf{k}i} \ln f_{\mathbf{k}i}
+ (1 - f_{\mathbf{k}i}) \ln (1 - f_{\mathbf{k}i}) \right],
```

and the free energy is {math}`F_\mathrm{el}(T) = E_\mathrm{el}(T) - T
S_\mathrm{el}(T)`. These are Eqs. (11) and (12) of C. Wolverton and A.
Zunger, Phys. Rev. B **52**, 8813 (1995).

The heat capacity at constant volume is
{math}`C_\mathrm{el} = dE_\mathrm{el}/dT`. Differentiating
{math}`E_\mathrm{el}` gives two terms. One term comes from the change of the
occupations at a fixed chemical potential. The other term comes from the
change of the chemical potential with temperature, which is fixed by the
condition that {math}`N` does not change. Together they give

```{math}
C_\mathrm{el}(T) = \frac{1}{k_\mathrm{B} T^2}
\left( A_2 - \frac{A_1^2}{A_0} \right),
\qquad
A_n = g \sum_\mathbf{k} w_\mathbf{k} \sum_i
f_{\mathbf{k}i} (1 - f_{\mathbf{k}i}) (\epsilon_{\mathbf{k}i} - \mu)^n.
```

The {math}`A_1^2/A_0` term is the contribution of the change of the chemical
potential. It is small when the density of states is nearly constant near
the Fermi level, and it grows with the slope of the density of states there.
Phonopy computes the heat capacity from this expression.

At low temperature, the entropy and the heat capacity are both linear in
{math}`T`,

```{math}
S_\mathrm{el} \approx C_\mathrm{el} \approx \frac{\pi^2}{3}
k_\mathrm{B}^2 T D(E_\mathrm{F}),
```

where {math}`D(E_\mathrm{F})` is the density of states at the Fermi level
per cell, counted for both spins.

### The reference at 0 K

The free energy is reported as {math}`F_\mathrm{el}(T) - F_\mathrm{el}(0)`.
The total energy of the electronic structure calculation already contains
the energy of the electrons at 0 K. Adding {math}`F_\mathrm{el}(T) -
F_\mathrm{el}(0)` to that total energy adds only the part that depends on
temperature. In a quasi-harmonic calculation, the static energy
{math}`U(V)` is the total energy, and the electronic free energy at volume
{math}`V` is

```{math}
U(V) + F_\mathrm{el}(T; V) - F_\mathrm{el}(0; V).
```

Use the total energy extrapolated to zero smearing as {math}`U(V)`, such
as `energy(sigma->0)` of VASP.

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

The tetrahedron method integrates only over an energy window around the
Fermi level. The states below the window are fully occupied at every
temperature and add a constant to the energy, which cancels in
{math}`F_\mathrm{el}(T) - F_\mathrm{el}(0)`. The states above the window are
empty. By default, the half-width of the window is 16
{math}`k_\mathrm{B} T` at the highest temperature, and at least 0.5 eV. The
width is set by the heat capacity, whose integrand
{math}`f(1-f)(\epsilon-\mu)^2` decays most slowly away from the Fermi level.
The density of states is sampled on an energy grid of 0.5 meV spacing in the
window. The `window` and `energy_spacing` parameters change these values.

The `symmetrize_tetrahedra` parameter averages the tetrahedron weights over
the point group. See {ref}`migration_v5` for this option and its default in
the next major version.

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

The results are per cell for which the eigenvalues were computed.

For VASP, `phonopy-vasp-efe` reads `vasprun.xml` files and writes the
electronic states of all of them to `electronic_states.hdf5`:

```
% phonopy-vasp-efe --es vasprun.xml-{00..10}
```

The k-points, the mesh and the crystal structure are stored when the
`vasprun.xml` describes a regular k-point mesh. Read the file with
`read_electronic_states_hdf5`, which returns a list of `ElectronicStates`.

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
```

Each field is an array with one value per temperature. The temperatures do
not have to include 0 K; the value at 0 K is computed anyway as the
reference of the free energy. To compare with the phonon thermal properties
of phonopy, which are in J/K/mol, multiply the entropy and the heat capacity
by `get_physical_units().EvTokJmol * 1000` from `phonopy.physical_units`.

`compute_thermal_properties_by_kpoint_sum` of `phonopy.electron.kpoint_sum`
takes the same arguments and computes the thermal properties by the k-point
sum. Both functions return `ElectronicThermalProperties`.

## Use in quasi-harmonic calculations

The electronic contributions enter a quasi-harmonic calculation in three
ways:

- `run_qha` takes the electronic states of all volumes as
  `electronic_structures`. It adds the free energy, the entropy and the
  heat capacity of the electrons to those of the phonons. See
  {ref}`phonopy_qha_electronic_structures`.
- `phonopy-vasp-efe` without `--es` writes the electronic free energies of
  all volumes to `fe-v.dat`, which `phonopy-qha --efe` reads. This file has
  only the free energy, so the entropy and the heat capacity of the
  electrons are not included in every output of `phonopy-qha`. See
  {ref}`phonopy_qha_efe_option`.
- `phonopy-anisotropic-qha` adds the electronic free energy when its dataset
  has the electronic states. It needs the k-point mesh with the states. See
  {ref}`aniso-thermal-expansion`.

In all three, the electronic states have to be computed for the same
crystal structures as the phonons.
