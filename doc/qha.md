(phonopy_qha)=

# Quasi harmonic approximation

This page describes how to calculate the thermal expansion and the heat
capacity at constant pressure (Cp) of a crystal with phonopy. The calculation
needs a phonon calculation at each of several volumes of the crystal.

(phonopy_qha_method)=

## Gibbs free energy and thermal expansion

At each volume, the phonons are calculated in the harmonic approximation. The
phonon frequencies change with the volume, so the phonon Helmholtz free energy
{math}`F_\mathrm{phonon}(T;\,V)` depends on both the temperature {math}`T` and
the volume {math}`V`. When the phonons are the only source of the temperature
dependence, the Gibbs free energy at the temperature {math}`T` and the pressure
{math}`p` is approximated by

```{math}
G(T, p) = \min_V \left[ U(V) + F_\mathrm{phonon}(T;\,V) + pV \right],
```

where {math}`U(V)` is the total energy of the static lattice at the volume
{math}`V`. {math}`U(V)` does not depend on the temperature. The temperature
dependence comes from {math}`F_\mathrm{phonon}(T;\,V)`. {math}`\min_V` means the
minimum of the function in the brackets with respect to {math}`V`. At each
temperature, the values of the function in the brackets at the calculated
volumes are fitted to an equation of state. The volume at the minimum of this
equation of state is the equilibrium volume {math}`V_0(T)`.

The volume dependence of {math}`F_\mathrm{phonon}` is different from that of
{math}`U`. Therefore {math}`V_0` is different from the volume at the minimum of
{math}`U(V) + pV` even at 0 K. With increasing temperature, the volume
dependence of {math}`F_\mathrm{phonon}` changes, and {math}`V_0(T)` changes with
it. The volumetric thermal expansion coefficient is calculated from this change
as {math}`\beta = (1/V_0)\, dV_0/dT`.

On this page, this calculation is called the quasi-harmonic approximation
(QHA). The phonons are harmonic at each volume, and the temperature changes the
phonons only through the change of the volume. The term is used for a wider
range of calculations in the literature. {ref}`aniso-thermal-expansion` uses the
same approximation, and it minimizes the free energy over the lattice
parameters instead of the volume.

(phonopy_qha_calculations_at_each_volume)=

## Calculations at each volume

At each volume, phonopy calculates the thermal properties
({ref}`thermal_properties_tag`) and writes them to `thermal_properties.yaml`.
The calculator that gives the forces, such as VASP, also gives the total
energy of the static lattice at each volume. The fitting to the equation of state needs at
least five volumes. {ref}`workflow` shows how the calculations at the different
volumes are combined.

The `phonopy-qha` command reads the `thermal_properties.yaml` files and a file
of volumes and total energies of the static lattice. The Python API `run_qha`
({ref}`phonopy_qha_python_api`) takes one `Phonopy` instance per volume. It also
calculates the lattice parameters of the unit cell of each instance as functions
of temperature.

(phonopy_qha_electronic_free_energy)=

## Electronic free energy

In a metal, the electrons near the Fermi level are excited thermally. The free
energy of these electrons depends on the temperature and the volume, and it
changes the equilibrium volume, the thermal expansion coefficient and the heat
capacity. In an insulator whose band gap is much larger than
{math}`k_\mathrm{B} T`, few electrons are excited, and this section can be
skipped.

The electronic free energy is added to the function in the brackets:

```{math}
G(T, p) = \min_V \left[ U(V) + F_\mathrm{el}(T;\,V) - F_\mathrm{el}(0;\,V)
+ F_\mathrm{phonon}(T;\,V) + pV \right].
```

Phonopy calculates {math}`F_\mathrm{el}(T;\,V)` in the fixed density-of-states
approximation. The electronic eigenvalues are calculated once at each volume,
and only the occupation of the states changes with temperature.
{math}`F_\mathrm{el}(0;\,V)` is subtracted because {math}`U(V)` already
contains the energy of the electrons at 0 K. For this reason, {math}`U(V)` has
to be the total energy at zero electronic temperature, such as
`energy(sigma->0)` of VASP. The equations are in
{ref}`electronic_thermal_properties`.

The electronic eigenvalues have to be calculated for the same cell as the one
to which the phonon thermal properties are normalized. When phonopy was run
with `PRIMITIVE_AXES` or `--pa`, this cell is the primitive cell. With a
different cell, the electronic and the phonon free energies refer to different
numbers of atoms.

For VASP, `phonopy-vasp-efe` reads the `vasprun.xml` of each volume. Its output
depends on which of the following is used:

- `phonopy-qha --efe` reads `fe-v.dat` written by `phonopy-vasp-efe`. This file
  has {math}`U(V) + F_\mathrm{el}(T;\,V) - F_\mathrm{el}(0;\,V)` at each
  temperature and volume. See {ref}`phonopy_qha_efe_option`.
- `run_qha` reads `electronic_states.hdf5` written by `phonopy-vasp-efe --es`.
  This file has the electronic eigenvalues, and `run_qha` calculates the free
  energy, the entropy and the heat capacity of the electrons from them. See
  {ref}`phonopy_qha_electronic_structures`.

`fe-v.dat` has only the free energies. For this reason, the quantities that
`phonopy-qha` calculates from the entropy and the heat capacity at each volume
do not include the electrons. The following table shows which results include
the electronic contribution.

| Result                                                | `phonopy-qha --efe`                         | `run_qha`                    |
| ----------------------------------------------------- | ------------------------------------------- | ---------------------------- |
| {math}`V_0(T)`, {math}`\beta`, {math}`G`, {math}`B_T` | included                                    | included                     |
| {math}`C_p` from {math}`-T\partial^2 G/\partial T^2`  | included (`Cp-temperature.dat`)             | not calculated               |
| {math}`C_p` from the polynomial fits                  | phonons only (`Cp-temperature_polyfit.dat`) | included (`heat_capacity_P`) |
| Thermodynamic Grüneisen parameter                     | phonons only (`gruneisen-temperature.dat`)  | included                     |
| {math}`S(T, p)`, {math}`H(T, p)`                      | not written                                 | included                     |

`phonopy-qha` will not be changed to include the electronic entropy and heat
capacity, because the command is planned to be removed in a future version.
When the electronic contribution to {math}`C_p` or to the Grüneisen parameter
is needed, use `run_qha`. With `phonopy-qha --efe`, use `Cp-temperature.dat`
for {math}`C_p`.

## The `phonopy-qha` command

```{note}
The `phonopy-qha` command is planned to be removed in a future version. For
new calculations, use the Python API `run_qha` ({ref}`phonopy_qha_python_api`).
`phonopy-qha` does not include the electronic entropy and heat capacity; see
{ref}`phonopy_qha_electronic_free_energy`.
```

`phonopy-qha` is the script to run the fitting and calculations. Mind that at
least 5 volume points are needed for the fitting.

An example of the usage for `example/Si-QHA` is as follows.

To watch selected plots:

```
% phonopy-qha -p e-v.dat thermal_properties.yaml-{-{5..1},{0..5}}
```

```{figure} Si-QHA.png

```

Without plots:

```
% phonopy-qha e-v.dat thermal_properties.yaml-{-{5..1},{0..5}}
```

The first argument is the filename of volume-energy data (in the above example,
`e-v.dat`). The volumes and energies are given in {math}`\text{Angstrom}^3` and
eV, respectively. These energies are only dependent on volume but not on
temperature unless using `--efe` option. Therefore in the simplest case, these
are taken as the total energies of the static lattice. An example of the
volume-energy file is:

```
#   cell volume   energy of cell other than phonon
     140.030000           -42.132246
     144.500000           -42.600974
     149.060000           -42.949142
     153.720000           -43.188162
     158.470000           -43.326751
     163.320000           -43.375124
     168.270000           -43.339884
     173.320000           -43.230619
     178.470000           -43.054343
     183.720000           -42.817825
     189.070000           -42.527932
```

Lines starting with `#` are ignored.

The following arguments of `phonopy-qha` are the filenames of
`thermal_properties.yaml`'s calculated at the volumes given in the volume-energy
file. These filenames have to be ordered in the same order as the volumes
written in the volume-energy file. Since the volume v.s. free energy fitting is
done at each temperature given in `thermal_properties.yaml`, all
`thermal_properties.yaml`'s have to be calculated in the same temperature ranges
and with the same temperature step. `phonopy-qha` can calculate thermal
properties at constant pressure up to the temperature point that is one point
less than that in `thermal_properties.yaml` because of the numerical
differentiation with respect to temperature points. Therefore
`thermal_properties.yaml` has to be calculated up to higher temperatures than
that expected by `phonopy-qha`.

Another example for Aluminum is found in the `example/Al-QHA` directory.

If the condition under pressure is expected, {math}`PV` terms may be included in
the energies, or equivalent effect is applied using `--pressure` option.

The electronic free energy is included with the `--efe` option; see
{ref}`phonopy_qha_electronic_free_energy`.

(phonopy_qha_options)=

### Options

#### `-h`

Show help. The available options are shown. Without any option, the results are
saved into text files in simple data format.

#### `--tmax`

The maximum temperature calculated is specified. This temperature has to be
lower than the maximum temperature calculated in `thermal_properties.yaml` to
let at least one temperature points fewer. The default value is `--tmax=1000`.

#### `--pressure`

Pressure is specified in GPa. This corresponds to the {math}`pV` term in the
equation of {math}`G(T, p)` at the top of this page. Note that bulk modulus obtained
with this option than 0 GPa is incorrect.

#### `-b`

Fitting volume-energy data to an EOS, and show bulk modulus (without considering
phonons). This is made by:

```
% phonopy-qha -b e-v.dat
```

#### `--eos`

EOS is chosen among `vinet`, `birch_murnaghan`, and `murnaghan`. The default EOS
is `vinet`.

```
% phonopy-qha --eos='birch_murnaghan' -b e-v.dat
```

#### `-p`

The fitting results, volume-temperature relation, and thermal expansion
coefficient are plotted on the display.

#### `-s`

The calculated values are written into files.

#### `--sparse`

This is used with `-s` or `-p` to thin out the number of plots of the fitting
results at temperatures. For example with `--sparse=10`, 1 in 10 temperature
curves is only plotted.

(phonopy_qha_efe_option)=

#### `--efe`

**Experimental**

Temperature dependent energies other than phonon free energy are included with
this option. This is used such as:

```
% phonopy-qha -p --tmax=1300 --efe fe-v.dat e-v.dat thermal_properties.yaml-{00..10}
```

```{figure} Cu-QHA.png

```

The temperature dependent energies are stored in `fe-v.dat`. The file format is:

```
# volume:       43.08047896     43.97798894     44.87549882     45.77300889     46.67051887     47.56802885     48.46553883     49.36304881     50.26055878     51.15806876     52.05557874
#    T(K)     Free energies
    0.0000     -17.27885993    -17.32227490    -17.34336569    -17.34479760    -17.32843604    -17.29673896    -17.25081954    -17.19263337    -17.12356816    -17.04467997    -16.95752155
   10.0000     -17.27886659    -17.32228126    -17.34337279    -17.34481060    -17.32844885    -17.29675204    -17.25083261    -17.19264615    -17.12358094    -17.04469309    -16.95753464
   20.0000     -17.27887453    -17.32228804    -17.34338499    -17.34482383    -17.32846353    -17.29676491    -17.25084547    -17.19265900    -17.12359399    -17.04470709    -16.95754774
...
```

The first column gives temperatures in K and the following columns give the
energies other than the phonon free energy in eV, at temperatures and at unit
(primitive) cell volumes. For the electronic free energy, each value is
{math}`U(V) + F_\mathrm{el}(T; V) - F_\mathrm{el}(0; V)`, where {math}`U(V)`
is the static energy. The lines starting with `#` are ignored. This file
doesn't contain the information about cell volumes. Instead, the volumes are
obtained from `e-v.dat` file. The energies in `e-v.dat` are not used when
`--efe` option is used, because `fe-v.dat` already contains {math}`U(V)`. The
temperature points are expected to be the same as those in
`thermal_properties.yaml` at least up to the maximum temperature specified for
`phonopy-qha`.

An example is given in `example/Cu-QHA`. Its `fe-v.dat` contains the static
energies plus the electronic free energies of the fixed density-of-states
approximation. The equations are given in
{ref}`electronic_thermal_properties`. For VASP,
`phonopy-vasp-efe` creates `fe-v.dat` and `e-v.dat`, and is used as:

```
% phonopy-vasp-efe --tmax=1500 vasprun.xml-{00..10}
```

where `vasprun.xml-{00..10}` have to be computed for the cell described in
{ref}`phonopy_qha_electronic_free_energy`. When the eigenvalues were computed
for a different cell, the energies in `fe-v.dat` and `e-v.dat` can instead be
scaled to that cell afterwards.

`phonopy-vasp-efe` integrates by the linear tetrahedron method when the
`vasprun.xml` describes a regular k-point mesh, and by the k-point sum
otherwise. How it chooses, and the `--k-point-sum`, `--electronic-window`,
`--electronic-spacing` and `--symmetrize-tetrahedra` options, are described
in {ref}`electronic_thermal_properties_integration`.

With `--efe`, some output files do not include the electronic entropy and heat
capacity. The table in {ref}`phonopy_qha_electronic_free_energy` lists which
files include them.

(phonopy_qha_output_files)=

### Output files

The physical units of V and T are {math}`\text{Angstrom}^3` and K, respectively.
The unit of eV for Helmholtz and Gibbs energies, J/K/mol for {math}`C_V` and
entropy, GPa for bulk modulus and pressure are used.

- Bulk modulus {math}`B_T` (GPa) vs {math}`T` (`bulk_modulus-temperature.*`)
- Gibbs free energy {math}`G` (eV) vs {math}`T` (`gibbs-temperature.*`)
- Heat capacity at constant pressure {math}`C_p` (J/K/mol) vs {math}`T` computed
  by {math}`-T\frac{\partial^2 G}{\partial T^2}` from three {math}`G(T)` points
  (`Cp-temperature.*`)
- Heat capacity at constant pressure {math}`C_p` (J/K/mol) vs {math}`T` computed
  by polynomial fittings of {math}`C_V(V)` (`Cv-volume.dat`) and {math}`S(V)`
  (`entropy-volume.dat`) for {math}`\partial S/\partial V`
  (`dsdv-temperature.dat`) and numerical differentiation of
  {math}`\partial V/\partial T`, e.g., see Eq.(5) of PRB **81**, 174301 by Togo
  _et al._ (`Cp-temperature_polyfit.*`). This may give smoother {math}`C_p` than
  that from {math}`-T\frac{\partial^2 G}{\partial T^2}`.
- Volumetric thermal expansion coefficient {math}`\beta` vs {math}`T` computed
  by numerical differentiation (`thermal_expansion.*`)
- Volume vs {math}`T` (`volume-temperature.*`)
- Thermodynamics Grüneisen parameter {math}`\gamma = V\beta B_T/C_V` (no unit)
  vs {math}`T` (`gruneisen-temperature.dat`)
- Helmholtz free energy (eV) vs volume (`helmholtz-volume.*`). When `--pressure`
  option is specified, energy offset of {math}`pV` is added. See also the
  equation of {math}`G(T, p)` at the top of this page.

(phonopy_qha_python_api)=

## Python API: `run_qha`

```{warning}
This API is experimental. The function and class names, arguments, and
return values described in this section may change in future releases
without the usual deprecation process.
```

`phonopy.run_qha` is the successor of the deprecated `PhonopyQHA` class. It
takes one `Phonopy` instance per volume point (with force constants set),
computes mesh sampling and thermal properties internally on a given
temperature grid, fits the total free energy to an equation of state at each
temperature, and returns an immutable `QHAResult` dataclass. File writers
live in `phonopy.qha.output` and plotting functions in `phonopy.qha.plot`;
both take a `QHAResult` as the first argument.

```python
import numpy as np
import phonopy
from phonopy import run_qha
from phonopy.qha.output import (
    write_lattice_parameters_temperature,
    write_volume_temperature,
)
from phonopy.qha.plot import plot_qha

phonopys = [phonopy.load(f"phonopy_params-{i:02d}.yaml") for i in range(11)]
internal_energies = np.loadtxt("e-v.dat")[:, 1]  # eV, one per volume
temperatures = np.arange(0, 1101, 10)
result = run_qha(phonopys, temperatures, internal_energies, mesh=100.0)

print(result.equilibrium_volumes)  # V(T)
write_volume_temperature(result)  # volume-temperature.dat
if result.lattice is not None:
    write_lattice_parameters_temperature(result)
plot_qha(result).show()
```

Remarks:

- `temperatures` must be strictly ascending. One temperature point is
  consumed by the numerical differentiations, so supply one more point than
  the temperature range of interest. All temperature-indexed arrays of
  `QHAResult` share the same length.
- `internal_energies` are the total energies of the static lattice
  {math}`U(V)` in eV with shape `(volumes,)`, calculated, e.g., by
  first-principles calculations or by machine learning potentials. All energies and volumes of
  `run_qha` refer to the primitive cell, to which the phonon thermal
  properties are normalized. Temperature dependence of the electronic
  system is supported through `electronic_structures` (see
  {ref}`phonopy_qha_electronic_structures`); temperature-dependent
  free-energy arrays like `fe-v.dat` are not accepted.
- `pressure` (GPa) and `eos` (`vinet`, `birch_murnaghan`, `murnaghan`) work
  as in `phonopy-qha`.
- `QHAResult` is in eV, angstrom and K throughout. Entropies and heat
  capacities are in eV/K and bulk moduli in eV/angstrom^3, where the
  deprecated `PhonopyQHA` reported J/K/mol and GPa. The writers in
  `phonopy.qha.output` convert back, so the files written by `phonopy-qha`
  are unchanged. To read the arrays in the older units, multiply by
  `EvTokJmol * 1000` or by `EVAngstromToGPa` from
  `phonopy.physical_units.get_physical_units`.
- {math}`C_p` is computed by the polynomial-fitting method (see
  {ref}`phonopy_qha_output_files`) and is available as
  `result.heat_capacity_P.heat_capacities`. The
  {math}`-T\partial^2 G/\partial T^2` variant of the legacy API is not
  provided.
- System entropy and enthalpy at constant pressure are
  `result.entropy_temperature` (eV/K) and
  `result.enthalpy_temperature` (eV). They are obtained from a
  degree-4 polynomial fit of {math}`S(V)` evaluated at {math}`V_eq(T,p)`,
  not from a numerical derivative of {math}`G(T)`. Then
  {math}`H = G + TS`. The deprecated `PhonopyQHA` exposes the same
  quantities as `entropy_temperature` and `enthalpy_temperature`; those
  properties are unavailable when temperature-dependent electronic
  free-energy arrays are supplied, because {math}`S_\mathrm{el}` is not
  stored on that path.
- The file formats written by `phonopy.qha.output` are identical to those of
  `phonopy-qha` for the shared quantities.

(phonopy_qha_electronic_structures)=

### Electronic free energies from eigenvalues

Instead of preparing a `fe-v.dat` file with `phonopy-vasp-efe` and the
`--efe` option of `phonopy-qha`, the electronic thermal properties can be
computed inside `run_qha` from the electronic states at each volume point.
For VASP, collect the states once with

```
% phonopy-vasp-efe --es vasprun.xml-{00..10}
```

(`--es` is short for `--write-electronic-states`). This writes
`electronic_states.hdf5`, which contains the electronic states together with
the volumes and the static energies (sigma->0) of all volume points, instead
of computing `fe-v.dat`. The k-points, the mesh and the cell are stored with
the states when the `vasprun.xml` describes a regular k-point mesh, so that
`run_qha` integrates by the linear tetrahedron method. Load the file with
`read_electronic_states_hdf5`:

```python
from phonopy.electron.states import read_electronic_states_hdf5

electronic_structures = read_electronic_states_hdf5("electronic_states.hdf5")
result = run_qha(phonopys, temperatures, electronic_structures=electronic_structures)
```

The eigenvalues must be computed for the cell described in
{ref}`phonopy_qha_electronic_free_energy`. Since the electronic
states carry the volumes and the total energies of the static lattice,
`internal_energies` may be omitted (an explicitly given array takes
precedence). `run_qha` checks the volumes against the primitive cell volumes
of `phonopys`, which protects against ordering mistakes.

For a calculator other than VASP, build one `ElectronicStates` per volume
point from its output; the fields are listed in
{ref}`electronic_thermal_properties_input`. Give `kpoints`, `mesh` and
`cell` as well. Without them, `run_qha` uses the k-point sum, which needs far
more k-points to converge than the tetrahedron method.

The approximation used for the electrons is described in
{ref}`phonopy_qha_electronic_free_energy`, and the integration over the
Brillouin zone in {ref}`electronic_thermal_properties_integration`. The
`electronic_window`, `electronic_spacing` and `symmetrize_tetrahedra`
parameters of `run_qha` are passed to the tetrahedron method.

(phonopy_qha_lattice_parameters)=

### Lattice parameters a(T), b(T), c(T)

```{note}
The setting assumed here is that the structures of the volume series were
prepared by relaxing the cell shape under hydrostatic pressures (or
equivalently at fixed volumes) using the static total energy only --
without the phonon contribution -- e.g., by first-principles calculations
or machine learning potentials, so that the axial ratios may vary along
the volume series, e.g., {math}`c/a` of a hexagonal crystal changing with
volume. This feature does **not** optimize {math}`a`, {math}`b`, {math}`c`
with the phonon contribution included: the cell shape at each volume is
fixed to the input shape and the phonon free energy optimizes the volume
alone, i.e., the cell shape is a function of volume determined by the
static total energy. An anisotropic QHA that minimizes
{math}`F(a, b, c;\,T)` in the full lattice-parameter space is a different
calculation and is beyond the scope of this feature. That calculation is
described in {ref}`aniso-thermal-expansion`.
```

Since each `Phonopy` instance carries its unit cell, the lattice parameters
at each volume point are known. `run_qha` propagates their volume
dependence to temperature through the equilibrium volumes {math}`V_0(T)`,
giving {math}`a(T)`, {math}`b(T)`, {math}`c(T)` with generally anisotropic
thermal expansion.

The cell volume is modeled as {math}`V = k\,abc`, where {math}`a`,
{math}`b`, {math}`c` are the lattice-vector lengths of the input unit
cells and {math}`k` is a geometric constant containing the cell-angle
factor. Since {math}`V` is the primitive cell volume, {math}`k` also
absorbs the unit-cell to primitive-cell volume ratio. {math}`k` is
determined from the input cells and no crystal-system flag is needed:
cubic cells simply give constant axial ratios and hexagonal cells give
{math}`b = a`. The axial ratios {math}`b/a` and {math}`c/a` are fitted as
polynomials of {math}`V` (degree `lattice_fit_degree`, default 2), and the
lattice parameters are recovered so that {math}`k\,a(V)\,b(V)\,c(V) = V`
holds exactly. Evaluating them at the equilibrium volumes
{math}`V_0(T)` gives {math}`a(T)`, {math}`b(T)`, {math}`c(T)`, and the
linear thermal expansion coefficients
{math}`\alpha_a = (1/a)\,\mathrm{d}a/\mathrm{d}T` etc. follow by central
differences, satisfying {math}`\alpha_a + \alpha_b + \alpha_c = \beta`.

The model requires the cell angles to be independent of volume. For
triclinic and monoclinic crystals, where cell angles are free parameters,
lattice-parameter fitting is therefore skipped with a warning and
`result.lattice` is `None` (the crystal system is determined from the
symmetry of the input cells). In addition, the constancy of {math}`k` over
the volume points is verified, and the fitting is also skipped when this
check fails. This happens when the unit cells are given in a setting whose
angles vary with volume even though the crystal system is compatible, for
example rhombohedral primitive cells, or fcc primitive cells of a crystal
strained along a conventional axis; give the unit cells in a fixed-angle
setting such as the conventional cell in such cases.

The results are stored in `result.lattice` (`QHALatticeData`):
`lattice_parameters` with shape `(temperatures, 3)` and
`axial_thermal_expansions` with shape `(temperatures, 3)`. The
corresponding output files are
`lattice_parameters-temperature.dat` (columns {math}`T`, {math}`a`,
{math}`b`, {math}`c`) by `write_lattice_parameters_temperature` and
`axial_thermal_expansion.dat` (columns {math}`T`, {math}`\alpha_a`,
{math}`\alpha_b`, {math}`\alpha_c`, {math}`\alpha_a+\alpha_b+\alpha_c`) by
`write_axial_thermal_expansion`. The plotting counterparts are
`plot_lattice_parameters` and `plot_axial_thermal_expansion`.

### Migration from `PhonopyQHA` (deprecated)

The `PhonopyQHA` class is deprecated and will be removed in a future major
release. The correspondence between the legacy attributes and the new API
is:

| `PhonopyQHA` (deprecated)    | New API                                    |
| ---------------------------- | ------------------------------------------ |
| `volume_temperature`         | `QHAResult.equilibrium_volumes`            |
| `gibbs_temperature`          | `QHAResult.gibbs_free_energies`            |
| `entropy_temperature`        | `QHAResult.entropy_temperature`            |
| `enthalpy_temperature`       | `QHAResult.enthalpy_temperature`           |
| `bulk_modulus_temperature`   | `QHAResult.bulk_moduli`                    |
| `thermal_expansion`          | `QHAResult.thermal_expansion`              |
| `heat_capacity_P_polyfit`    | `QHAResult.heat_capacity_P.heat_capacities`|
| `heat_capacity_P_numerical`  | not provided (use `heat_capacity_P`)       |
| `gruneisen_temperature`      | `QHAResult.gruneisen_parameters`           |
| `helmholtz_volume`           | `QHAResult.helmholtz_volume`               |
| `write_*` methods            | functions in `phonopy.qha.output`          |
| `plot_*` methods             | functions in `phonopy.qha.plot`            |
| `bulk_modulus` (E-V fitting) | `phonopy.qha.core.BulkModulus`             |

Three of these replacements change units. `QHAResult.entropy_temperature`
and `QHAResult.heat_capacity_P.heat_capacities` are in eV/K where
`PhonopyQHA` reported J/K/mol, and `QHAResult.bulk_moduli` is in
eV/angstrom^3 where `bulk_modulus_temperature` reported GPa. Code that
prints or plots these values has to convert them.

Behavioral differences: `run_qha` does not accept temperature-dependent
electronic free-energy arrays (the `fe-v.dat` style input of the legacy
API); supply `electronic_structures` instead (see
{ref}`phonopy_qha_electronic_structures`). With the electronic
contributions included there, `QHAResult.gruneisen_parameters` and
`heat_capacity_P` differ from the legacy behavior, where the Grüneisen
parameters used the phonon-only {math}`C_V` and the polyfit {math}`C_p`
was unavailable.
