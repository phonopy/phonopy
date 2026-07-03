(migration_v5)=

# Migrating toward phonopy v5

Phonopy is moving to a model where each analysis returns a
self-contained *result object* and the `Phonopy` instance holds only
the calculation inputs (cells, symmetry, force constants, NAC
parameters). See {ref}`development` for the architectural background
and the deprecation policy.

As part of this migration, a group of legacy `Phonopy` methods has
been deprecated. They still work in the v4.x series but emit
`DeprecationWarning` and are scheduled for removal in v5.0. This page
lists each deprecated method and its replacement so that existing
scripts can be updated ahead of v5.0.

The replacements are available now: every `run_*` method returns its
result object, and the corresponding property on `Phonopy` exposes the
same object. You can migrate today on v4.x without waiting for v5.0.

```{note}
v5.0 has not been released yet. This page is provisional: the set of
deprecated methods, their recommended replacements, and the removal
timing may still change before v5.0. The replacement APIs are already
available in v4.x and are intended to be stable, but some of their
method and attribute names may still be adjusted before v5.0. Treat
the v5.0 removal schedule as a plan rather than a guarantee, and always
check the {ref}`changelog` of the version you upgrade to for the final
names and list.
```

```{contents}
:depth: 2
:local:
```

## Result accessors: `get_*_dict()` to result-object properties

Each `get_*_dict()` method returned a plain dictionary. The
replacement is the result object returned by the matching `run_*`
method (also reachable through the property of the same name); its
attributes replace the former dict keys.

| Deprecated method | Replacement (result object): attributes |
|-------------------|------------------------------------------|
| `get_band_structure_dict()` | `band_structure` (`BandStructure`): `qpoints`, `distances`, `frequencies`, `eigenvectors`, `group_velocities` |
| `get_mesh_dict()` | `mesh` (`Mesh`): `qpoints`, `weights`, `frequencies`, `eigenvectors`, `group_velocities` |
| `get_qpoints_dict()` | `qpoints` (`QpointsPhonon`): `frequencies`, `eigenvectors`, `group_velocities`, `dynamical_matrices` |
| `get_total_dos_dict()` | `total_dos` (`TotalDos`): `frequency_points`, `dos` |
| `get_projected_dos_dict()` | `projected_dos` (`ProjectedDos`): `frequency_points`, `projected_dos` |
| `get_thermal_properties_dict()` | `thermal_properties` (`ThermalProperties`): `temperatures`, `free_energy`, `entropy`, `heat_capacity` |
| `get_thermal_displacements_dict()` | `thermal_displacements` (`ThermalDisplacements`): `temperatures`, `thermal_displacements` |
| `get_thermal_displacement_matrices_dict()` | `thermal_displacement_matrices` (`ThermalDisplacementMatrices`): `temperatures`, `thermal_displacement_matrices`, `thermal_displacement_matrices_cif` |

### Example: band structure

**Deprecated:**

```python
ph.run_band_structure(paths)
d = ph.get_band_structure_dict()
frequencies = d["frequencies"]
```

**Replacement:**

```python
bs = ph.run_band_structure(paths)   # returns a BandStructure
frequencies = bs.frequencies
# or, equivalently, through the property:
frequencies = ph.band_structure.frequencies
```

### Example: thermal properties

**Deprecated:**

```python
ph.run_thermal_properties()
d = ph.get_thermal_properties_dict()
free_energy = d["free_energy"]
```

**Replacement:**

```python
tp = ph.run_thermal_properties()    # returns a ThermalProperties
free_energy = tp.free_energy
```

## Single-q one-off evaluators: use `run_qpoints`

The single-q convenience methods are replaced by `run_qpoints`, which
returns a `QpointsPhonon` result object indexed by q-point.

| Deprecated method | Replacement |
|-------------------|-------------|
| `get_frequencies(q)` | `run_qpoints([q]).frequencies[0]` |
| `get_frequencies_with_eigenvectors(q)` | `run_qpoints([q], with_eigenvectors=True)`: `frequencies[0]`, `eigenvectors[0]` |
| `get_dynamical_matrix_at_q(q)` | `run_qpoints([q], with_dynamical_matrices=True).dynamical_matrices[0]` |
| `get_group_velocity_at_q(q)` | `run_qpoints([q], with_group_velocities=True).group_velocities[0]` |

**Deprecated:**

```python
frequencies = ph.get_frequencies(q)
```

**Replacement:**

```python
frequencies = ph.run_qpoints([q]).frequencies[0]
```

## Other deprecated analysis methods

| Deprecated method | Replacement |
|-------------------|-------------|
| `get_modulated_supercells()` | `run_modulations()`, then `modulation.modulated_supercells` |
| `get_modulations_and_supercell()` | `modulation.modulations`, `modulation.supercell` |
| `set_irreps(...)` | `run_irreps(...)` |
| `get_moment()` | `run_moment().moment` (or the `moment` property) |
| `get_dynamic_structure_factor()` | `run_dynamic_structure_factor()`: `qpoints`, `dynamic_structure_factors` |
| `get_Debye_frequency()` | `total_dos.debye_frequency` |
| `set_Debye_frequency()` | `run_total_dos()`, then `total_dos.run_debye_frequency(num_atoms)` and read `total_dos.debye_frequency` |
| `produce_force_constants(forces=...)` | set the `forces` setter, then call `produce_force_constants()` |

## Deprecated class: `PhonopyQHA` to `run_qha`

The `PhonopyQHA` class is deprecated. The replacement is the function
`phonopy.run_qha`, which takes one `Phonopy` instance per volume point,
computes the thermal properties internally, and returns an immutable
`QHAResult` dataclass. File writers and plotters became free functions in
`phonopy.qha.output` and `phonopy.qha.plot`. See
{ref}`phonopy_qha_python_api` for the full description including the new
lattice-parameter output.

| Deprecated (`PhonopyQHA`)    | Replacement                                |
|------------------------------|--------------------------------------------|
| `PhonopyQHA(volumes, ...)`   | `run_qha(phonopys, ...)` (`QHAResult`)     |
| `volume_temperature`         | `QHAResult.equilibrium_volumes`            |
| `gibbs_temperature`          | `QHAResult.gibbs_free_energies`            |
| `bulk_modulus_temperature`   | `QHAResult.bulk_moduli`                    |
| `thermal_expansion`          | `QHAResult.thermal_expansion`              |
| `heat_capacity_P_polyfit`    | `QHAResult.heat_capacity_P.heat_capacities`|
| `heat_capacity_P_numerical`  | not provided (use `heat_capacity_P`)       |
| `gruneisen_temperature`      | `QHAResult.gruneisen_parameters`           |
| `helmholtz_volume`           | `QHAResult.helmholtz_volume`               |
| `write_*` methods            | functions in `phonopy.qha.output`          |
| `plot_*` methods             | functions in `phonopy.qha.plot`            |
| `bulk_modulus` (E-V fitting) | `phonopy.qha.core.BulkModulus`             |

**Deprecated:**

```python
qha = PhonopyQHA(
    volumes=volumes,
    electronic_energies=energies,
    temperatures=temperatures,
    free_energy=fe_phonon,
    cv=cv,
    entropy=entropy,
)
volume_temperature = qha.volume_temperature
qha.write_volume_temperature()
```

**Replacement:**

```python
from phonopy import run_qha
from phonopy.qha.output import write_volume_temperature

result = run_qha(phonopys, temperatures, energies)  # returns a QHAResult
volume_temperature = result.equilibrium_volumes
write_volume_temperature(result)
```

The `phonopy-qha` command-line script is not affected; it keeps working
on the legacy implementation until a successor is provided.

## Deprecated utility method

| Deprecated method | Replacement |
|-------------------|-------------|
| `Phonopy.copy()` | `Phonopy.replicate()` |

`copy()` and `replicate()` both build a new instance from the same
init parameters; neither carries over internal state such as force
constants or NAC parameters. `copy()` is deprecated already in v4.x to
extend the notice period.

## Removed: the `factor` argument

The `factor` argument of `Phonopy(...)` and `phonopy.load(...)` has
already been removed (it raised an error in recent v4.x). Set the
frequency unit conversion factor through the `unit_conversion_factor`
setter instead.

**Removed:**

```python
ph = Phonopy(unitcell, supercell_matrix, factor=521.471)
```

**Replacement:**

```python
ph = Phonopy(unitcell, supercell_matrix)
ph.unit_conversion_factor = 521.471
```

## Surfacing the warnings in existing code

Python hides `DeprecationWarning` by default in many contexts. To see
which deprecated calls a script makes before v5.0 removes them, run it
with warnings made visible:

```bash
python -W default::DeprecationWarning your_script.py
```

In a test suite, configure pytest to error on the phonopy
deprecations, for example:

```ini
[pytest]
filterwarnings =
    error::DeprecationWarning:phonopy
```
