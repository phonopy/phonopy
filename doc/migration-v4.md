(migration_v4)=

# Migrating from phonopy v3 to v4

Phonopy v4 introduces several behaviour changes that affect existing
command lines, scripts, and saved workflows. This page lists the changes
in roughly decreasing order of user impact and shows how to update
existing usage.

```{contents}
:depth: 2
:local:
```

## Command line split: `phonopy` and `phonopy-init`

The `phonopy` command has been split into two commands:

- `phonopy-init` -- setup operations: generate supercells with
  displacements, create `FORCE_SETS` / `FORCE_CONSTANTS` files from
  external calculator results, inspect crystal symmetry.
- `phonopy` -- phonon calculation from a `phonopy.yaml`-like file.

The deprecated `phonopy-load` command is kept as an alias of `phonopy`.

Practically, setup-related flags (`-d`, `--rd`, `-c`, `--dim`, `-f`,
`--fz`, `--fc`, `--symmetry`) moved from `phonopy` to `phonopy-init`.
Running them on `phonopy` reports a migration error pointing here.

### Update existing shell scripts

**v3:**

```bash
phonopy -d --dim 2 2 2 -c POSCAR-unitcell
phonopy -f vasprun.xml-{001,002}
phonopy --fc vasprun.xml
phonopy --symmetry -c POSCAR-unitcell
```

**v4:**

```bash
phonopy-init -d --dim 2 2 2 -c POSCAR-unitcell
phonopy-init -f vasprun.xml-{001,002}
phonopy-init --fc vasprun.xml
phonopy-init --symmetry -c POSCAR-unitcell
```

For phonon calculation (band structure, DOS, thermal properties, ...)
nothing changes:

```bash
phonopy band.conf
```

## `primitive_matrix` default changed to `"auto"`

In v3 the default for `primitive_matrix` was the 3x3 identity matrix,
i.e. no transformation was applied and the input unit cell was used
as the primitive cell as-is. In v4 the default is `"auto"`: phonopy
detects the primitive cell from crystal symmetry via spglib and
transforms the input cell accordingly.

When the auto-detected matrix is not the identity, the q-point
convention and band layout differ from v3 even though the input file
and command line are unchanged. Phonopy emits a runtime warning
(`PrimitiveMatrixAutoDefaultWarning`) in that situation, showing the
resolved matrix and pointing to this page.

### Update existing command lines

The `--pa` option (alias of `PRIMITIVE_AXES`) now defaults to `auto`
instead of the identity matrix. Existing command lines that relied on
the v3 default and did not pass `--pa` will silently switch to
auto-detection.

**v3** (no flag; input cell used as the primitive cell):

```bash
phonopy band.conf
```

**v4** (recommended new default, automatic primitive detection):

```bash
phonopy band.conf
```

**v4** (to keep v3 behaviour explicitly):

```bash
phonopy --pa P band.conf
```

### Update existing API calls

**v3:**

```python
ph = Phonopy(unitcell, supercell_matrix)  # input cell used as the primitive cell
```

**v4 (recommended new default, automatic primitive detection):**

```python
ph = Phonopy(unitcell, supercell_matrix)
# or, equivalently
ph = Phonopy(unitcell, supercell_matrix, primitive_matrix="auto")
```

**v4 (to keep v3 behaviour explicitly):**

```python
ph = Phonopy(unitcell, supercell_matrix, primitive_matrix="P")
```

### When the default does not change behaviour

If the input unit cell is already the primitive cell, auto-detection
returns the identity matrix (i.e. the input cell is used unchanged)
and no warning is emitted.

If the calculation loads a `phonopy.yaml` that already records a
`primitive_matrix`, that stored value takes priority over the new
default. Workflows driven by saved YAML files therefore reproduce v3
results exactly.

## `--nac` removed

`--nac` was removed because non-analytical term correction is now
enabled automatically whenever the necessary data is available: a
`BORN` file in the working directory or `nac_params` stored in a
`phonopy.yaml`-like file. Pass `--nonac` to disable NAC explicitly.

**v3:**

```bash
phonopy --nac band.conf
```

**v4:**

```bash
phonopy band.conf           # NAC auto-detected from BORN / phonopy.yaml
phonopy --nonac band.conf   # explicit opt-out
```

## Mesh fallback for length-based input

When a sampling mesh is specified by a length (float) and the resulting
regular grid breaks the primitive-cell point-group symmetry, v4 rebuilds
the grid as a *generalized regular grid* (GR-grid) anchored to the
conventional cell, in order to keep full point-group symmetry. The
actual `mesh_numbers` (`D_diag`) used internally may differ from the
naive regular grid that `length2mesh` would have produced.

v3 instead kept the requested regular grid but handled the broken
symmetry incorrectly: incompatible rotations were silently dropped
inside spglib, and the resulting ir-grid did not actually respect the
symmetry it claimed. v4 fixes this. For the same length input, v4
therefore yields:

- different `mesh_numbers` (those of the GR-grid),
- somewhat different DOS and thermal-property values, reflecting the
  corrected symmetry handling.

The change is announced at runtime via `MeshGRGridFallbackWarning`,
which reports the requested mesh, the GR-grid `D_diag`, and the reason
for the substitution.

When the grid symmetry is broken, the v3 numerical results cannot be
exactly reproduced under v4. As a workaround, specifying the mesh
explicitly by integer numbers (e.g. `MESH = 12 12 12` or
`mesh=[12, 12, 12]`) keeps the requested grid and falls back to
time-reversal-only symmetry.

## symfc primitive-cell fallback

When the input "primitive" cell does not match the primitive cell that
symfc finds internally, v4 automatically falls back to full force
constants instead of compact ones.

## Renamed / relocated modules and functions

The following Python imports changed:

| v3 | v4 |
|----|----|
| `from phonopy.structure.grid_points import ...` | replaced by `phonopy.phonon.grid.BZGrid` (migrated from phono3py) |
| `Phonopy.run_dynamical_matrix_solver_c(...)` | `Phonopy.get_dynamical_matrices_at_qpoints(...)` |

There are no deprecation shims; the old names are gone in v4.

## Rust backend (`phonors`) is now the default

The Rust kernels in [phonors](https://github.com/phonopy/phonors) are
the default backend in v4. `phonors` is a required runtime dependency
and is installed automatically with phonopy. The C extension is still
built and kept as a legacy backend that users can opt back into.

What this means for existing code:

- `Phonopy(...)`, `phonopy.load(...)`, and the CLI all run on the Rust
  backend out of the box. Numerical results are bit-identical to the
  C path on every kernel that has parity tests.
- Performance is generally similar to or better than the C extension
  thanks to rayon-based parallelism. The startup banner now prints
  `Rust backend (phonors) using rayon (N threads).`
- The C extension is still built by default and remains selectable
  per call via `lang="C"` / `--legacy-backend`.

### `--rust` is deprecated (no-op)

`--rust` used to enable the experimental Rust backend. In v4 it is a
deprecated no-op: the Rust backend is already active. The flag still
parses (so v3 command lines do not error) but emits a
`DeprecationWarning` and will be removed in a future release.

### `--legacy-backend` opts back into the C extension

To keep using the C kernels (for example to compare against v3
numerical output, or to work around a hypothetical phonors regression),
pass `--legacy-backend` on the CLI or `lang="C"` in the Python API.
The conf-file equivalent is `LEGACY_BACKEND = .true.`.

**v3 (or v4 with explicit opt-in):**

```bash
phonopy --legacy-backend band.conf
```

```python
ph = Phonopy(unitcell, supercell_matrix, lang="C")
ph = phonopy.load("phonopy.yaml", lang="C")
```

### Optional C-extension builds

Setting `PHONOPY_NO_C_EXT=1` at build time still skips the C extension
entirely; in that case `lang="C"` silently falls back to Rust with a
one-time notice.
