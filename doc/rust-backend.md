(rust_backend)=

# Rust backend (experimental)

The computationally heavy parts of phonopy (dynamical-matrix builders,
reciprocal dipole-dipole, force-constant symmetrization, BZ-grid helpers,
smallest-vector search, tetrahedron-method weights, ...) have been
implemented as a C extension module called from Python through the Python/C
API. An alternative Rust implementation is now available experimentally.
The Rust path is distributed as a separate Python extension module,
`phonors`, maintained in its own top-level repository at
[github.com/phonopy/phonors](https://github.com/phonopy/phonors) and built
with [maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/). It is
experimental: behaviour is validated against the C path by the regression
tests under `test/`, but the C extension remains the default and both paths
are kept in the source tree for cross-checking.

```{contents}
:depth: 2
:local:
```

## Installation

The Rust backend is not bundled with the standard phonopy wheel and conda
package. It is installed as the separate `phonors` package, which has to
be built from a clone of the
[phonors repository](https://github.com/phonopy/phonors).

### Requirements

- A Rust toolchain (stable, edition 2021, `rustc >= 1.75`). The
  easiest way to install it is via [rustup](https://rustup.rs/).
- [maturin](https://www.maturin.rs/) 1.7 or newer (available on PyPI
  and conda-forge).
- Python 3.10 or newer. The extension is built against the stable ABI
  (`abi3-py310`), so one build works for all Python 3.10+ interpreters.
- A working phonopy source checkout and its usual build/runtime
  dependencies (see {ref}`install_from_source`).

(rust_backend_install)=

### Build and install

Clone the `phonors` repository alongside phonopy and build the extension
in editable mode with `maturin develop`:

```bash
% git clone https://github.com/phonopy/phonors.git
% cd phonors
% maturin develop --release
```

After a successful build, the module should import from any Python process

```python
import phonors
```

The phonopy Python layer imports `phonors` lazily and only when the Rust
backend is selected, so installations without the extension continue to
work on the C path.

### Optional: native CPU tuning

By default, `maturin develop --release` builds with the Rust baseline
target (x86-64 v1 on x86_64, Armv8.0 on aarch64), so the resulting
module runs on any CPU of that architecture. For a local build that
will only run on the current machine, enabling the host CPU's full
instruction set can recover a few percent of wall-clock:

```bash
% RUSTFLAGS='-C target-cpu=native' maturin develop --release
```

## Usage

Once `phonors` is installed, the Rust backend is selected through the
`--rust` flag on the command line or the `lang` keyword on the Python
API. The C backend remains the default.

(rust_backend_lang_dispatch)=

### Python API

The constructor and loader take a `lang` keyword:

```python
import phonopy

ph_c = phonopy.load("phonopy_disp.yaml", lang="C")     # default
ph_r = phonopy.load("phonopy_disp.yaml", lang="Rust")  # experimental
```

The current value is exposed as the read-only `Phonopy.lang` property.
Valid values are `"C"` (default) and `"Rust"`.

`lang` is threaded internally to every lang-aware consumer, including
`Primitive`, `Symmetry`, `ShortestPairs`, `BZGrid`, `GridMatrix`,
`DynamicalMatrix*`, `DynmatToForceConstants`, `Mesh`, `IterMesh`,
`QpointsPhonon`, `DerivativeOfDynamicalMatrix`, `TetrahedronMethod`,
`TetrahedronDOSAccumulator`, `TotalDos`, `ProjectedDos`, and
`ThermalProperties`. No per-call plumbing from the user side is required.

### Command line

Pass `--rust` to the `phonopy` (or `phonopy-load`) command:

```bash
% phonopy --rust ...
```

When the flag is set, the run header prints

```
Rust backend (phonors) using rayon (N threads).
```

instead of the OpenMP banner from the C build, where `N` follows rayon's
defaults.

### Thread pool

The Rust kernels parallelize with [rayon](https://docs.rs/rayon/),
which uses its own thread pool. The thread count is controlled by
`RAYON_NUM_THREADS` (not `OMP_NUM_THREADS`, which only affects the C
path):

```bash
% RAYON_NUM_THREADS=8 phonopy --rust ...
```

NumPy/SciPy BLAS multithreading used by phonon diagonalization is
independent and is controlled by the BLAS library's own variables
(e.g. `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).

### Dispatch tracing

To verify that a given code path is actually running on the Rust
backend, set `PHONOPY_TRACE_LANG=1` before launching:

```bash
% PHONOPY_TRACE_LANG=1 phonopy --rust ...
[phonopy.lang] dispatch name=Phonopy.__init__ lang=Rust
[phonopy.lang] dispatch name=Primitive.__init__ lang=Rust
[phonopy.lang] dispatch name=BZGrid.__init__ lang=Rust
...
```

Each lang-aware call site emits one line to stderr at construction time
(and the dynamical-matrix builders emit one per call). The trace is
silent by default. It is attached to a dedicated logger
(`phonopy.lang`) and is independent of the rest of the run log, so
enabling it does not affect normal output.

(rust_backend_no_c_ext)=

## Building phonopy without the C extension

For Rust-only deployments (or to validate that every dispatch site has
a Rust path), phonopy can be installed with the C extension skipped.
Install `phonors` first as described above, then build phonopy with
the env var set:

```bash
% PHONOPY_NO_C_EXT=1 pip install -e . -vvv
```

When the env var is set, `CMakeLists.txt` returns early and neither
`phonopy._phonopy` nor `phonopy._recgrid` is built. At runtime,
`Phonopy()` and `phonopy.load()` detect the missing C extension, emit
a one-time `[phonopy] C extension ... is not available; falling back
to lang='Rust' ...` message, and silently flip `lang="C"` requests to
`lang="Rust"`. `phonors` therefore becomes a hard requirement for
this build; an informative `ImportError` is raised if neither backend
is available.

To restore the C extension, simply rebuild without the env var:

```bash
% pip install -e . -vvv
```

```{note}
This option is intended for testing the Rust path and for packagers
who want a Rust-only wheel. For day-to-day use the regular install
(with the C extension) remains the recommended path.
```

## Scope

Rust kernels are wired through the lang dispatch for the following
groups:

- Grid construction (`BZGrid`, `GridMatrix`: SNF, grid addresses,
  reciprocal / transform rotations, ir-grid mapping).
- Dipole-dipole and charge-sum kernels for the non-analytical term
  correction (Wang and Gonze variants).
- Tetrahedron-method weights (`tetrahedra_relative_grid_address`,
  `TetrahedronDOSAccumulator`).
- Force-constant utilities (`compute_permutation`,
  `perm_trans_symmetrize_fc` / `_compact_fc`, `transpose_compact_fc`,
  `distribute_fc2`, drift FC computation).
- Smallest-vector search (`set_smallest_vectors_{sparse,dense}`).
- Dynamical-matrix construction at arbitrary q-points
  (`dynamical_matrices_at_qpoints[_gonze]`), `transform_dynmat_to_fc`,
  and the derivative of the dynamical matrix (`derivative_dynmat`).

### Known limitations

- **Dynamical-matrix diagonalization.** LAPACK calls (phonon
  eigenproblem) stay in Python/NumPy by design. They are not part of
  the Rust port.
- **Pure-numpy paths.** Routines that are already light-weight in
  NumPy (`ThermalProperties`, per-omega tetrahedron lookups) accept
  `lang` for API consistency but do not call into `phonors`.

If any of these code paths is reached with `lang="Rust"`, phonopy
transparently uses the C (or Python) implementation for that step.

```{warning}
The Rust backend is experimental. The default in `Phonopy()` and
`phonopy.load()` remains `lang="C"`. Numerical parity with the C path
is verified for the dispatch sites that have been ported, but APIs and
behaviour may still change.
```
