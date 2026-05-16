(rust_backend)=

# Rust backend

The computationally heavy parts of phonopy (dynamical-matrix builders,
reciprocal dipole-dipole, force-constant symmetrization, BZ-grid helpers,
smallest-vector search, tetrahedron-method weights, ...) are implemented
as a Rust extension module called from Python through PyO3. The Rust
path is distributed as a separate Python extension module, `phonors`,
maintained in its own top-level repository at
[github.com/phonopy/phonors](https://github.com/phonopy/phonors) and
built with [maturin](https://www.maturin.rs/) and
[PyO3](https://pyo3.rs/).

Since v4, `phonors` is the default backend and a required runtime
dependency of phonopy. The legacy C extension is still built by default
and can be selected per call via `lang="C"` or the `--legacy-backend`
CLI flag; both paths are kept in the source tree for cross-checking.
Numerical parity with the C path is verified by the regression tests
under `test/`.

```{contents}
:depth: 2
:local:
```

## Installation

`phonors` is listed in `pyproject.toml` as a required dependency, so
`pip install phonopy` (and the conda-forge package once updated)
pulls it in automatically. No extra step is needed for normal use.

(rust_backend_install)=

### Building `phonors` from source

When you want to track the development version of `phonors`, or to
build with custom Rust flags, install it from a source clone instead
of the PyPI wheel.

Requirements:

- A Rust toolchain (stable, edition 2021, `rustc >= 1.75`). The
  easiest way to install it is via [rustup](https://rustup.rs/).
- [maturin](https://www.maturin.rs/) 1.7 or newer (available on PyPI
  and conda-forge).
- Python 3.10 or newer. The extension is built against the stable ABI
  (`abi3-py310`), so one build works for all Python 3.10+ interpreters.

Clone the repository and build the extension in editable mode:

```bash
% git clone https://github.com/phonopy/phonors.git
% cd phonors
% maturin develop --release
```

After a successful build, the module should import from any Python process

```python
import phonors
```

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

The Rust backend is active by default; no flag or keyword is required.
The legacy C backend is selected per call through the `lang` keyword on
the Python API or the `--legacy-backend` flag on the CLI.

(rust_backend_lang_dispatch)=

### Python API

The constructor and loader take a `lang` keyword:

```python
import phonopy

ph = phonopy.load("phonopy_disp.yaml")                # lang="Rust" (default)
ph_c = phonopy.load("phonopy_disp.yaml", lang="C")    # legacy C backend
```

The current value is exposed as the read-only `Phonopy.lang` property.
Valid values are `"Rust"` (default) and `"C"`.

`lang` is threaded internally to every lang-aware consumer, including
`Primitive`, `Symmetry`, `ShortestPairs`, `BZGrid`, `GridMatrix`,
`DynamicalMatrix*`, `DynmatToForceConstants`, `Mesh`, `IterMesh`,
`QpointsPhonon`, `DerivativeOfDynamicalMatrix`, `TetrahedronMethod`,
`TetrahedronDOSAccumulator`, `TotalDos`, `ProjectedDos`, and
`ThermalProperties`. No per-call plumbing from the user side is required.

### Command line

The default `phonopy` invocation runs on Rust and prints

```
Rust backend (phonors) using rayon (N threads).
```

in the run header, where `N` follows rayon's defaults. To opt back into
the C extension, pass `--legacy-backend` (conf-file equivalent:
`LEGACY_BACKEND = .true.`):

```bash
% phonopy --legacy-backend ...
```

The old `--rust` flag still parses but is a deprecated no-op (the Rust
backend is already active) and emits a `DeprecationWarning`. It will be
removed in a future release.

### Thread pool

The Rust kernels parallelize with [rayon](https://docs.rs/rayon/),
which uses its own thread pool. The thread count is controlled by
`RAYON_NUM_THREADS` (not `OMP_NUM_THREADS`, which only affects the C
path):

```bash
% RAYON_NUM_THREADS=8 phonopy ...
```

NumPy/SciPy BLAS multithreading used by phonon diagonalization is
independent and is controlled by the BLAS library's own variables
(e.g. `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`).

### Dispatch tracing

To verify that a given code path is actually running on the Rust
backend, set `PHONOPY_TRACE_LANG=1` before launching:

```bash
% PHONOPY_TRACE_LANG=1 phonopy ...
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
a Rust path), phonopy can be installed with the C extension skipped:

```bash
% PHONOPY_NO_C_EXT=1 pip install -e . -vvv
```

When the env var is set, `CMakeLists.txt` returns early and neither
`phonopy._phonopy` nor `phonopy._recgrid` is built. At runtime,
`Phonopy()` and `phonopy.load()` detect the missing C extension, emit
a one-time `[phonopy] C extension ... is not available; falling back
to lang='Rust' ...` message, and silently flip `lang="C"` requests to
`lang="Rust"`.

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
