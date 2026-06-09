(development)=

# Development

This page documents the architectural direction of the phonopy code
base and the rules that new contributions are expected to follow.
For the practical contribution workflow (formatting, pre-commit,
tests), see the `Development` section of the
[README](https://github.com/phonopy/phonopy#development).

```{contents}
:depth: 2
:local:
```

## Architecture principles

The `Phonopy` class is being reduced step by step to a lean
calculation context (cells, symmetry, force constants, NAC
parameters). New code should follow these rules:

- A new `run_*` method returns a self-contained result object. It
  must not store the result as a new `self._*` attribute of
  `Phonopy`.
- `write_*` / `plot_*` functionality belongs to the module of the
  corresponding analysis or to its result object, not to `Phonopy`.
  During the transition, a thin delegating method on `Phonopy` is
  acceptable.
- Setters of calculation inputs must invalidate derived state via
  `Phonopy._invalidate_derived`. Do not add ad-hoc cache-clearing
  logic next to it.

## Architecture migration toward v5 and v6

Historically, every analysis stored its result as a mutable
attribute of the single `Phonopy` instance. This made stale-state
bugs easy to introduce and hard to see. The migration moves phonopy
to a model where analyses return self-contained result objects and
`Phonopy` holds only the calculation inputs. The planned stages:

1. (v4.x) New result-object APIs are introduced alongside existing
   ones. Derived-state invalidation is centralized: mutating a
   calculation input clears all results derived from it (done).
2. (v5.0) Legacy duplicated APIs (`get_*` / `set_*` methods that
   have property equivalents, already-raising init arguments, ...)
   emit `DeprecationWarning` or are removed.
3. (v5.x to v6) Analysis results move off `Phonopy` attributes onto
   self-contained result objects returned by `run_*` methods.

## Deprecation policy

- A replacement API lands first; the old API becomes a thin shim
  that delegates to the new implementation and emits
  `DeprecationWarning`. Two parallel implementations are never
  carried.
- The deprecation window is at most two major versions, e.g.,
  deprecated in v5.x and removed in v6.0. A deprecation may start
  earlier (already in v4.x) to extend the notice period.
- Example: `Phonopy.copy` is deprecated in v4.x in favor of
  `Phonopy.replicate`, which constructs a new instance from the same
  init parameters without carrying over internal state.
