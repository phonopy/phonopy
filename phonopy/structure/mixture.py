"""Utilities for mixed-species (site-mixture) cells.

This module hosts helpers that operate on PhonopyAtoms whose species
table contains merged weighted mixtures (e.g. alloy or solid-solution
sites): the mapping between phonopy sites and the constituent-expanded
row layout assumed by VASP-style outputs, and reductions of expanded
calculator data back to per-site quantities.

Cell construction utilities (``build_mixture_cell``,
``build_species_table_from_mixtures``) live with the cell / atoms
modules; this module focuses on operations that interpret the resulting
mixture structure.

"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy.structure.atoms import PhonopyAtoms


def build_mixtures_from_groups(
    symbols: Sequence[str],
    groups: Sequence[Sequence[int]],
    weights: NDArray[np.double],
) -> list[list[tuple[str, float]]]:
    """Assemble per-site mixtures from grouped atom indices and weights.

    Each group lists the atom indices that have been collapsed into a single
    site (typically by position overlap). For every group this returns one
    ``(symbol, weight)`` list describing the constituents of that site, in the
    group's index order. The result is the input layout expected by
    :func:`phonopy.structure.atoms.build_species_table_from_mixtures`.

    Weights are validated up front so that misconfigured site-mixture
    inputs are reported with the offending atom indices: an
    isolated atom (singleton group) must carry weight 1.0, and the weights of
    an overlapping group must sum to 1.0. This entry-level check complements,
    and does not replace, the constituent-sum check performed later by
    ``build_species_table_from_mixtures``.

    Parameters
    ----------
    symbols : sequence of str
        Per-atom chemical symbols in the original cell order.
    groups : sequence of sequence of int
        Groups of atom indices to merge into one site each.
    weights : NDArray[np.double]
        Per-atom mixture weights in the original cell order.

    Returns
    -------
    list[list[tuple[str, float]]]
        One ``(symbol, weight)`` list per group, in group order.

    """
    mixtures: list[list[tuple[str, float]]] = []
    for group in groups:
        wsum = float(weights[list(group)].sum())
        if len(group) == 1:
            if not np.isclose(weights[group[0]], 1.0):
                raise ValueError(
                    f"Weight of non-overlapping atom at index {group[0]} "
                    f"must be 1.0, got {weights[group[0]]}."
                )
        elif not np.isclose(wsum, 1.0):
            raise ValueError(
                f"Weights of overlapping atoms at indices {list(group)} must "
                f"sum to 1.0, got {wsum}."
            )
        mixtures.append([(symbols[i], float(weights[i])) for i in group])
    return mixtures


def iter_mixture_expansion_blocks(
    cell: PhonopyAtoms,
) -> Iterator[tuple[str, float, NDArray[np.int64]]]:
    """Yield ``(symbol, weight, atom_indices)`` blocks in expansion order.

    This is the single source of truth for the order in which a (possibly
    mixed) cell is expanded into per-constituent rows. The canonical order is:
    for each ``cell.species_table`` entry, one block per constituent -- a
    single ``(symbol, 1.0)`` for a non-mixture species, or each
    ``(symbol, weight)`` pair (in mixture order) for a mixed species -- paired
    with the indices of the atoms that reference the species. Both
    :func:`get_mixture_expansion` (per-row site/weight arrays) and the VASP
    POSCAR row layout build on this ordering, so they cannot drift apart.

    Parameters
    ----------
    cell : PhonopyAtoms
        Cell whose expansion order is to be reported. May or may not contain
        mixed-species sites.

    Yields
    ------
    symbol : str
        Constituent element symbol of the block.
    weight : float
        Constituent weight (1.0 for a non-mixture species).
    atom_indices : NDArray[np.int64]
        Ascending indices of the atoms in ``cell`` referencing the species.

    """
    species_ids = cell.species_ids
    for sid, sp in enumerate(cell.species_table):
        atom_idx = np.where(species_ids == sid)[0]
        if sp.mixture is None:
            yield sp.symbol, 1.0, atom_idx
        else:
            for sym, weight in sp.mixture:
                yield sym, float(weight), atom_idx


def get_mixture_expansion(
    cell: PhonopyAtoms,
) -> tuple[NDArray[np.int64], NDArray[np.double]]:
    """Return the (site_index, weight) order used to expand a mixture cell.

    The expansion mirrors the per-row layout of POSCAR / vasprun.xml
    produced by ``write_vasp(..., expand_mixtures=True)``: for each entry
    of ``cell.species_table``, every atom that references the entry is
    emitted once per constituent (in mixture order), each at the original
    site coordinate. This shared layout is also assumed by VASP
    `vasprun.xml` force parsing and by the expanded `FORCE_SETS` formats.

    For a non-mixture cell the result is the identity mapping
    ``(np.arange(natom), ones)``.

    Parameters
    ----------
    cell : PhonopyAtoms
        Cell whose expansion order is to be reported. May or may not
        contain mixed-species sites.

    Returns
    -------
    site_indices : NDArray[np.int64], shape (n_expanded,)
        Site index in ``cell`` for each expanded row.
    weights : NDArray[np.double], shape (n_expanded,)
        Constituent weight (1.0 for non-mixture sites).

    """
    site_indices: list[int] = []
    weights: list[float] = []
    for _symbol, weight, atom_idx in iter_mixture_expansion_blocks(cell):
        for j in atom_idx:
            site_indices.append(int(j))
            weights.append(weight)

    return (
        np.array(site_indices, dtype="int64"),
        np.array(weights, dtype="double"),
    )


def reduce_mixture_forces(
    forces: Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | Sequence[NDArray[np.double]]
    | NDArray[np.double],
    cell: PhonopyAtoms,
    mode: Literal["weighted_sum", "sum"] = "weighted_sum",
) -> NDArray[np.double]:
    """Reduce expanded calculator forces to per-site forces.

    Forces returned by a calculator that received a mixture-expanded
    supercell carry one vector per constituent row. This routine
    collapses them to one vector per phonopy site.

    Two reduction conventions are supported:

    * ``"weighted_sum"`` (default): per-site force is the weighted sum

          F_site = sum_k (w_k * F_k)

      where ``w_k`` is the mixture weight stored in
      ``cell.species_table``. Use this when the calculator returns
      single-potential forces that have not yet been folded with mixture
      weights.
    * ``"sum"``: per-site force is the plain sum

          F_site = sum_k F_k

      Use this for VASP, where the VCA tag in INCAR averages potentials
      and the per-row forces in vasprun.xml already incorporate the
      mixture weights. A second multiplication by ``w_k`` would
      double-apply the weights.

    Parameters
    ----------
    forces : Sequence or NDArray[np.double]
        Either ``(..., n_expanded, 3)`` raw expanded forces, or
        ``(..., n_sites, 3)`` already-reduced forces. The latter is
        passed through unchanged so callers can be agnostic. Accepts a
        nested Python sequence, a sequence of arrays, or a single ndarray.
    cell : PhonopyAtoms
        Mixture cell that defines the expansion order.
    mode : {"weighted_sum", "sum"}, optional
        Reduction convention. Default is ``"weighted_sum"``.

    Returns
    -------
    NDArray[np.double]
        Forces of shape ``(..., n_sites, 3)``.

    """
    if mode not in ("weighted_sum", "sum"):
        raise ValueError(f'mode must be "weighted_sum" or "sum", got {mode!r}.')

    forces_arr: NDArray[np.double] = np.asarray(forces, dtype="double")
    n_sites = len(cell)
    site_indices, weights = get_mixture_expansion(cell)
    n_expanded = site_indices.size

    if forces_arr.shape[-2] == n_sites:
        return forces_arr
    if forces_arr.shape[-2] != n_expanded:
        raise ValueError(
            f"forces second-to-last axis ({forces_arr.shape[-2]}) must equal "
            f"either n_sites ({n_sites}) or n_expanded ({n_expanded})."
        )

    if mode == "weighted_sum":
        contribution = forces_arr * weights[:, None]
    else:  # mode == "sum"
        contribution = forces_arr

    # For each phonopy site, gather the expanded rows that belong to it
    # and sum them along the constituent axis.
    out_shape = forces_arr.shape[:-2] + (n_sites, 3)
    out = np.empty(out_shape, dtype="double")
    for i in range(n_sites):
        out[..., i, :] = contribution[..., site_indices == i, :].sum(axis=-2)
    return out
