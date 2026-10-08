# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for mixed-species (site-mixture) cells.

The cell construction utilities (``build_mixture_cell``,
``merge_weighted_species``, ``build_species_table_from_mixtures``) live with
the cell / atoms modules.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


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
