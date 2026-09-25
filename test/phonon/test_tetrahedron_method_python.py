# SPDX-License-Identifier: BSD-3-Clause
"""Tests for TetrahedronMethod, the pure-Python tetrahedron method."""

import numpy as np
import pytest

from phonopy.phonon.tetrahedron_method import (
    TetrahedronMethod,
    _get_relative_grid_addresses_from_main_diagonal,
)

freqs = [7.75038996, 8.45225776]
# Frequencies at the vertices of the 24 tetrahedra, the central one first.
tetra_freqs = [
    [8.31845176, 8.69248151, 8.78939432, 8.66179133],
    [8.31845176, 8.69248151, 8.57211855, 8.66179133],
    [8.31845176, 8.3073908, 8.78939432, 8.66179133],
    [8.31845176, 8.3073908, 8.16360975, 8.66179133],
    [8.31845176, 8.15781566, 8.57211855, 8.66179133],
    [8.31845176, 8.15781566, 8.16360975, 8.66179133],
    [8.31845176, 8.3073908, 8.16360975, 7.23665561],
    [8.31845176, 8.15781566, 8.16360975, 7.23665561],
    [8.31845176, 8.69248151, 8.57211855, 8.25247917],
    [8.31845176, 8.15781566, 8.57211855, 8.25247917],
    [8.31845176, 8.15781566, 7.40609306, 8.25247917],
    [8.31845176, 8.15781566, 7.40609306, 7.23665561],
    [8.31845176, 8.69248151, 8.78939432, 8.55165578],
    [8.31845176, 8.3073908, 8.78939432, 8.55165578],
    [8.31845176, 8.3073908, 7.56474684, 8.55165578],
    [8.31845176, 8.3073908, 7.56474684, 7.23665561],
    [8.31845176, 8.69248151, 8.60076148, 8.55165578],
    [8.31845176, 8.69248151, 8.60076148, 8.25247917],
    [8.31845176, 7.72920193, 8.60076148, 8.55165578],
    [8.31845176, 7.72920193, 8.60076148, 8.25247917],
    [8.31845176, 7.72920193, 7.56474684, 8.55165578],
    [8.31845176, 7.72920193, 7.56474684, 7.23665561],
    [8.31845176, 7.72920193, 7.40609306, 8.25247917],
    [8.31845176, 7.72920193, 7.40609306, 7.23665561],
]
iw_I_ref = [0.37259443, 1.79993056]
iw_J_ref = [0.05740597, 0.76331859]


def _central_vertex_first() -> np.ndarray:
    """Return 24 tetrahedra with the vertex [0, 0, 0] first in each.

    The reference weights were made with the central vertex first.

    shape=(24, 4, 3)

    """
    relative_grid_address, central_indices = (
        _get_relative_grid_addresses_from_main_diagonal(0)
    )
    return np.array(
        [
            np.roll(t, -ci, axis=0)
            for t, ci in zip(relative_grid_address, central_indices, strict=True)
        ]
    )


@pytest.mark.parametrize("value,ref", [("I", iw_I_ref), ("J", iw_J_ref)])
def test_TetrahedronMethod(value: str, ref: list[float]):
    """Weights at several frequencies and at one."""
    thm = TetrahedronMethod(None, relative_grid_address=_central_vertex_first())
    thm.set_tetrahedra_omegas(tetra_freqs)
    thm.run(freqs, value=value)
    np.testing.assert_allclose(thm.get_integration_weight(), ref, atol=1e-5)
    for f, r in zip(freqs, ref, strict=True):
        thm.run(f, value=value)
        np.testing.assert_allclose(thm.get_integration_weight(), r, atol=1e-5)


@pytest.mark.parametrize("value", ["I", "J"])
def test_TetrahedronMethod_repeated_set(value: str):
    """Two copies of the same 24 tetrahedra give the weight of one copy."""
    table = _central_vertex_first()
    thm = TetrahedronMethod(None, relative_grid_address=table)
    thm.set_tetrahedra_omegas(tetra_freqs)
    thm.run(freqs, value=value)
    single = thm.get_integration_weight()

    thm2 = TetrahedronMethod(None, relative_grid_address=np.concatenate([table, table]))
    thm2.set_tetrahedra_omegas(np.concatenate([tetra_freqs, tetra_freqs]))
    thm2.run(freqs, value=value)
    np.testing.assert_allclose(thm2.get_integration_weight(), single, atol=1e-14)


def test_TetrahedronMethod_central_vertex_found():
    """The central vertex is found wherever it sits in each tetrahedron."""
    relative_grid_address, central_indices = (
        _get_relative_grid_addresses_from_main_diagonal(0)
    )
    thm = TetrahedronMethod(None)
    np.testing.assert_array_equal(thm.tetrahedra, relative_grid_address)
    np.testing.assert_array_equal(thm._central_indices, central_indices)
