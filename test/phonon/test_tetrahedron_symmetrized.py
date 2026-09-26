# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the tetrahedra rotated by the point group."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from phonopy.phonon.grid import (
    BZGrid,
    get_grid_point_from_address,
    get_neighboring_grid_points,
)
from phonopy.phonon.tetrahedron_method import (
    TetrahedronMethod,
    _get_tetrahedra_relative_grid_address,
    get_integration_weights,
    get_symmetrized_tetrahedra_relative_grid_address,
    get_tetrahedra_frequencies,
    get_tetrahedra_relative_gr_grid_address,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

pytest.importorskip("phonors")


def _cell(name: str) -> PhonopyAtoms:
    a = 4.0
    if name == "sc":
        return PhonopyAtoms(["Cu"], cell=np.eye(3) * a, scaled_positions=[[0, 0, 0]])
    if name == "fcc":
        lattice = (np.ones((3, 3)) - np.eye(3)) * a / 2
        return PhonopyAtoms(["Cu"], cell=lattice, scaled_positions=[[0, 0, 0]])
    if name == "bcc":
        lattice = (np.ones((3, 3)) - 2 * np.eye(3)) * a / 2
        return PhonopyAtoms(["Fe"], cell=lattice, scaled_positions=[[0, 0, 0]])
    return PhonopyAtoms(
        ["Ti", "Ti"],
        cell=[[2.95, 0, 0], [-1.475, 2.95 * np.sqrt(3) / 2, 0], [0, 0, 4.68]],
        scaled_positions=[[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]],
    )


def _bz_grid(name: str, mesh: list[int]) -> BZGrid:
    cell = _cell(name)
    return BZGrid(
        mesh,
        lattice=cell.cell,
        symmetry_dataset=Symmetry(cell).dataset,
        use_grg=True,
    )


def _band(name: str, bz_grid: BZGrid) -> NDArray[np.double]:
    """Return a band invariant under the point group on all BZ-grid points.

    Sum of cos(2 pi t.q) over the lattice vectors t of the three shortest
    lengths, cut by length so that the set of t is closed under the group.

    shape=(bz_grid_points, 1)

    """
    lattice = _cell(name).cell
    r = range(-4, 5)
    t = np.array([[i, j, k] for i in r for j in r for k in r])
    lengths = np.linalg.norm(t @ lattice, axis=1)
    cutoff = np.unique(np.round(lengths, 6))[3] + 1e-6
    t, lengths = t[lengths < cutoff], lengths[lengths < cutoff]
    q = bz_grid.addresses @ bz_grid.QDinv.T
    band = (np.exp(-lengths) * np.cos(2 * np.pi * q @ t.T)).sum(axis=1)
    return np.array(band[:, None], dtype="double", order="C")


def _max_star_difference(bz_grid: BZGrid, weights: NDArray[np.double]) -> float:
    """Return max |w(Rq) - w(q)| over R and q, relative to max |w|.

    weights : shape=(regular_grid_points, sampling_points, num_band), rows in
        the order of the GR-grid index.

    """
    addresses = bz_grid.addresses[bz_grid.grg2bzg]
    diff = 0.0
    for r in bz_grid.rotations:
        rotated = get_grid_point_from_address(
            addresses @ r.T, bz_grid.D_diag, lang="Python"
        )
        diff = max(diff, float(np.abs(weights[rotated] - weights).max()))
    return diff / float(np.abs(weights).max())


@pytest.mark.parametrize(
    "name,mesh,num_sets",
    [
        ("sc", [4, 4, 4], 4),
        ("fcc", [4, 4, 4], 1),
        ("bcc", [4, 4, 4], 12),
        ("hcp", [6, 6, 4], 6),
    ],
)
def test_get_symmetrized_tetrahedra_relative_grid_address(
    name: str, mesh: list[int], num_sets: int
):
    """Number of distinct sets of 24 tetrahedra in the point-group orbit."""
    bz_grid = _bz_grid(name, mesh)
    relative_grid_address = get_symmetrized_tetrahedra_relative_grid_address(bz_grid)
    assert relative_grid_address.shape == (24 * num_sets, 4, 3)
    np.testing.assert_array_equal(relative_grid_address[:, 0], 0)


@pytest.mark.parametrize("symmetrize_tetrahedra", [False, True])
def test_get_tetrahedra_relative_gr_grid_address(symmetrize_tetrahedra: bool):
    """The 24 tetrahedra in GR-grid coordinates, or their point-group orbit."""
    bz_grid = _bz_grid("hcp", [6, 6, 4])
    if symmetrize_tetrahedra:
        expected = get_symmetrized_tetrahedra_relative_grid_address(bz_grid)
    else:
        expected = np.dot(
            _get_tetrahedra_relative_grid_address(bz_grid.microzone_lattice),
            bz_grid.P.T,
        )
    np.testing.assert_array_equal(
        get_tetrahedra_relative_gr_grid_address(
            bz_grid, symmetrize_tetrahedra=symmetrize_tetrahedra
        ),
        expected,
    )


@pytest.mark.parametrize("function", ["I", "J"])
def test_integration_weights_symmetrized_hcp(function: str):
    """Symmetrized weights agree over the star; the default ones do not."""
    bz_grid = _bz_grid("hcp", [6, 6, 4])
    band = _band("hcp", bz_grid)
    sampling_points = np.linspace(band.min(), band.max(), 20)
    kwargs = {"function": function, "lang": "Rust"}
    weights = get_integration_weights(sampling_points, band, bz_grid, **kwargs)
    weights_sym = get_integration_weights(
        sampling_points, band, bz_grid, symmetrize_tetrahedra=True, **kwargs
    )
    assert _max_star_difference(bz_grid, weights) > 1e-3
    assert _max_star_difference(bz_grid, weights_sym) < 1e-12


def test_integration_weights_symmetrized_fcc():
    """In fcc the default tetrahedra are already symmetric."""
    bz_grid = _bz_grid("fcc", [4, 4, 4])
    band = _band("fcc", bz_grid)
    sampling_points = np.linspace(band.min(), band.max(), 20)
    weights = get_integration_weights(sampling_points, band, bz_grid, lang="Rust")
    weights_sym = get_integration_weights(
        sampling_points, band, bz_grid, lang="Rust", symmetrize_tetrahedra=True
    )
    np.testing.assert_allclose(weights_sym, weights, atol=1e-12)


@pytest.mark.parametrize("name,mesh", [("bcc", [4, 4, 4]), ("hcp", [6, 6, 4])])
def test_get_tetrahedra_frequencies(name: str, mesh: list[int]):
    """Frequencies at the vertices, for 24 tetrahedra and for 24 * n."""
    bz_grid = _bz_grid(name, mesh)
    rng = np.random.default_rng(0)
    frequencies = rng.random((len(bz_grid.addresses), 3))
    tables = (
        get_tetrahedra_relative_gr_grid_address(bz_grid),
        get_tetrahedra_relative_gr_grid_address(bz_grid, symmetrize_tetrahedra=True),
    )
    for table in tables:
        for gp in (0, 5, len(bz_grid.addresses) - 1):
            vertex_frequencies = get_tetrahedra_frequencies(
                gp, bz_grid, table, frequencies
            )
            assert vertex_frequencies.shape == (3, len(table), 4)
            vertices = get_neighboring_grid_points(gp, table, bz_grid)
            np.testing.assert_array_equal(
                vertex_frequencies, np.moveaxis(frequencies[vertices], -1, 0)
            )


@pytest.mark.parametrize("function", ["I", "J"])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_TetrahedronMethod_matches_rust(function: str, symmetrize: bool):
    """The pure-Python weights equal the Rust ones, also where vertices nearly tie.

    The band is rounded to integers and shifted by noise of 1e-12, so that many
    vertex values differ by less than the guard of 1e-10 and the weights are
    evaluated next to them. Both implementations drop the vanishing
    denominators there; without that the weights diverge.

    """
    bz_grid = _bz_grid("hcp", [6, 6, 4])
    rng = np.random.default_rng(0)
    band = np.round(4 * _band("hcp", bz_grid))
    sampling_points = np.unique(band)
    # Noise per GR-grid point, so translationally equivalent BZ-grid points,
    # which the two implementations may pick differently, keep equal values.
    noise = 1e-12 * rng.standard_normal((np.prod(bz_grid.D_diag), 1))
    band += noise[bz_grid.bzg2grg]
    weights_rust = get_integration_weights(
        sampling_points,
        band,
        bz_grid,
        function=function,
        lang="Rust",
        symmetrize_tetrahedra=symmetrize,
    )
    table = get_tetrahedra_relative_gr_grid_address(
        bz_grid, symmetrize_tetrahedra=symmetrize
    )
    thm = TetrahedronMethod(None, relative_grid_address=table)
    for i, gp in enumerate(bz_grid.grg2bzg):
        vertex_band = get_tetrahedra_frequencies(gp, bz_grid, table, band)
        thm.set_tetrahedra_omegas(vertex_band[0])
        thm.run(sampling_points, value=function)
        np.testing.assert_allclose(
            thm.get_integration_weight(), weights_rust[i, :, 0], atol=1e-12
        )


@pytest.mark.parametrize("function", ["I", "J"])
def test_TetrahedronMethod_at_vertices(function: str):
    """The pure-Python weights equal the Rust ones at sampling points on vertices.

    The sampling points are the values of the band at grid points, as the
    frequencies of the grid point itself are in the isotope scattering, so they
    equal vertex values bitwise.

    """
    bz_grid = _bz_grid("hcp", [6, 6, 4])
    band = _band("hcp", bz_grid)
    sampling_points = band[bz_grid.grg2bzg[:10], 0]
    weights = get_integration_weights(
        sampling_points, band, bz_grid, function=function, lang="Rust"
    )
    table = get_tetrahedra_relative_gr_grid_address(bz_grid)
    thm = TetrahedronMethod(None, relative_grid_address=table)
    for i, gp in enumerate(bz_grid.grg2bzg):
        vertex_band = get_tetrahedra_frequencies(gp, bz_grid, table, band)
        thm.set_tetrahedra_omegas(vertex_band[0])
        thm.run(sampling_points, value=function)
        np.testing.assert_allclose(
            thm.get_integration_weight(), weights[i, :, 0], atol=1e-12
        )
