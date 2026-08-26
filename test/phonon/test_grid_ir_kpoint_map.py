# SPDX-License-Identifier: BSD-3-Clause
"""Tests for mapping a calculator's ir-kpoints onto a BZGrid.

Kept out of test_grid.py, which skips itself without the C extension because
most of its tests drive the C kernels directly. get_ir_kpoint_map has no C
path, so gating it on C would leave it untested in exactly the environment
that is becoming the default.

"""

from __future__ import annotations

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.grid import (
    BZGrid,
    get_grid_shift_from_kpoints,
    get_ir_grid_points,
    get_ir_kpoint_map,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

# AlN, P6_3mc. Hexagonal on purpose: the doubly degenerate branches make it
# the case where an eigenvector-order convention would be ambiguous, and its
# 5x5x4 mesh is the one test_grid.py already uses, so the 15 irreducible grid
# points below can be checked against test_aln_BZGrid_with_shift.
MESH = [5, 5, 4]
NUM_IR_GRID_POINTS = 15


def _bz_grid(aln_cell: PhonopyAtoms, **kwargs) -> BZGrid:
    """Return the AlN 5x5x4 BZGrid these tests map onto."""
    return BZGrid(
        MESH,
        lattice=aln_cell.cell,
        symmetry_dataset=Symmetry(aln_cell).dataset,
        **kwargs,
    )


def _ir_kpoints(bz_grid: BZGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return ir-kpoints and normalized weights as a calculator would give them.

    Derived from the grid rather than hard-coded, so that these tests measure
    the mapping instead of a k-point list that would have to be regenerated
    whenever a grid convention changes.

    """
    ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
    addresses = bz_grid.addresses[bz_grid.grg2bzg[ir_grid_points]]
    kpoints = (2 * addresses + bz_grid.PS) / 2.0 @ bz_grid.QDinv.T
    return kpoints, ir_grid_weights / ir_grid_weights.sum()


def test_get_ir_kpoint_map(aln_cell: PhonopyAtoms):
    """Test mapping ir-kpoints already in the grid's own order."""
    bz_grid = _bz_grid(aln_cell)
    ir_grid_points, _, _ = get_ir_grid_points(bz_grid)
    assert len(ir_grid_points) == NUM_IR_GRID_POINTS

    kpoints, weights = _ir_kpoints(bz_grid)
    id_map = get_ir_kpoint_map(kpoints, weights, bz_grid)

    np.testing.assert_array_equal(id_map, np.arange(NUM_IR_GRID_POINTS))


def test_get_ir_kpoint_map_reorders(aln_cell: PhonopyAtoms):
    """Test that an arbitrary k-point order is recovered.

    A calculator is under no obligation to list its k-points in phonopy's
    order, which is the whole reason the mapping exists.

    """
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)
    permutation = np.random.default_rng(0).permutation(len(kpoints))

    id_map = get_ir_kpoint_map(kpoints[permutation], weights[permutation], bz_grid)

    # Applying the map to the permuted k-points undoes the permutation.
    np.testing.assert_array_equal(permutation[id_map], np.arange(len(kpoints)))


def test_get_ir_kpoint_map_accepts_counts_or_weights(aln_cell: PhonopyAtoms):
    """Test that weights may be counts or weights normalized to one."""
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)
    _, ir_grid_weights, _ = get_ir_grid_points(bz_grid)

    np.testing.assert_array_equal(
        get_ir_kpoint_map(kpoints, ir_grid_weights, bz_grid),
        get_ir_kpoint_map(kpoints, weights, bz_grid),
    )


def test_get_ir_kpoint_map_maps_a_shifted_grid(aln_cell: PhonopyAtoms):
    """Test that a mesh shifted by half a division maps as a centred one does.

    The half shift moves every k-point equally, so the addresses come back
    after subtracting it. AlN takes a shift along c, which its point group
    preserves.

    """
    shifted = _bz_grid(aln_cell, is_shift=[0, 0, 1])
    kpoints, weights = _ir_kpoints(shifted)
    ir_grid_points, _, _ = get_ir_grid_points(shifted)

    id_map = get_ir_kpoint_map(kpoints, weights, shifted)

    np.testing.assert_array_equal(id_map, np.arange(len(ir_grid_points)))


def test_get_ir_kpoint_map_rejects_a_grid_shifted_differently(aln_cell: PhonopyAtoms):
    """Test that k-points off a grid's own shift raise rather than map.

    The centred k-points sit half a division from every point of the shifted
    grid, which is as far off as a k-point can be.

    """
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)
    shifted = _bz_grid(aln_cell, is_shift=[0, 0, 1])

    with pytest.raises(ValueError, match="does not lie on the grid"):
        get_ir_kpoint_map(kpoints, weights, shifted)


def test_get_grid_shift_from_kpoints(aln_cell: PhonopyAtoms):
    """Test that the shift is read off the k-points themselves.

    What a calculator writes about its mesh does not say where it placed the
    points, so the points have to.

    """
    centred = _bz_grid(aln_cell)
    shifted = _bz_grid(aln_cell, is_shift=[0, 0, 1])

    np.testing.assert_array_equal(
        get_grid_shift_from_kpoints(_ir_kpoints(centred)[0], centred), [0, 0, 0]
    )
    # Read against the centred grid: QDinv does not depend on the shift, which
    # is what lets the shift be found before the grid that carries it is built.
    np.testing.assert_array_equal(
        get_grid_shift_from_kpoints(_ir_kpoints(shifted)[0], centred), [0, 0, 1]
    )


def test_get_grid_shift_from_kpoints_rejects_an_irregular_list(aln_cell: PhonopyAtoms):
    """Test that k-points that are neither on nor half off the grid raise."""
    bz_grid = _bz_grid(aln_cell)
    kpoints, _ = _ir_kpoints(bz_grid)

    with pytest.raises(ValueError, match="not a regular mesh"):
        get_grid_shift_from_kpoints(kpoints + 0.01, bz_grid)


def test_get_ir_kpoint_map_rejects_off_grid_kpoints(aln_cell: PhonopyAtoms):
    """Test that k-points off the mesh raise rather than being rounded onto it.

    This is what a band path or an explicit k-point list looks like from
    inside the mapping.

    """
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)

    with pytest.raises(ValueError, match="does not lie on the grid"):
        get_ir_kpoint_map(kpoints + 0.01, weights, bz_grid)


def test_get_ir_kpoint_map_rejects_monkhorst_pack_kpoints(aln_cell: PhonopyAtoms):
    """Test that a half-shifted Monkhorst-Pack mesh misses a centred grid.

    A Monkhorst-Pack mesh with even divisions is written by VASP exactly as
    one with odd divisions -- mode 'm', zero shift -- and only its k-point
    coordinates differ, every one of them landing on a half-integer address.
    Nothing in the input description separates the two, so the k-points
    themselves have to, which is what get_grid_shift_from_kpoints reads and
    what the grid then has to be built with. Handed the centred grid instead,
    the mapping raises rather than pairing eigenvalues with the wrong points.

    Measured on a real 8x8x8 Monkhorst-Pack calculation, whose k-points are
    at 0.0625 + n/8.

    """
    bz_grid = _bz_grid(aln_cell)
    _, weights = _ir_kpoints(bz_grid)
    kpoints, _ = _ir_kpoints(bz_grid)
    monkhorst_pack = kpoints + 0.5 / bz_grid.D_diag

    with pytest.raises(ValueError, match="does not lie on the grid"):
        get_ir_kpoint_map(monkhorst_pack, weights, bz_grid)


def test_get_ir_kpoint_map_rejects_wrong_kpoint_count(aln_cell: PhonopyAtoms):
    """Test that a disagreeing symmetry reduction raises."""
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)

    with pytest.raises(ValueError, match="symmetry reductions disagree"):
        get_ir_kpoint_map(kpoints[:-1], weights[:-1], bz_grid)


def test_get_ir_kpoint_map_rejects_wrong_weights(aln_cell: PhonopyAtoms):
    """Test that weights disagreeing with the grid's raise.

    The number of k-points can be right while the weights are wrong, which is
    what a symmetry reduction that spglib and the calculator disagree about
    looks like. This is the check that catches it.

    """
    bz_grid = _bz_grid(aln_cell)
    kpoints, _ = _ir_kpoints(bz_grid)
    uniform = np.full(len(kpoints), 1.0 / len(kpoints))

    with pytest.raises(ValueError, match="symmetry weights disagree"):
        get_ir_kpoint_map(kpoints, uniform, bz_grid)


def test_get_ir_kpoint_map_rejects_duplicate_kpoints(aln_cell: PhonopyAtoms):
    """Test that two k-points reducing to one grid point raise."""
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)
    duplicated = np.vstack([kpoints[:-1], kpoints[0:1]])

    with pytest.raises(ValueError, match="both map to"):
        get_ir_kpoint_map(duplicated, weights, bz_grid)


def test_get_ir_kpoint_map_generalized_regular_grid(ph_tio2: Phonopy):
    """Test the mapping on a grid whose generating matrix is not diagonal.

    Anatase TiO2 is body-centred tetragonal, so a mesh that is regular in the
    conventional setting becomes a generalized regular grid in the primitive
    one. Here neither P nor Q of the Smith normal form is the identity, which
    is what separates the general formula from the diagonal special case:
    ``kpoints @ grid_matrix`` misses an integer address by half a division on
    this grid.

    The grid matrix below is what phonopy generates at this mesh length and
    is also what a VASP calculation on this cell writes into
    ``input/kpoints/basis_vectors`` as ``rint(inv(basis_vectors.T))``.

    """
    primitive = ph_tio2.primitive
    bz_grid = BZGrid(
        30.0,
        lattice=primitive.cell,
        symmetry_dataset=ph_tio2.primitive_symmetry.dataset,
        use_grg=True,
    )
    assert bz_grid.grid_matrix is not None
    np.testing.assert_array_equal(
        bz_grid.grid_matrix, [[0, 8, 8], [8, 0, 8], [3, 3, 0]]
    )
    assert not np.array_equal(bz_grid.Q, np.eye(3, dtype=int))
    assert not np.array_equal(bz_grid.P, np.eye(3, dtype=int))

    ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
    assert len(ir_grid_points) == 50
    assert ir_grid_weights.sum() == 384

    addresses = bz_grid.addresses[bz_grid.grg2bzg[ir_grid_points]]
    kpoints = addresses @ bz_grid.QDinv.T
    weights = ir_grid_weights / ir_grid_weights.sum()

    np.testing.assert_array_equal(
        get_ir_kpoint_map(kpoints, weights, bz_grid), np.arange(50)
    )

    permutation = np.random.default_rng(2).permutation(50)
    id_map = get_ir_kpoint_map(kpoints[permutation], weights[permutation], bz_grid)
    np.testing.assert_array_equal(permutation[id_map], np.arange(50))


def test_get_ir_kpoint_map_rejects_bad_shapes(aln_cell: PhonopyAtoms):
    """Test the shape guards on the two input arrays."""
    bz_grid = _bz_grid(aln_cell)
    kpoints, weights = _ir_kpoints(bz_grid)

    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        get_ir_kpoint_map(kpoints[:, :2], weights, bz_grid)
    with pytest.raises(ValueError, match="one value per k-point"):
        get_ir_kpoint_map(kpoints, weights[:-1], bz_grid)
