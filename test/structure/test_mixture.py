# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.structure.mixture."""

import numpy as np
import pytest

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import build_mixture_cell
from phonopy.structure.mixture import get_mixture_expansion, reduce_mixture_forces


def test_get_mixture_expansion_GeSn_unitcell():
    """GeSn 50/50 unitcell expands to (Ge@site0, Ge@site1, Sn@site0, Sn@site1)."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])
    site_indices, weights = get_mixture_expansion(mixed_cell)

    np.testing.assert_array_equal(site_indices, [0, 1, 0, 1])
    np.testing.assert_allclose(weights, [0.5, 0.5, 0.5, 0.5])


def test_get_mixture_expansion_no_mixture_is_identity():
    """A regular cell expands to the identity mapping with unit weights."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Si", "Ge"],
    )
    site_indices, weights = get_mixture_expansion(cell)

    np.testing.assert_array_equal(site_indices, [0, 1])
    np.testing.assert_allclose(weights, [1.0, 1.0])


def test_get_mixture_expansion_mixed_with_pure_site():
    """A cell with one mixture site and one pure site preserves species_table order."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        symbols=["Ge", "Sn", "Si"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 1.0])
    # mixed_cell has 2 sites: site 0 = GeSn (mixture), site 1 = Si.
    site_indices, weights = get_mixture_expansion(mixed_cell)

    # Iteration order is species_table order; the mixture and the pure
    # species are emitted in whichever order the table holds them. The
    # contract is only that the (site, weight) pairs are consistent.
    assert sorted(zip(site_indices.tolist(), weights.tolist(), strict=True)) == [
        (0, 0.5),
        (0, 0.5),
        (1, 1.0),
    ]


def test_reduce_mixture_forces_GeSn_weighted_sum():
    """Per-site force is the weight-averaged sum over constituents."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    # Expanded forces: rows [Ge@s0, Ge@s1, Sn@s0, Sn@s1].
    expanded = np.array(
        [
            [1.0, 0.0, 0.0],  # Ge@s0
            [0.0, 2.0, 0.0],  # Ge@s1
            [3.0, 0.0, 0.0],  # Sn@s0
            [0.0, 4.0, 0.0],  # Sn@s1
        ],
        dtype="double",
    )

    reduced = reduce_mixture_forces(expanded, mixed_cell)

    np.testing.assert_allclose(
        reduced,
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
    )


def test_reduce_mixture_forces_sum_mode():
    """``mode="sum"`` adds constituent forces without reapplying weights.

    This is the VASP convention: vasprun.xml forces already incorporate
    the per-row mixture weight factor, so a plain sum across constituents at
    each site is the correct reduction.

    """
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    expanded = np.array(
        [
            [1.0, 0.0, 0.0],  # Ge@s0
            [0.0, 2.0, 0.0],  # Ge@s1
            [3.0, 0.0, 0.0],  # Sn@s0
            [0.0, 4.0, 0.0],  # Sn@s1
        ],
        dtype="double",
    )

    reduced = reduce_mixture_forces(expanded, mixed_cell, mode="sum")

    np.testing.assert_allclose(
        reduced,
        [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
    )


def test_reduce_mixture_forces_invalid_mode_raises():
    """Unknown reduction mode is rejected."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])
    forces = np.zeros((4, 3), dtype="double")
    with pytest.raises(ValueError, match='mode must be "weighted_sum" or "sum"'):
        reduce_mixture_forces(forces, mixed_cell, mode="weird")


def test_reduce_mixture_forces_passes_through_already_reduced():
    """Forces with shape (n_sites, 3) are returned unchanged."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    site_forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="double")
    reduced = reduce_mixture_forces(site_forces, mixed_cell)
    np.testing.assert_allclose(reduced, site_forces)


def test_reduce_mixture_forces_batched():
    """Reduction works across a leading num_supercells axis."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    expanded = np.zeros((3, 4, 3), dtype="double")
    expanded[0] = [[1, 0, 0], [0, 2, 0], [3, 0, 0], [0, 4, 0]]
    expanded[1] = [[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]
    expanded[2] = [[2, 0, 0], [0, 4, 0], [4, 0, 0], [0, 8, 0]]

    reduced = reduce_mixture_forces(expanded, mixed_cell)

    assert reduced.shape == (3, 2, 3)
    np.testing.assert_allclose(reduced[0], [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    np.testing.assert_allclose(reduced[1], [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(reduced[2], [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]])


def test_reduce_mixture_forces_shape_mismatch_raises():
    """Forces whose -2 axis does not match n_expanded or n_sites raise."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    bad = np.zeros((5, 3), dtype="double")
    with pytest.raises(ValueError, match="must equal"):
        reduce_mixture_forces(bad, mixed_cell)


def test_get_displacements_and_forces_handles_asymmetric_shape():
    """Type-1 dataset with raw expanded forces returns asymmetric arrays.

    For a mixture supercell the dataset stores per-site displacements
    and per-row (expanded) forces, so the helper must accept different
    second-axis lengths between the two arrays.

    """
    from phonopy.structure.dataset import get_displacements_and_forces

    # 2 mixture sites, n_expanded = 4 (Ge@s0, Ge@s1, Sn@s0, Sn@s1).
    raw_forces_disp0 = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype="double",
    )
    raw_forces_disp1 = np.array(
        [
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype="double",
    )
    dataset = {
        "natom": 2,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0.0, 0.0]),
                "forces": raw_forces_disp0,
            },
            {
                "number": 1,
                "displacement": np.array([0.0, 0.01, 0.0]),
                "forces": raw_forces_disp1,
            },
        ],
    }

    disps, forces = get_displacements_and_forces(dataset)

    assert disps.shape == (2, 2, 3)
    np.testing.assert_allclose(disps[0, 0], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(disps[1, 1], [0.0, 0.01, 0.0])
    assert forces is not None
    assert forces.shape == (2, 4, 3)
    np.testing.assert_allclose(forces[0], raw_forces_disp0)
    np.testing.assert_allclose(forces[1], raw_forces_disp1)
