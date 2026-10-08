# SPDX-License-Identifier: BSD-3-Clause
"""Tests for forces of the merge scheme of site mixture."""

import numpy as np
import pytest

from phonopy.api_phonopy import _reduce_dataset_forces_to_sites
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, merge_weighted_species


def _get_GeSn_weighted_cell() -> PhonopyAtoms:
    """Return GeSn 50/50 with atoms Ge@s0, Ge@s1, Sn@s0, Sn@s1."""
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
    return apply_site_mixture(cell, [0.5, 0.5, 0.5, 0.5])


# Forces on the atoms [Ge@s0, Ge@s1, Sn@s0, Sn@s1].
_atom_forces = np.array(
    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
    dtype="double",
)


@pytest.mark.parametrize(
    "mode,site_forces",
    [
        ("weighted_sum", [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ("sum", [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0]]),
    ],
)
def test_reduce_dataset_forces_to_sites_type1(mode, site_forces):
    """Forces on the atoms of each site are summed, with weights or not.

    ``mode="sum"`` is the VASP convention: vasprun.xml forces already
    incorporate the weight of each species row.

    """
    cell = _get_GeSn_weighted_cell()
    _, site_indices = merge_weighted_species(cell)
    dataset = {
        "natom": 2,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0, 0]),
                "forces": _atom_forces,
            }
        ],
    }
    reduced = _reduce_dataset_forces_to_sites(dataset, cell, site_indices, mode=mode)
    np.testing.assert_allclose(reduced["first_atoms"][0]["forces"], site_forces)
    # The raw forces in the input dataset are kept.
    np.testing.assert_allclose(dataset["first_atoms"][0]["forces"], _atom_forces)


def test_reduce_dataset_forces_to_sites_type2():
    """Forces of a type-2 dataset are summed per site for every snapshot."""
    cell = _get_GeSn_weighted_cell()
    _, site_indices = merge_weighted_species(cell)
    dataset = {
        "displacements": np.zeros((2, 2, 3)),
        "forces": np.array([_atom_forces, 2 * _atom_forces]),
    }
    reduced = _reduce_dataset_forces_to_sites(dataset, cell, site_indices, mode="sum")
    np.testing.assert_allclose(
        reduced["forces"],
        [
            [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
            [[8.0, 0.0, 0.0], [0.0, 12.0, 0.0]],
        ],
    )


def test_reduce_dataset_forces_to_sites_shape_mismatch_raises():
    """Forces must be those on the atoms of the unmerged supercell."""
    cell = _get_GeSn_weighted_cell()
    _, site_indices = merge_weighted_species(cell)
    dataset = {
        "natom": 2,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0, 0]),
                "forces": np.zeros((2, 3)),
            }
        ],
    }
    with pytest.raises(RuntimeError, match="do not match the 4 atoms"):
        _reduce_dataset_forces_to_sites(dataset, cell, site_indices)


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
