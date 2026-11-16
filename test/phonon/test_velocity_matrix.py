# SPDX-License-Identifier: BSD-3-Clause
"""Tests for VelocityMatrix."""

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.grid import BZGrid
from phonopy.phonon.velocity_matrix import VelocityMatrix


def test_velocity_matrices_are_not_projected(ph_si: Phonopy):
    """Test that run does not overwrite the diagonal of velocity matrices.

    Group velocities are projected by the little group of q, but velocity
    matrices are not. The q-point is on Gamma-L, where the projection changes
    the group velocities of some bands, as the test asserts. At a q-point where
    the projection does nothing, this test would pass with the overwritten
    diagonal.

    """
    bz_grid = BZGrid(
        [9, 9, 9],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    q_point = np.array([1, 1, 1], dtype="double") / 9

    velocity.run([q_point])

    expected = velocity._calculate_velocity_matrix_at_q(q_point)
    expected_diagonal = np.einsum("aii->ia", expected).real
    projected = expected_diagonal @ velocity._get_projector(q_point).T

    assert not np.allclose(expected_diagonal, projected)
    np.testing.assert_allclose(velocity.velocity_matrices[0], expected, atol=1e-10)
    np.testing.assert_allclose(velocity.group_velocities[0], projected, atol=1e-10)


def test_velocity_matrices_degeneracy_lifting_direction(ph_si: Phonopy):
    """Test degeneracy_lifting_direction of VelocityMatrix.run by Si.

    The q-point is on Gamma-L, where transverse bands are doubly degenerate,
    so that velocity matrices depend on the lifting direction, as the test
    asserts. ``perturbation`` is a deprecated alias.

    """
    bz_grid = BZGrid(
        [9, 9, 9],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    q_points = [[0.1, 0.1, 0.1]]

    velocity.run(q_points, degeneracy_lifting_direction=[1, 0, 0])
    vm_100 = velocity.velocity_matrices
    velocity.run(q_points, degeneracy_lifting_direction=[0, 1, 0])
    vm_010 = velocity.velocity_matrices
    assert vm_100 is not None and vm_010 is not None
    assert not np.allclose(vm_100, vm_010, atol=1e-2)

    with pytest.warns(DeprecationWarning, match="degeneracy_lifting_direction"):
        velocity.run(q_points, perturbation=[1, 0, 0])
    np.testing.assert_array_equal(velocity.velocity_matrices, vm_100)
