# SPDX-License-Identifier: BSD-3-Clause
"""Tests for VelocityMatrix."""

import numpy as np

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
