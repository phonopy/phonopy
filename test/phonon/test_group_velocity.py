# SPDX-License-Identifier: BSD-3-Clause
"""Tests for group velocity calculation."""

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.group_velocity import GroupVelocity


def test_gv_nacl(ph_nacl: Phonopy):
    """Test of GroupVelocity by NaCl (Gonze-Lee NAC).

    With the analytic Gonze-Lee derivative now available, this test
    goes through ``_get_dD_analytical``.  The reference values were
    originally captured against ``_get_dD_FD`` and agree with the
    analytic path to within ``atol=1e-5``.

    """
    gv_ref = [
        14.90162220,
        14.90162220,
        14.90162220,
        14.90162220,
        14.90162220,
        14.90162220,
        24.77046520,
        24.77046520,
        24.77046520,
        -2.17239664,
        -2.17239664,
        -2.17239664,
        -2.17239664,
        -2.17239664,
        -2.17239664,
        -3.05277585,
        -3.05277585,
        -3.05277585,
    ]
    gv = GroupVelocity(ph_nacl.dynamical_matrix, symmetry=ph_nacl.primitive_symmetry)
    gv.run([[0.1, 0.1, 0.1]])
    np.testing.assert_allclose(gv.group_velocities[0].ravel(), gv_ref, atol=1e-5)
    # for line in gv.group_velocities[0]:
    #     print("".join(["%.8f, " % v for v in line]))


def test_gv_nacl_wang(ph_nacl_wang: Phonopy):
    """Test of GroupVelocity by NaCl with Wang's NAC method.

    This test should pass _get_dD_analytical.

    """
    gv_ref = [
        14.56800976,
        14.56800976,
        14.56800976,
        14.56800976,
        14.56800976,
        14.56800976,
        25.16351730,
        25.16351730,
        25.16351730,
        1.51378156,
        1.51378156,
        1.51378156,
        1.51378156,
        1.51378156,
        1.51378156,
        -7.84946438,
        -7.84946438,
        -7.84946438,
    ]
    gv = GroupVelocity(
        ph_nacl_wang.dynamical_matrix, symmetry=ph_nacl_wang.primitive_symmetry
    )
    gv.run([[0.1, 0.1, 0.1]])
    # for line in gv.group_velocities[0]:
    #     print("".join(["%.8f, " % v for v in line]))
    np.testing.assert_allclose(gv.group_velocities[0].ravel(), gv_ref, atol=1e-5)


def test_gv_si(ph_si: Phonopy):
    """Test of GroupVelocity by Si.

    This test should pass _get_dD_analytical.

    """
    gv_ref = [
        17.06443768,
        17.06443768,
        17.06443768,
        17.06443768,
        17.06443768,
        17.06443768,
        46.95145125,
        46.95145125,
        46.95145125,
        -3.59278449,
        -3.59278449,
        -3.59278449,
        -2.39847202,
        -2.39847202,
        -2.39847202,
        -2.39847202,
        -2.39847202,
        -2.39847202,
    ]
    gv = GroupVelocity(ph_si.dynamical_matrix, symmetry=ph_si.primitive_symmetry)
    gv.run([[0.1, 0.1, 0.1]])
    np.testing.assert_allclose(gv.group_velocities[0].ravel(), gv_ref, atol=1e-5)


def test_gv_si_little_group(ph_si: Phonopy):
    """Test symmetrization of GroupVelocity by Si at Gamma-L and W.

    At these q-points, the result changes when the operations are selected
    by R^T q = q instead of R q = q.

    """
    v1, v2, v3, v4 = 17.06443768, 46.95145125, 3.59278449, 2.39847202
    gv_gl = np.array([v1, v1, v2, -v3, -v4, -v4])[:, None] * [-1, 1, 1]
    v1, v2, v3 = 24.82100783, 16.28626763, 1.04815965
    gv_w = np.array([[0, 0, -v1], [0, 0, v1], [0, 0, -v2], [0, 0, v2]])
    gv_w = np.vstack([gv_w, [[0, 0, -v3], [0, 0, v3]]])
    gv = GroupVelocity(ph_si.dynamical_matrix, symmetry=ph_si.primitive_symmetry)
    gv.run([[0.1, 0, 0], [0.5, 0.25, 0.75]])
    np.testing.assert_allclose(gv.group_velocities[0], gv_gl, atol=1e-5)
    np.testing.assert_allclose(gv.group_velocities[1], gv_w, atol=1e-5)


def test_gv_si_degeneracy_lifting_direction(ph_si: Phonopy):
    """Test degeneracy_lifting_direction of GroupVelocity.run by Si.

    The q-point is on Gamma-L, where transverse bands are doubly degenerate.
    Without symmetrization, their group velocities depend on the lifting
    direction, as the test asserts, and analytical and finite-difference
    derivatives have to agree for the same direction.

    """
    q_points = [[0.1, 0.1, 0.1]]
    gv = GroupVelocity(ph_si.dynamical_matrix)
    gv.run(q_points, degeneracy_lifting_direction=[1, 0, 0])
    gv_100 = gv.group_velocities
    gv.run(q_points, degeneracy_lifting_direction=[0, 1, 0])
    gv_010 = gv.group_velocities
    assert gv_100 is not None and gv_010 is not None
    assert not np.allclose(gv_100, gv_010, atol=1e-2)

    gv_fd = GroupVelocity(ph_si.dynamical_matrix, q_length=1e-5)
    gv_fd.run(q_points, degeneracy_lifting_direction=[1, 0, 0])
    np.testing.assert_allclose(gv_fd.group_velocities, gv_100, atol=1e-4)


def test_gv_si_perturbation_is_deprecated(ph_si: Phonopy):
    """Test that perturbation of GroupVelocity.run is a deprecated alias."""
    q_points = [[0.1, 0.1, 0.1]]
    gv = GroupVelocity(ph_si.dynamical_matrix)
    gv.run(q_points, degeneracy_lifting_direction=[1, 0, 0])
    gv_ref = gv.group_velocities
    gv.run(q_points)
    assert not np.allclose(gv.group_velocities, gv_ref, atol=1e-2)

    with pytest.warns(DeprecationWarning, match="degeneracy_lifting_direction"):
        gv.run(q_points, perturbation=[1, 0, 0])
    np.testing.assert_array_equal(gv.group_velocities, gv_ref)

    with pytest.raises(ValueError, match="cannot be given together"):
        gv.run(q_points, perturbation=[1, 0, 0], degeneracy_lifting_direction=[1, 0, 0])


@pytest.mark.parametrize("q_length", [None, 1e-5])
def test_gv_si_many_qpoints(ph_si: Phonopy, q_length: float | None):
    """Test that GroupVelocity.run at many q-points agrees with one by one.

    Derivatives of dynamical matrices of all q-points are computed before
    the loop over q-points. The group velocities differ among the q-points,
    as the test asserts, so that results stored in a wrong order are
    detected. The first two q-points have degenerate bands.

    """
    q_points = [[0.1, 0.1, 0.1], [0.5, 0.25, 0.75], [0.1, 0.2, 0.3]]
    gv = GroupVelocity(ph_si.dynamical_matrix, q_length=q_length)
    gv.run(q_points)
    gv_at_once = gv.group_velocities
    assert gv_at_once is not None
    assert gv_at_once.shape == (3, 6, 3)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not np.allclose(gv_at_once[i], gv_at_once[j], atol=1e-2)

    for i, q_point in enumerate(q_points):
        gv.run([q_point])
        assert gv.group_velocities is not None
        np.testing.assert_allclose(
            gv_at_once[i], gv.group_velocities[0], rtol=0, atol=1e-8
        )
