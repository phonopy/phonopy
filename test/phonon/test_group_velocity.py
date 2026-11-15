# SPDX-License-Identifier: BSD-3-Clause
"""Tests for group velocity calculation."""

import numpy as np

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
