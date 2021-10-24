"""Tests for group velocity calculation."""
import numpy as np

from phonopy import Phonopy
from phonopy.phonon.group_velocity import GroupVelocity


def test_gv_nacl(ph_nacl: Phonopy):
    """Test of GroupVelocity."""
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
