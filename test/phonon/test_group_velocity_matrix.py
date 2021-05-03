import numpy as np
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.group_velocity_matrix import GroupVelocityMatrix

def test_gvm_nacl(ph_nacl):
    qpoints = [[0.1, 0.1, 0.1]]
    gv = GroupVelocity(ph_nacl.dynamical_matrix,
                       symmetry=ph_nacl.primitive_symmetry)
    gv.run(qpoints)
    gvm = GroupVelocityMatrix(ph_nacl.dynamical_matrix,
                              symmetry=ph_nacl.primitive_symmetry)
    gvm.run(qpoints)
    gvs = []
    for mat in gvm.group_velocity_matrices[0]:
        gvs.append(np.diagonal(mat.real))
    gvs = np.transpose(gvs)
    np.testing.assert_allclose(gvs, gv.group_velocities[0], atol=1e-5)
