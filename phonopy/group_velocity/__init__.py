# Copyright (C) 2013 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from phonopy.units import VaspToTHz

class GroupVelocity:
    """
    d omega   ----
    ------- = \  / omega
    d q        \/q

    Gradient of omega in reciprocal space.

             d D(q)
    <e(q,nu)|------|e(q,nu)>
              d q
    """
    
    def __init__(self,
                 dynamical_matrix,
                 primitive,
                 q_points=None,
                 q_length=1e-4,
                 factor=VaspToTHz):
        """
        q_points is a list of sets of q-point and q-direction:
        [[q-point, q-direction], [q-point, q-direction], ...]

        q_length is used such as D(q + q_length) - D(q - q_length).
        """
        self._dynmat = dynamical_matrix
        self._reciprocal_lattice = np.linalg.inv(primitive.get_cell())
        self._q_points = q_points
        self._q_length = q_length
        self._factor = factor
        self._group_velocity = None
        self._eigenvectors = None
        self._frequencies = None
        if self._q_points is not None:
            self._set_group_velocity()
            
    def set_q_points(self, q_points):
        self._q_points = q_points
        self._set_group_velocity()

    def set_q_length(self, q_length):
        self._q_length = q_length

    def get_group_velocity(self):
        return self._group_velocity

    def _set_group_velocity(self):
        v_g = []
        for (q, n) in self._q_points:
            dD_at_q = get_group_velocity(q,
                                         n,
                                         self._q_length,
                                         self._dynmat,
                                         self._reciprocal_lattice,
                                         self._factor,
                                         self._frequencies,
                                         self._eigenvectors)
            v_g.append(dD_at_q)
        self._group_velocity = np.array(v_g)

def get_group_velocity(q,
                       n, # direction of dq
                       q_length, # finite distance in q
                       dynamical_matrix,
                       reciprocal_lattice,
                       factor=None,
                       frequencies=None,
                       eigenvectors=None):

    if frequencies is None or eigenvectors is None:
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * factor
    else:
        eigvecs = eigenvectors
        freqs = frequencies

    dD_at_q = []
    for dD_i in get_dD(np.array(q),
                       n,
                       dynamical_matrix,
                       reciprocal_lattice,
                       q_length): # (x, y, z)
        dD_i_at_q = [np.vdot(eigvec, np.dot(dD_i, eigvec)).real
                     for eigvec in eigvecs.T]
        dD_at_q.append(np.array(dD_i_at_q) / freqs / 2 * factor ** 2)
    return dD_at_q
        
def get_dD(q, n, dynamical_matrix, reciprocal_lattice, q_length):
    # The names of *c mean something in Cartesian.
    dynmat = dynamical_matrix
    rlat = reciprocal_lattice
    rlat_inv = np.linalg.inv(rlat)
    nc = np.dot(n, rlat)
    dqc = q_length * nc / np.linalg.norm(nc)
    ddm = []
    for dqc_i in np.diag(dqc):
        dq_i = np.dot(dqc_i, rlat_inv)
        dynmat.set_dynamical_matrix(q - dq_i)
        dm1 = dynmat.get_dynamical_matrix()
        dynmat.set_dynamical_matrix(q + dq_i)
        dm2 = dynmat.get_dynamical_matrix()
        ddm.append(dm2 - dm1)
    return [ddm_i / dpc_i for (ddm_i, dpc_i) in zip(ddm, 2 * dqc)]
