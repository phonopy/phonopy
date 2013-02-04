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
                 phonon,
                 q_points=None,
                 q_length=1e-4,
                 factor=VaspToTHz):
        """
        q_points is a list of sets of q-point and q-direction:
        [[q-point, q-direction], [q-point, q-direction], ...]
        """
        
        self._phonon = phonon
        self._dynmat = phonon.get_dynamical_matrix()
        self._reciprocal_lattice = np.linalg.inv(
            self._phonon.get_primitive().get_cell())
        self._q_points = q_points
        self._q_length = q_length
        self._factor = factor
        self._group_velocity = None
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
            self._dynmat.set_dynamical_matrix(q)
            dm = self._dynmat.get_dynamical_matrix()
            eigvals, eigvecs = np.linalg.eigh(dm)
            dD = self._get_dD(np.array(q), np.array(n))
            dD_at_q = []
            for dD_i in dD: # (x, y, z)
                dD_i_at_q = np.array([np.vdot(eigvec, np.dot(dD_i, eigvec)).real
                                      for eigvec in eigvecs.T])
                dD_at_q.append(dD_i_at_q / np.sqrt(np.abs(eigvals)) /
                               2 * self._factor)
            v_g.append(dD_at_q)
        self._group_velocity = np.array(v_g)

    def _get_dD(self, q, n):
        rlat = self._reciprocal_lattice
        nc = np.dot(n, rlat)
        dqc = self._q_length * nc / np.linalg.norm(nc)
        ddm = []
        for dqc_i in np.diag(dqc):
            dq_i = np.dot(dqc_i, np.linalg.inv(rlat))
            self._dynmat.set_dynamical_matrix(q - dq_i)
            dm1 = self._dynmat.get_dynamical_matrix()
            self._dynmat.set_dynamical_matrix(q + dq_i)
            dm2 = self._dynmat.get_dynamical_matrix()
            ddm.append(dm2 - dm1)
        return [ddm_i / dpc_i for (ddm_i, dpc_i) in zip(ddm, dqc)]
