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
        for q in self._q_points:
            dD_at_q = get_group_velocity(q,
                                         self._dynmat,
                                         self._reciprocal_lattice,
                                         q_length=self._q_length,
                                         factor=self._factor,
                                         frequencies=self._frequencies,
                                         eigenvectors=self._eigenvectors)
            v_g.append(dD_at_q)
        self._group_velocity = np.array(v_g)

def get_group_velocity(q, # q-point
                       dynamical_matrix,
                       reciprocal_lattice,
                       q_length=1e-4, # finite distance in q
                       factor=VaspToTHz,
                       frequencies=None,
                       eigenvectors=None):
    """
    If frequencies and eigenvectors are supplied they are used
    instead of calculating them at q-point (but not at q+dq and q-dq).

    reciprocal lattice has to be given as
    [[a_x, b_x, c_x],
     [a_y, b_y, c_y],
     [a_z, b_z, c_z]]
    """

    if frequencies is None or eigenvectors is None:
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * factor
    else:
        eigvecs = eigenvectors
        freqs = frequencies

    dD_at_q = np.zeros((3, len(freqs)), dtype=float)
    for i, dD_i in enumerate(get_dD(np.array(q),
                                    q_length,
                                    dynamical_matrix,
                                    reciprocal_lattice)): # (x, y, z)
        dD_at_q[i] = [np.vdot(eigvec, np.dot(dD_i, eigvec)).real
                     for eigvec in eigvecs.T]
        dD_at_q[i] *= factor ** 2 / freqs / 2 / (q_length * 2)
    return dD_at_q
        
def get_dD(q, q_length, dynamical_matrix, reciprocal_lattice):
    # The names of *c mean something in Cartesian.
    dynmat = dynamical_matrix
    rlat_inv = np.linalg.inv(reciprocal_lattice)
    ddm = []
    for dqc_i in (np.eye(3) * q_length):
        dq_i = np.dot(rlat_inv, dqc_i)
        dynmat.set_dynamical_matrix(q - dq_i)
        dm1 = dynmat.get_dynamical_matrix()
        dynmat.set_dynamical_matrix(q + dq_i)
        dm2 = dynmat.get_dynamical_matrix()
        ddm.append(dm2 - dm1)
    return np.array(ddm)
