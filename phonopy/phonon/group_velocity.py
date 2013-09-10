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
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

def get_group_velocity(q, # q-point
                       dynamical_matrix,
                       q_length=None, # finite distance in q
                       frequency_factor_to_THz=VaspToTHz):
    """
    If frequencies and eigenvectors are supplied they are used
    instead of calculating them at q-point (but not at q+dq and q-dq).

    reciprocal lattice has to be given as
    [[a_x, b_x, c_x],
     [a_y, b_y, c_y],
     [a_z, b_z, c_z]]
    """

    gv = GroupVelocity(dynamical_matrix,
                       [q],
                       q_length=q_length,
                       frequency_factor_to_THz=frequency_factor_to_THz)
    return gv.get_group_velocity()[0]

directions_all = np.array([[1, 0, 0],  # x
                           [0, 2, 0],  # y
                           [0, 0, 3],  # z
                           [0, 2, 3],  # yz
                           [1, 0, 3],  # zx
                           [1, 2, 0],  # xy
                           [1, 2, 3]]) # xyz
directions = np.array([[1, 0, 0],  # x
                       [0, 1, 0],  # y
                       [0, 0, 1],  # z
                       [1, 0, 1],  # zx
                       [0, 1, 1]]) # yz
            

def degenerate_sets(freqs, cutoff=1e-4):
    indices = []
    done = []
    for i in range(len(freqs)):
        if i in done:
            continue
        else:
            f_set = [i]
            done.append(i)
        for j in range(i + 1, len(freqs)):
            if (np.abs(freqs[f_set] - freqs[j]) < cutoff).any():
                f_set.append(j)
                done.append(j)
        indices.append(f_set[:])

    return indices

def delta_dynamical_matrix(q,
                           delta_q,
                           dynmat):
    dynmat.set_dynamical_matrix(q - delta_q)
    dm1 = dynmat.get_dynamical_matrix()
    dynmat.set_dynamical_matrix(q + delta_q)
    dm2 = dynmat.get_dynamical_matrix()
    return dm2 - dm1


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
                 q_points=None,
                 q_length=None,
                 frequency_factor_to_THz=VaspToTHz):
        """
        q_points is a list of sets of q-point and q-direction:
        [[q-point, q-direction], [q-point, q-direction], ...]

        q_length is used such as D(q + q_length) - D(q - q_length).
        """
        self._dynmat = dynamical_matrix
        primitive = dynamical_matrix.get_primitive()
        self._reciprocal_lattice_inv = primitive.get_cell()
        self._q_points = q_points
        self._q_length = q_length
        if q_length is None:
            self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        else:
            self._ddm = None
        self._factor = frequency_factor_to_THz
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
        for q in self._q_points:
            dD_at_q = self._set_group_velocity_at_q(q)
            v_g.append(dD_at_q)
        self._group_velocity = np.array(v_g)

    def _set_group_velocity_at_q(self, q):
        self._dynmat.set_dynamical_matrix(q)
        dm = self._dynmat.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
    
        gv = np.zeros((len(freqs), 3), dtype='double')
        deg_sets = degenerate_sets(freqs)

        ddm_dirs = self._get_dD(np.array(q)) # x, y, z, yz, zx, xy, xyz
        pos = 0
        for deg in deg_sets:
            gv_dirs = np.zeros((len(deg), len(directions_all)), dtype='double')
            for i, ddm in enumerate(ddm_dirs):
                gv_dirs[:, i] = self._perturb_D(ddm, eigvecs[:, deg])

            gv[pos:pos+len(deg)] = self._sort_gv(np.array(gv_dirs)) / [1, 2, 3]
            pos += len(deg)

        for i in range(3):
            gv[:, i] *= self._factor ** 2 / freqs / 2

        return gv

    def _get_dD(self, q):
        if self._q_length is None:
            return self._get_dD_analytical(q)
        else:
            return self._get_dD_FD(q)
    
    def _get_dD_FD(self, q): # finite difference
        ddm = []
        for dqc_i in (directions_all * self._q_length):
            dq = np.dot(self._reciprocal_lattice_inv, dqc_i)
            ddm.append(delta_dynamical_matrix(q,
                                              dq,
                                              self._dynmat) /
                       (self._q_length * 2))
        return np.array(ddm)
    
    def _get_dD_analytical(self, q):
        self._ddm.run(q)
        ddm = self._ddm.get_derivative_of_dynamical_matrix()
        ddm_dirs = np.zeros((len(directions_all),) + ddm.shape[1:],
                            dtype='complex128')
        for i, d in enumerate(directions_all):
            for j in range(3):
                ddm_dirs[i] += d[j] * ddm[j]
        return ddm_dirs
    
    def _perturb_D(self, dD, eigsets):
        eigvals = np.linalg.eigvalsh(
            np.dot(eigsets.T.conj(), np.dot(dD, eigsets)))
        return eigvals.real
        
    def _sort_gv(self, gv_dirs):
        num_deg = len(gv_dirs)
        gv = np.zeros((num_deg, 5), dtype='double')
        gv[:, 2] = gv_dirs[:, 2]

        for i in (0, 1): # x, y
            done_x = [False] * num_deg
            done_xz = [False] * num_deg
            for j in range(num_deg):
                indices = self._search(j, gv_dirs, i, done_x, done_xz)
                gv[j, i] = gv_dirs[indices[1], i]
                gv[j, 3 + i] = gv_dirs[indices[0], 3 + i]

        return gv[:, :3]
                
    def _search(self, i, gv, i_xy, done_x, done_xz):
        num_deg = len(gv)
        z = gv[i, 2]
        min_delta = None
        min_indices = None

        for j in range(num_deg):
            z_x = gv[j, 3 + i_xy]
            for k in range(num_deg):
                x = gv[k, i_xy]
                delta = x + z - z_x
                if abs(delta) < min_delta or min_indices is None:
                    if done_x[k] or done_xz[j]:
                        continue
                    else:
                        min_delta = abs(delta)
                        min_indices = [j, k]

        done_x[min_indices[1]] = True
        done_xz[min_indices[0]] = True

        return min_indices

            
