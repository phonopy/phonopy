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
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.degeneracy import degenerate_sets

def get_group_velocity(q, # q-point
                       dynamical_matrix,
                       q_length=None, # finite distance in q
                       symmetry=None,
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
                       q_length=q_length,
                       symmetry=symmetry,
                       frequency_factor_to_THz=frequency_factor_to_THz)
    gv.set_q_points([q])
    return gv.get_group_velocity()[0]

def delta_dynamical_matrix(q,
                           delta_q,
                           dynmat):
    dynmat.set_dynamical_matrix(q - delta_q)
    dm1 = dynmat.get_dynamical_matrix()
    dynmat.set_dynamical_matrix(q + delta_q)
    dm2 = dynmat.get_dynamical_matrix()
    return dm2 - dm1


class GroupVelocity(object):
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
                 q_length=None,
                 symmetry=None,
                 frequency_factor_to_THz=VaspToTHz,
                 cutoff_frequency=1e-4):
        """
        q_points is a list of sets of q-point and q-direction:
        [[q-point, q-direction], [q-point, q-direction], ...]

        q_length is used such as D(q + q_length) - D(q - q_length).
        """
        self._dynmat = dynamical_matrix
        primitive = dynamical_matrix.get_primitive()
        self._reciprocal_lattice_inv = primitive.get_cell()
        self._reciprocal_lattice = np.linalg.inv(self._reciprocal_lattice_inv)
        self._q_length = q_length
        if q_length is None:
            self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        else:
            self._ddm = None
        self._symmetry = symmetry
        self._factor = frequency_factor_to_THz
        self._cutoff_frequency = cutoff_frequency

        self._directions = np.array([[1, 2, 3],
                                     [1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]], dtype='double')
        self._directions[0] /= np.linalg.norm(self._directions[0])

        self._q_points = None
        self._group_velocity = None
        self._perturbation = None

    def set_q_points(self, q_points, perturbation=None):
        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(
                self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])
        self._set_group_velocity()

    def set_q_length(self, q_length):
        self._q_length = q_length

    def get_group_velocity(self):
        return self._group_velocity

    def _set_group_velocity(self):
        gv = [self._set_group_velocity_at_q(q) for q in self._q_points]
        self._group_velocity = np.array(gv)

    def _set_group_velocity_at_q(self, q):
        self._dynmat.set_dynamical_matrix(q)
        dm = self._dynmat.get_dynamical_matrix()
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
        gv = np.zeros((len(freqs), 3), dtype='double')
        deg_sets = degenerate_sets(freqs)

        ddms = self._get_dD(np.array(q))
        pos = 0
        for deg in deg_sets:
            gv[pos:pos+len(deg)] = self._perturb_D(ddms, eigvecs[:, deg])
            pos += len(deg)

        for i, f in enumerate(freqs):
            if f > self._cutoff_frequency:
                gv[i, :] *= self._factor ** 2 / f / 2
            else:
                gv[i, :] = 0

        if self._perturbation is None:
            return self._symmetrize_group_velocity(gv, q)
        else:
            return gv

    def _symmetrize_group_velocity(self, gv, q):
        rotations = []
        for r in self._symmetry.get_reciprocal_operations():
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < self._symmetry.get_symmetry_tolerance()).all():
                rotations.append(r)

        gv_sym = np.zeros_like(gv)
        for r in rotations:
            r_cart = similarity_transformation(self._reciprocal_lattice, r)
            gv_sym += np.dot(r_cart, gv.T).T

        return gv_sym / len(rotations)
    
    def _get_dD(self, q):
        if self._q_length is None:
            return self._get_dD_analytical(q)
        else:
            return self._get_dD_FD(q)
    
    def _get_dD_FD(self, q): # finite difference
        ddm = []
        for dqc in self._directions * self._q_length:
            dq = np.dot(self._reciprocal_lattice_inv, dqc)
            ddm.append(delta_dynamical_matrix(q, dq, self._dynmat) /
                       self._q_length / 2)
        return np.array(ddm)
    
    def _get_dD_analytical(self, q):
        self._ddm.run(q)
        ddm = self._ddm.get_derivative_of_dynamical_matrix()
        ddm_dirs = np.zeros((len(self._directions),) + ddm.shape[1:],
                            dtype='complex128')
        for i, dq in enumerate(self._directions):
            for j in range(3):
                ddm_dirs[i] += dq[j] * ddm[j]
        return ddm_dirs
    
    def _perturb_D(self, ddms, eigsets):
        eigvals, eigvecs = np.linalg.eigh(
            np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))

        gv = []
        rot_eigsets = np.dot(eigsets, eigvecs)
        for ddm in ddms[1:]:
            gv.append(
                np.diag(np.dot(rot_eigsets.T.conj(),
                               np.dot(ddm, rot_eigsets))).real)
        
        return np.transpose(gv)
