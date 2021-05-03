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
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.phonon.group_velocity import GroupVelocity

def get_group_velocity_matrices(q,  # q-points
                                dynamical_matrix,
                                q_length=None,  # finite distance in q
                                symmetry=None,
                                frequency_factor_to_THz=VaspToTHz):
    """Returns group velocity matrices at a q-points."""
    gv = GroupVelocityMatrix(dynamical_matrix,
                             q_length=q_length,
                             symmetry=symmetry,
                             frequency_factor_to_THz=frequency_factor_to_THz)
    gv.run([q])
    return gv.group_velocities[0]



class GroupVelocityMatrix(GroupVelocity):
    """Class to calculate group velocities matricies of phonons

     v_qjj' = 1/(2*sqrt(omega_qj*omega_qj')) * <e(q,j)|dD/dq|e(q,j')>

    Attributes
    ----------
    group_velocity_matrices : ndarray
        shape=(q-points, 3, bands, bands), order='C'
        dtype=complex that is "c%d" % (np.dtype('double').itemsize * 2)

    """

    def __init__(self,
                 dynamical_matrix,
                 q_length=None,
                 symmetry=None,
                 frequency_factor_to_THz=VaspToTHz,
                 cutoff_frequency=1e-4):
        self._dynmat = None
        self._reciprocal_lattice = None
        self._q_length = None
        self._ddm = None
        self._symmetry = None
        self._factor = None
        self._cutoff_frequency = None
        self._directions = None
        self._q_points = None
        self._perturbation = None

        GroupVelocity.__init__(
            self,
            dynamical_matrix,
            q_length=q_length,
            symmetry=symmetry,
            frequency_factor_to_THz=frequency_factor_to_THz,
            cutoff_frequency=cutoff_frequency)

        self._group_velocity_matrices = None
        self._dtype_complex = "c%d" % (np.dtype('double').itemsize * 2)

    def run(self, q_points, perturbation=None):
        """Group velocities matrices are computed at q-points.

        Calculated group velocities are stored in
        self._group_velocity_matrices.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like
            Direction in fractional coordinates of reciprocal space.

        """

        self._q_points = q_points
        self._perturbation = perturbation
        if perturbation is None:
            # Give an random direction to break symmetry
            self._directions[0] = np.array([1, 2, 3])
        else:
            self._directions[0] = np.dot(
                self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])

        gvm = [self._calculate_group_velocity_matrix_at_q(q)
               for q in self._q_points]
        self._group_velocity_matrices = np.array(
            gvm, dtype=self._dtype_complex, order='C')

    @property
    def group_velocity_matrices(self):
        return self._group_velocity_matrices

    def _calculate_group_velocity_matrix_at_q(self, q):
        self._dynmat.run(q)
        dm = self._dynmat.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor

        deg_sets = degenerate_sets(freqs)

        ddms = self._get_dD(np.array(q))

        rot_eigvecs = np.zeros((len(freqs), len(freqs)),
                               dtype=self._dtype_complex, order='C')
        for deg in deg_sets:
            rot_eigvecs[:, deg] = self._rot_eigsets(ddms, eigvecs[:, deg])

        for i, f in enumerate(freqs):
            if f > self._cutoff_frequency:
                freqs[i] = 1 / np.sqrt(2 * f)
            else:
                freqs[i] = 0
        freqs = np.diag(freqs)

        rot_eigvecs = np.dot(rot_eigvecs,freqs)

        gvm = []

        for ddm in ddms[1:]:
            ddm = ddm*(self._factor ** 2)
            gvm.append(np.dot(rot_eigvecs.T.conj(), np.dot(ddm,rot_eigvecs)))

        if self._perturbation is None:
            if self._symmetry is None:
                return gvm
            else:
                return self._symmetrize_group_velocity_matrix(gvm, q)
        else:
            return gvm

        return gvm


    def _symmetrize_group_velocity_matrix(self, gv, q):
        """Symmetrize obtained group velocity matrices using:
                 -  site symmetries
                 -  band hermicity
        """

        # site symmetries
        rotations = []
        for r in self._symmetry.reciprocal_operations:
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - np.dot(r, q_in_BZ)
            if (np.abs(diff) < self._symmetry.tolerance).all():
                rotations.append(r)

        gv_sym = np.zeros_like(gv)
        for r in rotations:
            r_cart = similarity_transformation(self._reciprocal_lattice, r)
            gv_sym += np.einsum('ij,jkl->ikl',r_cart,gv)
        gv_sym = gv_sym / len(rotations)

        # band hermicity
        gv_sym= (gv_sym + gv_sym.transpose(0, 2 ,1).conj()) / 2

        return gv_sym


    def _rot_eigsets(self, ddms, eigsets):
        """Treat degeneracy

        Eigenvectors of degenerates bands in eigsets are rotated to make
        the velocity analytical in a specified direction (self._directions[0]).

        ddms : array-like
            List of delta (derivative or finite difference) of dynamical
            matrices along several q-directions for perturbation.
        eigsets : array-like
            List of phonon eigenvectors of degenerate bands.

        """

        eigvals, eigvecs = np.linalg.eigh(
            np.dot(eigsets.T.conj(), np.dot(ddms[0], eigsets)))

        gv = []
        rot_eigsets = np.dot(eigsets, eigvecs)

        return rot_eigsets
