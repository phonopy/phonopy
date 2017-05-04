# Copyright (C) 2014 Atsushi Togo
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
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell
from phonopy.harmonic.force_constants import distribute_force_constants

def get_commensurate_points(supercell_matrix): # wrt primitive cell
    rec_primitive = Atoms(numbers=[1],
                          scaled_positions=[[0, 0, 0]],
                          cell=np.diag([1, 1, 1]),
                          pbc=True)
    rec_supercell = get_supercell(rec_primitive, supercell_matrix.T)
    q_pos = rec_supercell.get_scaled_positions()
    return np.where(q_pos > 1 - 1e-15, q_pos - 1, q_pos)

class DynmatToForceConstants(object):
    def __init__(self,
                 primitive,
                 supercell,
                 frequencies=None,
                 eigenvectors=None,
                 symprec=1e-5):
        self._primitive = primitive
        self._supercell = supercell
        supercell_matrix = np.linalg.inv(self._primitive.get_primitive_matrix())
        supercell_matrix = np.rint(supercell_matrix).astype('intc')
        self._commensurate_points = get_commensurate_points(supercell_matrix)
        (self._shortest_vectors,
         self._multiplicity) = primitive.get_smallest_vectors()
        self._dynmat = None
        n = self._supercell.get_number_of_atoms()
        self._force_constants = np.zeros((n, n, 3, 3),
                                         dtype='double', order='C')
        itemsize = self._force_constants.itemsize
        self._dtype_complex = ("c%d" % (itemsize * 2))

        if frequencies is not None and eigenvectors is not None:
            self.set_dynamical_matrices(frequencies, eigenvectors)

    def run(self):
        self._inverse_transformation()
        self._distribute_force_constants()

    def get_force_constants(self):
        return self._force_constants

    def get_commensurate_points(self):
        return self._commensurate_points

    def get_dynamical_matrices(self):
        return self._dynmat

    def set_dynamical_matrices(self,
                               frequencies_at_qpoints=None,
                               eigenvectors_at_qpoints=None,
                               dynmat=None):
        if dynmat is None:
            dm = []
            for frequencies, eigvecs in zip(frequencies_at_qpoints,
                                            eigenvectors_at_qpoints):
                eigvals = frequencies ** 2 * np.sign(frequencies)
                dm.append(
                    np.dot(np.dot(eigvecs, np.diag(eigvals)), eigvecs.T.conj()))
        else:
            dm = dynmat

        self._dynmat = np.array(dm, dtype=self._dtype_complex, order='C')

    def _inverse_transformation(self):
        s2p = self._primitive.get_supercell_to_primitive_map()
        p2s = self._primitive.get_primitive_to_supercell_map()
        p2p = self._primitive.get_primitive_to_primitive_map()

        fc = self._force_constants
        m = self._primitive.get_masses()
        N = (self._supercell.get_number_of_atoms() /
             self._primitive.get_number_of_atoms())

        for p_i, s_i in enumerate(p2s):
            for s_j, p_j in enumerate([p2p[i] for i in s2p]):
                coef = np.sqrt(m[p_i] * m[p_j]) / N
                fc[s_i, s_j] = self._sum_q(p_i, s_j, p_j) * coef

    def _distribute_force_constants(self):
        s2p = self._primitive.get_supercell_to_primitive_map()
        p2s = self._primitive.get_primitive_to_supercell_map()
        positions = self._supercell.get_scaled_positions()
        lattice = self._supercell.get_cell().T
        diff = positions - positions[p2s[0]]
        trans = np.array(diff[np.where(s2p == p2s[0])[0]],
                         dtype='double', order='C')
        rotations = np.array([np.eye(3, dtype='intc')] * len(trans),
                             dtype='intc', order='C')
        distribute_force_constants(self._force_constants,
                                   range(self._supercell.get_number_of_atoms()),
                                   p2s,
                                   lattice,
                                   positions,
                                   rotations,
                                   trans,
                                   1e-5)

    def _sum_q(self, p_i, s_j, p_j):
        multi = self._multiplicity[s_j, p_i]
        pos = self._shortest_vectors[s_j, p_i, :multi]
        sum_q = np.zeros((3, 3), dtype=self._dtype_complex, order='C')
        phases = -2j * np.pi * np.dot(self._commensurate_points, pos.T)
        phase_factors = np.exp(phases).sum(axis=1) / multi
        for i, coef in enumerate(phase_factors):
            sum_q += self._dynmat[i,
                                  (p_i * 3):(p_i * 3 + 3),
                                  (p_j * 3):(p_j * 3 + 3)] * coef
        return sum_q.real
