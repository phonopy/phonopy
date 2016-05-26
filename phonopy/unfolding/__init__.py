# Copyright (C) 2015 Atsushi Togo
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
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.structure.cells import get_supercell

class Unfolding(object):
    def __init__(self,
                 phonon,
                 supercell_matrix,
                 ideal_positions,
                 atom_mapping,
                 qpoints):
        self._phonon = phonon
        self._supercell_matrix = np.array(supercell_matrix, dtype='intc')
        self._ideal_positions = ideal_positions
        self._atom_mapping = atom_mapping
        self._qpoints = qpoints
        self._symprec = self._phonon.get_symmetry().get_symmetry_tolerance()
        
        self._trans_s = None
        self._trans_p = None
        self._comm_points = None
        self._index_set = None
        self._freqs = None
        self._eigvecs = None
        self._N = None

        self._weights = None
        self._q_index = None

    def __iter__(self):
        return self

    def run(self, verbose=False):
        self.prepare()
        self._q_index = 0
        for x in self:
            if verbose:
                print(self._q_index)

    def __next__(self):
        if self._q_index == len(self._eigvecs):
            raise StopIteration
        else:
            self._weights[self._q_index] = self._get_unfolding_weight()
            self._q_index += 1
            return self._weights[self._q_index - 1]

    def next(self):
        return self.__next__()

    def prepare(self):
        self._comm_points = get_commensurate_points(self._supercell_matrix)
        self._set_translations()
        self._set_shifted_index_set()
        self._solve_phonon()
        self._weights = np.zeros(
            (len(self._eigvecs), self._eigvecs[0].shape[0], self._N),
            dtype='double')

    def get_translations(self):
        return self._trans_s

    def get_commensurate_points(self):
        return self._comm_points

    def get_shifted_index_set(self):
        return self._index_set

    def get_unfolding_weights(self):
        return self._weights

    def get_frequencies(self):
        return self._freqs

    def _set_translations(self):
        pcell = Atoms(numbers=[1],
                      scaled_positions=[[0, 0, 0]],
                      cell=np.diag([1, 1, 1]),
                      pbc=True)
        smat = self._supercell_matrix
        self._trans_s = get_supercell(pcell, smat).get_scaled_positions()
        self._trans_p = np.dot(self._trans_s, self._supercell_matrix.T)
        self._N = len(self._trans_s)

    def _set_shifted_index_set(self):
        index_set = np.zeros((self._N, len(self._ideal_positions) * 3),
                             dtype='intc')
        for i, shift in enumerate(self._trans_s):
            for j, p in enumerate(self._ideal_positions - shift):
                diff = self._ideal_positions - p
                diff -= np.rint(diff)
                k = np.nonzero((np.abs(diff) < self._symprec).all(axis=1))[0][0]
                l = self._atom_mapping[k]
                index_set[i, j * 3:(j + 1) * 3] = np.arange(l * 3, (l + 1) * 3)
        self._index_set = index_set

    def _solve_phonon(self):
        if (self._phonon.set_qpoints_phonon(self._qpoints, is_eigenvectors=True)):
            self._freqs, self._eigvecs = self._phonon.get_qpoints_phonon()
        else:
            print("Solving phonon failed.")
            return False

    def _get_unfolding_weight(self):
        eigvecs = self._eigvecs[self._q_index]
        weights = np.zeros((eigvecs.shape[0], self._N), dtype='complex128')
        for shift, indices in zip(self._trans_p, self._index_set):
            dot_eigs = np.einsum(
                'ij,ij->j', eigvecs.conj(), eigvecs[indices, :])
            for i, G in enumerate(self._comm_points):
                phase = np.exp(2j * np.pi * np.dot(G, shift))
                weights[:, i] += dot_eigs * phase
        weights /= self._N

        # # Strainghtforward norm calculation (equivalent speed)
        # for i, G in enumerate(self._comm_points):
        #     eigvecs_shifted = np.zeros_like(eigvecs)
        #     for shift, indices in zip(self._trans_p, self._index_set):
        #         phase = np.exp(2j * np.pi * np.dot(G, shift))
        #         eigvecs_shifted += eigvecs[indices, :] * phase
        #     weights[:, i] = np.einsum(
        #             'ij,ij->j', eigvecs_shifted.conj(), eigvecs_shifted)
        # weights /= self._N**2

        if (weights.imag > 1e-5).any():
            print("Phonopy warning: Encountered imaginary values.")

        return weights.real
