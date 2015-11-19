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
from phonopy.structure.atoms import Atoms
from phonopy.structure.cells import get_supercell

class Unfolding:
    def __init__(self,
                 phonon,
                 supercell_matrix,
                 ideal_positions,
                 atom_mapping,
                 bands):
        self._phonon = phonon
        self._supercell_matrix = supercell_matrix
        self._ideal_positions = ideal_positions
        self._atom_mapping = atom_mapping
        self._bands = bands
        self._symprec = self._phonon.get_symmetry().get_symmetry_tolerance()
        self._translations = None
        self._index_set = None
        self._qpoints = None
        self._distances = None
        self._freqs = None
        self._eigvecs = None

    def run(self):
        self._comm_points = get_commensurate_points(self._supercell_matrix)
        self._set_translations()
        self._set_shifted_index_set()
        self._solve_phonon()
        self._set_unfolding_weights()

    def get_translations(self):
        return self._translations

    def get_commensurate_points(self):
        return self._comm_points

    def get_shifted_index_set(self):
        return self._index_set

    def get_unfolding_weights(self):
        return self._unfolding_weights

    def get_frequencies(self):
        return self._freqs[0]

    def _set_translations(self):
        pcell = Atoms(numbers=[1],
                      scaled_positions=[[0, 0, 0]],
                      cell=np.diag([1, 1, 1]),
                      pbc=True)
        smat = self._supercell_matrix
        self._translations = get_supercell(pcell, smat).get_scaled_positions()

    def _set_shifted_index_set(self):
        index_set = []
        for shift in self._translations:
            index_set.append(self._shift_indices(shift))
        self._index_set = np.array(index_set)

    def _shift_indices(self, shift):
        positions = self._ideal_positions
        shifted = positions.copy() - shift
        indices = np.zeros(len(positions) * 3, dtype='intc')
        for i, p in enumerate(shifted):
            diff = positions - p
            diff -= np.rint(diff)
            j = np.nonzero((np.abs(diff) < self._symprec).all(axis=1))[0][0]
            indices[i * 3:(i + 1) * 3] = [j * 3 + k for k in range(3)]
        return indices
        
    def _solve_phonon(self):
        if (self._phonon.set_band_structure(self._bands, is_eigenvectors=True)):
            (self._qpoints,
             self._distances,
             self._freqs,
             self._eigvecs) = self._phonon.get_band_structure()
        else:
            print("Solving phonon failed.")
            return False

    def _set_unfolding_weights(self):
        unfolding_weights = np.zeros((len(self._eigvecs[0]),
                                      self._eigvecs[0][0].shape[0],
                                      len(self._comm_points)), dtype='double')
        trans = np.dot(self._supercell_matrix, self._translations.T).T
        for i, eigvecs in enumerate(self._eigvecs[0]):
            unfolding_weights[i] = self._get_unfolding_weight(eigvecs, trans)
        self._unfolding_weights = unfolding_weights

    def _get_unfolding_weight(self, eigvecs, trans):
        weights = np.zeros((eigvecs.shape[0], len(self._comm_points)),
                           dtype='complex128')
        N = len(self._comm_points)
        for i, G in enumerate(self._comm_points):
            for shift, indices in zip(trans, self._index_set):
                phase = np.exp(2j * np.pi * np.dot(G, shift))
                eigvecs_shifted = eigvecs[indices, :]
                weights[:, i] += np.einsum(
                    'ij,ij->j', eigvecs.conj(), eigvecs_shifted) * phase / N

            ## Strainghtforward norm calculation (equivalent speed)
            # eigvecs_shifted = np.zeros_like(eigvecs)
            # for shift, indices in zip(trans, self._index_set):
            #     phase = np.exp(2j * np.pi * np.dot(G, shift))
            #     eigvecs_shifted += eigvecs[indices, :] * phase
            # weights[:, i] = np.einsum(
            #         'ij,ij->j', eigvecs_shifted.conj(), eigvecs_shifted) / N**2

        if (weights.imag > 1e-5).any():
            print("Phonopy warning: Encountered imaginary values.")
            return [None] * len(weights)

        return weights.real
