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
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone

class Unfolding:
    def __init__(self,
                 phonon,
                 phonon_ideal,
                 atom_mapping,
                 bands):
        self._phonon = phonon
        self._phonon_ideal = phonon_ideal
        self._atom_mapping = atom_mapping
        self._bands = bands
        self._symprec = self._phonon.get_symmetry().get_symmetry_tolerance()
        self._primitive = self._phonon_ideal.get_primitive()
        smat = np.linalg.inv(self._primitive.get_primitive_matrix())
        self._supercell_matrix = np.rint(smat).astype('intc')
        self._translations = None
        self._index_set = None
        self._qpoints = None
        self._distances = None
        self._freqs = None
        self._eigvecs = None

    def run(self):
        comm_points = get_commensurate_points(self._supercell_matrix)
        prim_vecs = np.linalg.inv(self._primitive.get_cell()).T
        comm_points_bz = get_qpoints_in_Brillouin_zone(prim_vecs, comm_points)
        self._comm_points = np.array([q[0] for q in comm_points_bz])
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
        translations = get_supercell(pcell, smat).get_scaled_positions()
        translations -= np.floor(translations)
        self._translations = translations

    def _set_shifted_index_set(self):
        index_set = []
        for shift in self._translations:
            index_set.append(self._shift_indices(shift))
        self._index_set = np.array(index_set)

    def _shift_indices(self, shift):
        positions = self._phonon_ideal.get_supercell().get_scaled_positions()
        shifted = positions.copy() - shift
        indices = []
        for p in shifted:
            diff = positions - p
            diff -= np.rint(diff)
            indices.append(
                np.nonzero((np.abs(diff) < self._symprec).all(axis=1))[0][0])
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
        unfolding_weights = []
        for i, eigvecs in enumerate(self._eigvecs[0]):
            print("%d " % i)
            weights_at_q = []
            for eigvec in eigvecs.T:
                weights_at_q.append(self._get_unfolding_weight(eigvec))
            unfolding_weights.append(weights_at_q)
        self._unfolding_weights = np.array(unfolding_weights)
        self._unfolding_weights /= self._unfolding_weights.shape[1]

    def _get_unfolding_weight(self, eigvec):
        trans = np.dot(self._supercell_matrix, self._translations.T).T
        weights = []
        for G in self._comm_points:
            w_sum = 0.0
            for shift, indices in zip(trans, self._index_set):
                phase = np.exp(2j * np.pi * np.dot(G, shift))
                eigvec_shifted = eigvec.reshape((-1, 3))[indices].ravel()
                w_sum += np.vdot(eigvec, eigvec_shifted) * phase
            weights.append(w_sum)

        weights = np.array(weights)
        if (weights.imag > 1e-5).any():
            print("Phonopy warning: Encountered imaginary values.")
            return [None] * len(weights)

        return weights.real
