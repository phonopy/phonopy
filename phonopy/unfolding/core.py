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
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell


class Unfolding(object):
    """

    Implementation of an unfolding method by
    P. B. Allen et al., Phys. Rev. B 87, 085322 (2013)

    T(r_i) in this implementation is defined as

        T(r_i) f(x) = f(x - r_i).

    The sign is opposite from that written in the Allen's paper.
    Bloch wave is defined in the same way for phase convention

        Psi_k(x + r) = exp(ikr) Psi_k(x).

    By these, sign of phase in Eq.(3) (Eq.(7) as well) is opposite.


    """

    def __init__(self,
                 phonon,
                 supercell_matrix,
                 ideal_positions,
                 atom_mapping,
                 qpoints):
        """

        Parameters
        ----------
        phonon : Phonopy
            Phonopy object made with supercell as the primitive cell.
        supercell_matrix : array_like
            Matrix that represents the primitive translation enforced within
            the supercell. This works like an inverse primitive matrix.
            shape=(3, 3), dtype='intc'
        ideal_positions : array_like
            shape=(3, 3), dtype='intc'
        atom_mapping : list
            Atomic index mapping from ideal_positions to supercell atoms in
            phonon. None is used for Vacancies.
        qpoints : array_like
            q-points in reciprocal primitive cell coordinates
            shape=(num_qpoints, 3), dtype='double'

        """

        self._phonon = phonon
        self._supercell_matrix = np.array(supercell_matrix, dtype='intc')
        self._ideal_positions = np.array(ideal_positions, dtype='double')
        self._qpoints_p = qpoints  # in PBZ
        self._qpoints_s = self._get_qpoints_in_SBZ()  # in SBZ
        self._symprec = self._phonon.symmetry.get_symmetry_tolerance()

        self._frequencies = None
        self._eigvecs = None  # This may have anormal array shape.
        self._unfolding_weights = None
        self._q_count = None  # As counter for iterator

        # Commensurate q-vectors in PBZ
        self._comm_points = get_commensurate_points(self._supercell_matrix)

        self._trans_s = None  # in SC (see docstring in _set_translations)
        self._trans_p = None  # in PC (see docstring in _set_translations)
        self._N = None
        self._set_translations()

        self._atom_mapping = None
        self._index_map_inv = None
        self._set_index_map(atom_mapping)

    def __iter__(self):
        return self

    def run(self):
        self.prepare()
        for x in self:
            pass

    def __next__(self):
        if self._q_count == len(self._qpoints_s):
            self._eigvecs = None
            raise StopIteration

        self._solve_phonon()
        self._unfolding_weights[self._q_count] = self._get_unfolding_weights()
        self._q_count += 1
        return self._unfolding_weights[self._q_count - 1]

    def next(self):
        return self.__next__()

    def prepare(self):
        self._q_count = 0
        self._unfolding_weights = np.zeros(
            (len(self._qpoints_s),
             self._phonon.supercell.get_number_of_atoms() * 3), dtype='double')
        self._frequencies = np.zeros_like(self._unfolding_weights)

    @property
    def commensurate_points(self):
        return self._comm_points

    def get_commensurate_points(self):
        return self.commensurate_points

    @property
    def unfolding_weights(self):
        return self._unfolding_weights

    def get_unfolding_weights(self):
        return self.unfolding_weights

    @property
    def frequencies(self):
        return self._frequencies

    def get_frequencies(self):
        return self.frequencies

    def _get_qpoints_in_SBZ(self):
        qpoints = np.dot(self._qpoints_p, self._supercell_matrix)
        qpoints -= np.rint(qpoints)
        return qpoints

    def _set_translations(self):
        """Set primitive translations in supercell

        _trans_s
            Translations with respect to supercell basis vectors
        _trans_p
            Translations with respect to primitive cell basis vectors
        _N
            Number of the translations = det(supercel_matrix)

        """

        pcell = PhonopyAtoms(numbers=[1],
                             scaled_positions=[[0, 0, 0]],
                             cell=np.diag([1, 1, 1]))
        smat = self._supercell_matrix
        self._trans_s = get_supercell(pcell, smat).get_scaled_positions()
        self._trans_p = np.dot(self._trans_s, self._supercell_matrix.T)
        self._N = len(self._trans_s)

    def _set_index_map(self, atom_mapping):
        """T(r_i) in Eq.(3) is given as permutation of atom indices.

        _index_set : ndarray
            For each translation (shift), atomic indices of the positions
            (_ideal_positions - shift) are searched and stored. The indices
            are used to select eigenvectors, by which T(r_i)|KJ> is
            represented.
            shape=(num_trans, num_sites), dtype='intc'

        """

        lattice = self._phonon.supercell.get_cell()
        natom = len(self._ideal_positions)
        index_map_inv = np.zeros((self._N, natom), dtype='intc')
        for i, shift in enumerate(self._trans_s):
            for j, p in enumerate(self._ideal_positions - shift):  # minus r_i
                diff = self._ideal_positions - p
                diff -= np.rint(diff)
                dist = np.sqrt((np.dot(diff, lattice) ** 2).sum(axis=1))
                # k is index in _ideal_positions.
                k = np.where(dist < self._symprec)[0][0]
                index_map_inv[i, j] = k
        self._index_map_inv = index_map_inv

        self._atom_mapping = np.zeros(len(atom_mapping), dtype='int')
        for i, idx in enumerate(atom_mapping):
            if idx is None:
                self._atom_mapping[i] = -1
            else:
                self._atom_mapping[i] = idx

    def _solve_phonon(self):
        self._phonon.run_qpoints(self._qpoints_s[self._q_count],
                                 with_eigenvectors=True)
        qpt = self._phonon.get_qpoints_dict()
        self._frequencies[self._q_count] = qpt['frequencies'][0]
        eigvecs = qpt['eigenvectors'][0]

        if -1 in self._atom_mapping:
            shape = list(eigvecs.shape)
            shape[0] += 3
            self._eigvecs = np.zeros(tuple(shape), dtype=eigvecs.dtype)
            self._eigvecs[:-3, :] = eigvecs
        else:
            self._eigvecs = eigvecs

    def _get_unfolding_weights(self):
        """Calculate Eq. (7)

        k = K + G + g

        K -> _qpoints_s[_q_index] (in SBZ)
        k -> _qpoints_p[_q_index] (in PBZ)
        G -> _comm_points (in PBZ)
        g -> a reciprocal lattice point in PBZ (unused explicitly)
        j -> Primitive translations in supercell (_trans_p)
        J -> Band indices of supercell phonon modes (axis=1 or eigvecs)

        The phase factor corresponding to K is not included in eigvecs
        with our choice of dynamical matrix.

        """

        eigvecs = self._eigvecs
        dtype = "c%d" % (np.dtype('double').itemsize * 2)
        q_p = self._qpoints_p[self._q_count]  # k
        q_s = self._qpoints_s[self._q_count]  # K
        diff = q_p - np.dot(q_s, np.linalg.inv(self._supercell_matrix))

        # Search G points corresponding to k = G + K
        for G in self._comm_points:
            d = diff - G
            d -= np.rint(d)
            if (np.abs(d) < 1e-5).all():
                break

        e = np.zeros((len(self._atom_mapping) * 3, eigvecs.shape[1]),
                     dtype=dtype)
        phases = np.exp(2j * np.pi * np.dot(self._trans_p, G))

        for phase, indices in zip(
                phases, self._atom_mapping[self._index_map_inv]):
            eig_indices = (
                np.c_[indices * 3, indices * 3 + 1, indices * 3 + 2]).ravel()
            e += eigvecs[eig_indices, :] * phase
        e /= self._N
        weights = (e.conj() * e).sum(axis=0)

        # indices = self._atom_mapping
        # eig_indices_r = (
        #     np.c_[indices * 3, indices * 3 + 1, indices * 3 + 2]).ravel()
        # for phase, indices in zip(
        #         phases, self._atom_mapping[self._index_map_inv]):
        #     eig_indices_l = (
        #         np.c_[indices * 3, indices * 3 + 1, indices * 3 + 2]).ravel()
        #     e += eigvecs[eig_indices_l, :] * eigvecs[eig_indices_r, :].conj() * phase
        # e /= self._N
        # weights = e.sum(axis=0)

        if (weights.imag > 1e-5).any():
            print("Phonopy warning: Encountered imaginary values.")

        return weights.real
