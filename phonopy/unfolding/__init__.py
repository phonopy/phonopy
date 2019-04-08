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

        """

        self._phonon = phonon
        self._supercell_matrix = np.array(supercell_matrix, dtype='intc')
        self._ideal_positions = np.array(ideal_positions, dtype='double')
        self._atom_mapping = atom_mapping
        self._qpoints = qpoints
        self._symprec = self._phonon.symmetry.get_symmetry_tolerance()

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
        self._set_index_set()
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
                             cell=np.diag([1, 1, 1]),
                             pbc=True)
        smat = self._supercell_matrix
        self._trans_s = get_supercell(pcell, smat).get_scaled_positions()
        self._trans_p = np.dot(self._trans_s, self._supercell_matrix.T)
        self._N = len(self._trans_s)

    def _set_index_set(self):
        """T(r_i) in Eq.(3) is given as permutation of atom indices.

        _index_set : ndarray
            For each translation (shift), atomic indices of the positions
            (_ideal_positions - shift) are searched and stored. The indices
            are used to select eigenvectors, by which T(r_i)|KJ> is
            represented.
            shape=(num_trans, num_sites), dtype='intc'

        """

        lattice = self._phonon.supercell.get_cell()
        natom = self._phonon.supercell.get_number_of_atoms()
        index_set = np.zeros((self._N, natom), dtype='intc')
        for i, shift in enumerate(self._trans_s):
            for j, p in enumerate(self._ideal_positions - shift):
                diff = self._ideal_positions - p
                diff -= np.rint(diff)
                dist = np.sqrt((np.dot(diff, lattice) ** 2).sum(axis=1))

                # k is index in _ideal_positions.
                k = np.where(dist < self._symprec)[0][0]

                # _atom_mapping from _ideal_positions to eigenvectors.
                if self._atom_mapping[k] is not None:
                    index_set[i, j] = self._atom_mapping[k]

        self._index_set = index_set

    def _solve_phonon(self):
        self._phonon.run_qpoints(self._qpoints, with_eigenvectors=True)
        qpt = self._phonon.get_qpoints_dict()
        self._freqs = qpt['frequencies']
        self._eigvecs = qpt['eigenvectors']

    def _get_unfolding_weight(self):
        """Calculate Eq. (7)

        K -> _qpoints[_q_index]
        G -> _comm_points
        j -> Primitive translations in supercell (_trans_p)
        J -> Band indices of supercell phonon modes (axis=1 or eigvecs)

        """

        eigvecs = self._eigvecs[self._q_index]
        dtype = "c%d" % (np.dtype('double').itemsize * 2)
        weights = np.zeros((eigvecs.shape[0], self._N), dtype=dtype)

        # Loop over r_j in Eq.(7)
        for shift, indices in zip(self._trans_p, self._index_set):
            eig_indices = (
                np.c_[indices * 3, indices * 3 + 1, indices * 3 + 2]).ravel()
            # Braket in Eq. (7). Results are given for bands (J).
            dot_eigs = (eigvecs.conj() * eigvecs[eig_indices]).sum(axis=0)
            # Phase in Eq. (7)
            phases = np.exp(2j * np.pi * np.dot(self._comm_points, shift))
            weights += np.outer(dot_eigs, phases)
        weights /= self._N

        if (weights.imag > 1e-5).any():
            print("Phonopy warning: Encountered imaginary values.")

        return weights.real
