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

class DerivativeOfDynamicalMatrix:
    def __init__(self, dynamical_matrix):
        self._dynmat = dynamical_matrix
        (self._smallest_vectors,
         self._multiplicity) = self._dynmat.get_shortest_vectors()
        self._force_constants = self._dynmat.get_force_constants()
        self._scell = self._dynmat.get_supercell()
        self._pcell = self._dynmat.get_primitive()

        self._p2s_map = self._dynmat.get_primitive_to_supercell_map()
        self._s2p_map = self._dynmat.get_supercell_to_primitive_map()
        self._mass = self._pcell.get_masses()
        self._nac = False

        self._derivative_order = 1
        self._ddm = None

    def run(self, q):
        self._run_py(q)

    def set_derivative_order(self, order):
        self._derivative_order = order
        
    def get_derivative_of_dynamical_matrix(self):
        return self._ddm
        
    def _run_py(self, q):
        fc = self._force_constants
        vecs = self._smallest_vectors
        multiplicity = self._multiplicity
        num_patom = len(self._p2s_map)
        num_satom = len(self._s2p_map)

        # The first "3" used for Catesian index of a vector
        ddm = np.zeros((3, 3 * num_patom, 3 * num_patom), dtype=complex)

        for i, j in list(np.ndindex(num_patom, num_patom)):
            s_i = self._p2s_map[i]
            s_j = self._p2s_map[j]
            mass = np.sqrt(self._mass[i] * self._mass[j])
            ddm_local = np.zeros((3, 3, 3), dtype='complex128')

            for k in range(num_satom):
                if s_j != self._s2p_map[k]:
                    continue

                multi = multiplicity[k, i]
                vecs_multi = vecs[k, i, :multi]
                phase_multi = np.exp([np.vdot(vec, q) * 2j * np.pi
                                      for vec in vecs_multi])
                vecs_multi_cart = np.dot(vecs_multi, self._pcell.get_cell())
                coef = (2j * np.pi * vecs_multi_cart) ** self._derivative_order
                for l in range(3):
                    ddm_local[l] += (fc[s_i, k] / mass *
                                     (coef[:, l] * phase_multi).sum() / multi)

            ddm[:, (i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)] = ddm_local

        # Impose Hermite condition
        self._ddm = np.array([(ddm[i] + ddm[i].conj().T) / 2 for i in range(3)])
