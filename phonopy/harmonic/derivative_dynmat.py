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

        self._derivative_order = 1
        self._ddm = None

    def run(self, q, q_direction=None):
        try:
            import phonopy._phonopy as phonoc
            self._run_c(q, q_direction=q_direction)
        except ImportError:
            self._run_py(q, q_direction=q_direction)

    def set_derivative_order(self, order):
        self._derivative_order = order
        
    def get_derivative_of_dynamical_matrix(self):
        return self._ddm

    def _run_c(self, q, q_direction=None):
        import phonopy._phonopy as phonoc
        num_patom = len(self._p2s_map)
        ddm_real = np.zeros((3, num_patom * 3, num_patom * 3), dtype='double')
        ddm_imag = np.zeros((3, num_patom * 3, num_patom * 3), dtype='double')
        mass = self._pcell.get_masses()
        fc = self._force_constants
        vectors = self._smallest_vectors
        multiplicity = self._multiplicity
        if self._dynmat.is_nac():
            born = self._dynmat.get_born_effective_charges()
            dielectric = self._dynmat.get_dielectric_constant()
            nac_factor = self._dynmat.get_nac_factor()
            if q_direction is None:
                q_dir = None
            else:
                q_dir = np.array(q_direction, dtype='double', order='C')
        else:
            born = None
            dielectric = None
            nac_factor = 0
            q_dir = None

        phonoc.derivative_dynmat(ddm_real,
                                 ddm_imag,
                                 fc,
                                 np.array(q, dtype='double'),
                                 np.array(self._pcell.get_cell().T,
                                          dtype='double', order='C'),
                                 vectors,
                                 multiplicity,
                                 mass,
                                 self._s2p_map,
                                 self._p2s_map,
                                 nac_factor,
                                 born,
                                 dielectric,
                                 q_dir)
        self._ddm = np.array([ddm_real[i] + ddm_real[i].T +
                              1j * (ddm_imag[i] - ddm_imag[i].T)
                              for i in range(3)]) / 2
        
    def _run_py(self, q, q_direction=None):
        if self._dynmat.is_nac():
            if q_direction is None:
                fc_nac = self._nac(q)
                d_nac = self._d_nac(q)
            else:
                fc_nac = self._nac(q_direction)
                d_nac = self._d_nac(q_direction)

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

                if self._dynmat.is_nac():
                    fc_elem = fc[s_i, k] + fc_nac[i, j]
                else:
                    fc_elem = fc[s_i, k]
                        
                for l in range(3):
                    ddm_elem = fc_elem * (coef[:, l] * phase_multi).sum()
                    if self._dynmat.is_nac() and self._derivative_order == 1:
                        ddm_elem += d_nac[l, i, j] * phase_multi.sum()

                    ddm_local[l] +=  ddm_elem / mass / multi
                                         

            ddm[:, (i * 3):(i * 3 + 3), (j * 3):(j * 3 + 3)] = ddm_local

        # Impose Hermite condition
        self._ddm = np.array([(ddm[i] + ddm[i].conj().T) / 2 for i in range(3)])

    def _nac(self, q_direction):
        """nac_term = (A1 (x) A2) / B * coef.
        """
        num_atom = self._pcell.get_number_of_atoms()
        nac_q = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
        if (np.abs(q_direction) < 1e-5).all():
            return nac_q

        rec_lat = np.linalg.inv(self._pcell.get_cell())
        nac_factor = self._dynmat.get_nac_factor()
        Z = self._dynmat.get_born_effective_charges()
        e = self._dynmat.get_dielectric_constant()
        q = np.dot(rec_lat, q_direction)

        B = self._B(e, q)
        for i in range(num_atom):
            A_i = self._A(q, Z, i)
            for j in range(num_atom):
                A_j = self._A(q, Z, j)
                nac_q[i, j] = np.outer(A_i, A_j) / B

        num_satom = self._scell.get_number_of_atoms()
        N = num_satom / num_atom

        return nac_q * nac_factor / N
    
    def _d_nac(self, q_direction):
        num_atom = self._pcell.get_number_of_atoms()
        d_nac_q = np.zeros((3, num_atom, num_atom, 3, 3), dtype='double')
        if (np.abs(q_direction) < 1e-5).all():
            return d_nac_q

        rec_lat = np.linalg.inv(self._pcell.get_cell())
        nac_factor = self._dynmat.get_nac_factor()
        Z = self._dynmat.get_born_effective_charges()
        e = self._dynmat.get_dielectric_constant()
        q = np.dot(rec_lat, q_direction)

        B = self._B(e, q)
        for xyz in range(3):
            dB = self._dB(e, q, xyz)
            for i in range(num_atom):
                A_i = self._A(q, Z, i)
                dA_i = self._dA(Z, i, xyz)
                for j in range(num_atom):
                    A_j = self._A(q, Z, j)
                    dA_j = self._dA(Z, j, xyz)
                    d_nac_q[xyz, i, j] = (
                        (np.outer(dA_i, A_j) + np.outer(A_i, dA_j)) / B -
                        np.outer(A_i, A_j) * dB / B ** 2)

        num_satom = self._scell.get_number_of_atoms()
        N = num_satom / num_atom
        return d_nac_q * nac_factor / N

    def _A(self, q, Z, atom_num):
        return np.dot(q, Z[atom_num])

    def _B(self, epsilon, q):
        return np.dot(q, np.dot(epsilon, q))

    def _dA(self, Z, atom_num, xyz):
        return Z[atom_num, xyz, :]

    def _dB(self, epsilon, q, xyz):
        e = epsilon
        return np.dot(e[xyz], q) * 2
