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

cube_vertices = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 1, 0],
                          [0, 0, 1],
                          [1, 0, 1],
                          [0, 1, 1],
                          [1, 1, 1]], dtype='intc')

class TetrahedronMethod:
    def __init__(self,
                 primitive_vectors,
                 mesh):
        self._reclat = primitive_vectors # column vectors
        self._mesh = mesh
        self._vertices = None
        self._relative_grid_address = None
        self._central_indices = None
        self._tetrahedra_omegas = None
        self._sort_indices = None
        self._omegas = None
        self._create_tetrahedra()
        self._set_relative_grid_address()

    def run(self, omega, value='I'):
        sum_value = 0.0
        self._omega = omega
        for omegas, indices, ci in zip(self._tetrahedra_omegas,
                                       self._sort_indices,
                                       self._central_indices):
            self._vertices_omegas = omegas[indices]
            i_where = np.where(self._omega < self._vertices_omegas)[0]
            if len(i_where):
                i = i_where[0]
            else:
                i = 4

            if value == 'I':
                sum_value += self._I(i, np.where(indices==ci)[0][0]) * self._g(i)
            elif value == 'J':
                sum_value += self._J(i, np.where(indices==ci)[0][0]) * self._n(i)
            else:
                assert False

        return sum_value / 6

    def get_tetrahedra(self):
        return self._relative_grid_address

    def set_tetrahedra_omegas(self, tetrahedra_omegas):
        self._tetrahedra_omegas = tetrahedra_omegas
        self._sort_indices = np.argsort(self._tetrahedra_omegas, axis=1)
        
    def _create_tetrahedra(self):
        #
        #     6-------7
        #    /|      /|
        #   / |     / |
        #  4-------5  |
        #  |  2----|--3
        #  | /     | /
        #  |/      |/
        #  0-------1
        #
        # i: vec        neighbours
        # 0: O          1, 2, 4    
        # 1: a          0, 3, 5
        # 2: b          0, 3, 6
        # 3: a + b      1, 2, 7
        # 4: c          0, 5, 6
        # 5: c + a      1, 4, 7
        # 6: c + b      2, 4, 7
        # 7: c + a + b  3, 5, 6
        a, b, c = self._reclat.T
        diag_vecs = np.array([ a + b + c,  # 0-7
                              -a + b + c,  # 1-6
                               a - b + c,  # 2-5
                               a + b - c]) # 3-4
        shortest_index = np.argmin(np.sum(diag_vecs ** 2, axis=1))
        # vertices = [np.zeros(3), a, b, a + b, c, c + a, c + b, c + a + b]
        if shortest_index == 0:
            pairs = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
            tetras = np.sort([[0, 7] + list(x) for x in pairs])
        elif shortest_index == 1:
            pairs = ((0, 2), (0, 4), (2, 3), (3, 7), (4, 5), (5, 7))
            tetras = np.sort([[1, 6] + list(x) for x in pairs])
        elif shortest_index == 2:
            pairs = ((0, 1), (0, 4), (1, 3), (3, 7), (4, 6), (6, 7))
            tetras = np.sort([[2, 5] + list(x) for x in pairs])
        elif shortest_index == 3:
            pairs = ((0, 1), (0, 2), (1, 5), (2, 6), (5, 7), (6, 7))
            tetras = np.sort([[3, 4] + list(x) for x in pairs])
        else:
            assert False

        self._vertices = tetras

    def _set_relative_grid_address(self):
        relative_grid_address = np.zeros((24, 4, 3), dtype='intc')
        central_indices = np.zeros(24, dtype='intc')
        pos = 0
        for i in range(8):
            cube_shifted = cube_vertices - cube_vertices[i]
            for tetra in self._vertices:
                if i in tetra:
                    central_indices[pos] = np.where(tetra==i)[0][0]
                    relative_grid_address[pos, :, :] = cube_shifted[tetra]
                    pos += 1
        self._relative_grid_address = relative_grid_address
        self._central_indices = central_indices

    def _f(self, n, m):
        return ((self._omega - self._vertices_omegas[m]) /
                (self._vertices_omegas[n] - self._vertices_omegas[m]))

    def _I(self, i, ci):
        if i == 0:
            return self._I_0()
        elif i == 1:
            if ci == 0:
                return self._I_10()
            elif ci == 1:
                return self._I_11()
            elif ci == 2:
                return self._I_12()
            elif ci == 3:
                return self._I_13()
            else:
                assert False
        elif i == 2:
            if ci == 0:
                return self._I_20()
            elif ci == 1:
                return self._I_21()
            elif ci == 2:
                return self._I_22()
            elif ci == 3:
                return self._I_23()
            else:
                assert False
        elif i == 3:
            if ci == 0:
                return self._I_30()
            elif ci == 1:
                return self._I_31()
            elif ci == 2:
                return self._I_32()
            elif ci == 3:
                return self._I_33()
            else:
                assert False
        elif i == 4:
            return self._I_4()
        else:
            assert False

    def _n(self, i):
        if i == 0:
            return self._n_0()
        elif i == 1:
            return self._n_1()
        elif i == 2:
            return self._n_2()
        elif i == 3:
            return self._n_3()
        elif i == 4:
            return self._n_4()
        else:
            assert False

    def _g(self, i):
        if i == 0:
            return self._g_0()
        elif i == 1:
            return self._g_1()
        elif i == 2:
            return self._g_2()
        elif i == 3:
            return self._g_3()
        elif i == 4:
            return self._g_4()
        else:
            assert False
    
    def _n_0(self):
        """omega < omega1"""
        return 0.0

    def _n_1(self):
        """omega1 < omega < omega2"""
        return self._f(1, 0) * self._f(2, 0) * self._f(3, 0)

    def _n_2(self):
        """omega2 < omega < omega3"""
        return (self._f(3, 1) * self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2))
                
    def _n_3(self):
        """omega2 < omega < omega3"""
        return (1.0 - self._f(0, 3) * self._f(1, 3) * self._f(2, 3))

    def _n_4(self):
        """omega4 < omega"""
        return 1.0

    def _g_0(self):
        """omega < omega1"""
        return 0.0

    def _g_1(self):
        """omega1 < omega < omega2"""
        return 3 * self._n_1() / (self._omega - self._vertices_omegas[0])

    def _g_2(self):
        """omega2 < omega < omega3"""
        return 3 / (self._vertices_omegas[3] - self._vertices_omegas[0]) * (
            self._f(1, 2) * self._f(2, 0) +
            self._f(2, 1) * self._f(1, 3))

    def _g_3(self):
        """omega3 < omega < omega4"""
        return 3 * (1.0 - self._n_3()) / (self._vertices_omegas[3] - self._omega)

    def _g_4(self):
        """omega4 < omega"""
        return 0.0

    def _J0(self):
        return 0.0
    
    def _J_10(self):
        return (1.0 + self._f(0, 1) + self._f(0, 2) + self._f(0, 3)) / 4

    def _J_11(self):
        return self._f(1, 0) / 4

    def _J_12(self):
        return self._f(2, 0) / 4

    def _J_13(self):
        return self._f(3, 0) / 4

    def _J_20(self):
        return (self._f(3, 1) * self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (1.0 + self._f(0, 3)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                (1.0 + self._f(0, 3) + self._f(0, 2))) / 4

    def _J_21(self):
        return (self._f(3, 1) * self._f(2, 1) *
                (1.0 + self._f(1, 3) + self._f(1, 2)) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (self._f(1, 3) + self._f(1, 2)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                self._f(1, 2)) / 4

    def _J_22(self):
        return (self._f(3, 1) * self._f(2, 1) *
                self._f(2, 1) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                self._f(2, 1) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                (self._f(2, 1) + self._f(2, 0))) / 4

    def _J_23(self):
        return (self._f(3, 1) * self._f(2, 1) *
                self._f(3, 2) +
                self._f(3, 0) * self._f(1, 3) * self._f(2, 1) *
                (self._f(3, 2) + self._f(3, 0)) +
                self._f(3, 0) * self._f(2, 0) * self._f(1, 2) *
                self._f(3, 0)) / 4

    def _J_30(self):
        return ((1.0 - self._f(0, 3) ** 2 * self._f(1, 3) * self._f(2, 3)) /
                4 / self._n_3())

    def _J_31(self):
        return ((1.0 - self._f(0, 3) * self._f(1, 3) ** 2 * self._f(2, 3)) /
                4 / self._n_3())

    def _J_32(self):
        return ((1.0 + self._f(0, 3) * self._f(1, 3) * self._f(2, 3) ** 2) /
                4 / self._n_3())

    def _J_33(self):
        return ((1.0 - self._f(0, 3) * self._f(1, 3) * self._f(2, 3) *
                 (1.0 + self._f(3, 0) + self._f(3, 1) + self._f(3, 2))) /
                4 / self._n_3())

    def _J_4(self):
        return 0.25

    def _I_0(self):
        return 0.0
    
    def _I_10(self):
        return (self._f(0, 1) + self._f(0, 2) + self._f(0, 3)) / 3

    def _I_11(self):
        return self._f(1, 0) / 3

    def _I_12(self):
        return self._f(2, 0) / 3

    def _I_13(self):
        return self._f(3, 0) / 3

    def _I_20(self):
        return (self._f(0, 3) +
                self._f(0, 2) * self._f(2, 0) * self._f(1, 2) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_21(self):
        return (self._f(1, 2) +
                self._f(1, 3) ** 2 * self._f(2, 1) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_22(self):
        return (self._f(2, 1) +
                self._f(2, 0) ** 2 * self._f(1, 2) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3
                
    def _I_23(self):
        return (self._f(3, 0) +
                self._f(3, 1) * self._f(1, 3) * self._f(2, 1) /
                (self._f(1, 2) * self._f(2, 0) + self._f(2, 1) * self._f(1, 3))
                ) / 3

    def _I_30(self):
        return self._f(0, 3) / 3

    def _I_31(self):
        return self._f(1, 3) / 3

    def _I_32(self):
        return self._f(2, 3) / 3

    def _I_33(self):
        return (self._f(3, 0) + self._f(3, 1) + self._f(3, 2)) / 3

    def _I_4(self):
        return 0.0

