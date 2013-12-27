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

    def run(self):
        self._create_tetrahedrons()
        self._set_relative_grid_address()

    def get_tetrahedrons(self):
        return self._relative_grid_address
        
    def _create_tetrahedrons(self):
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
        pos = 0
        for i in range(8):
            cube_shifted = cube_vertices - cube_vertices[i]
            for tetra in self._vertices:
                if i in tetra:
                    relative_grid_address[pos, :, :] = cube_shifted[tetra]
                    pos += 1
        self._relative_grid_address = relative_grid_address
