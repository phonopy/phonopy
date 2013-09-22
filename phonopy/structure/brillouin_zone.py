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
from phonopy.structure.cells import get_reduced_bases

search_space = np.array([
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1]], dtype='intc')

def get_qpoints_in_Brillouin_zone(primitive_vectors,
                                  qpoints,
                                  tolerance=1e-5):
    bz = BrillouinZone(primitive_vectors, tolerance=tolerance)
    bz.run(qpoints)
    return bz.get_shortest_qpoints()

class BrillouinZone:
    def __init__(self,
                 primitive_vectors,
                 tolerance=1e-5):
        self._primitive_vectors = primitive_vectors # column vectors
        self._tolerance = tolerance
        self._reduced_bases = get_reduced_bases(primitive_vectors.T,
                                                self._tolerance).T
        self._primitive_vectors_inv = np.linalg.inv(self._primitive_vectors)
        self._reduced_bases_inv = np.linalg.inv(self._reduced_bases)

        self._shortest_qpoints = None

    def run(self, qpoints):
        reduced_qpoints = np.dot(self._reduced_bases_inv,
                                 np.dot(self._primitive_vectors, qpoints.T)).T
        self._shortest_qpoints = []
        for q in reduced_qpoints:
            distances = np.array([(np.dot(self._reduced_bases, q + g) ** 2).sum()
                                  for g in search_space], dtype='double')
            min_dist = min(distances)
            shortest_indices = [i for i, d in enumerate(distances - min_dist)
                                if abs(d) < self._tolerance ** 2]

            self._shortest_qpoints.append(
                np.dot(self._primitive_vectors_inv,
                       np.dot(self._reduced_bases,
                              (search_space[shortest_indices] + q).T)).T)

    def get_shortest_qpoints(self):
        return self._shortest_qpoints

    
if __name__ == '__main__':
    from phonopy.interface.vasp import read_vasp
    from phonopy.structure.symmetry import Symmetry, get_lattice_vector_equivalence
    from phonopy.structure.spglib import get_ir_reciprocal_mesh
    import sys

    cell = read_vasp(sys.argv[1])
    symmetry = Symmetry(cell)
    mesh = [10, 10, 10]
    mapping_table, grid_addrees = get_ir_reciprocal_mesh(
        mesh,
        cell,
        is_shift=[0, 0, 0])

    ir_grid_points = np.unique(mapping_table)
    qpoints = grid_addrees[ir_grid_points] / np.array(mesh, dtype='double')

    primitive_vectors = np.linalg.inv(cell.get_cell())
    bz = BrillouinZone(primitive_vectors)
    bz.run(qpoints)
    sv = bz.get_qpoints()
    for q, vs in zip(qpoints, sv):
        print q
        for v in vs:
            print v, np.linalg.norm(np.dot(primitive_vectors, v))

    rotations = symmetry.get_reciprocal_operations()
    print get_lattice_vector_equivalence(rotations)
    
