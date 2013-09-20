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

class BrillouinZone:
    def __init__(self,
                 primitive_vectors,
                 mesh,
                 qpoints):
        self._primitive_vectors = primitive_vectors # column vectors
        self._mesh = mesh
        longest = max([np.linalg.norm(vec) for vec in primitive_vectors.T])
        self._reduced_bases = get_reduced_bases(primitive_vectors,
                                                longest / max(mesh) / 10)
        self._qpoints = np.dot(np.linalg.inv(self._reduced_bases),
                               np.dot(primitive_vectors, qpoints.T)).T

        print self._qpoints
                 
if __name__ == '__main__':
    from phonopy.interface.vasp import read_vasp
    from phonopy.structure.spglib import get_ir_reciprocal_mesh
    import sys

    cell = read_vasp(sys.argv[1])
    mesh = [10, 10, 10]
    mapping_table, grid_addrees = get_ir_reciprocal_mesh(
        mesh,
        cell,
        is_shift=[0, 0, 0])

    ir_grid_points = np.unique(mapping_table)
    qpoints = grid_addrees[ir_grid_points] / np.array(mesh, dtype='double')
    
    bz = BrillouinZone(np.linalg.inv(cell.get_cell()),
                       mesh,
                       qpoints)

