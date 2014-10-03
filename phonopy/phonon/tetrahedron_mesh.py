# copyright (C) 2013 Atsushi Togo
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

import sys
import numpy as np
from phonopy.structure.tetrahedron_method import TetrahedronMethod

def get_tetrahedra_frequencies(gp,
                               mesh,
                               grid_order,
                               grid_address,
                               relative_grid_address,
                               gp_ir_index,
                               frequencies):
    t_frequencies = np.zeros((frequencies.shape[1], 24, 4), dtype='double')
    for i, t in enumerate(relative_grid_address):
        address = t + grid_address[gp]
        neighbors = np.dot(address % mesh, grid_order)
        t_frequencies[:, i, :] = frequencies[gp_ir_index[neighbors]].T
    return t_frequencies

# def __init__(self,
#              cell,
#              mesh,
#              grid_address,
#              ir_grid_points,
#              grid_mapping_table,
#              grid_order,
#              frequencies
             
class TetrahedronMesh:
    def __init__(self, mesh_object):
        self._mesh_object = mesh_object
        self._grid_address = None
        self._grid_order = None
        self._ir_grid_points = None
        self._ir_grid_weights = None
        self._gp_ir_index = None

        self._cell = None
        self._frequencies = None
        self._eigenvalues = None

        self._tm = None
        self._tetrahedra_frequencies = None
        self._integration_weights = None
        self._relative_grid_address = None

        self._total_dos = None
        self._partial_dos = None

        self._prepare()
        
    def get_total_dos(self):
        return self._total_dos

    def get_partial_dos(self):
        return self._partial_dos

    def get_integration_weights(self):
        return self._integration_weights

    def get_frequency_points(self):
        return self._frequency_points

    def run_at_frequencies(self,
                           value='I',
                           division_number=201,
                           frequency_points=None):                            
        if frequency_points is None:
            max_frequency = np.amax(self._frequencies)
            min_frequency = np.amin(self._frequencies)
            self._frequency_points = np.linspace(min_frequency,
                                                 max_frequency,
                                                 division_number)
        else:
            self._frequency_points = frequency_points

        num_ir_grid_points = len(self._ir_grid_points)
        num_band = self._cell.get_number_of_atoms() * 3
        num_freqs = len(self._frequency_points)
        self._integration_weights = np.zeros(
            (num_freqs, num_band, num_ir_grid_points), dtype='double')

        reciprocal_lattice = np.linalg.inv(self._cell.get_cell())
        self._tm = TetrahedronMethod(reciprocal_lattice, mesh=self._mesh)
        self._relative_grid_address = self._tm.get_tetrahedra()

        for i, gp in enumerate(self._ir_grid_points):
            self._set_tetrahedra_frequencies(gp)
            for ib, frequencies in enumerate(self._tetrahedra_frequencies):
                self._tm.set_tetrahedra_omegas(frequencies)
                self._tm.run(self._frequency_points, value=value)
                iw = self._tm.get_integration_weight()
                self._integration_weights[:, ib, i] = iw

        self._integration_weights /= np.prod(self._mesh)

    def _prepare(self):
        mo = self._mesh_object
        self._cell = mo.get_dynamical_matrix().get_primitive()
        self._mesh = mo.get_mesh_numbers()
        self._grid_address = mo.get_grid_address()
        self._ir_grid_points = mo.get_ir_grid_points()
        self._grid_order = [1, self._mesh[0], self._mesh[0] * self._mesh[1]]

        grid_mapping_table = mo.get_grid_mapping_table()
        self._gp_ir_index = np.zeros_like(grid_mapping_table)
        count = 0
        for i, gp in enumerate(grid_mapping_table):
            if i == gp:
                self._gp_ir_index[i] = count
                count += 1
            else:
                self._gp_ir_index[i] = self._gp_ir_index[grid_mapping_table[i]]

        self._frequencies = mo.get_frequencies()

    def _set_tetrahedra_frequencies(self, gp):
        self._tetrahedra_frequencies = get_tetrahedra_frequencies(
            gp,
            self._mesh,
            self._grid_order,
            self._grid_address,
            self._relative_grid_address,
            self._gp_ir_index,
            self._frequencies)
