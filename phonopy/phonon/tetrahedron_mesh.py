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
from phonopy.phonon.mesh import shift2boolean, has_mesh_symmetry
from phonopy.structure.spglib import get_stabilized_reciprocal_mesh, relocate_BZ_grid_address
from phonopy.units import VaspToTHz
from phonopy.structure.tetrahedron_method import TetrahedronMethod

class TetrahedronMesh:
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 rotations, # Point group operations in real space
                 shift=None,
                 is_time_reversal=True,
                 is_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 factor=VaspToTHz):
        self._dynamical_matrix = dynamical_matrix
        self._mesh = mesh
        self._rotations = rotations
        self._shift = shift
        self._is_time_reversal = is_time_reversal
        self._is_symmetry = is_symmetry
        self._is_gamma_center = is_gamma_center
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor

        self._grid_address = None
        self._grid_mapping_table = None
        self._grid_order = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._cell = dynamical_matrix.get_primitive()
        self._frequencies = None
        self._eigenvalues = None
        self._eigenvectors = None

        self._tm = None
        self._tetrahedra_frequencies = None
        
    def run(self):
        self._set_grid_points()
        self._set_phonon()
        self._tm = TetrahedronMethod(np.linalg.inv(self._cell.get_cell()),
                                     self._mesh)
        n = 201
        max_frequency = np.amax(self._frequencies)
        freq_points = np.linspace(0, 10, n)
        max_i = np.where(max_frequency < freq_points)[0][0]
            
        dos = np.zeros(n, dtype='double')
        for i, (gp, weight) in enumerate(
            zip(self._ir_grid_points, self._ir_grid_weights)):
            print "# %d/%d" % (i + 1, len(self._ir_grid_weights))
            sys.stdout.flush()
            self._set_tetrahedra_frequencies(gp)
            for frequencies in self._tetrahedra_frequencies:
                self._tm.set_tetrahedra_omegas(frequencies)
                for j, f in enumerate(freq_points[:max_i]):
                    dos[j] += self._tm.run(f, value='g') * weight

        dos /= np.prod(self._mesh)
        for f, d in zip(freq_points, dos):
            print f, d

    def _set_grid_points(self):
        mesh = self._mesh
        
        if not self._is_symmetry:
            print "Disabling mesh symmetry is not supported."
        assert has_mesh_symmetry(mesh, self._rotations), \
            "Mesh numbers don't have proper symmetry."
            
        self._is_shift = shift2boolean(mesh,
                                       q_mesh_shift=self._shift,
                                       is_gamma_center=self._is_gamma_center)

        if not self._is_shift:
            print "Only mesh shift of 0 or 1/2 is allowed."
            print "Mesh shift is set [0, 0, 0]."

        self._grid_mapping_table, grid_address = get_stabilized_reciprocal_mesh(
            mesh,
            self._rotations,
            is_shift=self._is_shift,
            is_time_reversal=self._is_time_reversal)

        if grid_address[1, 0] == 1:
            self._grid_order = [1, mesh[0], mesh[0] * mesh[1]]
        else:
            self._grid_order = [mesh[2] * mesh[1], mesh[2], 1]

        self._grid_address = relocate_BZ_grid_address(
            grid_address,
            mesh,
            np.linalg.inv(self._cell.get_cell()),
            is_shift=self._is_shift)[0][:np.prod(mesh)]

        shift = [s * 0.5 for s in self._is_shift]
        self._qpoints = (self._grid_address + shift) / mesh

        self._ir_grid_points = np.unique(self._grid_mapping_table)
        weights = np.zeros_like(self._grid_mapping_table)
        for gp in self._grid_mapping_table:
            weights[gp] += 1
        self._ir_grid_weights = weights[self._ir_grid_points]

    def _set_phonon(self):
        self._eigenvalues = np.zeros(
            (np.prod(self._mesh), self._cell.get_number_of_atoms() * 3),
            dtype='double')
        self._frequencies = np.zeros_like(self._eigenvalues)
        for gp in np.unique(self._grid_mapping_table):
            self._dynamical_matrix.set_dynamical_matrix(self._qpoints[gp])
            dm = self._dynamical_matrix.get_dynamical_matrix()
            self._eigenvalues[gp] = np.linalg.eigvalsh(dm).real
        self._frequencies = np.array(np.sqrt(abs(self._eigenvalues)) *
                                     np.sign(self._eigenvalues)) * self._factor

    def _set_tetrahedra_frequencies(self, gp):
        neighbors = np.zeros((24, 4), dtype='intc')
        frequencies = np.zeros((self._frequencies.shape[1], 24, 4),
                               dtype='double')
        for i, t in enumerate(self._tm.get_tetrahedra()):
            address = t + self._grid_address[gp]
            neighbors[i] = np.dot(address % self._mesh, self._grid_order)
            frequencies[:, i, :] = self._frequencies[
                self._grid_mapping_table[neighbors[i]]].T
        self._tetrahedra_frequencies = frequencies

