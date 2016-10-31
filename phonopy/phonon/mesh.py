# Copyright (C) 2011 Atsushi Togo
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
from phonopy.units import VaspToTHz
from phonopy.structure.grid_points import GridPoints

class Mesh(object):
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 group_velocity=None,
                 rotations=None, # Point group operations in real space
                 factor=VaspToTHz,
                 use_lapack_solver=False):

        self._mesh = np.array(mesh, dtype='intc')
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor
        self._cell = dynamical_matrix.get_primitive()
        self._dynamical_matrix = dynamical_matrix
        self._use_lapack_solver = use_lapack_solver

        self._gp = GridPoints(self._mesh,
                              np.linalg.inv(self._cell.get_cell()),
                              q_mesh_shift=shift,
                              is_gamma_center=is_gamma_center,
                              is_time_reversal=is_time_reversal,
                              rotations=rotations,
                              is_mesh_symmetry=is_mesh_symmetry)

        self._qpoints = self._gp.get_ir_qpoints()
        self._weights = self._gp.get_ir_grid_weights()

        self._frequencies = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._set_phonon()

        self._group_velocities = None
        if group_velocity is not None:
            self._set_group_velocities(group_velocity)

    def get_dynamical_matrix(self):
        return self._dynamical_matrix
        
    def get_mesh_numbers(self):
        return self._mesh
        
    def get_qpoints(self):
        return self._qpoints

    def get_weights(self):
        return self._weights

    def get_grid_address(self):
        return self._gp.get_grid_address()

    def get_ir_grid_points(self):
        return self._gp.get_ir_grid_points()
    
    def get_grid_mapping_table(self):
        return self._gp.get_grid_mapping_table()

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_frequencies(self):
        return self._frequencies

    def get_group_velocities(self):
        return self._group_velocities
    
    def get_eigenvectors(self):
        """
        Eigenvectors is a numpy array of three dimension.
        The first index runs through q-points.
        In the second and third indices, eigenvectors obtained
        using numpy.linalg.eigh are stored.
        
        The third index corresponds to the eigenvalue's index.
        The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        """
        return self._eigenvectors

    def write_hdf5(self):
        import h5py
        with h5py.File('mesh.hdf5', 'w') as w:
            w.create_dataset('mesh', data=self._mesh)
            w.create_dataset('qpoint', data=self._qpoints)
            w.create_dataset('weight', data=self._weights)
            w.create_dataset('frequency', data=self._frequencies)
            if self._eigenvectors is not None:
                w.create_dataset('eigenvector', data=self._eigenvectors)
            if self._group_velocities is not None:
                w.create_dataset('group_velocity', data=self._group_velocities)

    def write_yaml(self):
        w = open('mesh.yaml', 'w')
        eigenvalues = self._eigenvalues
        natom = self._cell.get_number_of_atoms()
        rec_lattice = np.linalg.inv(self._cell.get_cell()) # column vectors

        w.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        w.write("nqpoint: %-7d\n" % self._qpoints.shape[0])
        w.write("reciprocal_lattice:\n")
        for vec, axis in zip(rec_lattice.T, ('a*', 'b*', 'c*')):
            w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" %
                    (tuple(vec) + (axis,)))
        w.write("natom:   %-7d\n" % natom)
        w.write(str(self._cell))
        w.write("\n")
        w.write("phonon:\n")

        for i, q in enumerate(self._qpoints):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            w.write("  weight: %-5d\n" % self._weights[i])
            w.write("  band:\n")

            for j, freq in enumerate(self._frequencies[i]):
                w.write("  - # %d\n" % (j + 1))
                w.write("    frequency:  %15.10f\n" % freq)

                if self._group_velocities is not None:
                    w.write("    group_velocity: ")
                    w.write("[ %13.7f, %13.7f, %13.7f ]\n" %
                            tuple(self._group_velocities[i, j]))

                if self._is_eigenvectors:
                    w.write("    eigenvector:\n")
                    for k in range(natom):
                        w.write("    - # atom %d\n" % (k+1))
                        for l in (0,1,2):
                            w.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i,k*3+l,j].real,
                                     self._eigenvectors[i,k*3+l,j].imag))
            w.write("\n")

    def _set_phonon(self):
        num_band = self._cell.get_number_of_atoms() * 3
        num_qpoints = len(self._qpoints)

        self._eigenvalues = np.zeros((num_qpoints, num_band), dtype='double')
        self._frequencies = np.zeros_like(self._eigenvalues)
        if self._is_eigenvectors or self._use_lapack_solver:
            self._eigenvectors = np.zeros(
                (num_qpoints, num_band, num_band,), dtype='complex128')

        if self._use_lapack_solver:
            from phono3py.phonon.solver import get_phonons_at_qpoints
            get_phonons_at_qpoints(self._frequencies,
                                   self._eigenvectors,
                                   self._dynamical_matrix,
                                   self._qpoints,
                                   self._factor,
                                   nac_q_direction=None,
                                   lapack_zheev_uplo='L')
            self._eigenvalues = np.array(self._frequencies ** 2 *
                                         np.sign(self._frequencies),
                                         dtype='double',
                                         order='C') / self._factor ** 2
            if not self._is_eigenvectors:
                self._eigenvalues = None
        else:
            for i, q in enumerate(self._qpoints):
                self._dynamical_matrix.set_dynamical_matrix(q)
                dm = self._dynamical_matrix.get_dynamical_matrix()
                if self._is_eigenvectors:
                    eigvals, self._eigenvectors[i] = np.linalg.eigh(dm)
                    self._eigenvalues[i] = eigvals.real
                else:
                    self._eigenvalues[i] = np.linalg.eigvalsh(dm).real
            self._frequencies = np.array(np.sqrt(abs(self._eigenvalues)) *
                                         np.sign(self._eigenvalues),
                                         dtype='double',
                                         order='C') * self._factor


    def _set_group_velocities(self, group_velocity):
        group_velocity.set_q_points(self._qpoints)
        self._group_velocities = group_velocity.get_group_velocity()
