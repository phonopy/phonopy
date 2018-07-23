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


class MeshBase(object):
    """Base class of Mesh and IterMesh classes

    Attributes
    ----------
    mesh_numbers: ndarray
        Mesh numbers along a, b, c axes.
        dtype='intc'
        shape=(3,)
    qpoints: ndarray
        q-points in reduced coordinates of reciprocal lattice
        dtype='double'
        shape=(ir-grid points, 3)
    weights: ndarray
        Geometric q-point weights. Its sum is the number of grid points.
        dtype='intc'
        shape=(ir-grid points,)
    grid_address: ndarray
        Addresses of all grid points represented by integers.
        dtype='intc'
        shape=(prod(mesh_numbers), 3)
    ir_grid_points: ndarray
        Indices of irreducibple grid points in grid_address.
        dtype='intc'
        shape=(ir-grid points,)
    grid_mapping_table: ndarray
        Index mapping table from all grid points to ir-grid points.
        dtype='intc'
        shape=(prod(mesh_numbers),)
    dynamical_matrix: DynamicalMatrix
        Dynamical matrix instance to compute dynamical matrix at q-points.

    """
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 rotations=None,  # Point group operations in real space
                 factor=VaspToTHz):
        self._mesh = np.array(mesh, dtype='intc')
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor
        self._cell = dynamical_matrix.get_primitive()
        self._dynamical_matrix = dynamical_matrix

        self._gp = GridPoints(self._mesh,
                              np.linalg.inv(self._cell.get_cell()),
                              q_mesh_shift=shift,
                              is_gamma_center=is_gamma_center,
                              is_time_reversal=(is_time_reversal and
                                                is_mesh_symmetry),
                              rotations=rotations,
                              is_mesh_symmetry=is_mesh_symmetry)

        self._qpoints = self._gp.qpoints
        self._weights = self._gp.weights

        self._frequencies = None
        self._eigenvectors = None

        self._q_count = 0

    @property
    def mesh_numbers(self):
        return self._mesh

    def get_mesh_numbers(self):
        return self.mesh_numbers

    @property
    def qpoints(self):
        return self._qpoints

    def get_qpoints(self):
        return self.qpoints

    @property
    def weights(self):
        return self._weights

    def get_weights(self):
        return self.weights

    @property
    def grid_address(self):
        return self._gp.grid_address

    def get_grid_address(self):
        return self.grid_address

    @property
    def ir_grid_points(self):
        return self._gp.ir_grid_points

    def get_ir_grid_points(self):
        return self.ir_grid_points

    @property
    def grid_mapping_table(self):
        return self._gp.grid_mapping_table

    def get_grid_mapping_table(self):
        return self.grid_mapping_table

    @property
    def dynamical_matrix(self):
        return self._dynamical_matrix

    def get_dynamical_matrix(self):
        return self.dynamical_matrix


class Mesh(MeshBase):
    """Class for phonons on mesh grid

    Frequencies and eigenvectors can be also accessible by iterator
    representation to be compatible with IterMesh.

    Attributes
    ----------
    frequencies: ndarray
        Phonon frequencies at ir-grid points. Imaginary frequenies are
        represented by negative real numbers.
        dtype='double'
        shape=(ir-grid points, bands)
    eigenvectors: ndarray
        Phonon eigenvectors at ir-grid points. See the data structure at
        np.linalg.eigh.
        dtype='complex128'
        shape=(ir-grid points, bands, bands)
    group_velocities: ndarray
        Phonon group velocities at ir-grid points.
        dtype='double'
        shape=(ir-grid points, bands, 3)
    More attributes from MeshBase should be watched.

    """
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 group_velocity=None,
                 rotations=None,  # Point group operations in real space
                 factor=VaspToTHz,
                 use_lapack_solver=False):
        MeshBase.__init__(self,
                          dynamical_matrix,
                          mesh,
                          shift=shift,
                          is_time_reversal=is_time_reversal,
                          is_mesh_symmetry=is_mesh_symmetry,
                          is_eigenvectors=is_eigenvectors,
                          is_gamma_center=is_gamma_center,
                          rotations=rotations,
                          factor=factor)

        self._group_velocity = group_velocity
        self._group_velocities = None
        self._use_lapack_solver = use_lapack_solver

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._q_count == len(self._qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            i = self._q_count
            self._q_count += 1
            if self._eigenvectors is None:
                return self._frequencies[i], None
            else:
                return self._frequencies[i], self._eigenvectors[i]

    def run(self):
        self._set_phonon()
        if self._group_velocity is not None:
            self._set_group_velocities(self._group_velocity)

    @property
    def frequencies(self):
        return self._frequencies

    def get_frequencies(self):
        return self.frequencies

    @property
    def eigenvectors(self):
        """
        Eigenvectors is a numpy array of three dimension.
        The first index runs through q-points.
        In the second and third indices, eigenvectors obtained
        using numpy.linalg.eigh are stored.

        The third index corresponds to the eigenvalue's index.
        The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        """
        return self._eigenvectors

    def get_eigenvectors(self):
        return self.eigenvectors

    @property
    def group_velocities(self):
        return self._group_velocities

    def get_group_velocities(self):
        return self.group_velocities

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
        natom = self._cell.get_number_of_atoms()
        rec_lattice = np.linalg.inv(self._cell.get_cell())  # column vectors
        distances = np.sqrt(
            np.sum(np.dot(self._qpoints, rec_lattice.T) ** 2, axis=1))

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

        for i, (q, d) in enumerate(zip(self._qpoints, distances)):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            w.write("  distance_from_gamma: %12.9f\n" % d)
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
                        for l in (0, 1, 2):
                            w.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i, k*3+l, j].real,
                                     self._eigenvectors[i, k*3+l, j].imag))
            w.write("\n")

    def _set_phonon(self):
        num_band = self._cell.get_number_of_atoms() * 3
        num_qpoints = len(self._qpoints)

        self._frequencies = np.zeros((num_qpoints, num_band), dtype='double')
        if self._is_eigenvectors or self._use_lapack_solver:
            dtype = "c%d" % (np.dtype('double').itemsize * 2)
            self._eigenvectors = np.zeros(
                (num_qpoints, num_band, num_band,), dtype=dtype)

        if self._use_lapack_solver:
            from phono3py.phonon.solver import get_phonons_at_qpoints
            get_phonons_at_qpoints(self._frequencies,
                                   self._eigenvectors,
                                   self._dynamical_matrix,
                                   self._qpoints,
                                   self._factor,
                                   nac_q_direction=None,
                                   lapack_zheev_uplo='L')
        else:
            for i, q in enumerate(self._qpoints):
                self._dynamical_matrix.set_dynamical_matrix(q)
                dm = self._dynamical_matrix.get_dynamical_matrix()
                if self._is_eigenvectors:
                    eigvals, self._eigenvectors[i] = np.linalg.eigh(dm)
                    eigenvalues = eigvals.real
                else:
                    eigenvalues = np.linalg.eigvalsh(dm).real
                self._frequencies[i] = np.array(np.sqrt(abs(eigenvalues)) *
                                                np.sign(eigenvalues),
                                                dtype='double',
                                                order='C') * self._factor

    def _set_group_velocities(self, group_velocity):
        group_velocity.set_q_points(self._qpoints)
        self._group_velocities = group_velocity.get_group_velocity()


class IterMesh(MeshBase):
    """Generator class for phonons on mesh grid

    Not like as Mesh class, frequencies and eigenvectors are not
    stored, instead generated by iterator. This may be used for
    saving memory space even with very dense samplig mesh.

    Attributes
    ----------
    Attributes from MeshBase should be watched.

    """
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 shift=None,
                 is_time_reversal=True,
                 is_mesh_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 rotations=None,  # Point group operations in real space
                 factor=VaspToTHz):
        MeshBase.__init__(self,
                          dynamical_matrix,
                          mesh,
                          shift=shift,
                          is_time_reversal=is_time_reversal,
                          is_mesh_symmetry=is_mesh_symmetry,
                          is_eigenvectors=is_eigenvectors,
                          is_gamma_center=is_gamma_center,
                          rotations=rotations,
                          factor=factor)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._q_count == len(self._qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            q = self._qpoints[self._q_count]
            self._dynamical_matrix.set_dynamical_matrix(q)
            dm = self._dynamical_matrix.get_dynamical_matrix()
            if self._is_eigenvectors:
                eigvals, eigenvectors = np.linalg.eigh(dm)
                eigenvalues = eigvals.real
            else:
                eigenvalues = np.linalg.eigvalsh(dm).real
            frequencies = np.array(np.sqrt(abs(eigenvalues)) *
                                   np.sign(eigenvalues),
                                   dtype='double',
                                   order='C') * self._factor
            self._q_count += 1
            return frequencies, eigenvectors
