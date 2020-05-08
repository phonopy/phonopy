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


class QpointsPhonon(object):
    """Calculate phonons at specified qpoints

    Attributes
    ----------
    frequencies : ndarray
        Phonon frequencies. Imaginary frequenies are represented by
        negative real numbers.
        shape=(qpoints, bands), dtype='double'
    eigenvectors : ndarray
        Phonon eigenvectors. None if eigenvectors are not stored.
        shape=(qpoints, bands, bands), dtype='complex'
    group_velocities : ndarray
        Phonon group velocities. None if group velocities are not
        calculated.
        shape=(qpoints, bands, 3), dtype='double'
    dynamical_matrices : ndarray
        Dynamical matrices at q-points.
        shape=(qpoints, bands, bands), dtype='double'

    """

    def __init__(self,
                 qpoints,
                 dynamical_matrix,
                 nac_q_direction=None,
                 with_eigenvectors=False,
                 group_velocity=None,
                 with_dynamical_matrices=False,
                 factor=VaspToTHz):
        primitive = dynamical_matrix.get_primitive()
        self._natom = primitive.get_number_of_atoms()
        self._masses = primitive.get_masses()
        self._symbols = primitive.get_chemical_symbols()
        self._positions = primitive.get_scaled_positions()
        self._lattice = primitive.get_cell()

        self._qpoints = qpoints
        self._dynamical_matrix = dynamical_matrix
        self._nac_q_direction = nac_q_direction
        self._with_eigenvectors = with_eigenvectors
        self._group_velocity = group_velocity
        self._with_dynamical_matrices = with_dynamical_matrices
        self._factor = factor

        self._group_velocities = None
        self._eigenvectors = None
        self._frequencies = None
        self._dynamical_matrices = None

        self._run()

    @property
    def frequencies(self):
        return self._frequencies

    def get_frequencies(self):
        return self.frequencies

    @property
    def eigenvectors(self):
        return self._eigenvectors

    def get_eigenvectors(self):
        return self.eigenvectors

    @property
    def group_velocities(self):
        return self._group_velocities

    def get_group_velocities(self):
        return self.group_velocities

    @property
    def dynamical_matrices(self):
        return self._dynamical_matrices

    def write_hdf5(self):
        import h5py
        with h5py.File('qpoints.hdf5', 'w') as w:
            w.create_dataset('qpoint', data=self._qpoints)
            w.create_dataset('frequency', data=self._frequencies)
            if self._with_eigenvectors:
                w.create_dataset('eigenvector', data=self._eigenvectors)
            if self._group_velocities is not None:
                w.create_dataset('group_velocity', data=self._group_velocities)
            if self._with_dynamical_matrices:
                w.create_dataset('dynamical_matrix',
                                 data=self._dynamical_matrices)

    def write_yaml(self):
        w = open('qpoints.yaml', 'w')
        w.write("nqpoint: %-7d\n" % len(self._qpoints))
        w.write("natom:   %-7d\n" % self._natom)
        rec_lattice = np.linalg.inv(self._lattice)  # column vectors
        w.write("reciprocal_lattice:\n")
        for vec, axis in zip(rec_lattice.T, ('a*', 'b*', 'c*')):
            w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" %
                    (tuple(vec) + (axis,)))
        w.write("phonon:\n")

        for i, q in enumerate(self._qpoints):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            if self._with_dynamical_matrices:
                w.write("  dynamical_matrix:\n")
                for row in self._dynamical_matrices[i]:
                    w.write("  - [ ")
                    for j, elem in enumerate(row):
                        w.write("%15.10f, %15.10f" % (elem.real, elem.imag))
                        if j == len(row) - 1:
                            w.write(" ]\n")
                        else:
                            w.write(", ")

            w.write("  band:\n")
            for j, freq in enumerate(self._frequencies[i]):
                w.write("  - # %d\n" % (j + 1))
                w.write("    frequency: %15.10f\n" % freq)

                if self._group_velocities is not None:
                    w.write("    group_velocity: [ %13.7f, %13.7f, %13.7f ]\n"
                            % tuple(self._group_velocities[i, j]))

                if self._with_eigenvectors:
                    w.write("    eigenvector:\n")
                    for k in range(self._natom):
                        w.write("    - # atom %d\n" % (k + 1))
                        for l in (0, 1, 2):
                            w.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i][k * 3 + l, j].real,
                                     self._eigenvectors[i][k * 3 + l, j].imag))
            w.write("\n")

    def _run(self):
        if self._group_velocity is not None:
            self._group_velocity.run(
                self._qpoints, perturbation=self._nac_q_direction)
            self._group_velocities = self._group_velocity.group_velocities

        if self._with_dynamical_matrices:
            dynamical_matrices = []

        self._frequencies = []
        if self._with_eigenvectors:
            self._eigenvectors = []

        for q in self._qpoints:
            dm = self._get_dynamical_matrix(q)
            if self._with_dynamical_matrices:
                dynamical_matrices.append(dm)
            if self._with_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(dm)
                self._eigenvectors.append(eigvecs)
            else:
                eigvals = np.linalg.eigvalsh(dm)
            eigvals = eigvals.real
            self._frequencies.append(np.sqrt(np.abs(eigvals)) *
                                     np.sign(eigvals) * self._factor)

        self._frequencies = np.array(self._frequencies,
                                     dtype='double', order='C')
        dtype = "c%d" % (np.dtype('double').itemsize * 2)
        if self._with_eigenvectors:
            self._eigenvectors = np.array(self._eigenvectors,
                                          dtype=dtype, order='C')
        if self._with_dynamical_matrices:
            self._dynamical_matrices = np.array(dynamical_matrices,
                                                dtype=dtype, order='C')

    def _get_dynamical_matrix(self, q):
        if (self._dynamical_matrix.is_nac() and
            self._nac_q_direction is not None and
            (np.abs(q) < 1e-5).all()):
            self._dynamical_matrix.run(
                q, q_direction=self._nac_q_direction)
        else:
            self._dynamical_matrix.run(q)
        return self._dynamical_matrix.dynamical_matrix
