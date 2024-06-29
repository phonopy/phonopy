"""Phonon calculation on sampling mesh."""

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

import warnings

import numpy as np

from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    run_dynamical_matrix_solver_c,
)
from phonopy.structure.grid_points import GridPoints
from phonopy.units import VaspToTHz


class MeshBase:
    """Base class of Mesh and IterMesh classes.

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

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        is_gamma_center=False,
        rotations=None,  # Point group operations in real space
        factor=VaspToTHz,
    ):
        """Init method."""
        self._mesh = np.array(mesh, dtype="intc")
        self._with_eigenvectors = with_eigenvectors
        self._factor = factor
        self._cell = dynamical_matrix.primitive
        self._dynamical_matrix = dynamical_matrix

        self._gp = GridPoints(
            self._mesh,
            np.linalg.inv(self._cell.cell),
            q_mesh_shift=shift,
            is_gamma_center=is_gamma_center,
            is_time_reversal=(is_time_reversal and is_mesh_symmetry),
            rotations=rotations,
            is_mesh_symmetry=is_mesh_symmetry,
        )

        self._qpoints = self._gp.qpoints
        self._weights = self._gp.weights

        self._frequencies = None
        self._eigenvectors = None

        self._q_count = 0

    @property
    def mesh_numbers(self):
        """Return mesh numbers."""
        return self._mesh

    def get_mesh_numbers(self):
        """Return mesh numbers."""
        warnings.warn(
            "MeshBase.get_mesh_numbers() is deprecated. Use mesh_numbers attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mesh_numbers

    @property
    def qpoints(self):
        """Return (irreducible) q-points."""
        return self._qpoints

    def get_qpoints(self):
        """Return (irreducible) q-points."""
        warnings.warn(
            "MeshBase.get_qpoints() is deprecated. Use qpoints attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.qpoints

    @property
    def weights(self):
        """Return (irreducible) weights of q-points."""
        return self._weights

    def get_weights(self):
        """Return (irreducible) weights of q-points."""
        warnings.warn(
            "MeshBase.get_weights()) is deprecated. Use weights attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.weights

    @property
    def grid_address(self):
        """Return mesh grid addresses."""
        return self._gp.grid_address

    def get_grid_address(self):
        """Return mesh grid addresses."""
        warnings.warn(
            "MeshBase.get_grid_address()) is deprecated. Use grid_address attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grid_address

    @property
    def ir_grid_points(self):
        """Return irreducible grid indices."""
        return self._gp.ir_grid_points

    def get_ir_grid_points(self):
        """Return irreducible grid indices."""
        warnings.warn(
            "MeshBase.get_ir_grid_points() is deprecated. "
            "Use ir_grid_points attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ir_grid_points

    @property
    def grid_mapping_table(self):
        """Return grid index mapping table."""
        return self._gp.grid_mapping_table

    def get_grid_mapping_table(self):
        """Return grid index mapping table."""
        warnings.warn(
            "MeshBase.get_grid_mapping_table() is deprecated. "
            "Use grid_mapping_table attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.grid_mapping_table

    @property
    def dynamical_matrix(self):
        """Return dynamical matrix class instance."""
        return self._dynamical_matrix

    def get_dynamical_matrix(self):
        """Return dynamical matrix class instance."""
        warnings.warn(
            "MeshBase.get_dynamical_matrix() is deprecated. "
            "Use dynamical_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dynamical_matrix

    @property
    def with_eigenvectors(self):
        """Whether eigenvectors are calculated or not."""
        return self._with_eigenvectors


class Mesh(MeshBase):
    """Class for phonons on mesh grid.

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
        shape=(ir-grid points, bands, bands)
        dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
        order='C'
    group_velocities: ndarray
        Phonon group velocities at ir-grid points.
        shape=(ir-grid points, bands, 3)
        dtype='double'
    More attributes from MeshBase should be watched.

    """

    def __init__(
        self,
        dynamical_matrix,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        is_gamma_center=False,
        group_velocity=None,
        rotations=None,  # Point group operations in real space
        factor=VaspToTHz,
    ):
        """Init method."""
        super().__init__(
            dynamical_matrix,
            mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            with_eigenvectors=with_eigenvectors,
            is_gamma_center=is_gamma_center,
            rotations=rotations,
            factor=factor,
        )

        self._group_velocity = group_velocity
        self._group_velocities = None

    def __iter__(self):
        """Define iterator over q-points.

        Initially, all phonons are computed and stored in arrays.
        Then this is just used as an iterator to return exisiting results.
        The purpose of this iterator is compatible use of IterMesh.

        """
        if self._frequencies is None:
            self.run()
        return self

    def __next__(self):
        """Return phonon frequencies and eigenvectors at each q-point."""
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
        """Calculate phonons at all required q-points."""
        self._set_phonon()
        if self._group_velocity is not None:
            self._set_group_velocities(self._group_velocity)

    @property
    def frequencies(self):
        """Return phonon frequencies."""
        if self._frequencies is None:
            self.run()
        return self._frequencies

    def get_frequencies(self):
        """Return phonon frequencies."""
        warnings.warn(
            "Mesh.get_frequencies() is deprecated. " "Use frequencies attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frequencies

    @property
    def eigenvectors(self):
        """Return eigenvectors.

        Eigenvectors is a numpy array of three dimension.
        The first index runs through q-points.
        In the second and third indices, eigenvectors obtained
        using numpy.linalg.eigh are stored.

        The third index corresponds to the eigenvalue's index.
        The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].

        """
        if self._frequencies is None:
            self.run()
        return self._eigenvectors

    def get_eigenvectors(self):
        """Return eigenvectors."""
        warnings.warn(
            "Mesh.get_eigenvectors() is deprecated. " "Use eigenvectors attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eigenvectors

    @property
    def group_velocities(self):
        """Return group velocities."""
        if self._frequencies is None:
            self.run()
        return self._group_velocities

    def get_group_velocities(self):
        """Return group velocities."""
        warnings.warn(
            "Mesh.get_group_velocities() is deprecated. "
            "Use group_velocities attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.group_velocities

    def write_hdf5(self, filename="mesh.hdf5"):
        """Write results to hdf5 file."""
        import h5py

        with h5py.File(filename, "w") as w:
            w.create_dataset("mesh", data=self._mesh)
            w.create_dataset("qpoint", data=self._qpoints)
            w.create_dataset("weight", data=self._weights)
            w.create_dataset("frequency", data=self._frequencies)
            if self._eigenvectors is not None:
                w.create_dataset("eigenvector", data=self._eigenvectors)
            if self._group_velocities is not None:
                w.create_dataset("group_velocity", data=self._group_velocities)

    def write_yaml(self, filename="mesh.yaml"):
        """Write results to yaml file."""
        natom = len(self._cell)
        rec_lattice = np.linalg.inv(self._cell.cell)  # column vectors
        distances = np.sqrt(np.sum(np.dot(self._qpoints, rec_lattice.T) ** 2, axis=1))

        lines = []

        lines.append("mesh: [ %5d, %5d, %5d ]" % tuple(self._mesh))
        lines.append("nqpoint: %-7d" % self._qpoints.shape[0])
        lines.append("reciprocal_lattice:")
        for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*")):
            lines.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
        lines.append("natom:   %-7d" % natom)
        lines.append(str(self._cell))
        lines.append("")
        lines.append("phonon:")

        for i, (q, d) in enumerate(zip(self._qpoints, distances)):
            lines.append("- q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(q))
            lines.append("  distance_from_gamma: %12.9f" % d)
            lines.append("  weight: %-5d" % self._weights[i])
            lines.append("  band:")

            for j, freq in enumerate(self._frequencies[i]):
                lines.append("  - # %d" % (j + 1))
                lines.append("    frequency:  %15.10f" % freq)

                if self._group_velocities is not None:
                    lines.append(
                        "    group_velocity: "
                        "[ %13.7f, %13.7f, %13.7f ]"
                        % tuple(self._group_velocities[i, j])
                    )

                if self._with_eigenvectors:
                    lines.append("    eigenvector:")
                    for k in range(natom):
                        lines.append("    - # atom %d" % (k + 1))
                        for ll in (0, 1, 2):
                            lines.append(
                                "      - [ %17.14f, %17.14f ]"
                                % (
                                    self._eigenvectors[i, k * 3 + ll, j].real,
                                    self._eigenvectors[i, k * 3 + ll, j].imag,
                                )
                            )
            lines.append("")

        with open(filename, "w") as w:
            w.write("\n".join(lines))

    def _set_phonon(self):
        import phonopy._phonopy as phonoc

        num_band = len(self._cell) * 3
        num_qpoints = len(self._qpoints)

        self._frequencies = np.zeros((num_qpoints, num_band), dtype="double")
        if phonoc.use_openmp():
            dynmat = run_dynamical_matrix_solver_c(
                self._dynamical_matrix, self._qpoints
            )
            eigenvectors = dynmat
        elif self._with_eigenvectors:
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            eigenvectors = np.zeros(
                (
                    num_qpoints,
                    num_band,
                    num_band,
                ),
                dtype=dtype,
                order="C",
            )

        for i, q in enumerate(self._qpoints):
            if phonoc.use_openmp():
                dm = dynmat[i]
            else:
                self._dynamical_matrix.run(q)
                dm = self._dynamical_matrix.dynamical_matrix
            if self._with_eigenvectors:
                eigvals, eigenvectors[i] = np.linalg.eigh(dm)
                eigenvalues = eigvals.real
            else:
                eigenvalues = np.linalg.eigvalsh(dm).real
            self._frequencies[i] = (
                np.array(
                    np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues),
                    dtype="double",
                    order="C",
                )
                * self._factor
            )

        if self._with_eigenvectors:
            self._eigenvectors = eigenvectors

    def _set_group_velocities(self, group_velocity):
        group_velocity.run(self._qpoints)
        self._group_velocities = group_velocity.group_velocities


class IterMesh(MeshBase):
    """Generator class for phonons on mesh grid.

    Not like as Mesh class, frequencies and eigenvectors are not
    stored, instead generated by iterator. This may be used for
    saving memory space even with very dense samplig mesh.

    Attributes
    ----------
    Attributes from MeshBase should be watched.

    """

    def __init__(
        self,
        dynamical_matrix,
        mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=True,
        with_eigenvectors=False,
        is_gamma_center=False,
        rotations=None,  # Point group operations in real space
        factor=VaspToTHz,
    ):
        """Init method."""
        super().__init__(
            dynamical_matrix,
            mesh,
            shift=shift,
            is_time_reversal=is_time_reversal,
            is_mesh_symmetry=is_mesh_symmetry,
            with_eigenvectors=with_eigenvectors,
            is_gamma_center=is_gamma_center,
            rotations=rotations,
            factor=factor,
        )

    def __iter__(self):
        """Define iterator over q-points."""
        return self

    def __next__(self):
        """Calculate phonons at a q-point."""
        if self._q_count == len(self._qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            q = self._qpoints[self._q_count]
            self._dynamical_matrix.run(q)
            dm = self._dynamical_matrix.dynamical_matrix
            if self._with_eigenvectors:
                eigvals, eigenvectors = np.linalg.eigh(dm)
                eigenvalues = eigvals.real
            else:
                eigenvalues = np.linalg.eigvalsh(dm).real
            frequencies = (
                np.array(
                    np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues),
                    dtype="double",
                    order="C",
                )
                * self._factor
            )
            self._q_count += 1
            return frequencies, eigenvectors
