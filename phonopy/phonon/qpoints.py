"""Phonon calculation at specific q-points."""

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

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixNAC,
    run_dynamical_matrix_solver_c,
)
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import Primitive


class QpointsPhonon:
    """Calculate phonons at specified qpoints.

    Attributes
    ----------
    frequencies : ndarray
        Phonon frequencies. Imaginary frequenies are represented by
        negative real numbers. Unit conversion factor is multipled.
        shape=(qpoints, bands), dtype='double'
    eigenvectors : ndarray
        Phonon eigenvectors. None when with_eigenvectors=False.
        shape=(qpoints, bands, bands), dtype='complex'
    eigenvalues : ndarray
        Phonon eigenvvalues. Unit conversion factor is not multipled.
        shape=(qpoints, bands), dtype='double'
    group_velocities : ndarray
        Phonon group velocities. None if group velocities are not
        calculated.
        shape=(qpoints, bands, 3), dtype='double'
    dynamical_matrices : ndarray
        Dynamical matrices at q-points.
        shape=(qpoints, bands, bands), dtype='double'

    """

    def __init__(
        self,
        qpoints,
        dynamical_matrix: Union[DynamicalMatrix, DynamicalMatrixNAC],
        nac_q_direction: ArrayLike | None = None,
        with_eigenvectors: bool = False,
        group_velocity: GroupVelocity | None = None,
        with_dynamical_matrices: bool = False,
        factor: float | None = None,
    ):
        """Init method."""
        primitive: Primitive = dynamical_matrix.primitive
        self._natom = len(primitive)
        self._masses = primitive.masses
        self._symbols = primitive.symbols
        self._positions = primitive.scaled_positions
        self._lattice = primitive.cell

        self._qpoints = qpoints
        self._dynamical_matrix = dynamical_matrix
        self._nac_q_direction = nac_q_direction
        self._with_eigenvectors = with_eigenvectors
        self._gv_obj: Optional[GroupVelocity] = group_velocity
        self._with_dynamical_matrices = with_dynamical_matrices
        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor

        self._group_velocities: NDArray | None = None
        self._eigenvectors: NDArray | None = None
        self._eigenvalues: NDArray
        self._frequencies: NDArray
        self._dynamical_matrices: NDArray | None = None

        self._run()

    @property
    def frequencies(self) -> NDArray:
        """Return frequencies."""
        return self._frequencies

    @property
    def eigenvalues(self) -> NDArray:
        """Return eigenvalues."""
        return self._eigenvalues

    @property
    def eigenvectors(self) -> NDArray | None:
        """Return eigenvectors."""
        return self._eigenvectors

    @property
    def group_velocities(self) -> NDArray | None:
        """Return group velocities."""
        return self._group_velocities

    @property
    def dynamical_matrices(self):
        """Return DynamicalMatrix class instance."""
        return self._dynamical_matrices

    def write_hdf5(self, filename: str | os.PathLike = "qpoints.hdf5"):
        """Write results in hdf5."""
        import h5py

        with h5py.File(filename, "w") as w:
            w.create_dataset("reciprocal_lattice", data=np.linalg.inv(self._lattice.T))
            w.create_dataset("masses", data=self._masses)
            w.create_dataset("qpoint", data=self._qpoints)
            w.create_dataset("frequency", data=self._frequencies)
            if self._with_eigenvectors:
                w.create_dataset("eigenvector", data=self._eigenvectors)
            if self._group_velocities is not None:
                w.create_dataset("group_velocity", data=self._group_velocities)
            if self._with_dynamical_matrices:
                w.create_dataset("dynamical_matrix", data=self._dynamical_matrices)

    def write_yaml(self, filename: str | os.PathLike = "qpoints.yaml"):
        """Write results in yaml."""
        w = open(filename, "w")
        w.write("nqpoint: %-7d\n" % len(self._qpoints))
        w.write("natom:   %-7d\n" % self._natom)
        rec_lattice = np.linalg.inv(self._lattice)  # column vectors
        w.write("reciprocal_lattice:\n")
        for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*")):
            w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" % (tuple(vec) + (axis,)))
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
                    w.write(
                        "    group_velocity: [ %13.7f, %13.7f, %13.7f ]\n"
                        % tuple(self._group_velocities[i, j])
                    )

                if self._with_eigenvectors:
                    w.write("    eigenvector:\n")
                    for k in range(self._natom):
                        w.write("    - # atom %d\n" % (k + 1))
                        for ll in (0, 1, 2):
                            w.write(
                                "      - [ %17.14f, %17.14f ]\n"
                                % (
                                    self._eigenvectors[i][k * 3 + ll, j].real,
                                    self._eigenvectors[i][k * 3 + ll, j].imag,
                                )
                            )
            w.write("\n")

    def _run(self):
        if self._gv_obj is not None:
            self._gv_obj.run(self._qpoints, perturbation=self._nac_q_direction)
            self._group_velocities = self._gv_obj.group_velocities

        if self._with_dynamical_matrices:
            dynamical_matrices = []

        num_band = self._natom * 3
        num_qpoints = len(self._qpoints)
        self._frequencies = np.zeros((num_qpoints, num_band), dtype="double")
        self._eigenvalues = np.zeros((num_qpoints, num_band), dtype="double")
        dynmat = run_dynamical_matrix_solver_c(
            self._dynamical_matrix, self._qpoints, self._nac_q_direction
        )
        eigenvectors = dynmat

        for i, _q in enumerate(self._qpoints):
            dm = dynmat[i]
            if self._with_dynamical_matrices:
                dynamical_matrices.append(dm)
            if self._with_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(dm)  # type: ignore
                eigenvectors[i] = eigvecs
            else:
                eigvals = np.linalg.eigvalsh(dm)  # type: ignore
            eigvals = eigvals.real
            self._eigenvalues[i] = eigvals
            self._frequencies[i] = (
                np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor
            )

        if self._with_eigenvectors:
            self._eigenvectors = eigenvectors

        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        if self._with_dynamical_matrices:
            self._dynamical_matrices = np.array(
                dynamical_matrices, dtype=dtype, order="C"
            )

    def _get_dynamical_matrix(self, q):
        if (
            isinstance(self._dynamical_matrix, DynamicalMatrixNAC)
            and self._nac_q_direction is not None
            and (np.abs(q) < 1e-5).all()
        ):
            self._dynamical_matrix.run(q, q_direction=self._nac_q_direction)
        else:
            self._dynamical_matrix.run(q)
        return self._dynamical_matrix.dynamical_matrix
