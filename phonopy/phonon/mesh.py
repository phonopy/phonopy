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

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    get_dynamical_matrices_at_qpoints,
)
from phonopy.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.physical_units import get_physical_units
from phonopy.structure.symmetry import Symmetry


class MeshSymmetryFallbackWarning(RuntimeWarning):
    """Issued when ``Mesh`` falls back to TR-only ir-grid reduction.

    BZGrid raises ``RuntimeError("Grid symmetry is broken by grid shift.")``
    when the requested half-grid shift is not preserved by the primitive
    cell's point group.  ``_MeshGrid`` catches that and rebuilds the BZGrid
    with point-group symmetry disabled (time-reversal symmetry stays on).
    Emitted as a separate subclass so CLI scripts can format the message
    nicely instead of the default ``warnings`` output.

    """


def _resolve_is_shift(
    mesh: NDArray[np.int64],
    q_mesh_shift: Sequence[float] | NDArray[np.double] | None,
    is_gamma_center: bool,
) -> list[int]:
    """Reproduce ``GridPoints._shift2boolean`` for the BZGrid path.

    Returns a 0/1 list of half-grid shift flags per axis.

    """
    if q_mesh_shift is None:
        diff = np.zeros(3, dtype="double")
    else:
        shift = np.array(q_mesh_shift, dtype="double")
        diff = np.abs(shift - np.rint(shift - 0.5))
    if is_gamma_center:
        return [int(d > 0.1) for d in diff]
    return [int(np.logical_xor(d > 0.1, mesh[i] % 2 == 0)) for i, d in enumerate(diff)]


class _MeshGrid:
    """Internal Mesh bookkeeper backed by BZGrid + get_ir_grid_points.

    Replaces ``phonopy.structure.grid_points.GridPoints`` inside ``Mesh``.
    Exposes the property surface that ``Mesh`` consumes
    (``mesh_numbers``, ``qpoints``, ``weights``, ``grid_address``,
    ``ir_grid_points``, ``grid_mapping_table``, ``is_shift``,
    ``reciprocal_lattice``).

    When the requested ``is_shift`` breaks point-group symmetry, BZGrid
    raises ``RuntimeError("Grid symmetry is broken by grid shift.")``.  We
    catch that, emit a warning, and fall back to a BZGrid built without
    point-group symmetry (time-reversal stays on).  The legacy spglib path
    silently dropped incompatible rotations and produced an ir-grid whose
    symmetry was not actually preserved; the fallback here is the explicit,
    correct behaviour.

    """

    def __init__(
        self,
        mesh: NDArray[np.int64],
        lattice: NDArray[np.double],
        primitive_symmetry: Symmetry | None,
        q_mesh_shift: Sequence[float] | NDArray[np.double] | None,
        is_gamma_center: bool,
        is_time_reversal: bool,
        is_mesh_symmetry: bool,
        lang: Literal["C", "Rust"] = "C",
    ) -> None:
        self._mesh = np.array(mesh, dtype="int64")
        self._reciprocal_lattice = np.linalg.inv(lattice)
        self._is_shift = _resolve_is_shift(self._mesh, q_mesh_shift, is_gamma_center)

        symmetry_dataset = (
            primitive_symmetry.dataset
            if (is_mesh_symmetry and primitive_symmetry is not None)
            else None
        )
        try:
            self._bzgrid = BZGrid(
                self._mesh,
                lattice=lattice,
                symmetry_dataset=symmetry_dataset,
                is_shift=self._is_shift,
                is_time_reversal=is_time_reversal,
                lang=lang,
            )
        except RuntimeError as exc:
            if "Grid symmetry is broken" in str(exc):
                warnings.warn(
                    "BZGrid construction with point-group symmetry failed for "
                    f"mesh={self._mesh.tolist()} shift={self._is_shift}: "
                    f"{exc}. Falling back to time-reversal-only ir-grid "
                    "reduction.",
                    MeshSymmetryFallbackWarning,
                    stacklevel=4,
                )
                self._bzgrid = BZGrid(
                    self._mesh,
                    lattice=lattice,
                    is_shift=self._is_shift,
                    is_time_reversal=is_time_reversal,
                    lang=lang,
                )
            else:
                raise

        ir_gp, ir_w, ir_map = get_ir_grid_points(self._bzgrid)
        self._ir_grid_points = ir_gp
        self._ir_weights = ir_w
        self._grid_mapping_table = ir_map
        # GR-grid addresses indexed by GR index.
        self._grid_address = np.ascontiguousarray(
            self._bzgrid.addresses[self._bzgrid.grg2bzg], dtype="int64"
        )
        # ir-qpoints: (grid_address + 0.5 * is_shift) / mesh_numbers
        shift_half = np.array(self._is_shift, dtype="double") * 0.5
        self._ir_qpoints = np.ascontiguousarray(
            (self._grid_address[self._ir_grid_points] + shift_half) / self._mesh,
            dtype="double",
        )

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        return self._mesh

    @property
    def reciprocal_lattice(self) -> NDArray[np.double]:
        return self._reciprocal_lattice

    @property
    def grid_address(self) -> NDArray[np.int64]:
        return self._grid_address

    @property
    def ir_grid_points(self) -> NDArray[np.int64]:
        return self._ir_grid_points

    @property
    def qpoints(self) -> NDArray[np.double]:
        return self._ir_qpoints

    @property
    def weights(self) -> NDArray[np.int64]:
        return self._ir_weights

    @property
    def grid_mapping_table(self) -> NDArray[np.int64]:
        return self._grid_mapping_table

    @property
    def is_shift(self) -> list[int]:
        return self._is_shift


class MeshDict(TypedDict):
    """Return type of Phonopy.get_mesh_dict for Mesh."""

    qpoints: NDArray[np.double]
    weights: NDArray[np.int64]
    frequencies: NDArray[np.double]
    eigenvectors: NDArray[np.cdouble] | None
    group_velocities: NDArray[np.double] | None


class IterMeshDict(TypedDict):
    """Return type of Phonopy.get_mesh_dict for IterMesh."""

    qpoints: NDArray[np.double]
    weights: NDArray[np.int64]


class MeshBase:
    """Base class of Mesh and IterMesh classes.

    Attributes
    ----------
    mesh_numbers: ndarray
        Mesh numbers along a, b, c axes.
        dtype='int64'
        shape=(3,)
    qpoints: ndarray
        q-points in reduced coordinates of reciprocal lattice
        dtype='double'
        shape=(ir-grid points, 3)
    weights: ndarray
        Geometric q-point weights. Its sum is the number of grid points.
        dtype='int64'
        shape=(ir-grid points,)
    grid_address: ndarray
        Addresses of all grid points represented by integers.
        dtype='int64'
        shape=(prod(mesh_numbers), 3)
    ir_grid_points: ndarray
        Indices of irreducibple grid points in grid_address.
        dtype='int64'
        shape=(ir-grid points,)
    grid_mapping_table: ndarray
        Index mapping table from all grid points to ir-grid points.
        dtype='int64'
        shape=(prod(mesh_numbers),)
    dynamical_matrix: DynamicalMatrix
        Dynamical matrix instance to compute dynamical matrix at q-points.

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        mesh: Sequence[int] | NDArray[np.int64],
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        is_gamma_center: bool = False,
        rotations: Sequence[Sequence[Sequence[int]]]
        | Sequence[NDArray[np.int64]]
        | NDArray[np.int64]
        | None = None,  # Point group operations in real space
        primitive_symmetry: Symmetry | None = None,
        factor: float | None = None,
        lang: Literal["C", "Rust"] | None = None,
    ) -> None:
        """Init method.

        ``primitive_symmetry`` is the ``Symmetry`` instance of the primitive
        cell.  When supplied (typically from ``Phonopy.primitive_symmetry``)
        downstream consumers such as the BZGrid-based DOS path can avoid
        recomputing it; ``rotations`` is still used for ir-grid reduction at
        construction time.  Default ``None`` keeps backward compatibility.

        ``lang`` selects the backend for the batched dynamical-matrix
        build.  When None (default) the value is inherited from
        ``dynamical_matrix.lang``; pass an explicit string to override.

        """
        self._mesh = np.array(mesh, dtype="int64")
        self._with_eigenvectors = with_eigenvectors
        if factor is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = factor
        self._cell = dynamical_matrix.primitive
        self._dynamical_matrix = dynamical_matrix
        self._primitive_symmetry = primitive_symmetry
        self._lang: Literal["C", "Rust"] = (
            lang if lang is not None else dynamical_matrix.lang
        )

        # ``rotations`` argument is now superseded by ``primitive_symmetry``
        # and ignored; kept on the signature for backward compatibility.
        _ = rotations
        self._gp = _MeshGrid(
            self._mesh,
            self._cell.cell,
            primitive_symmetry,
            q_mesh_shift=shift,
            is_gamma_center=is_gamma_center,
            is_time_reversal=(is_time_reversal and is_mesh_symmetry),
            is_mesh_symmetry=is_mesh_symmetry,
            lang=self._lang,
        )

        self._qpoints = self._gp.qpoints
        self._weights = self._gp.weights

        self._frequencies = None
        self._eigenvectors = None

        self._q_count = 0

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return mesh numbers."""
        return self._mesh

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return (irreducible) q-points."""
        return self._qpoints

    @property
    def weights(self) -> NDArray[np.int64]:
        """Return (irreducible) weights of q-points."""
        return self._weights

    @property
    def grid_address(self) -> NDArray[np.int64]:
        """Return mesh grid addresses."""
        return self._gp.grid_address

    @property
    def ir_grid_points(self) -> NDArray[np.int64]:
        """Return irreducible grid indices."""
        return self._gp.ir_grid_points

    @property
    def grid_mapping_table(self) -> NDArray[np.int64]:
        """Return grid index mapping table."""
        return self._gp.grid_mapping_table

    @property
    def is_shift(self) -> list[int] | None:
        """Return half-grid shift flags per axis (0 or 1)."""
        return self._gp.is_shift

    @property
    def primitive_symmetry(self) -> Symmetry | None:
        """Return the primitive-cell Symmetry passed at construction, if any."""
        return self._primitive_symmetry

    @property
    def dynamical_matrix(self) -> DynamicalMatrix:
        """Return dynamical matrix class instance."""
        return self._dynamical_matrix

    @property
    def with_eigenvectors(self) -> bool:
        """Whether eigenvectors are calculated or not."""
        return self._with_eigenvectors


class Mesh(MeshBase):
    """Class for phonons on mesh grid.

    Frequencies and eigenvectors can be also accessible by iterator
    representation to be compatible with IterMesh.

    Attributes
    ----------
    frequencies: ndarray
        Phonon frequencies at ir-grid points. Imaginary frequencies are
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
        dynamical_matrix: DynamicalMatrix,
        mesh: Sequence[int] | NDArray[np.int64],
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        is_gamma_center: bool = False,
        group_velocity: GroupVelocity | None = None,
        rotations: Sequence[Sequence[Sequence[int]]]
        | Sequence[NDArray[np.int64]]
        | NDArray[np.int64]
        | None = None,  # Point group operations in real space
        primitive_symmetry: Symmetry | None = None,
        factor: float | None = None,
        lang: Literal["C", "Rust"] | None = None,
    ) -> None:
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
            primitive_symmetry=primitive_symmetry,
            factor=factor,
            lang=lang,
        )

        self._group_velocity = group_velocity
        self._group_velocities: NDArray[np.double] | None = None

    def __iter__(self) -> Mesh:
        """Define iterator over q-points.

        Initially, all phonons are computed and stored in arrays.
        Then this is just used as an iterator to return existing results.
        The purpose of this iterator is compatible use of IterMesh.

        """
        if self._frequencies is None:
            self.run()
        return self

    def __next__(
        self,
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble] | None]:
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

    def run(self) -> None:
        """Calculate phonons at all required q-points."""
        self._set_phonon()
        if self._group_velocity is not None:
            self._set_group_velocities(self._group_velocity)

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies."""
        if self._frequencies is None:
            self.run()
        return self._frequencies  # type: ignore[return-value]

    @property
    def eigenvectors(self) -> NDArray[np.cdouble] | None:
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

    @property
    def group_velocities(self) -> NDArray[np.double] | None:
        """Return group velocities."""
        if self._frequencies is None:
            self.run()
        return self._group_velocities

    def write_hdf5(
        self,
        filename: str | os.PathLike = "mesh.hdf5",
        compression: Literal["gzip", "lzf"] | int | None = None,
    ) -> None:
        """Write results to hdf5 file."""
        import h5py

        with h5py.File(filename, "w") as w:
            w.create_dataset("mesh", data=self._mesh)
            w.create_dataset("qpoint", data=self._qpoints, compression=compression)
            w.create_dataset("weight", data=self._weights, compression=compression)
            w.create_dataset(
                "frequency", data=self._frequencies, compression=compression
            )
            if self._eigenvectors is not None:
                w.create_dataset(
                    "eigenvector", data=self._eigenvectors, compression=compression
                )
            if self._group_velocities is not None:
                w.create_dataset(
                    "group_velocity",
                    data=self._group_velocities,
                    compression=compression,
                )

    def write_yaml(self, filename: str | os.PathLike = "mesh.yaml") -> None:
        """Write results to yaml file."""
        natom = len(self._cell)
        rec_lattice = np.linalg.inv(self._cell.cell)  # column vectors
        distances = np.sqrt(np.sum(np.dot(self._qpoints, rec_lattice.T) ** 2, axis=1))

        lines = []

        lines.append("mesh: [ %5d, %5d, %5d ]" % tuple(self._mesh))
        lines.append("nqpoint: %-7d" % self._qpoints.shape[0])
        lines.append("reciprocal_lattice:")
        for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*"), strict=True):
            lines.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
        lines.append("natom:   %-7d" % natom)
        lines.append(str(self._cell))
        lines.append("")
        lines.append("phonon:")

        for i, (q, d) in enumerate(zip(self._qpoints, distances, strict=True)):
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

    def _set_phonon(self) -> None:
        from phonopy._lang import c_use_openmp

        num_band = len(self._cell) * 3
        num_qpoints = len(self._qpoints)

        # The batched DM build is used whenever a parallel kernel is
        # available: the C path requires OpenMP (compile-time), but the
        # Rust path is rayon-parallel without an extra runtime check.
        use_batched_build = self._lang == "Rust" or c_use_openmp()

        self._frequencies = np.zeros((num_qpoints, num_band), dtype="double")
        if use_batched_build:
            dynmat = get_dynamical_matrices_at_qpoints(
                self._dynamical_matrix,
                self._qpoints,
                lang=self._lang,
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
            if use_batched_build:
                dm = dynmat[i]
            else:
                self._dynamical_matrix.run(q)
                dm = self._dynamical_matrix.dynamical_matrix
            if self._with_eigenvectors:
                eigvals, eigenvectors[i] = np.linalg.eigh(dm)  # type: ignore
                eigenvalues = eigvals.real
            else:
                eigenvalues = np.linalg.eigvalsh(dm).real  # type: ignore
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

    def _set_group_velocities(self, group_velocity: GroupVelocity) -> None:
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
        dynamical_matrix: DynamicalMatrix,
        mesh: Sequence[int] | NDArray[np.int64],
        shift: Sequence[float] | NDArray[np.double] | None = None,
        is_time_reversal: bool = True,
        is_mesh_symmetry: bool = True,
        with_eigenvectors: bool = False,
        is_gamma_center: bool = False,
        rotations: Sequence[Sequence[Sequence[int]]]
        | Sequence[NDArray[np.int64]]
        | NDArray[np.int64]
        | None = None,  # Point group operations in real space
        primitive_symmetry: Symmetry | None = None,
        factor: float | None = None,
        lang: Literal["C", "Rust"] | None = None,
    ) -> None:
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
            primitive_symmetry=primitive_symmetry,
            factor=factor,
            lang=lang,
        )

    def __iter__(self) -> IterMesh:
        """Define iterator over q-points."""
        return self

    def __next__(
        self,
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble] | None]:
        """Calculate phonons at a q-point."""
        if self._q_count == len(self._qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            q = self._qpoints[self._q_count]
            self._dynamical_matrix.run(q)
            dm = self._dynamical_matrix.dynamical_matrix
            if self._with_eigenvectors:
                eigvals, eigenvectors = np.linalg.eigh(dm)  # type: ignore
                eigenvalues = eigvals.real
            else:
                eigenvalues = np.linalg.eigvalsh(dm).real  # type: ignore
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
