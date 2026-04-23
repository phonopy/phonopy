"""Conventinal and generalized regular grid routines."""

# Copyright (C) 2014 Atsushi Togo
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

from collections.abc import Sequence

import numpy as np
import spglib

from phonopy.structure.atoms import PhonopyAtoms

try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass

from numpy.typing import NDArray

from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.structure.cells import (
    determinant,
    estimate_supercell_matrix,
    estimate_supercell_matrix_from_pointgroup,
    get_cell_parameters,
    get_primitive_matrix_by_centring,
)
from phonopy.structure.snf import SNF3x3
from phonopy.structure.symmetry import (
    NosymDataset,
    Symmetry,
    collect_unique_rotations,
    get_lattice_vector_equivalence,
    get_pointgroup_operations,
)
from phonopy.utils import similarity_transformation


def length2mesh(
    length: float,
    lattice: Sequence[Sequence[float]] | NDArray[np.double],
    rotations: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Convert length to mesh for q-point sampling.

    This conversion for each reciprocal axis follows VASP convention by
        N = max(1, int(l * |a|^* + 0.5))
    'int' means rounding down, not rounding to nearest integer.

    Parameters
    ----------
    length : float
        Length having the unit of direct space length.
    lattice : array_like
        Basis vectors of primitive cell in row vectors.
        dtype='double', shape=(3, 3)
    rotations: array_like, optional
        Rotation matrices in real space. When given, mesh numbers that are
        symmetrically reasonable are returned. Default is None.
        dtype='int64', shape=(rotations, 3, 3)

    Returns
    -------
    array_like
        dtype=int, shape=(3,)

    """
    rec_lattice = np.array(np.linalg.inv(lattice), dtype="double")
    rec_lat_lengths = get_cell_parameters(rec_lattice.T)
    mesh_numbers = np.rint(rec_lat_lengths * length).astype(int)

    if rotations is not None:
        reclat_equiv = get_lattice_vector_equivalence(
            np.array([r.T for r in rotations], dtype="int64")
        )
        m = mesh_numbers
        mesh_equiv = [m[1] == m[2], m[2] == m[0], m[0] == m[1]]
        # Follow symmetry when distorted, and align to larger one.
        for i, pair in enumerate(([1, 2], [2, 0], [0, 1])):
            if reclat_equiv[i] and not mesh_equiv[i]:
                m[pair] = max(m[pair])

    return np.maximum(mesh_numbers, [1, 1, 1])


def get_qpoints(
    mesh_numbers: Sequence[int] | NDArray[np.int64],
    reciprocal_lattice: Sequence[Sequence[float]]
    | NDArray[np.double],  # column vectors
    q_mesh_shift: Sequence[float] | NDArray[np.double] | None = None,
    is_gamma_center: bool = True,
    is_time_reversal: bool = True,
    fit_in_BZ: bool = True,
    rotations: Sequence[Sequence[Sequence[int]]]
    | NDArray[np.int64]
    | None = None,  # Point group operations in real space
    is_mesh_symmetry: bool = True,
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return q-points and weights on a mesh sampling grid.

    See attributes of GriePoints, qpoints and wegiths.

    """
    gp = GridPoints(
        mesh_numbers,
        reciprocal_lattice,
        q_mesh_shift=q_mesh_shift,
        is_gamma_center=is_gamma_center,
        is_time_reversal=is_time_reversal,
        fit_in_BZ=fit_in_BZ,
        rotations=rotations,
        is_mesh_symmetry=is_mesh_symmetry,
    )

    return gp.qpoints, gp.weights


def extract_ir_grid_points(
    grid_mapping_table: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ir-grid points and weights in grid index mapping table."""
    ir_grid_points = np.array(np.unique(grid_mapping_table), dtype="int64")
    weights = np.zeros_like(grid_mapping_table)
    for gp in grid_mapping_table:
        weights[gp] += 1
    ir_weights = np.array(weights[ir_grid_points], dtype="int64")

    return ir_grid_points, ir_weights


class GridPoints:
    """Class to generate irreducible grid points on uniform mesh grids.

    Attributes
    ----------
    mesh_numbers: ndarray
        Mesh numbers along a, b, c axes.
        dtype='int64'
        shape=(3,)
    reciprocal_lattice: ndarray
        Basis vectors in reciprocal space. a*, b*, c* are given in column
        vectors.
        dtype='double'
        shape=(3, 3)
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
        Indices of irreducible grid points in grid_address.
        dtype='int64', shape=(ir-grid points,)
    grid_mapping_table: ndarray
        Index mapping table from all grid points to ir-grid points.
        dtype='int64', shape=(prod(mesh_numbers),)

    """

    def __init__(
        self,
        mesh_numbers: Sequence[int] | NDArray[np.int64],
        reciprocal_lattice: Sequence[Sequence[float]] | NDArray[np.double],
        q_mesh_shift: Sequence[float] | NDArray[np.double] | None = None,
        is_gamma_center: bool = True,
        is_time_reversal: bool = True,
        fit_in_BZ: bool = True,
        rotations: Sequence[Sequence[Sequence[int]]]
        | Sequence[NDArray[np.int64]]
        | NDArray[np.int64]
        | None = None,
        is_mesh_symmetry: bool = True,
    ) -> None:  # Except for time reversal symmetry
        """Init method.

        Note
        ----
        Uniform mesh grids are made according to Monkhorst-Pack scheme, i.e.,
        for odd (even) numbers, centre are (are not) sampled. The Gamma-centre
        sampling is supported by ``is_gamma_center=True``.

        Parameters
        ----------
        mesh_numbers: array_like
            Mesh numbers along a, b, c axes.
            dtype='int64'
            shape=(3, )
        reciprocal_lattice: array_like
            Basis vectors in reciprocal space. a*, b*, c* are given in column
            vectors.
            dtype='double'
            shape=(3, 3)
        q_mesh_shift: array_like, optional, default None (no shift)
            Mesh shifts along a*, b*, c* axes with respect to neighboring grid
            points from the original mesh (Monkhorst-Pack or Gamma center).
            0.5 gives half grid shift. Normally 0 or 0.5 is given.
            Otherwise q-points symmetry search is not performed.
            dtype='double'
            shape=(3, )
        is_gamma_center: bool, default False
            Uniform mesh grids are generated centring at Gamma point but not
            the Monkhorst-Pack scheme.
        is_time_reversal: bool, optional, default True
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included.
        fit_in_BZ: bool, optional, default True
        rotations: array_like, default None (only unitary operation)
            Rotation matrices in direct space. For each rotation matrix R,
            a point in crystallographic coordinates, x, is sent as x' = Rx.
            dtype='int64'
            shape=(rotations, 3, 3)
        is_mesh_symmetry: bool, optional, default True
            Whether symmetry search is done or not.

        """
        self._mesh = np.array(mesh_numbers, dtype="int64")
        self._rec_lat = np.array(reciprocal_lattice, dtype="double", order="C")
        self._is_shift = self._shift2boolean(
            q_mesh_shift, is_gamma_center=is_gamma_center
        )
        self._is_time_reversal = is_time_reversal
        self._fit_in_BZ = fit_in_BZ
        self._rotations = (
            np.array(rotations, dtype="int64") if rotations is not None else None
        )
        self._is_mesh_symmetry = is_mesh_symmetry

        self._ir_qpoints: NDArray[np.double]
        self._grid_address: NDArray[np.int64]
        self._ir_grid_points: NDArray[np.int64]
        self._ir_weights: NDArray[np.int64]
        self._grid_mapping_table: NDArray[np.int64]

        if self._is_shift is None:
            self._is_mesh_symmetry = False
            self._is_shift = self._shift2boolean(None)
            self._set_grid_points()
            self._ir_qpoints += q_mesh_shift / self._mesh
            self._fit_qpoints_in_BZ()
        else:  # zero or half shift
            self._set_grid_points()

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return mesh numbers."""
        return self._mesh

    @property
    def reciprocal_lattice(self) -> NDArray[np.double]:
        """Return reciprocal lattice in column vectors."""
        return self._rec_lat

    @property
    def grid_address(self) -> NDArray[np.int64]:
        """Return all grid point addresses."""
        return self._grid_address

    @property
    def ir_grid_points(self) -> NDArray[np.int64]:
        """Return ir-grid point indices."""
        return self._ir_grid_points

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return irreducible q-points."""
        return self._ir_qpoints

    @property
    def weights(self) -> NDArray[np.int64]:
        """Return weights of ir-grid points."""
        return self._ir_weights

    @property
    def grid_mapping_table(self) -> NDArray[np.int64]:
        """Return grid index mapping table."""
        return self._grid_mapping_table

    def _set_grid_points(self) -> None:
        if self._is_mesh_symmetry and self._has_mesh_symmetry():
            assert self._rotations is not None
            self._set_ir_qpoints(
                self._rotations, is_time_reversal=self._is_time_reversal
            )
        else:
            self._set_ir_qpoints(
                [np.eye(3, dtype="int64")], is_time_reversal=self._is_time_reversal
            )

    def _shift2boolean(
        self,
        q_mesh_shift: Sequence[float] | NDArray[np.double] | None,
        is_gamma_center: bool = False,
        tolerance: float = 1e-5,
    ) -> list | None:
        """Return bools of with or without half-shifts.

        Parameters
        ----------
        tolerance : float
            This is used to judge zero/half grid shift.

        """
        if q_mesh_shift is None:
            shift = np.zeros(3, dtype="double")
        else:
            shift = np.array(q_mesh_shift, dtype="double")

        diffby2 = np.abs(shift * 2 - np.rint(shift * 2))

        if (diffby2 < 0.01).all():  # zero or half shift
            diff = np.abs(shift - np.rint(shift))
            if is_gamma_center:
                is_shift = list(diff > 0.1)
            else:  # Monkhorst-pack
                is_shift = list(np.logical_xor((diff > 0.1), (self._mesh % 2 == 0)) * 1)
        else:
            is_shift = None

        return is_shift

    def _has_mesh_symmetry(self) -> bool:
        if self._rotations is None:
            return False
        m = self._mesh
        mesh_equiv = [m[1] == m[2], m[2] == m[0], m[0] == m[1]]
        lattice_equiv = get_lattice_vector_equivalence(
            np.array([r.T for r in self._rotations], dtype="int64", order="C")
        )
        return bool(np.extract(lattice_equiv, mesh_equiv).all())

    def _fit_qpoints_in_BZ(self) -> None:
        qpolongset_in_BZ = get_qpoints_in_Brillouin_zone(
            self._rec_lat, self._ir_qpoints
        )
        qpoints_in_BZ = np.array(
            [q_set[0] for q_set in qpolongset_in_BZ], dtype="double", order="C"
        )
        self._ir_qpoints = qpoints_in_BZ

    def _set_ir_qpoints(
        self,
        rotations: NDArray[np.int64] | list[NDArray[np.int64]],
        is_time_reversal: bool = True,
    ) -> None:
        grid_mapping_table, grid_address = spglib.get_stabilized_reciprocal_mesh(
            self._mesh,
            rotations,
            is_shift=self._is_shift,
            is_time_reversal=is_time_reversal,
            is_dense=True,
        )  # type: ignore
        # uintp to long
        grid_mapping_table = np.array(grid_mapping_table, dtype="int64")

        if self._fit_in_BZ:
            grid_address, _ = spglib.relocate_BZ_grid_address(
                grid_address,
                self._mesh,
                self._rec_lat,
                is_shift=self._is_shift,
                is_dense=True,
            )  # type: ignore
            self._grid_address = np.array(
                grid_address[: np.prod(self._mesh)], dtype="int64", order="C"
            )
        else:
            self._grid_address = np.array(grid_address, dtype="int64", order="C")

        self._ir_grid_points, self._ir_weights = extract_ir_grid_points(
            grid_mapping_table
        )

        shift = np.array(self._is_shift) * 0.5
        self._ir_qpoints = np.array(
            (self._grid_address[self._ir_grid_points] + shift) / self._mesh,
            dtype="double",
            order="C",
        )

        self._grid_mapping_table = grid_mapping_table


class GeneralizedRegularGridPoints:
    """Generalized regular grid points.

    Method strategy in suggest mode
    -------------------------------
    1. Create conventional unit cell using spglib.
    2. Sample regular grid points for the conventional unit cell (mesh_numbers)
    3. Transformation matrix from primitive to conventinal unit cell (inv_pmat)
    4. Get supercell multiplicities (mesh_numbers) from the conventional unit
       cell considering the lattice shape.
    5. mmat = (inv_pmat * mesh_numbers).T, which is related to the
       transformation from primitive cell to supercell.
    6. D = P.mmat.Q, where D = diag([n1, n2, n3])
    7. Grid points for primitive cell are
       [np.dot(Q, g) for g in ndindex((n1, n2, n3))].

    Method strategy in non-suggest mode
    -----------------------------------
    1. Find symmetry operations
    2. Determine point group and transformation matrix (tmat) from input cell
    3. Get supercell multiplicities (mesh_numbers) from the transformed cell
       considering the lattice shape.
    4. mmat = (tmat * mesh_numbers).T
    5. D = P.mmat.Q, where D = diag([n1, n2, n3])
    6. Grid points for primitive cell are
       [np.dot(Q, g) for g in ndindex((n1, n2, n3))].

    Attributes
    ----------
    grid_address : ndarray
        Grid addresses in integers.
        shape=(num_grid_points, 3), dtype='int64', order='C'
    qpoints : ndarray
        q-points with respect to basis vectors of input or standardized
        primitive cell.
        shape=(num_grid_points, 3), dtype='double', order='C'
    grid_matrix : ndarray
        Grid generating matrix.
        shape=(3,3), dtype='int64', order='C'
    matrix_to_primitive : ndarray or None
        None when ``suggest`` is False. Otherwise, transformation matrix from
        input cell to the suggested primitive cell.
        shape=(3,3), dtype='double', order='C'
    snf : SNF3x3
        SNF3x3 instance of grid generating matrix.

    """

    def __init__(
        self,
        cell: PhonopyAtoms,
        length: float,
        suggest: bool = True,
        is_time_reversal: bool = True,
        x_fastest: bool = True,
        symprec: float = 1e-5,
    ) -> None:
        """Init method.

        Parameters
        ----------
        cell : PhonopyAtoms
            Input cell.
        length : float
            Length having the unit of direct space length.
        suggest : bool, optional, default True
            With True, a standardized primitive cell is suggested and the grids
            are generated for it. With False, input cell is used.
        is_time_reversal: bool, optional, default True
            Time reversal symmetry is considered in symmetry search. By this,
            inversion symmetry is always included.
        x_fastest : bool, optional, default=True
            In grid generation, [[x, y, z], ...], x runs fastest when True,
            otherwise z runs fastest.

        """
        self._cell = cell
        self._length = length
        self._suggest = suggest
        self._is_time_reversal = is_time_reversal
        self._x_fastest = x_fastest
        self._grid_address: NDArray[np.int64]
        self._snf: SNF3x3
        self._transformation_matrix: NDArray[np.double]
        self._grid_matrix: NDArray[np.int64]
        self._reciprocal_operations: NDArray[np.int64]
        self._symmetry: Symmetry
        self._prepare(cell, length, symprec)
        self._generate_grid_points()
        self._generate_q_points()
        self._reciprocal_operations = get_reciprocal_operations(
            self._symmetry.symmetry_operations["rotations"],
            self._transformation_matrix,
            self._snf.D,
            self._snf.Q,
            is_time_reversal=self._is_time_reversal,
        )

    @property
    def grid_address(self) -> NDArray[np.int64]:
        """Return all grid point addresses."""
        return self._grid_address

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return all q-points."""
        return self._qpoints

    @property
    def grid_matrix(self) -> NDArray[np.int64]:
        """Return grid generating matrix."""
        return self._grid_matrix

    @property
    def transformation_matrix(self) -> NDArray[np.double]:
        """Return transformation matrix."""
        return self._transformation_matrix

    @property
    def snf(self) -> SNF3x3:
        """Return SNF3x3 instance of grid generating matrix."""
        return self._snf

    @property
    def mesh_numbers(self) -> list[int]:
        """Return mesh numbers for reciprocal conventional unit cell."""
        return self._mesh_numbers

    @property
    def reciprocal_operations(self) -> NDArray[np.int64]:
        """Return reciprocal point group operations."""
        return self._reciprocal_operations

    def _prepare(self, cell: PhonopyAtoms, length: float, symprec: float) -> None:
        """Define grid generating matrix and run the SNF."""
        # fsg_cell = cell.copy()
        # magmoms = fsg_cell.magnetic_moments
        # if magmoms is not None:
        #     fsg_cell.magnetic_moments = np.abs(magmoms)
        self._symmetry = Symmetry(cell, symprec=symprec)
        if self._suggest:
            self._set_grid_matrix_by_std_primitive_cell(cell, length)
        else:
            self._set_grid_matrix_by_input_cell(cell, length)
        self._snf = SNF3x3(self._grid_matrix)

    def _set_grid_matrix_by_std_primitive_cell(
        self, cell: PhonopyAtoms, length: float
    ) -> None:
        """Grid generating matrix based on standardized primitive cell."""
        assert not isinstance(self._symmetry.dataset, NosymDataset)
        tmat = self._symmetry.dataset.transformation_matrix
        spg_type = spglib.get_spacegroup_type(self._symmetry.dataset.hall_number)
        assert spg_type is not None
        centring = spg_type.international[0]
        pmat = get_primitive_matrix_by_centring(centring)
        conv_lat = np.dot(np.linalg.inv(tmat).T, cell.cell)
        num_cells = np.prod(length2mesh(length, conv_lat))
        self._mesh_numbers = estimate_supercell_matrix(
            self._symmetry.dataset,
            max_num_atoms=num_cells * len(self._symmetry.dataset.std_types),  # type: ignore
        )
        inv_pmat = np.linalg.inv(pmat)  # type: ignore
        inv_pmat_int = np.rint(inv_pmat).astype(int)
        assert (np.abs(inv_pmat - inv_pmat_int) < 1e-5).all()
        # transpose in reciprocal space
        self._grid_matrix = np.array(
            (inv_pmat_int * self._mesh_numbers).T, dtype="int64", order="C"
        )
        # From input lattice to the primitive lattice in real space
        self._transformation_matrix = np.array(
            np.dot(np.linalg.inv(tmat), pmat),  # type: ignore
            dtype="double",
            order="C",
        )

    def _set_grid_matrix_by_input_cell(
        self, input_cell: PhonopyAtoms, length: float
    ) -> None:
        """Grid generating matrix based on input cell."""
        pointgroup = spglib.get_pointgroup(
            self._symmetry.symmetry_operations["rotations"]
        )
        # tmat: From input lattice to point group preserving lattice
        tmat = pointgroup[2]  # type: ignore
        lattice = np.dot(input_cell.cell.T, tmat).T
        num_cells = np.prod(length2mesh(length, lattice))
        self._mesh_numbers = estimate_supercell_matrix_from_pointgroup(
            pointgroup[1],  # type: ignore
            lattice,
            max_num_cells=num_cells,  # type: ignore
        )
        # transpose in reciprocal space
        self._grid_matrix = np.array(
            np.multiply(tmat, self._mesh_numbers).T, dtype="int64", order="C"
        )
        self._transformation_matrix = np.eye(3, dtype="double", order="C")

    def _generate_grid_points(self) -> None:
        d = np.diagonal(self._snf.D)
        if self._x_fastest:
            # x runs fastest.
            z, y, x = np.meshgrid(range(d[2]), range(d[1]), range(d[0]), indexing="ij")
        else:
            # z runs fastest.
            x, y, z = np.meshgrid(range(d[0]), range(d[1]), range(d[2]), indexing="ij")
        self._grid_address = np.array(
            np.c_[x.ravel(), y.ravel(), z.ravel()], dtype="int64", order="C"
        )

    def _generate_q_points(self) -> None:
        D_inv = np.linalg.inv(self._snf.D)
        qpoints = np.dot(self._grid_address, np.dot(self._snf.Q, D_inv).T)
        qpoints -= np.rint(qpoints)
        self._qpoints = qpoints


def get_reciprocal_operations(
    rotations: NDArray[np.int64],
    transformation_matrix: NDArray[np.double],
    D: NDArray[np.int64],
    Q: NDArray[np.int64],
    is_time_reversal: bool = True,
) -> NDArray[np.int64]:
    """Generate reciprocal rotation matrices.

    Collect unique real space rotation matrices and transpose them.
    When is_time_reversal=True, inversion is added if it is not in the
    list of the rotation matrices.

    Parameters
    ----------
    rotations : ndarray
        Rotation matrices in real space. x' = Rx.
        shape=(rotations, 3, 3), dtype='int64'
    transformation_matrxi : array_like
        Transformation matrix of basis vectors in real space. Using this
        rotation matrices are transformed.
    D : array_like
        D of smith normal form 3x3.
        shape=(3, 3)
    Q : array_like
        Q of smith normal form 3x3.
        shape=(3, 3)
    is_time_reversal : bool
        When True, inversion operation is added.

    Returns
    -------
    rotations_for_Q : ndarray
        Rotation matrices in reciprocal space. Grid points are sent by the
        symmetrically equivalent grid points as follows:

        g' = (R_Q g) % diagonal(D)

        shape=(rotations, 3, 3), dtype='int64', order='C'

    """
    unique_rots: list[NDArray[np.int64]] = []
    tmat_inv: NDArray[np.double] = np.linalg.inv(transformation_matrix)  # type: ignore
    for r in collect_unique_rotations(rotations):
        _r = similarity_transformation(tmat_inv, r)
        _r_int = np.array(np.rint(_r), dtype="int64")
        assert (np.abs(_r - _r_int) < 1e-5).all()
        unique_rots.append(_r_int)

    _, rec_ops = get_pointgroup_operations(
        np.array(unique_rots, dtype="int64"), is_time_reversal=is_time_reversal
    )

    Q_inv = np.linalg.inv(Q)  # type: ignore
    rec_ops_Q = []
    for r in rec_ops:
        _r = similarity_transformation(Q_inv, r)
        _r = similarity_transformation(D, _r)
        _r_int = np.rint(_r).astype(int)
        assert (np.abs(_r - _r_int) < 1e-5).all()
        assert abs(determinant(_r_int)) == 1
        rec_ops_Q.append(_r_int)

    return np.array(rec_ops_Q, dtype="int64", order="C")
