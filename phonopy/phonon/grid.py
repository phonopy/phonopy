# SPDX-License-Identifier: BSD-3-Clause
"""Regular grid tools."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from spglib import SpglibDataset, SpglibMagneticDataset

from phonopy._lang import log_dispatch, resolve_lang
from phonopy.structure.cells import (
    determinant,
    estimate_supercell_matrix,
    get_cell_parameters,
    get_reduced_bases,
    is_primitive_cell,
)
from phonopy.structure.symmetry import NosymDataset, get_lattice_vector_equivalence
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


def get_ir_qpoints_and_weights(
    mesh: Sequence[int] | NDArray[np.int64],
    lattice: NDArray[np.double],
    primitive_symmetry: object | None = None,
    q_mesh_shift: Sequence[float] | NDArray[np.double] | None = None,
    is_time_reversal: bool = True,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return irreducible q-points and weights backed by BZGrid.

    Replaces the legacy ``phonopy.structure.grid_points.get_qpoints`` for
    consumers that only need (qpoints, weights).  When the requested shift
    breaks point-group symmetry the BZGrid construction falls back to
    time-reversal-only ir-grid reduction with a warning, mirroring the
    behaviour of ``Mesh._MeshGrid``.

    """
    # Lazy import to avoid a circular dependency with mesh.py.
    from phonopy.phonon.mesh import _MeshGrid

    grid = _MeshGrid(
        np.array(mesh, dtype="int64"),
        lattice,
        primitive_symmetry,  # type: ignore[arg-type]
        q_mesh_shift=q_mesh_shift,
        is_gamma_center=is_gamma_center,
        is_time_reversal=is_time_reversal,
        is_mesh_symmetry=is_mesh_symmetry,
        lang=lang,
    )
    return grid.qpoints, grid.weights


@dataclasses.dataclass(frozen=True)
class GridSymmetryDataset:
    """Symmetry dataset for grid generation."""

    rotations: NDArray[np.int64]
    translations: NDArray[np.double]
    transformation_matrix: NDArray[np.double]
    std_lattice: NDArray[np.double]
    std_types: NDArray[np.int64]
    hall_number: int


class BZGrid:
    """Data structure of BZ grid of primitive cell.

    Note when using SNF
    -------------------
    The grid lattice is generated against the conventional unit cell when using
    SNF. To make the grid lattice of the input cell commensurate with this generated
    grid lattice, the input cell is assumed to a primitive cell. When the input
    cell is not a primitive cell, it falls back to non-SNF grid generation.

    GR-grid and BZ-grid
    -------------------
    GR-grid address is defined by three integers of {0 <= m_i < D_diag[i]}.
    Therefore number of unique grid points represented by GR-grid is
    prod(D_diag).

    BZ-grid address is defined on GR-grid but to be closest to the origin
    in Cartesian coordinates of the reciprocal space in the periodic boundary
    condition of the reciprocal lattice. The translationally equivalent
    grid points on BZ surface can be equidistant from the origin.
    In this case, those all grid addresses are contained in the data structure
    of BZGrid. Therefore number of unique grid points represented by BZ-grid
    can be larger than prod(D_diag).

    The grid systems with (BZ-grid, BZG) and without (GR-grid, GRG) BZ surface
    are mutually related up to modulo D_diag. More precisely the conversion
    of grid addresses are performed as follows:

    From BZG to GRG
        gr_gp = get_grid_point_from_address(bz_grid.addresses[bz_gp], D_diag)
    and the shortcut is
        gr_gp = bz_grid.bzg2grg[bz_gp]

    From GRG to BZG
    When store_dense_gp_map=True,
        bz_gp = bz_grid.gp_map[gr_gp]
    When store_dense_gp_map=False,
        bz_gp = gr_gp
    The shortcut is
        bz_gp = bz_grid.grg2bzg[gr_gp]
    When translationally equivalent points exist on BZ surface, the one of them
    is chosen.

    Recovering reduced coordinates
    ------------------------------
    q-points with respect to the original reciprocal
    basis vectors are given by

    q = np.dot(Q, addresses[gp] / D_diag.astype('double'))

    for the Gamma centred grid. With shifted, where only half grid shifts
    that satisfy the symmetry are considered,

    q = np.dot(Q, (addresses[gp] + np.dot(P, s)) / D_diag.astype('double'))

    where s is the shift vectors that are 0 or 1/2. But it is more
    convenient to use the integer shift vectors S by 0 or 1, which gives

    q = (np.dot(Q, (2 * addresses[gp] + PS) / D_diag.astype('double') / 2))

    where PS = np.dot(P, s) * 2.

    Attributes
    ----------
    addresses : ndarray
    gp_map : ndarray
    bzg2grg : ndarray
    grg2bzg : ndarray
    store_dense_gp_map : bool, optional
    rotations : ndarray
    reciprocal_operations : ndarray
    D_diag : ndarray
    P : ndarray
    Q : ndarray
    PS : ndarray
    QDinv : ndarray
    grid_matrix : ndarray
    microzone_lattice : ndarray
    gp_Gamma : int

    """

    def __init__(
        self,
        mesh: float
        | np.number
        | Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray[np.int64],
        reciprocal_lattice: Sequence[Sequence[float]]
        | NDArray[np.double]
        | None = None,
        lattice: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
        symmetry_dataset: SpglibDataset
        | SpglibMagneticDataset
        | NosymDataset
        | None = None,
        transformation_matrix: Sequence[Sequence[float]]
        | NDArray[np.double]
        | None = None,
        is_shift: NDArray[np.int64] | Sequence[int] | None = None,
        is_time_reversal: bool = True,
        use_grg: bool = False,
        force_SNF: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
        store_dense_gp_map: bool = True,
        lang: Literal["C", "Rust"] = "Rust",
    ):
        """Init method.

        mesh : array_like or float
            Mesh numbers or length. shape=(3,), dtype='int64'
        reciprocal_lattice : array_like
            Reciprocal primitive basis vectors given as column vectors shape=(3,
            3), dtype='double', order='C'
        lattice : array_like
            Direct primitive basis vectors given as row vectors shape=(3, 3),
            dtype='double', order='C'
        symmetry_dataset : SpglibDataset, SpglibMagneticDataset, NosymDataset, optional
            Symmetry dataset (Symmetry.dataset) searched for the primitive cell
            corresponding to ``reciprocal_lattice`` or ``lattice``.
        transformation_matrix : array_like, optional
            Transformation matrix equivalent to ``transformation_matrix`` in
            spglib-dataset. This is only used when ``use_grg=True`` and
            ``symmetry_dataset`` is unspecified. Default is None.
        is_shift : array_like or None, optional
            [0, 0, 0] (or [False, False, False]) gives Gamma center mesh and
            value 1 (or True) gives half mesh shift along the basis vectors.
            Default is None. dtype='int64', shape=(3,)
        is_time_reveral : bool, optional
            Inversion symmetry is included in reciprocal point group. Default is
            True.
        use_grg : bool, optional
            Use generalized regular grid. Default is False. ``symmetry_dataset``
            or ``transformation_matrix`` have to be specified when
            ``use_grg=True``.
        force_SNF : bool, optional
            Enforce Smith normal form even when grid lattice of GR-grid is the
            same as the traditional grid lattice. Default is False.
        SNF_coordinates : str, optional
            `reciprocal` or `direct`. Space of coordinates to generate grid
            generating matrix either in direct or reciprocal space. The default
            is `reciprocal`.
        store_dense_gp_map : bool, optional
            See the detail in the docstring of `_relocate_BZ_grid_address`.
            Default is True.
        lang : {"C", "Rust"}, optional
            Backend selector for grid-related native routines. "C" uses
            ``phonopy._recgrid``; "Rust" uses ``phonors``. Default is
            "C".

        """
        lang = resolve_lang(lang)
        log_dispatch(lang, "BZGrid.__init__")

        self._lang: Literal["C", "Rust"] = lang
        if is_shift is None:
            self._is_shift = None
        else:
            self._is_shift = [v * 1 for v in is_shift]
        self._is_time_reversal = is_time_reversal
        self._store_dense_gp_map = store_dense_gp_map
        self._addresses: NDArray[np.int64]
        self._gp_map: NDArray[np.int64]
        self._grid_matrix = None
        self._D_diag = np.ones(3, dtype="int64")
        self._Q = np.eye(3, dtype="int64", order="C")
        self._P = np.eye(3, dtype="int64", order="C")
        self._QDinv: NDArray[np.double]
        self._microzone_lattice: NDArray[np.double]
        self._rotations: NDArray[np.int64]
        self._reciprocal_operations: NDArray[np.int64]
        self._rotations_cartesian: NDArray[np.double]
        self._gp_Gamma: int
        self._bzg2grg: NDArray[np.int64]
        self._grg2bzg: NDArray[np.int64]

        if reciprocal_lattice is not None:
            self._reciprocal_lattice = np.array(
                reciprocal_lattice, dtype="double", order="C"
            )
            self._lattice = np.array(
                np.linalg.inv(reciprocal_lattice), dtype="double", order="C"
            )
        elif lattice is not None:
            self._lattice = np.array(lattice, dtype="double", order="C")
            self._reciprocal_lattice = np.array(
                np.linalg.inv(lattice), dtype="double", order="C"
            )
        else:
            raise ValueError("Either reciprocal_lattice or lattice has to be given.")

        gm = GridMatrix(
            mesh,
            self._lattice,
            symmetry_dataset=symmetry_dataset,
            transformation_matrix=transformation_matrix,
            use_grg=use_grg,
            force_SNF=force_SNF,
            SNF_coordinates=SNF_coordinates,
            lang=lang,
        )
        self._symmetry_dataset = gm.grid_symmetry_dataset
        self._grid_matrix = gm.grid_matrix
        self._D_diag = gm.D_diag
        self._P = gm.P
        self._Q = gm.Q
        self._set_bz_grid()
        self._set_rotations()

    @property
    def D_diag(self) -> NDArray[np.int64]:
        """Diagonal elements of diagonal matrix after SNF: D=PAQ.

        This corresponds to the mesh numbers in transformed reciprocal
        basis vectors.
        shape=(3,), dtype='int64'

        """
        return self._D_diag

    @property
    def P(self) -> NDArray[np.int64]:
        """Left unimodular matrix after SNF: D=PAQ.

        Left unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._P

    @property
    def Q(self) -> NDArray[np.int64]:
        """Right unimodular matrix after SNF: D=PAQ.

        Right unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._Q

    @property
    def QDinv(self) -> NDArray[np.double]:
        """QD^-1.

        ndarray :
            shape=(3, 3), dtype='double', order='C'.

        """
        return self._QDinv

    @property
    def PS(self) -> NDArray[np.int64]:
        """Integer shift vectors of GRGrid."""
        if self._is_shift is None:
            return np.zeros(3, dtype="int64")
        else:
            return np.array(np.dot(self.P, self._is_shift), dtype="int64")

    @property
    def grid_matrix(self) -> NDArray[np.int64] | None:
        """Grid generating matrix to be represented by SNF.

        Grid generating matrix used for SNF.
        When SNF is used, ndarray, otherwise None.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._grid_matrix

    @property
    def addresses(self) -> NDArray[np.int64]:
        """BZ-grid addresses.

        Integer grid address of the points in Brillouin zone including
        surface. There are two types of address order by either
        `store_dense_gp_map` is True or False.
        shape=(np.prod(D_diag) + some on surface, 3), dtype='int64', order='C'.

        """
        return self._addresses

    @property
    def gp_map(self) -> NDArray[np.int64]:
        """Definitions of grid index.

        Grid point mapping table containing BZ surface. There are two types of
        address order by either `store_dense_gp_map` is True or False. See more
        detail in `_relocate_BZ_grid_address` docstring.

        """
        return self._gp_map

    @property
    def gp_Gamma(self) -> int:
        """Return grid point index of Gamma-point."""
        return self._gp_Gamma

    @property
    def bzg2grg(self) -> NDArray[np.int64]:
        """Transform grid point indices from BZG to GRG.

        Grid index mapping table from BZGrid to GRgrid.
        shape=(len(addresses), ), dtype='int64'.

        Equivalent to
            get_grid_point_from_address(
                self._addresses[bz_grid_index], self._D_diag)

        """
        return self._bzg2grg

    @property
    def grg2bzg(self) -> NDArray[np.int64]:
        """Transform grid point indices from GRG to BZG.

        Grid index mapping table from GRgrid to BZGrid. Unique one
        of translationally equivalent grid points in BZGrid is chosen.
        shape=(prod(D_diag), ), dtype='int64'.

        """
        return self._grg2bzg

    @property
    def microzone_lattice(self) -> NDArray[np.double]:
        """Basis vectors of microzone.

        Basis vectors of microzone of GR-grid in column vectors.
        shape=(3, 3), dtype='double', order='C'.

        """
        return self._microzone_lattice

    @property
    def reciprocal_lattice(self) -> NDArray[np.double]:
        """Reciprocal basis vectors of primitive cell.

        Reciprocal basis vectors of primitive cell in column vectors.
        shape=(3, 3), dtype='double', order='C'.

        """
        return self._reciprocal_lattice

    @property
    def store_dense_gp_map(self) -> bool:
        """Return gp_map type.

        See the detail in the docstring of `_relocate_BZ_grid_address`.

        """
        return self._store_dense_gp_map

    @property
    def rotations(self) -> NDArray[np.int64]:
        """Return rotation matrices for grid points.

        Rotation matrices for GR-grid addresses (g) defined as g'=Rg. This can
        be different from ``reciprocal_operations`` when GR-grid is used because
        grid addresses are defined on an oblique lattice.
        shape=(rotations, 3, 3), dtype='int64', order='C'.

        """
        return self._rotations

    @property
    def rotations_cartesian(self) -> NDArray[np.double]:
        """Return rotations in Cartesian coordinates."""
        return self._rotations_cartesian

    @property
    def reciprocal_operations(self) -> NDArray[np.int64]:
        """Return reciprocal rotations.

        Reciprocal space rotation matrices in fractional coordinates defined as
        q'=Rq.
        shape=(rotations, 3, 3), dtype='int64', order='C'.

        """
        return self._reciprocal_operations

    @property
    def grid_symmetry_dataset(
        self,
    ) -> GridSymmetryDataset:
        """Return minimum symmetry dataset used in this class."""
        return self._symmetry_dataset

    @property
    def lang(self) -> Literal["C", "Rust"]:
        """Return backend selector for grid-related native routines."""
        return self._lang

    def get_indices_from_addresses(
        self, addresses: NDArray[np.int64]
    ) -> int | NDArray[np.int64]:
        """Return BZ grid point indices from grid addresses.

        Parameters
        ----------
        addresses : ndarray
            Integer grid addresses.
            shape=(n, 3) or (3, ), where n is the number of grid points.

        Returns
        -------
        ndarray or int
            Grid point indices corresponding to the grid addresses. Each
            returned grid point index is one of those of the
            translationally equivalent grid points.
            shape=(n, ), dtype='int64' when multiple addresses are given.
            Otherwise one integer value is returned.

        """
        try:
            len(addresses[0])
        except TypeError:
            return int(
                self._grg2bzg[
                    get_grid_point_from_address(
                        addresses, self._D_diag, lang=self._lang
                    )
                ]
            )

        gps = [
            get_grid_point_from_address(adrs, self._D_diag, lang=self._lang)
            for adrs in addresses
        ]
        return np.array(self._grg2bzg[gps], dtype="int64")

    def _set_bz_grid(self) -> None:
        """Generate BZ grid addresses and grid point mapping table."""
        self._addresses, self._gp_map, self._bzg2grg = _relocate_BZ_grid_address(
            self._D_diag,
            self._Q,
            self._reciprocal_lattice,  # column vectors
            PS=self.PS,
            store_dense_gp_map=self._store_dense_gp_map,
            lang=self._lang,
        )
        if self._store_dense_gp_map:
            self._grg2bzg = np.array(self._gp_map[:-1], dtype="int64")
        else:
            self._grg2bzg = np.arange(np.prod(self._D_diag), dtype="int64")

        self._QDinv = np.array(
            self.Q * (1 / self.D_diag.astype("double")), dtype="double", order="C"
        )
        self._microzone_lattice = np.dot(
            self._reciprocal_lattice, np.dot(self._QDinv, self._P)
        )
        self._gp_Gamma = int(
            self._grg2bzg[
                get_grid_point_from_address([0, 0, 0], self._D_diag, lang=self._lang)
            ]
        )

    def _set_rotations(self) -> None:
        """Rotation matrices are transformed those for non-diagonal D matrix.

        Terminate when symmetry of grid is broken.

        """
        direct_rotations = np.ascontiguousarray(
            self._symmetry_dataset.rotations, dtype="int64"
        )
        if self._lang == "Rust":
            import phonors

            rec_ops = phonors.reciprocal_rotations(
                direct_rotations, bool(self._is_time_reversal)
            )
        else:
            import phonopy._recgrid as recgrid  # type: ignore[import-untyped]

            rec_ops_buf = np.zeros((48, 3, 3), dtype="int64", order="C")
            num_rec_rot = recgrid.reciprocal_rotations(
                rec_ops_buf, direct_rotations, int(self._is_time_reversal)
            )
            rec_ops = rec_ops_buf[:num_rec_rot]

        self._reciprocal_operations = np.ascontiguousarray(rec_ops, dtype="int64")
        self._rotations = self._get_GRG_rotations()
        self._rotations_cartesian = np.array(
            [
                similarity_transformation(self._reciprocal_lattice, r)
                for r in self._reciprocal_operations
            ],
            dtype="double",
            order="C",
        )
        if self._is_shift is not None:
            _check_grid_shift_symmetry(self._is_shift, self._rotations, self._P)

    def _get_GRG_rotations(self) -> NDArray[np.int64]:
        """Return rotation matrices in GR-grid."""
        rots = np.ascontiguousarray(self._reciprocal_operations, dtype="int64")
        d_diag = np.ascontiguousarray(self._D_diag, dtype="int64")
        q = np.ascontiguousarray(self._Q, dtype="int64")
        if self._lang == "Rust":
            import phonors

            return np.ascontiguousarray(
                phonors.transform_rotations(rots, d_diag, q), dtype="int64"
            )

        import phonopy._recgrid as recgrid  # type: ignore[import-untyped]

        rotations = np.zeros(rots.shape, dtype="int64", order="C")
        if not recgrid.transform_rotations(rotations, rots, d_diag, q):
            raise RuntimeError("Grid symmetry is broken. Use generalized regular grid.")
        return rotations


class GridMatrix:
    """Class to generate regular grid in reciprocal space.

    Attributes
    ----------
    D_diag : ndarray
    P : ndarray
    Q : ndarray
    grid_matrix : ndarray

    """

    def __init__(
        self,
        mesh: float
        | np.number
        | Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray[np.int64],
        lattice: Sequence[Sequence[float]] | NDArray[np.double],
        symmetry_dataset: SpglibDataset
        | SpglibMagneticDataset
        | NosymDataset
        | None = None,
        transformation_matrix: Sequence[Sequence[float]]
        | NDArray[np.double]
        | None = None,
        use_grg: bool = True,
        force_SNF: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        mesh : array_like or float
            Mesh numbers or length. With float number, either conventional or
            generalized regular grid is computed depending on the given flags
            (`use_grg`, `force_SNF`). Given ndarry with
                shape=(3,), dtype='int64': conventional regular grid shape=(3,
                3), dtype='int64': generalized regular grid
        lattice : array_like
            Primitive basis vectors in direct space given as row vectors.
            shape=(3, 3), dtype='double', order='C'
        symmetry_dataset : SpglibDataset, SpglibMagneticDataset, NosymDataset, optional
            Symmetry dataset of spglib (Symmetry.dataset) of primitive cell that
            has `lattice`. Default is None.
        transformation_matrix : array_like, optional
            Transformation matrix equivalent to ``transformation_matrix`` in
            spglib-dataset. This is only used when ``use_grg=True`` and
            ``symmetry_dataset`` is unspecified. Default is None.
        use_grg : bool, optional
            Use generalized regular grid. Default is False.
        force_SNF : bool, optional
            Enforce Smith normal form even when grid lattice of GR-grid is the
            same as the traditional grid lattice. Default is False.
        SNF_coordinates : str, optional
            `reciprocal` or `direct`. Space of coordinates to generate grid
            generating matrix either in direct or reciprocal space. The default
            is `reciprocal`.
        lang : {"C", "Rust"}, optional
            Backend selector for SNF. "C" uses ``phonopy._recgrid``; "Rust"
            uses ``phonors``. Default is "C".

        """
        log_dispatch(lang, "GridMatrix.__init__")

        self._lang: Literal["C", "Rust"] = lang
        self._mesh = mesh
        self._lattice = np.asarray(lattice, dtype="double", order="C")
        self._grid_matrix: NDArray[np.int64] | None = None
        self._D_diag = np.ones(3, dtype="int64")
        self._Q = np.eye(3, dtype="int64", order="C")
        self._P = np.eye(3, dtype="int64", order="C")

        _transformation_matrix = (
            np.asarray(transformation_matrix, dtype="double", order="C")
            if transformation_matrix is not None
            else None
        )
        self._mock_symmetry_dataset = get_mock_symmetry_dataset(
            lattice=self._lattice,
            transformation_matrix=_transformation_matrix,
            symmetry_dataset=symmetry_dataset,
        )
        self._set_mesh_numbers(
            mesh,
            use_grg=use_grg,
            force_SNF=force_SNF,
            coordinates=SNF_coordinates,
        )

    @property
    def grid_matrix(self) -> NDArray[np.int64] | None:
        """Grid generating matrix to be represented by SNF.

        Grid generating matrix used for SNF.
        When SNF is used, ndarray, otherwise None.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._grid_matrix

    @property
    def D_diag(self) -> NDArray[np.int64]:
        """Diagonal elements of diagonal matrix after SNF: D=PAQ.

        This corresponds to the mesh numbers in transformed reciprocal
        basis vectors.
        shape=(3,), dtype='int64'

        """
        return self._D_diag

    @property
    def P(self) -> NDArray[np.int64]:
        """Left unimodular matrix after SNF: D=PAQ.

        Left unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._P

    @property
    def Q(self) -> NDArray[np.int64]:
        """Right unimodular matrix after SNF: D=PAQ.

        Right unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._Q

    @property
    def grid_symmetry_dataset(
        self,
    ) -> GridSymmetryDataset:
        """Return symmetry dataset."""
        return self._mock_symmetry_dataset

    def _set_mesh_numbers(
        self,
        mesh: float
        | np.number
        | Sequence[int]
        | Sequence[Sequence[int]]
        | NDArray[np.int64],
        use_grg: bool = False,
        force_SNF: bool = False,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ) -> None:
        """Set mesh numbers from array or float value.

        self._grid_matrix and self._D_diag can be set.

        Four cases:
        1) Three integers are given.
            Use these numbers as regular grid.
        2) One number is given with no symmetry provided.
            Regular grid is computed from this value. Grid is generated so that
            distances in reciprocal space between neighboring grid points become
            similar.
        3) One number is given and use_grg=False.
            Regular grid is computed from this value and point group symmetry.
            Grid is generated so that distances in reciprocal space between
            neighboring grid points become similar.
        4) One number is given with symmetry provided and use_grg=True.
            Generalized regular grid is generated. However if the grid
            generating matrix is a diagonal matrix, use it as the D matrix
            of SNF and P and Q are set as identity matrices. Otherwise
            D, P, Q matrices are computed using SNF. Grid is generated so that
            basis vectors of supercell in direct space corresponding to this grid
            have similar lengths.

        """
        if isinstance(mesh, (int, float, np.number)):
            length = float(mesh)
            if use_grg:
                found_grg = self._run_grg(
                    length,
                    None,
                    force_SNF,
                    coordinates,
                )
            if not use_grg or not found_grg:
                mesh_numbers = length2mesh(
                    length,
                    self._lattice,
                    rotations=self._mock_symmetry_dataset.rotations,  # type: ignore[arg-type]
                )
                self._D_diag = np.array(mesh_numbers, dtype="int64")
        else:
            _mesh = np.asarray(mesh, dtype="int64", order="C")
            if _mesh.shape == (3, 3):
                self._run_grg(
                    None,
                    _mesh,
                    force_SNF,
                    coordinates,
                )
            if _mesh.shape == (3,):
                self._D_diag = np.array(mesh, dtype="int64")

        _check_grid_symmetry(
            self._mock_symmetry_dataset.rotations, self._D_diag, self._Q
        )

    def _run_grg(
        self,
        length: float | None,
        grid_matrix: NDArray[np.int64] | None,
        force_SNF: bool,
        coordinates: Literal["reciprocal", "direct"],
    ) -> bool:
        if is_primitive_cell(self._mock_symmetry_dataset.rotations):
            # self._D_diag or self._grid_matrix is set in this method.
            self._set_GRG_mesh(
                length=length,
                grid_matrix=grid_matrix,
                force_SNF=force_SNF,
                coordinates=coordinates,
            )
            return True

        warnings.warn(
            "Non primitive cell input. Unable to use GR-grid.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    def _set_GRG_mesh(
        self,
        length: float | None = None,
        grid_matrix: NDArray[np.int64] | None = None,
        force_SNF: bool = False,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ) -> None:
        """Set grid_matrix or D_diag with generalized regular grid.

        Microzone is defined as the regular grid of a conventional
        unit cell. To find the conventional unit cell, symmetry
        information is used.

        """
        if length is not None:
            _grid_matrix = self._get_grid_matrix(length, coordinates=coordinates)
        elif grid_matrix is not None:
            _grid_matrix = np.array(grid_matrix, dtype="int64", order="C")

        # If grid_matrix is a diagonal matrix, use it as D matrix.
        gm_diag = np.diagonal(_grid_matrix)
        if (np.diag(gm_diag) == _grid_matrix).all() and not force_SNF:
            self._D_diag = np.array(gm_diag, dtype="int64")
        else:
            A = np.ascontiguousarray(_grid_matrix, dtype="int64")
            if self._lang == "Rust":
                import phonors

                try:
                    d_diag, p, q = phonors.snf3x3(A)
                except (ValueError, RuntimeError) as exc:
                    msg = "SNF3x3 failed."
                    raise RuntimeError(msg) from exc

                self._D_diag[:] = d_diag
                self._P[:] = p
                self._Q[:] = q
            else:
                from phonopy import _recgrid as recgrid

                D_diag = np.zeros(3, dtype="int64")
                P = np.zeros((3, 3), dtype="int64", order="C")
                Q = np.zeros((3, 3), dtype="int64", order="C")
                if not recgrid.snf3x3(D_diag, P, Q, A.copy()):
                    msg = "SNF3x3 failed."
                    raise RuntimeError(msg)
                self._D_diag[:] = D_diag
                self._P[:] = P
                self._Q[:] = Q
            self._grid_matrix = _grid_matrix  # type: ignore[assignment]

    def _get_grid_matrix(
        self,
        length: float,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ) -> NDArray[np.int64]:
        """Return grid matrix.

        Grid is generated by the distance `length`. `coordinates` is used either
        the grid is defined by supercell in real space or mesh grid in reciprocal
        space.

        Note
        ----
        It is assumed that self._lattice is a primitive cell basis vectors.

        Parameters
        ----------
        length : float
            Distance measure to generate grid.
        coordinates : str, optional
            `reciprocal` (default) or `direct`.

        """
        tmat = self._mock_symmetry_dataset.transformation_matrix
        conv_lat = np.dot(np.linalg.inv(tmat).T, self._lattice)

        # GRG is wanted to be generated with respect to std_lattice if possible.
        if _can_use_std_lattice(
            conv_lat,
            tmat,
            self._mock_symmetry_dataset.std_lattice,
            self._mock_symmetry_dataset.rotations,
        ):
            conv_lat = self._mock_symmetry_dataset.std_lattice
            tmat = np.dot(self._lattice, np.linalg.inv(conv_lat)).T

        if coordinates == "direct":
            num_cells = int(np.prod(length2mesh(length, conv_lat)))
            max_num_atoms = num_cells * len(self._mock_symmetry_dataset.std_types)
            conv_mesh_numbers = estimate_supercell_matrix(
                cast(SpglibDataset, self._mock_symmetry_dataset),
                max_num_atoms=max_num_atoms,
                max_iter=200,
            )
        elif coordinates == "reciprocal":
            conv_mesh_numbers = length2mesh(length, conv_lat)  # type: ignore[assignment]
        else:
            raise TypeError('Expect "direct" or "reciprocal" for coordinates.')

        inv_tmat = np.linalg.inv(tmat)
        inv_tmat_int = np.rint(inv_tmat).astype(int)
        assert (np.abs(inv_tmat - inv_tmat_int) < 1e-5).all()
        grid_matrix = np.array(
            (inv_tmat_int * conv_mesh_numbers).T, dtype="int64", order="C"
        )
        return grid_matrix


def _check_grid_symmetry(
    direct_rotations: NDArray[np.int64] | Sequence,
    D_diag: NDArray[np.int64] | Sequence[int],
    Q: NDArray[np.int64] | Sequence[int],
) -> NDArray[np.int64]:
    """Check whether grid symmetry is satisfied.

    Return rotation matrices for test.

    """
    QDinv = np.array(Q) * (1 / np.array(D_diag, dtype="double"))
    rotations = []
    for r in direct_rotations:
        _r = np.linalg.inv(np.transpose(r) @ QDinv) @ QDinv
        rotations.append(np.rint(_r))
        if not np.allclose(_r, np.rint(_r), atol=1e-5):
            msg = "Grid symmetry is broken."
            raise RuntimeError(msg)
    return np.array(rotations, dtype="int64", order="C")


def _check_grid_shift_symmetry(
    is_shift: NDArray[np.int64] | Sequence[int],
    grg_rotations: NDArray[np.int64] | Sequence,
    P: NDArray[np.int64] | Sequence[int],
) -> None:
    """Check whether given shift satisfies the symmetry."""
    Pinv = np.rint(np.linalg.inv(P)).astype(int)
    assert determinant(Pinv) == 1
    S = np.array(is_shift, dtype=int)
    for r in grg_rotations:
        _S = np.dot(np.dot(Pinv, np.dot(r, P)), S)
        if not np.array_equal((S - _S) % 2, [0, 0, 0]):
            msg = "Grid symmetry is broken by grid shift."
            raise RuntimeError(msg)


def get_qpoints_from_bz_grid_points(
    gps: int | NDArray[np.int64], bz_grid: BZGrid
) -> NDArray[np.double]:
    """Return q-point(s) in reduced coordinates of grid point(s).

    Parameters
    ----------
    i_gps : int or ndarray
        BZ-grid index (int) or indices (ndarray).
    bz_grid : BZGrid
        BZ-grid instance.

    """
    return bz_grid.addresses[gps] @ bz_grid.QDinv.T


def get_grid_point_from_address_py(
    addresses: Sequence[int] | NDArray[np.int64],
    D_diag: NDArray[np.int64] | Sequence[int],
) -> NDArray[np.int64]:
    """Return GR-grid point index from addresses.

    Python version of get_grid_point_from_address.
    X runs first in XYZ
    In grid.c, Z first is possible with MACRO setting.

    addresses :
        shape=(..., 3)

    """
    return np.dot(np.mod(addresses, D_diag), [1, D_diag[0], D_diag[0] * D_diag[1]])


def get_grid_point_from_address(
    address: Sequence[int] | NDArray[np.int64],
    D_diag: Sequence[int] | NDArray[np.int64],
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Return GR grid-point indices of grid addresses.

    Parameters
    ----------
    address : array_like
        Grid address.
        shape=(3, ) or (n, 3), dtype='int64'
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int64'
    lang : {"C", "Rust"}
        Backend selector. "C" uses ``phonopy._recgrid``; "Rust" uses
        ``phonors``. Default is "C".

    Returns
    -------
    int
        GR-grid point index.
    or

    ndarray
        GR-grid point indices.
        shape=(n, ), dtype='int64'

    """
    if resolve_lang(lang) == "Rust":
        import phonors as backend
    else:
        import phonopy._recgrid as backend  # type: ignore[import-untyped,no-redef]

    adrs_array = np.ascontiguousarray(address, dtype="int64")
    mesh_array = np.ascontiguousarray(D_diag, dtype="int64")

    if adrs_array.ndim == 1:
        return backend.grid_index_from_address(adrs_array, mesh_array)

    gps = np.zeros(adrs_array.shape[0], dtype="int64")
    for i, adrs in enumerate(adrs_array):
        gps[i] = backend.grid_index_from_address(np.ascontiguousarray(adrs), mesh_array)
    return gps


def get_ir_grid_points(
    bz_grid: BZGrid,
    lang: Literal["C", "Rust"] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return ir-grid-points in generalized regular grid.

    bz_grid : BZGrid
        Data structure to represent BZ grid.
    lang : {"C", "Rust"} or None, optional
        Backend selector passed to ``_get_ir_grid_map``. If None,
        ``bz_grid.lang`` is used.

    Returns
    -------
    ir_grid_points : ndarray
        Irreducible grid point indices in GR-grid.
        shape=(num_ir_grid_points, ), dtype='int64'
    ir_grid_weights : ndarray
        Weights of irreducible grid points. Its sum is the number of
        grid points in GR-grid (prod(D_diag)).
        shape=(num_ir_grid_points, ), dtype='int64'
    ir_grid_map : ndarray
        Index mapping table to irreducible grid points from all grid points
        such as, [0, 0, 2, 3, 3, ...] in GR-grid.
        shape=(prod(D_diag), ), dtype='int64'

    """
    _lang = bz_grid.lang if lang is None else lang
    ir_grid_map = _get_ir_grid_map(
        bz_grid.D_diag, bz_grid.rotations, PS=bz_grid.PS, lang=_lang
    )
    ir_grid_points, ir_grid_weights = extract_ir_grid_points(ir_grid_map)

    return ir_grid_points, ir_grid_weights, ir_grid_map


def get_grid_points_by_rotations(
    bz_gp: int,
    bz_grid: BZGrid,
    reciprocal_rotations: NDArray[np.int64] | None = None,
    with_surface: bool = False,
    lang: Literal["C", "Rust"] | None = None,
) -> NDArray[np.int64]:
    """Return BZ-grid point indices rotated from a BZ-grid point index.

    Parameters
    ----------
    bz_gp : int
        BZ-grid point index.
    bz_grid : BZGrid
        Data structure to represent BZ grid.
    reciprocal_rotations : array_like or None, optional
        Rotation matrices {R} with respect to basis vectors of GR-grid.
        Defined by g'=Rg, where g is the grid point address represented by
        three integers in BZ-grid.
        dtype='int64', shape=(rotations, 3, 3)
    with_surface : Bool, optional
        This parameter affects to how to treat grid points on BZ surface.
        When False, rotated BZ surface points are moved to representative
        ones among translationally equivalent points to hold one-to-one
        correspondence to GR grid points. With True, BZ grid point indices
        having the rotated grid addresses are returned. Default is False.

    Returns
    -------
    rot_grid_indices : ndarray
        BZ-grid point indices obtained after rotating a grid point index.
        dtype='int64', shape=(rotations,)

    """
    _lang = bz_grid.lang if lang is None else lang

    if reciprocal_rotations is not None:
        rec_rots = reciprocal_rotations
    else:
        rec_rots = bz_grid.rotations

    if with_surface:
        return _get_grid_points_by_bz_rotations(bz_gp, bz_grid, rec_rots, lang=_lang)
    else:
        return _get_grid_points_by_rotations(bz_gp, bz_grid, rec_rots, lang=_lang)


def _get_grid_points_by_rotations(
    bz_gp: int,
    bz_grid: BZGrid,
    rotations: NDArray[np.int64],
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray:
    """Grid point rotations without surface treatment."""
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag, lang=lang)
    return bz_grid.grg2bzg[grgps]


def _get_grid_points_by_bz_rotations(
    bz_gp: int,
    bz_grid: BZGrid,
    rotations: NDArray[np.int64],
    lang: Literal["C", "Python", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Grid point rotations with surface treatment.

    ``lang="C"`` dispatches to the C extension backend,
    ``lang="Rust"`` to ``phonors``, and ``lang="Python"`` uses the
    pure-Python reference implementation.

    """
    if lang == "Python":
        return _get_grid_points_by_bz_rotations_py(bz_gp, bz_grid, rotations)
    return _get_grid_points_by_bz_rotations_c(bz_gp, bz_grid, rotations, lang=lang)


def _get_grid_points_by_bz_rotations_c(
    bz_gp: int,
    bz_grid: BZGrid,
    rotations: NDArray,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    if resolve_lang(lang) == "Rust":
        import phonors as backend
    else:
        import phonopy._recgrid as backend  # type: ignore[import-untyped,no-redef]

    addresses = np.ascontiguousarray(bz_grid.addresses, dtype="int64")
    gp_map = np.ascontiguousarray(bz_grid.gp_map, dtype="int64")
    d_diag = np.ascontiguousarray(bz_grid.D_diag, dtype="int64")
    ps = np.ascontiguousarray(bz_grid.PS, dtype="int64")
    bz_grid_type = int(bz_grid.store_dense_gp_map) + 1
    bzgps = np.zeros(len(rotations), dtype="int64")
    for i, r in enumerate(rotations):
        bzgps[i] = backend.rotate_bz_grid_index(
            int(bz_gp),
            np.ascontiguousarray(r, dtype="int64"),
            addresses,
            gp_map,
            d_diag,
            ps,
            bz_grid_type,
        )
    return bzgps


def _get_grid_points_by_bz_rotations_py(
    bz_gp: int, bz_grid: BZGrid, rotations: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return BZ-grid point indices generated by rotations.

    Rotated BZ-grid addresses are compared with translationally
    equivalent BZ-grid addresses to get the respective BZ-grid point
    indices.

    """
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag)
    bzgps = np.zeros(len(grgps), dtype="int64")
    if bz_grid.store_dense_gp_map:
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs, strict=True)):
            indices = np.where(
                (
                    bz_grid.addresses[bz_grid.gp_map[gp] : bz_grid.gp_map[gp + 1]]
                    == adrs
                ).all(axis=1)
            )[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = bz_grid.gp_map[gp] + indices[0]
    else:
        num_grgp = np.prod(bz_grid.D_diag)
        num_bzgp = num_grgp * 8
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs, strict=True)):
            gps = (
                np.arange(
                    bz_grid.gp_map[num_bzgp + gp],  # type: ignore
                    bz_grid.gp_map[num_bzgp + gp + 1],  # type: ignore
                )
                + num_grgp
            ).tolist()
            assert isinstance(gps, list)
            gps.insert(0, gp)
            indices = np.where((bz_grid.addresses[gps] == adrs).all(axis=1))[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = gps[indices[0]]

    return bzgps


def _get_grid_address(
    D_diag: NDArray[np.int64] | Sequence[int],
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Return generalized regular grid addresses.

    Parameters
    ----------
    D_diag : array_like
        Three integers that represent the generalized regular grid.
        shape=(3, ), dtype='int64'
    lang : {"C", "Rust"}
        Backend selector. Default is "C".

    Returns
    -------
    gr_grid_addresses : ndarray
        Integer triplets that represents grid point addresses in
        generalized regular grid.
        shape=(prod(D_diag), 3), dtype='int64'

    """
    d_diag = np.ascontiguousarray(D_diag, dtype="int64")
    if resolve_lang(lang) == "Rust":
        import phonors

        return np.asarray(phonors.gr_grid_addresses(d_diag), dtype="int64")

    import phonopy._recgrid as recgrid  # type: ignore[import-untyped]

    gr_grid_addresses = np.zeros((int(np.prod(d_diag)), 3), dtype="int64", order="C")
    recgrid.gr_grid_addresses(gr_grid_addresses, d_diag)
    return gr_grid_addresses


def _relocate_BZ_grid_address(
    D_diag: NDArray[np.int64] | Sequence[int],
    Q: NDArray[np.int64],
    reciprocal_lattice: NDArray[np.double],  # column vectors
    PS: NDArray[np.int64] | None = None,
    store_dense_gp_map: bool = False,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Grid addresses are relocated to be inside first Brillouin zone.

    Number of ir-grid-points inside Brillouin zone is returned.
    It is assumed that the following arrays have the shapes of
        bz_grid_address : (num_grid_points_in_FBZ, 3)

    Note that the shape of grid_address is (prod(mesh), 3) and the
    addresses in grid_address are arranged to be in parallelepiped
    made of reciprocal basis vectors. The addresses in bz_grid_address
    are inside the first Brillouin zone or on its surface. Each
    address in grid_address is mapped to one of those in
    bz_grid_address by a reciprocal lattice vector (including zero
    vector) with keeping element order. For those inside first BZ, the
    mapping is one-to-one. For those on the first BZ surface, more
    than one addresses in bz_grid_address that are equivalent by the
    reciprocal lattice translations are mapped to one address in
    grid_address. The bz_grid_address and bz_map are given in the
    following format depending on the choice of `store_dense_gp_map`.

    store_dense_gp_map = False
    --------------------------
    Those grid points on the BZ surface except for one of them are
    appended to the tail of this array, for which bz_grid_address has
    the following data storing:

    |------------------array size of bz_grid_address-------------------------|
    |--those equivalent to grid_address--|--those on surface except for one--|
    |-----array size of grid_address-----|

    Number of grid points stored in bz_grid_address is returned.
    bz_map is used to recover grid point index expanded to include BZ
    surface from grid address. The grid point indices are mapped to
    (mesh[0] * 2) x (mesh[1] * 2) x (mesh[2] * 2) space (bz_map).
    bz_map[(prod(mesh) * 8):(prod(mesh) * 9 + 1)] contains equivalent
    information to bz_map[:] with `store_dense_gp_map=True`.

    shape=(prod(mesh * 9) + 1, )

    store_dense_gp_map = True
    -------------------------
    The translationally equivalent grid points corresponding to one grid point
    on BZ surface are stored in continuously. If the multiplicity (number of
    equivalent grid points) is 1, 2, 1, 4, ... for the grid points,
    ``bz_map`` stores the multiplicities and the index positions of the first
    grid point of the equivalent grid points, i.e.,

    bz_map[:] = [0, 1, 3, 4, 8...]
    grid_address[0] -> bz_grid_address[0:1]
    grid_address[1] -> bz_grid_address[1:3]
    grid_address[2] -> bz_grid_address[3:4]
    grid_address[3] -> bz_grid_address[4:8]

    shape=(prod(mesh) + 1, )

    """
    if PS is None:
        ps = np.zeros(3, dtype="int64")
    else:
        ps = np.ascontiguousarray(PS, dtype="int64")

    reduced_basis, tmat_inv_int = get_reduced_bases_and_tmat_inv(reciprocal_lattice)
    q_eff = np.ascontiguousarray(tmat_inv_int @ Q, dtype="int64")
    d_diag = np.ascontiguousarray(D_diag, dtype="int64")
    rec = np.ascontiguousarray(reduced_basis, dtype="float64")
    bz_grid_type = int(store_dense_gp_map) + 1

    if resolve_lang(lang) == "Rust":
        import phonors

        addresses, bz_map, bzg2grg = phonors.bz_grid_addresses(
            d_diag, q_eff, ps, rec, bz_grid_type
        )
        bz_grid_addresses = np.ascontiguousarray(addresses, dtype="int64")
        bz_map = np.asarray(bz_map, dtype="int64")
        bzg2grg = np.asarray(bzg2grg, dtype="int64")
        return bz_grid_addresses, bz_map, bzg2grg

    import phonopy._recgrid as recgrid  # type: ignore[import-untyped]

    num_grg = int(np.prod(d_diag))
    bz_grid_addresses = np.zeros((num_grg * 8, 3), dtype="int64", order="C")
    bzg2grg = np.zeros(len(bz_grid_addresses), dtype="int64")
    if store_dense_gp_map:
        bz_map = np.zeros(num_grg + 1, dtype="int64")
    else:
        bz_map = np.zeros(num_grg * 9 + 1, dtype="int64")
    num_gp = recgrid.bz_grid_addresses(
        bz_grid_addresses,
        bz_map,
        bzg2grg,
        d_diag,
        q_eff,
        ps,
        rec,
        bz_grid_type,
    )
    bz_grid_addresses = np.ascontiguousarray(bz_grid_addresses[:num_gp], dtype="int64")
    bzg2grg = np.asarray(bzg2grg[:num_gp], dtype="int64")
    return bz_grid_addresses, bz_map, bzg2grg


def get_reduced_bases_and_tmat_inv(
    reciprocal_lattice: NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return reduced bases and inverse transformation matrix.

    Parameters
    ----------
    reciprocal_lattice : NDArray[np.double]
        Reciprocal lattice vectors in column vectors.
        shape=(3, 3), dtype='double'

    Returns
    -------
    reduced_basis : ndarray
        Reduced basis vectors in column vectors.
        shape=(3, 3), dtype='double', order='C'
    tmat_inv_int : ndarray
        Inverse transformation matrix in integer.
        This is used to transform reciprocal lattice vectors to
        conventional lattice vectors.
        shape=(3, 3), dtype='int64'

    """
    # Mpr^-1 = Lr^-1 Lp
    reclat_T = np.array(np.transpose(reciprocal_lattice), dtype="double", order="C")
    reduced_basis = get_reduced_bases(reclat_T)
    assert reduced_basis is not None, "Reduced basis is not found."
    tmat_inv = np.linalg.inv(reduced_basis.T) @ reclat_T.T
    tmat_inv_int = np.asarray(
        np.rint(tmat_inv).astype("int64"), dtype="int64", order="C"
    )
    assert (np.abs(tmat_inv - tmat_inv_int) < 1e-5).all()
    return np.array(reduced_basis.T, dtype="double", order="C"), tmat_inv_int


def _get_ir_grid_map(
    D_diag: NDArray[np.int64] | Sequence[int],
    grg_rotations: NDArray[np.int64],
    PS: NDArray[np.int64] | None = None,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Return mapping to irreducible grid points in GR-grid.

    Parameters
    ----------
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int64'
    grg_rotations : array_like
        GR-grid rotation matrices.
        dtype='int64', shape=(grg_rotations, 3)
    PS : array_like
        GR-grid shift defined.
        dtype='int64', shape=(3,)
    lang : {"C", "Rust"}
        Backend selector. Default is "C".

    Returns
    -------
    ir_grid_map : ndarray
        Grid point mapping from all indices to ir-gird-point indices in GR-grid.
        dtype='int64', shape=(prod(mesh),)

    """
    d_diag = np.ascontiguousarray(D_diag, dtype="int64")
    if PS is None:
        ps = np.zeros(3, dtype="int64")
    else:
        ps = np.ascontiguousarray(PS, dtype="int64")
    rots = np.ascontiguousarray(grg_rotations, dtype="int64")

    if resolve_lang(lang) == "Rust":
        import phonors

        map_vec, num_ir = phonors.ir_grid_map(rots, d_diag, ps)
        if num_ir > 0:
            return np.asarray(map_vec, dtype="int64")
        raise RuntimeError("_get_ir_grid_map failed to find ir-grid-points.")

    import phonopy._recgrid as recgrid  # type: ignore[import-untyped]

    ir_grid_map = np.zeros(int(np.prod(d_diag)), dtype="int64")
    num_ir = recgrid.ir_grid_map(ir_grid_map, d_diag, ps, rots)
    if num_ir > 0:
        return ir_grid_map
    raise RuntimeError("_get_ir_grid_map failed to find ir-grid-points.")


def _can_use_std_lattice(
    conv_lat: NDArray[np.double],
    tmat: NDArray[np.double],
    std_lattice: NDArray[np.double],
    rotations: NDArray[np.int64] | Sequence,
    symprec: float = 1e-5,
) -> bool:
    """Inspect if std_lattice can be used as conv_lat.

    r_s is the rotation matrix of conv_lat.
    Return if conv_lat rotated by det(r_s)*r_s and std_lattice are equivalent.
    det(r_s) is necessary to make improper rotation to proper rotation.

    """
    for r in rotations:
        r_s = similarity_transformation(tmat, r)  # type: ignore[arg-type]
        if np.allclose(
            np.linalg.det(r_s) * np.dot(np.transpose(conv_lat), r_s),
            np.transpose(std_lattice),
            atol=symprec,
        ):
            return True
    return False


def get_mock_symmetry_dataset(
    lattice: NDArray[np.double] | None = None,
    transformation_matrix: NDArray[np.double] | None = None,
    symmetry_dataset: SpglibDataset
    | SpglibMagneticDataset
    | NosymDataset
    | None = None,
) -> GridSymmetryDataset:
    """Return mock symmetry dataset."""
    if symmetry_dataset is None or isinstance(symmetry_dataset, NosymDataset):
        _tmat = (
            transformation_matrix
            if transformation_matrix is not None
            else np.eye(3, dtype="double", order="C")
        )
        _lattice = (
            lattice if lattice is not None else np.eye(3, dtype="double", order="C")
        )
        return _get_mock_symmetry_dataset_nosym(_lattice, _tmat)

    assert symmetry_dataset is not None
    sym_dataset = GridSymmetryDataset(
        rotations=np.asarray(symmetry_dataset.rotations, dtype="int64", order="C"),
        translations=np.asarray(
            symmetry_dataset.translations, dtype="double", order="C"
        ),
        transformation_matrix=np.asarray(
            symmetry_dataset.transformation_matrix, dtype="double", order="C"
        ),
        std_lattice=np.asarray(symmetry_dataset.std_lattice, dtype="double", order="C"),
        std_types=np.asarray(symmetry_dataset.std_types, dtype="int64"),
        hall_number=symmetry_dataset.hall_number,
    )
    return sym_dataset


def _get_mock_symmetry_dataset_nosym(
    lattice: NDArray[np.double], transformation_matrix: NDArray[np.double]
) -> GridSymmetryDataset:
    """Return mock symmetry_dataset containing transformation matrix.

    Assuming self._lattice as standardized cell, and inverse of
    transformation_matrix indicates original primitive lattice with respect
    to self._lattice.

    """
    tmat_inv = np.linalg.inv(transformation_matrix)
    tmat_inv_int = np.rint(tmat_inv).astype(int)
    if (tmat_inv - tmat_inv_int > 1e-8).all():
        msg = "Inverse of transformation matrix has to be an integer matrix."
        raise RuntimeError(msg)
    if determinant(tmat_inv_int) < 0:
        msg = "Determinant of transformation matrix has to be positive."
        raise RuntimeError(msg)
    if determinant(tmat_inv_int) < 1:
        msg = (
            "Determinant of inverse of transformation matrix has to "
            "be equal to or larger than 1."
        )
        raise RuntimeError(msg)

    sym_dataset = GridSymmetryDataset(
        rotations=np.eye(3, dtype="int64", order="C").reshape(1, 3, 3),
        translations=np.zeros((1, 3), dtype="double", order="C"),
        transformation_matrix=transformation_matrix,
        std_lattice=np.array(lattice, dtype="double", order="C"),
        std_types=np.array([1], dtype="int64"),
        hall_number=1,
    )
    return sym_dataset
