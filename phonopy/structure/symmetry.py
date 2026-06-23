"""Crystal symmetry routines."""

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

import dataclasses
from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
import spglib

try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass

from numpy.typing import NDArray
from spglib import SpglibDataset, SpglibMagneticDataset

from phonopy._lang import resolve_lang
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    Supercell,
    compute_all_sg_permutations,
    get_atom_order,
    get_primitive,
    get_supercell,
)
from phonopy.utils import similarity_transformation


class _SymmetryOperations(TypedDict):
    rotations: NDArray[np.int64]
    translations: NDArray[np.double]


@dataclasses.dataclass(eq=False, frozen=True)
class NosymDataset:
    """Symmetry dataset substitute used when symmetry analysis is disabled.

    Mimics the parts of an ``spglib`` dataset that phonopy consumes,
    populated with only the identity operation (and pure lattice
    translations of the supercell when applicable).

    """

    rotations: NDArray[np.int64]
    translations: NDArray[np.double]
    transformation_matrix: NDArray[np.double]
    pointgroup: str


class Symmetry:
    """Find and store crystal symmetry information of a cell.

    A ``Symmetry`` instance is built from a :class:`PhonopyAtoms` cell.
    It exposes the space-group and point-group operations, the
    international symbol, Wyckoff letters, atom-to-representative
    mapping, and related quantities. Backed by ``spglib`` for ordinary
    and magnetic cells, and falls back to a no-symmetry dataset when
    ``is_symmetry=False``.

    """

    def __init__(
        self,
        cell: PhonopyAtoms,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        s2p_map: NDArray[np.int64] | None = None,
        lang: Literal["C", "Rust"] = "Rust",
        distinguish_symbol_index: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        cell : PhonopyAtoms
            Crystal structure whose symmetry is analyzed.
        symprec : float, optional
            Tolerance used to find crystal symmetry. Default is 1e-5.
        is_symmetry : bool, optional
            Whether to perform symmetry analysis. When
            ``is_symmetry=False`` and ``s2p_map`` is given, pure
            translations inside ``cell`` are registered as symmetry
            operations. Default is True.
        s2p_map : ndarray, optional
            Equivalent to :attr:`phonopy.structure.cells.Primitive.s2p_map`.
        lang : {"C", "Rust"}, optional
            Backend used by helpers that have a Rust port (currently
            the atomic-permutation matcher). Default is ``"Rust"``.
        distinguish_symbol_index : bool, optional
            When True, atoms whose symbols differ only in the numeric
            suffix ("Cl" vs "Cl1") are treated as distinct species in
            the symmetry search. By default (False) the suffix is a
            calculator-facing label and such atoms are symmetry
            equivalent. Must be consistent with the primitive matrix:
            ``guess_primitive_matrix`` takes the same flag. Default is
            False.

        """
        self._cell = cell
        self._symprec = symprec
        self._distinguish_symbol_index = distinguish_symbol_index
        self._lang: Literal["C", "Rust"] = resolve_lang(lang)

        self._symmetry_operations: _SymmetryOperations
        self._international_table = None
        self._dataset: SpglibDataset | SpglibMagneticDataset | NosymDataset
        self._wyckoff_letters = None
        self._map_atoms: NDArray
        self._atomic_permutations: NDArray
        self._pointgroup_operations: NDArray
        self._reciprocal_operations: NDArray
        self._pointgroup_symbol: str
        self._independent_atoms: NDArray
        self._map_operations: NDArray

        magmom = cell.magnetic_moments

        if not is_symmetry:
            self._set_nosym(s2p_map)
        elif magmom is None:
            self._set_symmetry_dataset()
        else:
            self._set_symmetry_operations_with_magmoms()
        (
            self._pointgroup_operations,
            self._reciprocal_operations,
        ) = get_pointgroup_operations(self._symmetry_operations["rotations"])
        _ptg = spglib.get_pointgroup(self._pointgroup_operations)
        assert _ptg is not None
        ptg_symbol = _ptg[0]
        self._pointgroup_symbol = ptg_symbol.strip()
        self._set_atomic_permutations()
        self._set_independent_atoms()
        self._map_operations = self._get_map_operations_from_permutations()

    @property
    def symmetry_operations(self) -> _SymmetryOperations:
        """Return symmetry operations.

        Returns
        -------
        dict
            ``'rotations'`` : ndarray
                Matrix parts of the operations.
                ``shape=(num_operations, 3, 3)``, ``dtype='int64'``.
            ``'translations'`` : ndarray
                Vector parts of the operations.
                ``shape=(num_operations, 3)``, ``dtype='double'``.

        """
        return self._symmetry_operations

    def get_symmetry_operation(self, operation_number: int) -> _SymmetryOperations:
        """Return one symmetry operation as a ``{rotations, translations}`` dict.

        Parameters
        ----------
        operation_number : int
            Zero-based index of the operation in :attr:`symmetry_operations`.

        """
        operation = self._symmetry_operations
        return {
            "rotations": operation["rotations"][operation_number],
            "translations": operation["translations"][operation_number],
        }

    @property
    def pointgroup_operations(self) -> NDArray[np.int64]:
        """Return the crystallographic point-group operations.

        ``shape=(n_ops, 3, 3)``, ``dtype='int64'``.

        """
        return self._pointgroup_operations

    @property
    def pointgroup_symbol(self) -> str:
        """Return the Hermann-Mauguin symbol of the crystallographic point group."""
        return self._pointgroup_symbol

    def get_international_table(self) -> str | None:
        """Return the international symbol of the space group.

        Returns a string like ``"Fm-3m (225)"`` (Hermann-Mauguin symbol
        followed by the space-group number), or ``None`` when symmetry
        analysis has been disabled.

        """
        return self._international_table

    def get_Wyckoff_letters(self) -> list[str] | None:
        """Return Wyckoff letters of the atoms, or ``None`` if not available."""
        return self._wyckoff_letters

    @property
    def dataset(self) -> SpglibDataset | SpglibMagneticDataset | NosymDataset:
        """Return the raw spglib symmetry dataset.

        ``SpglibDataset`` for ordinary cells,
        ``SpglibMagneticDataset`` for cells with magnetic moments, or
        ``NosymDataset`` when ``is_symmetry=False``.

        """
        return self._dataset

    def get_independent_atoms(self) -> NDArray[np.int64]:
        """Return indices of symmetrically inequivalent atoms.

        ``shape=(n_independent,)``, ``dtype='int64'``.

        """
        return self._independent_atoms

    def get_map_atoms(self) -> NDArray[np.int64]:
        """Return ``equivalent_atoms`` of the spglib dataset.

        For each atom, returns the index of the symmetrically
        equivalent representative. For example, an 8-atom cell with two
        symmetrically distinct sites might give
        ``[0, 0, 0, 0, 4, 4, 4, 4]``.
        ``shape=(natom,)``, ``dtype='int64'``.

        """
        return self._map_atoms

    def get_map_operations(self) -> NDArray[np.int64]:
        """Return per-atom indices of operations mapping to the representative atom.

        Returns
        -------
        ndarray
            For each atom, the index (into :attr:`symmetry_operations`)
            of one symmetry operation that sends it to its equivalent
            representative atom. Only one such operation per atom is
            stored. Use :attr:`atomic_permutations` when all symmetry
            mapping information is needed.
            ``shape=(natom,)``, ``dtype='int64'``.

        """
        return self._map_operations

    def get_site_symmetry(self, atom_number: int) -> NDArray[np.int64]:
        """Return matrix parts of site-symmetry operations of one atom.

        Parameters
        ----------
        atom_number : int
            Zero-based atom index.

        Returns
        -------
        ndarray
            Rotation matrices that fix the given site (modulo lattice
            translation). ``shape=(n_site_ops, 3, 3)``,
            ``dtype='int64'``.

        """
        positions = self._cell.scaled_positions
        lattice = self._cell.cell
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]

        return self._get_site_symmetry(
            atom_number, lattice, positions, rotations, translations, self._symprec
        )

    @property
    def tolerance(self) -> float:
        """Return the symmetry-search tolerance (``symprec``) used at construction."""
        return self._symprec

    @property
    def reciprocal_operations(self) -> NDArray[np.int64]:
        """Return reciprocal-space point-group operations.

        Operations act on q-vectors as ``q' = R q``. This is the
        transpose of the convention shown in ITA (``q' = q R``).
        ``shape=(n_ops, 3, 3)``, ``dtype='int64'``.

        """
        return self._reciprocal_operations

    @property
    def atomic_permutations(self) -> NDArray[np.int64]:
        """Return per-operation atomic-index permutations.

        For each space-group operation, the new index of every atom
        under that operation, computed in real space (modulo lattice
        translation). ``shape=(n_operations, n_atoms)``,
        ``dtype='int64'``. See
        :func:`phonopy.utils.compute_all_sg_permutations` for the
        underlying routine.

        """
        return self._atomic_permutations

    def _set_atomic_permutations(self) -> None:
        positions = self._cell.scaled_positions
        lattice = np.array(self._cell.cell.T, dtype="double", order="C")
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]
        # When atoms can be co-located (mixtures or weighted species, or
        # suffix-distinguished species), match permutations within each
        # type class so position degeneracy does not mix species. The
        # types are the same per-atom labels spglib used to find the
        # operations. Ordinary cells pass None for unchanged behavior.
        if (
            self._cell.has_mixtures
            or self._cell.has_weighted_species
            or self._distinguish_symbol_index
        ):
            types = self._cell.totuple(
                distinguish_symbol_index=self._distinguish_symbol_index
            )[2]
        else:
            types = None
        self._atomic_permutations = compute_all_sg_permutations(
            positions,  # scaled positions
            rotations,  # scaled
            translations,  # scaled
            lattice,  # column vectors
            self._symprec,
            lang=self._lang,
            types=types,
        )

    def _get_site_symmetry(
        self,
        atom_number: int,
        lattice: NDArray[np.double],
        positions: NDArray[np.double],
        rotations: NDArray[np.int64],
        translations: NDArray[np.double],
        symprec: float,
    ) -> NDArray[np.int64]:
        pos = positions[atom_number]
        site_symmetries = []

        for r, t in zip(rotations, translations, strict=True):
            rot_pos = np.dot(pos, r.T) + t
            diff = pos - rot_pos
            diff -= np.rint(diff)
            diff = np.dot(diff, lattice)
            if np.linalg.norm(diff) < symprec:
                site_symmetries.append(r)

        return np.array(site_symmetries, dtype="int64")

    def _set_symmetry_dataset(self) -> None:
        _dataset = spglib.get_symmetry_dataset(
            self._cell.totuple(distinguish_symbol_index=self._distinguish_symbol_index),  # type: ignore[arg-type]
            self._symprec,
        )
        assert _dataset is not None
        self._dataset = _dataset

        self._symmetry_operations = {
            "rotations": np.asarray(self._dataset.rotations, dtype="int64", order="C"),
            "translations": np.asarray(
                self._dataset.translations, dtype="double", order="C"
            ),
        }
        self._international_table = "%s (%d)" % (
            self._dataset.international,
            self._dataset.number,
        )
        self._wyckoff_letters = self._dataset.wyckoffs[:]

        self._map_atoms = np.asarray(self._dataset.equivalent_atoms, dtype="int64")

    def _set_symmetry_operations_with_magmoms(self) -> None:
        _dataset = spglib.get_magnetic_symmetry_dataset(
            self._cell.totuple(distinguish_symbol_index=self._distinguish_symbol_index),  # type: ignore
            symprec=self._symprec,
        )
        assert _dataset is not None
        self._dataset = _dataset

        self._symmetry_operations = {
            "rotations": np.asarray(self._dataset.rotations, dtype="int64", order="C"),
            "translations": np.asarray(
                self._dataset.translations, dtype="double", order="C"
            ),
        }
        self._map_atoms = np.asarray(self._dataset.equivalent_atoms, dtype="int64")

    def _set_independent_atoms(self) -> None:
        indep_atoms = []
        for i, atom_map in enumerate(self._map_atoms):
            if i == atom_map:
                indep_atoms.append(i)
        self._independent_atoms = np.array(indep_atoms, dtype="int64")

    def _get_map_operations_from_permutations(self) -> NDArray[np.int64]:
        perm = self._atomic_permutations
        map_operations = np.zeros(perm.shape[1], dtype="int64")
        for i, eq_atom in enumerate(self._map_atoms):
            match = np.where(perm[:, i] == eq_atom)[0]
            assert len(match) != 0
            map_operations[i] = match[0]
        return map_operations

    def _set_nosym(self, s2p_map: NDArray[np.int64] | None = None) -> None:
        translations = []
        rotations = []

        if s2p_map is None:
            rotations.append(np.eye(3, dtype="int64"))
            translations.append(np.zeros(3, dtype="double"))
            self._map_atoms = np.arange(len(self._cell), dtype="int64")
        else:
            positions = self._cell.scaled_positions
            ipos0 = 0
            for i, j in enumerate(s2p_map):
                if j == 0:
                    ipos0 = i
                    break
            for i, p in zip(s2p_map, positions, strict=True):
                if i == 0:
                    trans = p - positions[ipos0]
                    trans -= np.floor(trans)
                    translations.append(trans)
                    rotations.append(np.eye(3, dtype="int64"))
            self._map_atoms = s2p_map
        self._symmetry_operations = {
            "rotations": np.asarray(rotations, dtype="int64", order="C"),
            "translations": np.asarray(translations, dtype="double", order="C"),
        }
        self._international_table = "P1 (1)"
        self._wyckoff_letters = ["a"] * len(self._cell)
        self._dataset = NosymDataset(
            rotations=self._symmetry_operations["rotations"],
            translations=self._symmetry_operations["translations"],
            transformation_matrix=np.eye(3, dtype="double", order="C"),
            pointgroup="1",
        )


def get_pointgroup_operations(
    rotations: NDArray[np.int64], is_time_reversal: bool = True
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return direct-space and reciprocal-space point-group operations.

    Parameters
    ----------
    rotations : ndarray
        Rotation matrices from space-group operations.
        ``shape=(n_sg, 3, 3)``, ``dtype='int64'``.
    is_time_reversal : bool, optional
        If True and inversion is not already present, the inversion of
        each rotation is added to the reciprocal operations to account
        for time-reversal symmetry. Default is True.

    Returns
    -------
    ptg_ops : ndarray
        Direct-space point-group operations (deduplicated rotations).
        ``shape=(n_ptg, 3, 3)``, ``dtype='int64'``.
    reciprocal_ops : ndarray
        Reciprocal-space point-group operations (transposes of
        ``ptg_ops``, optionally with their inversions appended).
        ``dtype='int64'``, ``order='C'``.

    """
    ptg_ops = collect_unique_rotations(rotations)
    reciprocal_rotations = [rot.T for rot in ptg_ops]

    if is_time_reversal:
        exist_r_inv = False
        for rot in ptg_ops:
            if (rot == -np.eye(3, dtype="int64")).all():
                exist_r_inv = True
                break
        if not exist_r_inv:
            reciprocal_rotations += [-rot.T for rot in ptg_ops]

    return ptg_ops, np.array(reciprocal_rotations, dtype="int64", order="C")


def collect_unique_rotations(rotations: NDArray[np.int64]) -> NDArray[np.int64]:
    """Deduplicate a list of rotation matrices.

    Parameters
    ----------
    rotations : ndarray
        Rotation matrices, possibly with duplicates.
        ``shape=(n, 3, 3)``, ``dtype='int64'``.

    Returns
    -------
    ndarray
        Unique rotation matrices in input order.
        ``shape=(n_unique, 3, 3)``, ``dtype='int64'``, ``order='C'``.

    """
    ptg_ops = []
    for rot in rotations:
        is_same = False
        for tmp_rot in ptg_ops:
            if (tmp_rot == rot).all():
                is_same = True
                break
        if not is_same:
            ptg_ops.append(rot)

    return np.array(ptg_ops, dtype="int64", order="C")


def get_lattice_vector_equivalence(point_symmetry: NDArray[np.int64]) -> list[bool]:
    """Return equivalences of basis-vector length pairs, ``(b==c, c==a, a==b)``.

    Change of basis is defined by ``(a', b', c') = (a, b, c) R``. The
    rotation matrices are scanned to see whether any of
    ``a -> b``, ``b -> c``, ``c -> a``, ``a -> -b``, ``b -> -c``, or
    ``c -> -a`` occurs.

    Parameters
    ----------
    point_symmetry : array_like
        Rotation matrices. ``shape=(n_rot, 3, 3)``, ``dtype=int``.

    Returns
    -------
    list of bool
        ``[b == c, c == a, a == b]``: True when the corresponding basis
        vectors are symmetry-equivalent in length.

    """
    # primitive_vectors: column vectors

    equivalence = [False, False, False]
    for r in point_symmetry:
        if (np.abs(r[:, 0]) == [0, 1, 0]).all():
            equivalence[2] = True
        if (np.abs(r[:, 0]) == [0, 0, 1]).all():
            equivalence[1] = True
        if (np.abs(r[:, 1]) == [1, 0, 0]).all():
            equivalence[2] = True
        if (np.abs(r[:, 1]) == [0, 0, 1]).all():
            equivalence[0] = True
        if (np.abs(r[:, 2]) == [1, 0, 0]).all():
            equivalence[1] = True
        if (np.abs(r[:, 2]) == [0, 1, 0]).all():
            equivalence[0] = True

    return equivalence


def elaborate_borns_and_epsilon(
    ucell: PhonopyAtoms,
    borns: NDArray[np.double],
    epsilon: NDArray[np.double],
    primitive_matrix: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    is_symmetry: bool = True,
    symmetrize_tensors: bool = False,
    symprec: float = 1e-5,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.int64]]:
    """Symmetrize Born effective charges and the dielectric tensor.

    Born effective charges of the symmetrically independent atoms in
    the primitive cell are extracted.

    Parameters
    ----------
    ucell : PhonopyAtoms
        Unit cell.
    borns : array_like
        Born effective charges of ``ucell``.
        ``shape=(n_atoms, 3, 3)``, ``dtype='double'``.
    epsilon : array_like
        Dielectric constant tensor.
        ``shape=(3, 3)``, ``dtype='double'``.
    primitive_matrix : array_like, optional
        Primitive matrix used to construct the primitive cell. Default
        is None.
    supercell_matrix : array_like, optional
        Supercell matrix used together with ``primitive_matrix`` to
        build the primitive cell. Default is None (1x1x1).
    is_symmetry : bool, optional
        Whether to use symmetry when extracting independent atoms.
        Default is True.
    symmetrize_tensors : bool, optional
        If True, also symmetrize the input ``borns`` and ``epsilon`` via
        :func:`symmetrize_borns_and_epsilon` before extracting the
        independent set. Default is False.
    symprec : float, optional
        Symmetry tolerance. Default is 1e-5.
    lang : {"C", "Rust"}, optional
        Backend implementation. Default is ``"Rust"``.

    Returns
    -------
    borns_indep : ndarray
        Born effective charges of the symmetrically independent atoms
        in the primitive cell.
        ``shape=(n_indep, 3, 3)``, ``dtype='double'``, ``order='C'``.
    epsilon : ndarray
        Dielectric constant tensor (symmetrized if
        ``symmetrize_tensors=True``, otherwise as given).
    indeps_in_supercell : ndarray
        Atom indices of the independent atoms in the supercell.

    Raises
    ------
    AssertionError
        If the number of atoms in ``ucell`` and ``borns`` disagree.

    Warns
    -----
    Emits a ``UserWarning`` if the input Born effective charges deviate
    significantly from their symmetrized values.

    """
    lang = resolve_lang(lang)

    assert len(borns) == len(ucell), "num_atom %d != len(borns) %d" % (
        len(ucell),
        len(borns),
    )

    if symmetrize_tensors:
        borns_, epsilon_ = symmetrize_borns_and_epsilon(
            borns, epsilon, ucell, symprec=symprec, is_symmetry=is_symmetry, lang=lang
        )
    else:
        borns_ = borns
        epsilon_ = epsilon

    indeps_in_supercell, indeps_in_unitcell = _extract_independent_atoms(
        ucell,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        is_symmetry=is_symmetry,
        symprec=symprec,
        lang=lang,
    )

    return (
        np.array(borns_[indeps_in_unitcell], dtype="double", order="C"),
        epsilon_,
        indeps_in_supercell,
    )


def symmetrize_borns_and_epsilon(
    borns: Sequence[Sequence[Sequence[float]]] | NDArray[np.double],
    epsilon: Sequence[Sequence[float]] | NDArray[np.double],
    ucell: PhonopyAtoms,
    primitive_matrix: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    primitive: PhonopyAtoms | None = None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    symprec: float = 1e-5,
    is_symmetry: bool = True,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Symmetrize Born effective charges and the dielectric tensor.

    Parameters
    ----------
    borns : array_like
        Born effective charges.
        ``shape=(unitcell_atoms, 3, 3)``, ``dtype='double'``.
    epsilon : array_like
        Dielectric constant tensor.
        ``shape=(3, 3)``, ``dtype='double'``.
    ucell : PhonopyAtoms
        Unit cell.
    primitive_matrix : array_like, optional
        Primitive matrix used to select Born effective charges in the
        primitive cell. ``shape=(3, 3)``, ``dtype='double'``. If None
        (default), Born effective charges in the unit cell are returned.
    primitive : PhonopyAtoms, optional
        Alternative to ``primitive_matrix``. Mp is given as
        ``Mp = (a_u, b_u, c_u)^-1 (a_p, b_p, c_p)``. The order of atoms
        in the returned Born effective charges is aligned with the
        atoms in this primitive cell. No rigid rotation of the crystal
        structure is assumed.
    supercell_matrix : array_like, optional
        Supercell matrix used to select Born effective charges in the
        primitive cell. The supercell matrix is needed because the
        primitive cell is constructed by first building the supercell
        from the unit cell and then extracting the primitive cell from
        the supercell. ``shape=(3, 3)``, ``dtype='int'``. If None
        (default), a 1x1x1 supercell is assumed.
    symprec : float, optional
        Symmetry tolerance. Default is 1e-5.
    is_symmetry : bool, optional
        Set to False to disable symmetrization. Default is True.
    lang : {"C", "Rust"}, optional
        Backend implementation. Default is ``"Rust"``.

    Returns
    -------
    borns : ndarray
        Symmetrized Born effective charges.
    epsilon : ndarray
        Symmetrized dielectric tensor.

    """
    lang = resolve_lang(lang)
    lattice = ucell.cell
    u_sym = Symmetry(ucell, is_symmetry=is_symmetry, symprec=symprec, lang=lang)
    rotations = u_sym.symmetry_operations["rotations"]
    translations = u_sym.symmetry_operations["translations"]
    ptg_ops = u_sym.pointgroup_operations
    epsilon_ = _symmetrize_2nd_rank_tensor(
        np.asarray(epsilon, dtype="double"), ptg_ops, lattice
    )
    borns_ = _take_average_of_borns(borns, rotations, translations, ucell, symprec)

    if (abs(borns - borns_) > 0.1).any():
        lines = [
            "Symmetry of Born effective charge is largely broken. The difference is:",
            "%s" % (borns - borns_),
        ]
        import warnings

        warnings.warn("\n".join(lines), stacklevel=2)

    if primitive_matrix is None and primitive is None:
        return borns_, epsilon_
    else:
        if primitive is not None:
            pmat = np.dot(np.linalg.inv(ucell.cell.T), primitive.cell.T)
        else:
            pmat = primitive_matrix

        scell, pcell = _get_supercell_and_primitive(
            ucell,
            primitive_matrix=pmat,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
            lang=lang,
        )

        idx = [scell.u2u_map[i] for i in scell.s2u_map[pcell.p2s_map]]
        borns_in_prim = np.array(borns_[idx], dtype="double", order="C")

        if primitive is None:
            return borns_in_prim, epsilon_
        else:
            idx2 = _get_mapping_between_cells(pcell, primitive)
            return np.array(borns_in_prim[idx2], dtype="double", order="C"), epsilon_


def _take_average_of_borns(
    borns: Sequence[Sequence[Sequence[float]]] | NDArray[np.double],
    rotations: NDArray[np.int64],
    translations: NDArray[np.double],
    cell: PhonopyAtoms,
    symprec: float,
) -> NDArray[np.double]:
    lattice = cell.cell
    positions = cell.scaled_positions
    # Per-atom species id, used to disambiguate co-located atoms of a site
    # mixture (e.g. Ge and Sn sharing a site): the symmetry-operation
    # pre-image of atom i must be the same species, which a position-only
    # match cannot guarantee. Within one cell the species id is exact.
    species_ids = cell.species_ids
    borns_ = np.zeros_like(borns)
    for i in range(len(borns)):
        for r, t in zip(rotations, translations, strict=True):
            diff = np.dot(positions, r.T) + t - positions[i]
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(diff, lattice), axis=1)
            matches = np.nonzero((dist < symprec) & (species_ids == species_ids[i]))[0]
            j = matches[0]
            r_cart = similarity_transformation(lattice.T, r)
            borns_[i] += similarity_transformation(r_cart, borns[j])
        borns_[i] /= len(rotations)

    sum_born = borns_.sum(axis=0) / len(borns_)
    borns_ -= sum_born

    return borns_


def _get_mapping_between_cells(
    cell_from: PhonopyAtoms, cell_to: PhonopyAtoms, symprec: float = 1e-5
) -> list[int]:
    # See get_atom_order for the definition of the returned order.
    order = get_atom_order(cell_from, cell_to, atol=symprec)
    if order is None:
        msg = "Index matching didn't go well."
        raise RuntimeError(msg)
    return order


def _symmetrize_2nd_rank_tensor(
    tensor: NDArray[np.double],
    symmetry_operations: NDArray[np.int64],
    lattice: NDArray[np.double],
) -> NDArray[np.double]:
    sym_cart = [similarity_transformation(lattice.T, r) for r in symmetry_operations]
    sum_tensor = np.zeros_like(tensor)
    for sym in sym_cart:
        sum_tensor += similarity_transformation(sym, tensor)
    return sum_tensor / len(symmetry_operations)


def _extract_independent_atoms(
    ucell: PhonopyAtoms,
    primitive_matrix: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    is_symmetry: bool = True,
    symprec: float = 1e-5,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.int64], list[int]]:
    scell, pcell = _get_supercell_and_primitive(
        ucell,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
        lang=lang,
    )
    p_sym = Symmetry(pcell, is_symmetry=is_symmetry, symprec=symprec, lang=lang)
    s_indep_atoms = np.array(
        pcell.p2s_map[p_sym.get_independent_atoms()], dtype="int64"
    )
    u_indep_atoms = [scell.u2u_map[x] for x in s_indep_atoms]

    return s_indep_atoms, u_indep_atoms


def _get_supercell_and_primitive(
    ucell: PhonopyAtoms,
    primitive_matrix: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64] | None = None,
    symprec: float = 1e-5,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[Supercell, Primitive]:
    if primitive_matrix is None:
        pmat = np.eye(3)
    else:
        pmat = primitive_matrix
    if supercell_matrix is None:
        smat = np.eye(3, dtype="int64")
    else:
        smat = supercell_matrix

    inv_smat = np.linalg.inv(smat)
    scell = get_supercell(ucell, smat, symprec=symprec)
    pcell = get_primitive(scell, np.dot(inv_smat, pmat), symprec=symprec, lang=lang)

    return scell, pcell
