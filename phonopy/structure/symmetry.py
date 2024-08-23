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

import warnings
from collections.abc import Sequence
from typing import Optional

try:
    from spglib import SpglibDataset
except ImportError:
    from types import SimpleNamespace as SpglibDataset

import numpy as np
import spglib

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    compute_all_sg_permutations,
    get_primitive,
    get_supercell,
)
from phonopy.utils import get_dot_access_dataset, similarity_transformation


class Symmetry:
    """Class to find and store crystal symmetry information."""

    def __init__(
        self,
        cell: PhonopyAtoms,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        s2p_map: Optional[np.ndarray] = None,
    ):
        """Init method.

        cell : PhonopyAtoms
            Crystal structure whose symmetry is analyzed.
        symprec : float, optional
            Tolerance used to find crystal symmetry. Default is 1e-5.
        is_symmetry : bool, optional
            With or without symmetry analysis. When `is_symmetry=False` and
            `p2s_map` is given, pure translations inside `cell` is registered
            as symmetry operations. Default is True.
        s2p_map : ndarray, optional
            This is equivalent to `Primitive.s2p_map`.

        """
        self._cell = cell
        self._symprec = symprec

        self._symmetry_operations: dict
        self._international_table = None
        self._dataset = None
        self._wyckoff_letters = None
        self._map_atoms = None
        self._atomic_permutations = None
        self._pointgroup_operations = None
        self._pointgroup = None
        self._independent_atoms = None
        self._map_operations = None

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
        ptg_symbol = spglib.get_pointgroup(self._pointgroup_operations)[0]
        self._pointgroup = ptg_symbol.strip()
        self._set_atomic_permutations()
        self._set_independent_atoms()
        self._map_operations = self._get_map_operations_from_permutations()

    @property
    def symmetry_operations(self):
        """Return symmetry operations.

        Returns
        -------
        dict
            'rotations': ndarray
                Matrix parts.
                shape(num_operations, 3, 3), dtype='intc',
             'translations': ndarray
                Vector parts.
                shape(num_operations, 3), dtype='double',

        """
        return self._symmetry_operations

    def get_symmetry_operations(self):
        """Return symmetry operations."""
        warnings.warn(
            "Symmetry.get_symmetry_operations() is deprecated."
            "Use symmetry_operations attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.symmetry_operations

    def get_symmetry_operation(self, operation_number):
        """Return one symmetry operation."""
        operation = self._symmetry_operations
        return {
            "rotations": operation["rotations"][operation_number],
            "translations": operation["translations"][operation_number],
        }

    @property
    def pointgroup_operations(self):
        """Return crystallographic point group operations."""
        return self._pointgroup_operations

    def get_pointgroup_operations(self):
        """Return crystallographic point group operations."""
        warnings.warn(
            "Symmetry.get_pointgroup_operations() is deprecated."
            "Use pointgroup_operations attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pointgroup_operations

    @property
    def pointgroup_symbol(self):
        """Return symbol of crystallographic point group."""
        return self._pointgroup

    def get_pointgroup(self):
        """Return symbol of crystallographic point group."""
        warnings.warn(
            "Symmetry.get_pointgroup() is deprecated."
            "Use pointgroup_symbol attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pointgroup

    def get_international_table(self):
        """Return international symbol of space group."""
        return self._international_table

    def get_Wyckoff_letters(self):
        """Return Wycloff letters."""
        return self._wyckoff_letters

    @property
    def dataset(self) -> SpglibDataset:
        """Return spglib dataset.

        This is raw data of symmetry.

        """
        return self._dataset

    def get_dataset(self):
        """Return spglib dataset."""
        warnings.warn(
            "Symmetry.get_dataset() is deprecated." "Use dataset attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dataset

    def get_independent_atoms(self):
        """Return symmetrically unique atoms."""
        return self._independent_atoms

    def get_map_atoms(self):
        """Return equivalent_atoms of spglib dataset.

        Returns
        -------
        spglib_dataset.equivalent_atoms
        e.g.,
            [0, 0, 0, 0, 4, 4, 4, 4]

        """
        return self._map_atoms

    def get_map_operations(self):
        """Return symmetry operation indices that map to respective equivalent atoms.

        Returns
        -------
        operations : ndarray
            Indices of symmetry operations that sent atoms to respective
            equivalent atoms. For each atom, only one of those symmetry
            operations is stored. When all symmetry mapping information
            is needed, ``atomic_permulations`` is used.
            shape=(atoms,), dtype='intc'

        """
        return self._map_operations

    def get_site_symmetry(self, atom_number):
        """Return matrix parts of site symmetry operations."""
        positions = self._cell.scaled_positions
        lattice = self._cell.cell
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]

        return self._get_site_symmetry(
            atom_number, lattice, positions, rotations, translations, self._symprec
        )

    @property
    def tolerance(self):
        """Return symmetry tolerance."""
        return self._symprec

    def get_symmetry_tolerance(self):
        """Return symmetry tolerance."""
        warnings.warn(
            "Symmetry.get_symmetry_tolerance() is deprecated."
            "Use tolerance attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tolerance

    @property
    def reciprocal_operations(self):
        """Return reciprocal space point group operations.

        Definition of operation:
        q' = Rq

        This is transpose of that shown in ITA (q' = qR).

        """
        return self._reciprocal_operations

    def get_reciprocal_operations(self):
        """Return reciprocal space point group operations."""
        warnings.warn(
            "Symmetry.get_reciprocal_operations() is deprecated."
            "Use reciprocal_operations attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.reciprocal_operations

    @property
    def atomic_permutations(self):
        """Return atomic index permutations by space group operations.

        shape=(operations, positions)

        See compute_all_sg_permutations.

        """
        return self._atomic_permutations

    def get_atomic_permutations(self):
        """Return atomic index permutations by space group operations."""
        warnings.warn(
            "Symmetry.get_atomic_permutations() is deprecated."
            "Use atomic_permutations attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.atomic_permutations

    def _set_atomic_permutations(self):
        positions = self._cell.scaled_positions
        lattice = np.array(self._cell.cell.T, dtype="double", order="C")
        rotations = self._symmetry_operations["rotations"]
        translations = self._symmetry_operations["translations"]
        self._atomic_permutations = compute_all_sg_permutations(
            positions,  # scaled positions
            rotations,  # scaled
            translations,  # scaled
            lattice,  # column vectors
            self._symprec,
        )

    def _get_site_symmetry(
        self, atom_number, lattice, positions, rotations, translations, symprec
    ):
        pos = positions[atom_number]
        site_symmetries = []

        for r, t in zip(rotations, translations):
            rot_pos = np.dot(pos, r.T) + t
            diff = pos - rot_pos
            diff -= np.rint(diff)
            diff = np.dot(diff, lattice)
            if np.linalg.norm(diff) < symprec:
                site_symmetries.append(r)

        return np.array(site_symmetries, dtype="intc")

    def _set_symmetry_dataset(self):
        self._dataset = get_dot_access_dataset(
            spglib.get_symmetry_dataset(self._cell.totuple(), self._symprec)
        )

        self._symmetry_operations = {
            "rotations": self._dataset.rotations,
            "translations": self._dataset.translations,
        }
        self._international_table = "%s (%d)" % (
            self._dataset.international,
            self._dataset.number,
        )
        self._wyckoff_letters = self._dataset.wyckoffs

        self._map_atoms = self._dataset.equivalent_atoms

    def _set_symmetry_operations_with_magmoms(self):
        self._dataset = get_dot_access_dataset(
            spglib.get_magnetic_symmetry_dataset(
                self._cell.totuple(), symprec=self._symprec
            )
        )
        self._symmetry_operations = {
            "rotations": self._dataset.rotations,
            "translations": self._dataset.translations,
        }
        self._map_atoms = self._dataset.equivalent_atoms

    def _set_independent_atoms(self):
        indep_atoms = []
        for i, atom_map in enumerate(self._map_atoms):
            if i == atom_map:
                indep_atoms.append(i)
        self._independent_atoms = np.array(indep_atoms, dtype="intc")

    def _set_map_operations(self):
        warnings.warn(
            "Symmetry._set_map_operations is deprecated."
            "This was replaced by "
            "_get_map_operations_from_permutations.",
            DeprecationWarning,
            stacklevel=2,
        )

        ops = self._symmetry_operations
        pos = self._cell.scaled_positions
        lattice = self._cell.cell
        map_operations = np.zeros(len(pos), dtype="intc")

        for i, eq_atom in enumerate(self._map_atoms):
            for j, (r, t) in enumerate(zip(ops["rotations"], ops["translations"])):
                diff = np.dot(pos[i], r.T) + t - pos[eq_atom]
                diff -= np.rint(diff)
                dist = np.linalg.norm(np.dot(diff, lattice))
                if dist < self._symprec:
                    map_operations[i] = j
                    break
        self._map_operations = map_operations

    def _get_map_operations_from_permutations(self):
        perm = self._atomic_permutations
        map_operations = np.zeros(perm.shape[1], dtype="intc")
        for i, eq_atom in enumerate(self._map_atoms):
            match = np.where(perm[:, i] == eq_atom)[0]
            assert len(match) != 0
            map_operations[i] = match[0]
        return map_operations

    def _set_nosym(self, s2p_map):
        translations = []
        rotations = []

        if s2p_map is None:
            rotations.append(np.eye(3, dtype="intc"))
            translations.append(np.zeros(3, dtype="double"))
            self._map_atoms = range(len(self._cell))
        else:
            positions = self._cell.scaled_positions
            for i, j in enumerate(s2p_map):
                if j == 0:
                    ipos0 = i
                    break
            for i, p in zip(s2p_map, positions):
                if i == 0:
                    trans = p - positions[ipos0]
                    trans -= np.floor(trans)
                    translations.append(trans)
                    rotations.append(np.eye(3, dtype="intc"))
            self._map_atoms = s2p_map
        self._symmetry_operations = {
            "rotations": np.array(rotations, dtype="intc"),
            "translations": np.array(translations, dtype="double"),
        }
        self._international_table = "P1 (1)"
        self._wyckoff_letters = ["a"] * len(self._cell)


def get_pointgroup_operations(rotations, is_time_reversal=True):
    """Return direct and reciprocal point group operations."""
    ptg_ops = collect_unique_rotations(rotations)
    reciprocal_rotations = [rot.T for rot in ptg_ops]

    if is_time_reversal:
        exist_r_inv = False
        for rot in ptg_ops:
            if (rot == -np.eye(3, dtype="intc")).all():
                exist_r_inv = True
                break
        if not exist_r_inv:
            reciprocal_rotations += [-rot.T for rot in ptg_ops]

    return (
        np.array(ptg_ops, dtype="intc"),
        np.array(reciprocal_rotations, dtype="intc"),
    )


def collect_unique_rotations(rotations):
    """Collect unique rotations in input rotation matrices."""
    ptg_ops = []
    for rot in rotations:
        is_same = False
        for tmp_rot in ptg_ops:
            if (tmp_rot == rot).all():
                is_same = True
                break
        if not is_same:
            ptg_ops.append(rot)

    return ptg_ops


def get_lattice_vector_equivalence(point_symmetry):
    """Return equivalences of basis vector length pairs, (b==c, c==a, a==b).

    Change of basis is defined by

    (a', b', c') = (a, b, c) R.

    Then, check rotation matrices if
        a -> b, b -> c, c -> a, a -> -b, b -> -c, and c -> -a,
    can happen.

    Parameters
    ----------
    point_symmetry : array_like
        Rotation matrices.
        shape=(n_rot, 3, 3), dtype=int.

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
    ucell,
    borns,
    epsilon,
    primitive_matrix=None,
    supercell_matrix=None,
    is_symmetry=True,
    symmetrize_tensors=False,
    symprec=1e-5,
):
    """Symmetrize Born effective charges and dielectric constants.

    Born effective charges of symmetrically independent atoms
    for primitive cell are extracted.


    Parameters
    ----------
    ucell : PhonopyAtoms
        Unit cell structure
    borns : array_like
        Born effective charges of ucell
    epsilon : array_like
        Dielectric constant tensor

    Returns
    -------
    ndarray
        Born effective charges of symmetrically independent atoms in primitive cell
        Dielectric constant
    ndarray
        Atomic index mapping table from supercell to primitive cell
        of independent atoms

    Raises
    ------
    AssertionError
        Inconsistency of number of atoms or Born effective charges.

    Warning
    -------
    Broken symmetry of Born effective charges

    """
    assert len(borns) == len(ucell), "num_atom %d != len(borns) %d" % (
        len(ucell),
        len(borns),
    )

    if symmetrize_tensors:
        borns_, epsilon_ = symmetrize_borns_and_epsilon(
            borns, epsilon, ucell, symprec=symprec, is_symmetry=is_symmetry
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
    )

    return borns_[indeps_in_unitcell].copy(), epsilon_, indeps_in_supercell


def symmetrize_borns_and_epsilon(
    borns: Sequence,
    epsilon: Sequence,
    ucell: PhonopyAtoms,
    primitive_matrix: Optional[Sequence] = None,
    primitive: PhonopyAtoms = None,
    supercell_matrix: Optional[Sequence] = None,
    symprec: float = 1e-5,
    is_symmetry: bool = True,
):
    """Symmetrize Born effective charges and dielectric tensor.

    Parameters
    ----------
    borns: array_like
        Born effective charges.
        shape=(unitcell_atoms, 3, 3)
        dtype='double'
    epsilon: array_like
        Dielectric constant
        shape=(3, 3)
        dtype='double'
    ucell: PhonopyAtoms
        Unit cell
    primitive_matrix: array_like, optional
        Primitive matrix. This is used to select Born effective charges in
        primitive cell. If None (default), Born effective charges in unit cell
        are returned.
        shape=(3, 3)
        dtype='double'
    primitive : PhonopyAtoms
        This is an alternative of giving primitive_matrix (Mp). Mp is given as
            Mp = (a_u, b_u, c_u)^-1 * (a_p, b_p, c_p).
        In addition, the order of atoms is alined to those of atoms in this
        primitive cell for Born effective charges. No rigid rotation of
        crystal structure is assumed.
    supercell_matrix: array_like, optional
        Supercell matrix. This is used to select Born effective charges in
        **primitive cell**. Supercell matrix is needed because primitive
        cell is created first creating supercell from unit cell, then
        the primitive cell is created from the supercell. If None (defautl),
        1x1x1 supercell is created.
        shape=(3, 3)
        dtype='int'
    symprec: float, optional
        Symmetry tolerance. Default is 1e-5
    is_symmetry: bool, optinal
        By setting False, symmetrization can be switched off. Default is True.

    """
    lattice = ucell.cell
    u_sym = Symmetry(ucell, is_symmetry=is_symmetry, symprec=symprec)
    rotations = u_sym.symmetry_operations["rotations"]
    translations = u_sym.symmetry_operations["translations"]
    ptg_ops = u_sym.pointgroup_operations
    epsilon_ = _symmetrize_2nd_rank_tensor(epsilon, ptg_ops, lattice)
    borns_ = _take_average_of_borns(borns, rotations, translations, ucell, symprec)

    if (abs(borns - borns_) > 0.1).any():
        lines = [
            "Symmetry of Born effective charge is largely broken. "
            "The difference is:",
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
        )

        idx = [scell.u2u_map[i] for i in scell.s2u_map[pcell.p2s_map]]
        borns_in_prim = borns_[idx].copy()

        if primitive is None:
            return borns_in_prim, epsilon_
        else:
            idx2 = _get_mapping_between_cells(pcell, primitive)
            return borns_in_prim[idx2].copy(), epsilon_


def _take_average_of_borns(
    borns: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    cell: PhonopyAtoms,
    symprec: float,
) -> np.ndarray:
    lattice = cell.cell
    positions = cell.scaled_positions
    borns_ = np.zeros_like(borns)
    for i in range(len(borns)):
        for r, t in zip(rotations, translations):
            diff = np.dot(positions, r.T) + t - positions[i]
            diff -= np.rint(diff)
            dist = np.sqrt(np.sum(np.dot(diff, lattice) ** 2, axis=1))
            j = np.nonzero(dist < symprec)[0][0]
            r_cart = similarity_transformation(lattice.T, r)
            borns_[i] += similarity_transformation(r_cart, borns[j])
        borns_[i] /= len(rotations)

    sum_born = borns_.sum(axis=0) / len(borns_)
    borns_ -= sum_born

    return borns_


def _get_mapping_between_cells(
    cell_from: PhonopyAtoms, cell_to: PhonopyAtoms, symprec: float = 1e-5
):
    indices = []
    lattice = cell_from.cell
    pos_from = cell_from.scaled_positions
    for p_to in cell_to.scaled_positions:
        diff = pos_from - p_to
        diff -= np.rint(diff)
        dist = np.sqrt(np.sum(np.dot(diff, lattice) ** 2, axis=1))
        ids = np.nonzero(dist < symprec)[0]
        if len(ids) == 1:
            indices.append(ids[0])
        else:
            msg = "Index matching didn't go well."
            raise RuntimeError(msg)
    return indices


def _symmetrize_2nd_rank_tensor(tensor, symmetry_operations, lattice):
    sym_cart = [similarity_transformation(lattice.T, r) for r in symmetry_operations]
    sum_tensor = np.zeros_like(tensor)
    for sym in sym_cart:
        sum_tensor += similarity_transformation(sym, tensor)
    return sum_tensor / len(symmetry_operations)


def _extract_independent_atoms(
    ucell, primitive_matrix=None, supercell_matrix=None, is_symmetry=True, symprec=1e-5
):
    scell, pcell = _get_supercell_and_primitive(
        ucell,
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        symprec=symprec,
    )
    p_sym = Symmetry(pcell, is_symmetry=is_symmetry, symprec=symprec)
    s_indep_atoms = pcell.p2s_map[p_sym.get_independent_atoms()]
    u_indep_atoms = [scell.u2u_map[x] for x in s_indep_atoms]

    return s_indep_atoms, u_indep_atoms


def _get_supercell_and_primitive(
    ucell, primitive_matrix=None, supercell_matrix=None, symprec=1e-5
):
    if primitive_matrix is None:
        pmat = np.eye(3)
    else:
        pmat = primitive_matrix
    if supercell_matrix is None:
        smat = np.eye(3, dtype="intc")
    else:
        smat = supercell_matrix

    inv_smat = np.linalg.inv(smat)
    scell = get_supercell(ucell, smat, symprec=symprec)
    pcell = get_primitive(scell, np.dot(inv_smat, pmat), symprec=symprec)

    return scell, pcell
