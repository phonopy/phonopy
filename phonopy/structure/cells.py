"""Primitive cell and supercell, and related utilities."""

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
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
import spglib

try:
    spglib.error.OLD_ERROR_HANDLING = False
except AttributeError:
    pass

from numpy.typing import NDArray
from spglib import SpglibDataset, SpglibMagneticDataset

from phonopy._lang import log_dispatch, resolve_lang
from phonopy.structure.atomic_data import get_atomic_data
from phonopy.structure.atoms import (
    PhonopyAtoms,
    _dedup_species,
    _Species,
    build_species_table_from_mixtures,
)
from phonopy.structure.mixture import build_mixtures_from_groups
from phonopy.structure.snf import SNF3x3


class Supercell(PhonopyAtoms):
    """Build supercell from supercell matrix and unit cell.

    Attributes
    ----------
    supercell_matrix : ndarray
    s2u_map : ndarray
    u2s_map : ndarray
    u2u_map : dicst

    """

    def __init__(
        self,
        unitcell: PhonopyAtoms,
        supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64],
        is_old_style: bool = True,
        symprec: float = 1e-5,
    ) -> None:
        """Init method.

        Note
        ----
        `is_old_style=True` invokes the following algorithm.
        In this function, unit cell is considered
          [1,0,0]
          [0,1,0]
          [0,0,1].
        Supercell matrix is given by relative ratio, e.g,
          [-1, 1, 1]
          [ 1,-1, 1]  is for FCC from simple cubic.
          [ 1, 1,-1].
        In this case multiplicities of surrounding simple lattice are [2,2,2].
        First, create supercell with surrounding simple lattice.
        Second, trim the surrounding supercell with the target lattice.

        `is_old_style=False` calls the Smith normal form.

        These two algorithm may order atoms in different ways. So for the
        backward compatibility, `is_old_style=True` is the default
        option. However the Smith normal form shows far better performance
        in case of very large supercell multiplicities.

        Parameters
        ----------
        unitcell: PhonopyAtoms
            Unit cell
        supercell_matrix: ndarray or list of list
            Transformation matrix from unit cell to supercell. The
            elements have to be integers.
            shape=(3,3)
        is_old_stype: bool
            This switches the algorithms. See Note.
        symprec: float, optional
            Tolerance to find overlapping atoms in supercell cell. The default
            values is 1e-5.

        """
        self._is_old_style = is_old_style
        self._s2u_map: NDArray
        self._u2s_map: NDArray
        self._u2u_map: dict[int, int]
        self._supercell_matrix = np.array(supercell_matrix, dtype="int64", order="C")
        self._create_supercell(unitcell, symprec)

    @property
    def supercell_matrix(self) -> NDArray[np.int64]:
        """Return supercell_matrix.

        Returns
        -------
        ndarray
            Supercell matrix.
            shape=(3, 3), dtype='int64'

        """
        return self._supercell_matrix

    @property
    def s2u_map(self) -> NDArray[np.int64]:
        """Return atomic index mapping table from supercell to unit cell.

        Each array index and the stored value correspond to the supercell atom
        and unit cell atom in supercell atomic indices in supercell atom index.

        Returns
        -------
        ndarray
            shape=(num_atoms_in_supercell, ), dtype='int64'

        """
        return self._s2u_map

    @property
    def u2s_map(self) -> NDArray[np.int64]:
        """Return atomic index mapping table from unit cell to supercell.

        Each array index and the stored value correspond to the unit cell atom
        and supecell atom in supercell atom index.

        Returns
        -------
        ndarray
            shape=(num_atoms_in_unitcell, ), dtype='int64'

        """
        return self._u2s_map

    @property
    def u2u_map(self) -> dict[int, int]:
        """Return atomic index mapping table from unit cell to unit cell.

        Returns
        -------
        dict
            Each key and value correspond to supercell atom index and unit cell
            atom index to represent an atom in unit cell.

        """
        return self._u2u_map

    def _create_supercell(self, unitcell: PhonopyAtoms, symprec: float) -> None:
        mat = self._supercell_matrix
        if self._is_old_style:
            P = None
            multi = self._get_surrounding_frame(mat)
            # trim_frame is to trim overlapping atoms.
            trim_frame = np.array(
                [
                    mat[0] / float(multi[0]),
                    mat[1] / float(multi[1]),
                    mat[2] / float(multi[2]),
                ]
            )
        else:
            # In the new style, it is unnecessary to trim atoms,
            if (np.diag(np.diagonal(mat)) != mat).any():
                snf = SNF3x3(mat)
                P = snf.P
                multi = np.diagonal(snf.D)
            else:
                P = None
                multi = np.diagonal(mat)
            trim_frame = np.eye(3)

        sur_cell, u2sur_map = self._get_simple_supercell(unitcell, multi, P)
        supercell, sur2s_map, mapping_table = _trim_cell(
            trim_frame, sur_cell, check_overlap=self._is_old_style, symprec=symprec
        )
        num_satom = len(supercell)
        num_uatom = len(unitcell)
        N = num_satom // num_uatom

        if N != determinant(self._supercell_matrix):
            msg = "\n".join(
                [
                    "Supercell creation failed.",
                    "Probably some atoms are overwrapped. "
                    "The mapping table is given below.",
                    str(mapping_table),
                ]
            )
            raise RuntimeError(msg)
        else:
            super().__init__(
                species_table=supercell.species_table,
                species_ids=supercell.species_ids,
                masses=supercell.masses,
                magnetic_moments=supercell.magnetic_moments,
                scaled_positions=supercell.scaled_positions,
                cell=supercell.cell,
            )
            self._u2s_map = np.array(np.arange(num_uatom) * N, dtype="int64")
            self._u2u_map = {j: i for i, j in enumerate(self._u2s_map)}
            self._s2u_map = np.array(u2sur_map[sur2s_map] * N, dtype="int64")

    def _get_simple_supercell(
        self,
        unitcell: PhonopyAtoms,
        multi: list[int] | NDArray[np.int64],
        P: NDArray[np.int64] | None,
    ) -> tuple[PhonopyAtoms, NDArray[np.int64]]:
        if self._is_old_style:
            mat = np.diag(multi)
        else:
            mat = self._supercell_matrix

        # Scaled positions within the frame, i.e., create a supercell that
        # is made simply to multiply the input cell.
        positions = unitcell.scaled_positions
        masses = unitcell.masses
        magmoms = unitcell.magnetic_moments
        lattice = unitcell.cell

        # Index of a axis runs fastest for creating lattice points.
        # See numpy.meshgrid document for the complicated index order for 3D
        b, c, a = np.meshgrid(range(multi[1]), range(multi[2]), range(multi[0]))
        lattice_points = np.c_[a.ravel(), b.ravel(), c.ravel()]

        if P is not None:
            # If supercell matrix is not a diagonal matrix,
            # Smith normal form is applied to find oblique basis vectors for
            # supercell and primitive cells, where their basis vectos are
            # parallel each other. By this reason, simple construction of
            # supercell becomes possible.
            P_inv = np.rint(np.linalg.inv(P)).astype(int)
            assert determinant(P_inv) == 1
            lattice_points = np.dot(lattice_points, P_inv.T)

        n = len(positions)
        n_l = len(lattice_points)
        # tile: repeat blocks
        # repeat: repeat each element
        positions_multi = np.dot(
            np.tile(lattice_points, (n, 1)) + np.repeat(positions, n_l, axis=0),
            np.linalg.inv(mat).T,
        )
        species_ids_multi = np.repeat(unitcell.species_ids, n_l)
        atom_map = np.repeat(np.arange(n), n_l)
        if masses is None:
            masses_multi = None
        else:
            masses_multi = np.repeat(masses, n_l)
        if magmoms is None:
            magmoms_multi = None
        elif magmoms.ndim == 1:
            magmoms_multi = np.repeat(magmoms, n_l)
        else:  # non-collinear
            magmoms_multi = [v for v in magmoms for _ in range(n_l)]

        simple_supercell = PhonopyAtoms(
            species_table=unitcell.species_table,
            species_ids=species_ids_multi,
            masses=masses_multi,
            magnetic_moments=magmoms_multi,
            scaled_positions=positions_multi,
            cell=np.dot(mat, lattice),
        )

        return simple_supercell, atom_map

    def _get_surrounding_frame(self, supercell_matrix: NDArray[np.int64]) -> list[int]:
        # Build a frame surrounding supercell lattice
        # For example,
        #  [2,0,0]
        #  [0,2,0] is the frame for FCC from simple cubic.
        #  [0,0,2]

        m = np.array(supercell_matrix)
        axes = np.array(
            [
                [0, 0, 0],
                m[:, 0],
                m[:, 1],
                m[:, 2],
                m[:, 1] + m[:, 2],
                m[:, 2] + m[:, 0],
                m[:, 0] + m[:, 1],
                m[:, 0] + m[:, 1] + m[:, 2],
            ]
        )
        frame = [max(axes[:, i]) - min(axes[:, i]) for i in (0, 1, 2)]
        return frame


class Primitive(PhonopyAtoms):
    """Build primitive cell."""

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive_matrix: Sequence[Sequence[float]] | NDArray,
        symprec: float = 1e-5,
        store_dense_svecs: bool = True,
        positions_to_reorder: NDArray[np.double] | None = None,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell
        primitive_matrix : list of list or ndarray
            Transformation matrix to transform supercell to primitive cell such
            as:
               np.dot(primitive_matrix.T, supercell.cell)
            shape=(3,3)
        symprec : float, optional
            Tolerance to find overlapping atoms in primitive cell. Default is
            1e-5.
        store_dense_svecs : bool, optional
            Shortest vectors are stored in a dense array. Default is True. See
            `ShortestPairs`.
        positions_to_reorder : NDArray[np.double], optional
            If atomic positions in a created primitive cell is known and the
            order of atoms is expected to be sure, these positions with the
            specific order is used after position matching between this data and
            generated positions. Default is None.
        lang : {"C", "Rust"}, optional
            Backend used by helpers that have a Rust port (currently the
            atomic-permutation matcher).  Default is "C".

        """
        self._primitive_matrix = np.array(primitive_matrix, dtype="double", order="C")
        self._symprec = symprec
        self._store_dense_svecs = store_dense_svecs
        self._lang: Literal["C", "Rust"] = resolve_lang(lang)
        log_dispatch(self._lang, "Primitive.__init__")
        self._p2s_map: NDArray
        self._s2p_map: NDArray
        self._p2p_map: dict[int, int]
        self._smallest_vectors: NDArray
        self._multiplicity: NDArray
        self._atomic_permutations: NDArray
        self._run(supercell, positions_to_reorder=positions_to_reorder)

    @property
    def primitive_matrix(self) -> NDArray[np.double]:
        """Return primitive_matrix.

        Returns
        -------
        ndarray
            Transformation matrix from supercell to primitive cell
            dtype='double'
            shape=(3,3)

        """
        return self._primitive_matrix

    @property
    def p2s_map(self) -> NDArray[np.int64]:
        """Return mapping table of atoms from primitive cell to supercell.

        Returns
        -------
        ndarray
            Mapping table from atoms in primitive cell to those in supercell.
            Supercell atomic indices are used.
            shape=(num_atoms_in_primitive_cell,), dtype='int64'

        """
        return self._p2s_map

    @property
    def s2p_map(self) -> NDArray[np.int64]:
        """Return mapping table of atoms from supercell to primitive cells.

        Returns
        -------
        ndarray
            Mapping table from atoms in supercell cell to those in primitive
            cell.  Supercell atomic indices are used.
            shape=(num_atoms_in_supercell, ), dtype='int64'

        """
        return self._s2p_map

    @property
    def p2p_map(self) -> dict[int, int]:
        """Return mapping table of indices in supercell and primitive cell.

        Returns
        -------
        dict
            Mapping of primitive cell atoms in supercell to those in primitive.
            cell.
            ex. {0: 0, 4: 1}

        """
        return self._p2p_map

    def get_smallest_vectors(self) -> tuple[NDArray[np.double], NDArray[np.int64]]:
        """Return shortest vectors and multiplicities.

        See also the docstring of `ShortestPairs`. The older less densen format
        is deprecated. The detailed explanation is found in `ShortestPairs`
        class.

        Returns
        -------
        tuple
            shortest_vectors : np.ndarray
                Shortest vectors of atomic pairs in supercell from an atom in
                the primitive cell to an atom in the supercell. The vectors are
                given in the coordinates with respect to the primitive cell
                basis vectors. In the dense format
                shape=(sum(multiplicities[:,:, 0], 3), dtype='double',
                dtype='double', order='C'.
            multiplicities : np.ndarray
                Number of equidistance shortest vectors. The last dimension
                indicates [0] multipliticy at the pair of atoms in the supercell
                and primitive cell, and [1] integral of multiplicities to this
                pair, i.e., which indicates address used in `shortest_vectors`.
                In the dense format, shape=(size_super, size_prim, 2),
                dtype='int64', order='C'.

        """
        return self._smallest_vectors, self._multiplicity

    @property
    def atomic_permutations(self) -> NDArray[np.int64]:
        """Return atomic index permutations by pure translations.

        Returns
        -------
        ndarray
            Atomic position permutation by pure translations is represented by
            changes of indices.
            dtype='int64'
            shape=(num_trans, num_atoms_in_supercell)
            ex.       supercell atomic indices
                     [[0, 1, 2, 3, 4, 5, 6, 7],
               trans  [1, 2, 3, 0, 5, 6, 7, 4],
              indices [2, 3, 0, 1, 6, 7, 4, 5],
                      [3, 0, 1, 2, 7, 4, 5, 6]]

        """
        return self._atomic_permutations

    @property
    def store_dense_svecs(self) -> bool:
        """Return whether shortest vectors are stored in dense array or not."""
        return self._store_dense_svecs

    def _run(
        self,
        supercell: PhonopyAtoms,
        positions_to_reorder: NDArray[np.double] | None = None,
    ) -> None:
        self._p2s_map = self._create_primitive_cell(
            supercell, positions_to_reorder=positions_to_reorder
        )
        self._s2p_map, self._p2p_map = self._map_atomic_indices(
            supercell.scaled_positions, supercell.species_ids
        )
        (self._smallest_vectors, self._multiplicity) = self._get_smallest_vectors(
            supercell
        )
        self._atomic_permutations = self._get_atomic_permutations(supercell)

    def _create_primitive_cell(
        self,
        supercell: PhonopyAtoms,
        positions_to_reorder: NDArray[np.double] | None = None,
    ) -> NDArray[np.int64]:
        trimmed_cell, p2s_map, _ = _trim_cell(
            self._primitive_matrix,
            supercell,
            symprec=self._symprec,
            positions_to_reorder=positions_to_reorder,
        )
        super().__init__(
            species_table=trimmed_cell.species_table,
            species_ids=trimmed_cell.species_ids,
            masses=trimmed_cell.masses,
            magnetic_moments=trimmed_cell.magnetic_moments,
            scaled_positions=trimmed_cell.scaled_positions,
            cell=trimmed_cell.cell,
        )
        return p2s_map

    def _map_atomic_indices(
        self, s_pos_orig: NDArray[np.double], species_ids: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], dict[int, int]]:
        frac_pos = np.dot(s_pos_orig, np.linalg.inv(self._primitive_matrix).T)

        p2s_positions = frac_pos[self._p2s_map]
        p2s_species = species_ids[self._p2s_map]
        s2p_map = []
        for s_pos, s_species in zip(frac_pos, species_ids, strict=True):
            # Compute distances from s_pos to all positions in _p2s_map.
            frac_diffs = p2s_positions - s_pos
            frac_diffs -= np.rint(frac_diffs)
            cart_diffs = np.dot(frac_diffs, self.cell)
            distances = np.sqrt((cart_diffs**2).sum(axis=1))
            # Match the same species, so co-located atoms of different
            # species map to their own representative (e.g. Ge and Sn at
            # the same site). Ordinary cells have one species per site, so
            # this is unchanged.
            indices = np.where(
                (distances < self._symprec) & (p2s_species == s_species)
            )[0]
            assert len(indices) == 1
            s2p_map.append(self._p2s_map[indices[0]])

        s2p_map = np.array(s2p_map, dtype="int64")
        p2p_map = dict([(j, i) for i, j in enumerate(self._p2s_map)])

        return s2p_map, p2p_map

    def _get_atomic_permutations(self, supercell: PhonopyAtoms) -> NDArray[np.int64]:
        positions = supercell.scaled_positions
        diff = positions - positions[self._p2s_map[0]]
        trans = np.array(
            diff[np.where(self._s2p_map == self._p2s_map[0])[0]],
            dtype="double",
            order="C",
        )
        rotations = np.array(
            [np.eye(3, dtype="int64")] * len(trans), dtype="int64", order="C"
        )
        atomic_permutations = compute_all_sg_permutations(
            positions,
            rotations,
            trans,
            np.array(supercell.cell.T, dtype="double", order="C"),
            self._symprec,
            supercell.permutation_types,
            lang=self._lang,
        )

        return atomic_permutations

    def _get_smallest_vectors(
        self, supercell: PhonopyAtoms
    ) -> tuple[NDArray[np.double], NDArray[np.int64]]:
        """Find shortest vectors.

        See the docstring of `ShortestPairs`.

        Note
        ----
        Returned shortest vectors are transformed to those in the primitive
        cell coordinates from those in the supercell coordinates in this
        method.

        """
        p2s_map = self._p2s_map
        supercell_pos = supercell.scaled_positions
        primitive_pos = supercell_pos[p2s_map]
        supercell_bases = supercell.cell
        primitive_bases = self._cell
        svecs, multi = get_smallest_vectors(
            supercell_bases,
            supercell_pos,
            primitive_pos,
            store_dense_svecs=self._store_dense_svecs,
            symprec=self._symprec,
            lang=self._lang,
        )
        trans_mat_float = np.dot(supercell_bases, np.linalg.inv(primitive_bases))
        trans_mat = np.rint(trans_mat_float).astype(int)
        assert (np.abs(trans_mat_float - trans_mat) < 1e-8).all()
        svecs = np.array(np.dot(svecs, trans_mat), dtype="double", order="C")
        return svecs, multi


class TrimmedCell(PhonopyAtoms):
    """Cell obtained by trimming overlapping atoms in a smaller lattice.

    The lattice of ``cell`` is transformed into a smaller one by
    ``relative_axes`` (``trimmed_lattice = relative_axes.T @ cell.cell``), and
    atoms that coincide on the same site under that smaller lattice are merged
    into a single representative. The trimmed cell is ``self``.

    Merging is species-aware: atoms of distinct species on the same site (e.g.
    "Cl" and "Cl1", or distinct concentrations) are kept separate, so the
    trimmed cell is always well-formed. The surviving atom count must equal the
    lattice volume ratio ``det(relative_axes)`` times the input count; a
    mismatch (distinct species folded onto one site, or overlapping atoms) is
    an error and raises ``RuntimeError``.

    Attributes
    ----------
    extracted_atoms : ndarray
        Indices into the input cell of the atoms kept in the trimmed cell.
    mapping_table : ndarray
        For each atom of the input cell, the input-cell index of the
        representative atom it is merged into.

    """

    def __init__(
        self,
        relative_axes: NDArray[np.double],
        cell: PhonopyAtoms,
        positions_to_reorder: NDArray[np.double] | None = None,
        check_overlap: bool = True,
        symprec: float = 1e-5,
    ) -> None:
        """Init method.

        Parameters
        ----------
        relative_axes: ndarray
            Transformation matrix to transform supercell to a smaller cell
            such as:
                trimmed_lattice = np.dot(relative_axes.T, cell.cell)
            shape=(3,3)
        cell: PhonopyAtoms
            Supercell.
        positions_to_reorder : NDArray[np.double] | None
            Expected positions after trimming. This is used to fix the order
            of atoms in trimmed cell. This may be used to get the same
            primitive cell generated from supercells having different shapes.
        check_overlap : bool, optional
            This flag can be set False, if the determinant of relative_axis
            is 1, e.g., when using SNF. Default is True.
        symprec: float, optional
            Tolerance to find overlapping atoms in the trimmed cell.
            Default is 1e-5.

        """
        self._run(cell, relative_axes, positions_to_reorder, check_overlap, symprec)

    @property
    def mapping_table(self) -> NDArray[np.int64]:
        """Return mapping table.

        mapping_table : ndarray
            The atomic indices of 'extracted_atom's of all atoms in the input
            cell.
            shape=(len(cell), ), dtype='int64'

        """
        return self._mapping_table

    @property
    def extracted_atoms(self) -> NDArray[np.int64]:
        """Return extracted atoms.

        Returns
        -------
        extracted_atoms : ndarray
            Indices of atomic indices of input cell that are in the trimmed
            cell.
            shape=(len(trimmed_cell), ), dtype='int64'

        """
        return self._extracted_atoms

    def _run(
        self,
        cell: PhonopyAtoms,
        relative_axes: NDArray[np.double],
        positions_to_reorder: NDArray[np.double] | None,
        check_overlap: bool,
        symprec: float,
    ) -> None:
        trimmed_lattice = np.dot(relative_axes.T, cell.cell)
        positions_in_new_lattice = np.dot(
            cell.scaled_positions, np.linalg.inv(relative_axes).T
        )
        positions_in_new_lattice -= np.floor(positions_in_new_lattice)

        (
            trimmed_positions,
            trimmed_masses,
            trimmed_magmoms,
            extracted_atoms,
            mapping_table,
        ) = self._extract(
            positions_in_new_lattice,
            trimmed_lattice,
            cell.masses,
            cell.magnetic_moments,
            check_overlap,
            symprec,
            cell.species_ids,
        )

        if positions_to_reorder is not None:
            ids = self._get_reorder_indices(
                positions_to_reorder, trimmed_positions, trimmed_lattice, symprec
            )
            trimmed_positions = trimmed_positions[ids]
            if trimmed_masses is not None:
                trimmed_masses = trimmed_masses[ids]
            if trimmed_magmoms is not None:
                trimmed_magmoms = trimmed_magmoms[ids]
            extracted_atoms = extracted_atoms[ids]

        # A correct trim reduces the cell by the lattice volume ratio
        # det(relative_axes). A different surviving count means atoms that
        # should have coincided did not: distinct species folded onto one site
        # (e.g. an incompatible primitive matrix) or overlapping atoms in a bad
        # supercell matrix. Either way it is an error.
        expected_natom = int(np.rint(len(cell) * np.linalg.det(relative_axes)))
        if len(extracted_atoms) != expected_natom:
            symbols = [cell.symbols[i] for i in extracted_atoms]
            msg = [
                "Cell trimming failed.",
                f"The input cell ({len(cell)} atoms) does not reduce to "
                f"{expected_natom} atoms by the lattice volume ratio; trimming "
                f"left {len(extracted_atoms)}.",
                "Atoms of distinct species were folded onto the same site, or "
                "atoms overlap. Trimmed cell:",
            ]
            for i, s in enumerate(symbols):
                msg.append(f"  {i + 1}: {s}")
            raise RuntimeError("\n".join(msg))
        super().__init__(
            species_table=cell.species_table,
            species_ids=cell.species_ids[extracted_atoms],
            masses=trimmed_masses,
            magnetic_moments=trimmed_magmoms,
            scaled_positions=trimmed_positions,
            cell=trimmed_lattice,
        )
        self._extracted_atoms = np.array(extracted_atoms, dtype="int64")
        self._mapping_table = mapping_table

    def _extract(
        self,
        positions_in_new_lattice: NDArray[np.double],
        trimmed_lattice: NDArray[np.double],
        masses: NDArray[np.double] | None,
        magmoms: NDArray[np.double] | None,
        check_overlap: bool,
        symprec: float,
        species_ids: NDArray[np.int64],
    ) -> tuple[
        NDArray[np.double],
        NDArray[np.double] | None,
        NDArray[np.double] | None,
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        num_atoms = 0
        extracted_atoms: list[int] = []
        mapping_table = np.arange(len(positions_in_new_lattice), dtype="int64")
        trimmed_positions = np.zeros_like(positions_in_new_lattice)
        trimmed_masses_list: list[float] | None = None if masses is None else []
        trimmed_magmoms_list: list | None = None if magmoms is None else []

        for i, pos in enumerate(positions_in_new_lattice):
            found_overlap = False
            if check_overlap and num_atoms > 0:
                diff = trimmed_positions[:num_atoms] - pos
                diff -= np.rint(diff)
                # Older numpy doesn't support axis argument.
                distances = np.sqrt(np.sum(np.dot(diff, trimmed_lattice) ** 2, axis=1))
                close = distances < symprec
                # Require the same species id so co-located atoms of different
                # species (e.g. Ge and Sn at one site) are kept separate.
                close = close & (species_ids[extracted_atoms] == species_ids[i])
                overlap_indices = np.where(close)[0]
                if len(overlap_indices) > 0:
                    assert len(overlap_indices) == 1
                    found_overlap = True
                    mapping_table[i] = extracted_atoms[overlap_indices[0]]

            if not found_overlap:
                trimmed_positions[num_atoms] = pos
                num_atoms += 1
                if masses is not None:
                    assert trimmed_masses_list is not None
                    trimmed_masses_list.append(masses[i])
                if magmoms is not None:
                    assert trimmed_magmoms_list is not None
                    trimmed_magmoms_list.append(magmoms[i])
                extracted_atoms.append(i)

        trimmed_masses = (
            None
            if trimmed_masses_list is None
            else np.array(trimmed_masses_list, dtype="double")
        )
        trimmed_magmoms = (
            None
            if trimmed_magmoms_list is None
            else np.array(trimmed_magmoms_list, dtype="double", order="C")
        )

        return (
            np.array(trimmed_positions[:num_atoms], dtype="double", order="C"),
            trimmed_masses,
            trimmed_magmoms,
            np.array(extracted_atoms, dtype="int64"),
            mapping_table,
        )

    def _get_reorder_indices(
        self,
        positions: NDArray[np.double],
        trimmed_positions: NDArray[np.double],
        trimmed_lattice: NDArray[np.double],
        symprec: float,
    ) -> list[int]:
        """Reorder trimmed cell by input primitive cell positions."""
        reorder_indices = []
        for pos in positions:
            diff = trimmed_positions - pos
            diff -= np.rint(diff)
            dist = np.sqrt(np.sum(np.dot(diff, trimmed_lattice) ** 2, axis=1))
            overlap_indices = np.where(dist < symprec)[0]
            assert len(overlap_indices) == 1
            reorder_indices.append(overlap_indices[0])
        return reorder_indices


def get_supercell(
    unitcell: PhonopyAtoms,
    supercell_matrix: Sequence[Sequence[int]] | NDArray[np.int64],
    is_old_style: bool = True,
    symprec: float = 1e-5,
) -> Supercell:
    """Create supercell."""
    return Supercell(
        unitcell, supercell_matrix, is_old_style=is_old_style, symprec=symprec
    )


def sort_positions_by_symbols(
    symbols: Sequence[str | int] | NDArray[np.int64],
    positions: NDArray[np.double] | None = None,
) -> tuple[list[int], list[str | int], NDArray[np.double] | None, list[int]]:
    """Sort atomic positions by symbols.

    Sort positions by symbols (using the order defined by reduced_symbols)
    using a stable sort algorithm. Written by @ExpHP, refactored by @atztogo.

    symbols = ["A", "B", "A", "B"]
    reduced_symbols = ["A", "B"]
    sort_keys = [0, 1, 0, 1]
    perm = [0, 2, 1, 3]
    counts_dict = {'A': 2, 'B': 2}
    counts_list = [2, 2]

    Parameters
    ----------
    symbols : list[str] or list[int] or NDArray[np.int64]
        Sequence of hashable objects. This may be a list of chemical symbols
        or numbers.
    positions : NDArray[np.double] or None, optional
        Atomic positions. When None, sorted_positions is also None.

    Returns
    -------
    sorted_positions = positions[perm]
    For the others, see the example above.

    Functions
    ---------
    _argsort_stable :
        Alternative to `np.argsort(keys)` that uses a stable sorting algorithm
        so that indices tied for the same value are listed in increasing order.

    """

    def _argsort_stable(keys):
        # Python's built-in sort algorithm is a stable sort
        return sorted(range(len(keys)), key=keys.__getitem__)

    # dict in Python 3.7 or later is ordered dict.
    reduced_symbols = list(dict.fromkeys(symbols))
    counts_dict = Counter(symbols)
    # list(counts_dict.values()) may be used...
    counts_list = [counts_dict[s] for s in reduced_symbols]
    sort_keys = [reduced_symbols.index(i) for i in symbols]
    perm = _argsort_stable(sort_keys)

    if positions is None:
        sorted_positions = None
    else:
        sorted_positions = positions[perm]

    return counts_list, reduced_symbols, sorted_positions, perm


def get_primitive(
    supercell: PhonopyAtoms,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R"]
    | Sequence[Sequence[float]]
    | NDArray[np.double],
    symprec: float = 1e-5,
    store_dense_svecs: bool = True,
    positions_to_reorder: NDArray[np.double] | None = None,
    lang: Literal["C", "Rust"] = "Rust",
) -> Primitive:
    """Create primitive cell."""
    pmat = get_primitive_matrix(primitive_matrix)
    assert pmat is not None
    return Primitive(
        supercell,
        pmat,
        symprec=symprec,
        store_dense_svecs=store_dense_svecs,
        positions_to_reorder=positions_to_reorder,
        lang=lang,
    )


def raise_if_suffixed_symbols(cell: PhonopyAtoms) -> None:
    """Raise if the cell carries suffixed symbols sharing an atomic number.

    Symmetry standardization rebuilds a cell from atomic numbers using spglib,
    so two species that share an atomic number but differ only by a symbol
    suffix (e.g. ``"Cl"`` and ``"Cl1"``) cannot be distinguished in the result.
    This guard refuses such cells rather than silently dropping the suffix
    labels. Site-mixture cells are rebuilt from the species table and keep
    their symbols, so they are exempt; this guard is meant only for the
    atomic-number reconstruction path.

    Parameters
    ----------
    cell : PhonopyAtoms
        Cell about to be reconstructed from atomic numbers.

    Raises
    ------
    ValueError
        If two species share an atomic number but carry different symbols.

    """
    symbol_of: dict[int, str] = {}
    for sp in cell.species_table:
        if sp.atomic_number is None:
            continue
        previous = symbol_of.get(sp.atomic_number)
        if previous is not None and previous != sp.symbol:
            raise ValueError(
                "Symmetry standardization cannot preserve suffixed symbols that "
                f"share an atomic number (e.g. '{previous}' and '{sp.symbol}'); "
                "the standardized cell is built from atomic numbers, which would "
                "drop the suffix labels."
            )
        symbol_of[sp.atomic_number] = sp.symbol


def get_standardized_cell(
    cell: PhonopyAtoms,
    dataset: SpglibDataset | SpglibMagneticDataset,
) -> PhonopyAtoms:
    """Build the standardized conventional cell from a spglib dataset.

    This is the inverse of PhonopyAtoms.totuple: it turns the spglib
    standardization result back into a PhonopyAtoms, preserving the
    species table (site-mixture weights / mixed species) and, for
    magnetic datasets, the standardized magnetic moments. The
    standardized cell may include a rigid rotation with respect to the
    input cell for which symmetry was analyzed.

    The dataset must have been produced from cell.totuple() (i.e. with
    distinguish_symbol_index=False, the default). Under that contract,
    dataset.std_types are species ids into cell.species_table exactly
    when cell.has_mixtures or cell.has_weighted_species, and atomic
    numbers otherwise.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input cell from which the symmetry dataset was computed. Its
        species table is reused to restore mixture weights / mixed
        species.
    dataset : SpglibDataset or SpglibMagneticDataset
        Symmetry dataset of spglib computed from cell.totuple().

    Returns
    -------
    PhonopyAtoms
        Standardized conventional unit cell.

    """
    std_positions = dataset.std_positions
    std_types = dataset.std_types
    _, _, _, perm = sort_positions_by_symbols(std_types, std_positions)

    if cell.is_site_mixture:
        # std_types are species ids into cell.species_table (totuple
        # hands species ids to spglib for such cells). Rebuilding with
        # the species table restores symbol, atomic number, mass, and
        # weight / mixture in one step.
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            species_table=cell.species_table,
            species_ids=std_types[perm],
        )

    # std_types are atomic numbers.
    raise_if_suffixed_symbols(cell)
    atom_data = get_atomic_data().atom_data
    if isinstance(dataset, SpglibDataset):
        return PhonopyAtoms(
            cell=dataset.std_lattice,
            scaled_positions=std_positions[perm],
            symbols=[atom_data[n][1] for n in std_types[perm]],
        )
    return PhonopyAtoms(
        cell=dataset.std_lattice,
        scaled_positions=std_positions[perm],
        symbols=[atom_data[n][1] for n in std_types[perm]],
        magnetic_moments=dataset.std_tensors[perm],
    )


def generate_standardized_cells(
    cell: PhonopyAtoms,
    dataset: SpglibDataset | SpglibMagneticDataset,
    symprec: float = 1e-5,
) -> tuple[PhonopyAtoms, PhonopyAtoms, NDArray[np.double]]:
    """Return the standardized conventional and primitive cells.

    Companion of get_standardized_cell that also derives the primitive
    cell. The primitive cell is obtained via get_primitive, which carries
    the species table, so mixture weights / mixed species propagate to
    the primitive cell as well.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input cell from which the symmetry dataset was computed.
    dataset : SpglibDataset or SpglibMagneticDataset
        Symmetry dataset of spglib computed from cell.totuple().
    symprec : float, optional
        Symmetry search tolerance forwarded to get_primitive. Default is
        1e-5.

    Returns
    -------
    tuple[PhonopyAtoms, PhonopyAtoms, NDArray[np.double]]
        (conventional, primitive, primitive_matrix). When the primitive
        matrix is the identity, primitive is the conventional cell.

    """
    conventional = get_standardized_cell(cell, dataset)
    pmat = _get_primitive_matrix_from_dataset(dataset)
    if (np.abs(pmat - np.eye(3)) < 1e-8).all():
        primitive = conventional
    else:
        primitive = get_primitive(conventional, primitive_matrix=pmat, symprec=symprec)
    return conventional, primitive, pmat


def _get_primitive_matrix_from_dataset(
    dataset: SpglibDataset | SpglibMagneticDataset,
) -> NDArray[np.double]:
    """Return the centring primitive matrix implied by a spglib dataset.

    The returned matrix maps the standardized conventional cell to its
    primitive cell. To obtain a primitive matrix relative to the input
    cell, compose it with the inverse transformation matrix (see
    guess_primitive_matrix).

    """
    spg_type = spglib.get_spacegroup_type(dataset.hall_number)
    assert spg_type is not None
    centring = spg_type.international[0]
    return get_primitive_matrix_by_centring(centring)


def build_mixture_cell(
    cell: PhonopyAtoms,
    weights: Sequence[float],
    symprec: float = 1e-5,
    sort_constituents: bool = True,
) -> PhonopyAtoms:
    """Merge overlapping atoms into mixed-species sites.

    Atoms whose fractional positions agree (modulo lattice translations)
    within ``symprec`` are collapsed into a single site whose species is the
    weighted mixture of the constituents. Each non-overlapping atom is
    preserved verbatim and its weight must equal 1.0.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input unit cell. Must not already contain mixed-species sites.
    weights : sequence of float
        Per-atom mixture weights in the input order. Length must equal
        ``len(cell)``. Within each overlapping atom group the weights must
        sum to 1.0; for an isolated atom the weight must be exactly 1.0.
        Note that this is a per-atom convention and is distinct from the
        VASP ``INCAR`` ``VCA`` tag, which lists one weight per element row
        in POSCAR.
    symprec : float
        Tolerance for treating two fractional positions as overlapping.
    sort_constituents : bool, optional
        Forwarded to ``build_species_table_from_mixtures``. When True
        (default), constituents of every merged site are sorted
        alphabetically by symbol so that two sites differing only in input
        order share one species. Pass False to keep the input order.

    Returns
    -------
    PhonopyAtoms
        Cell with one atom per merged group. Mixed sites carry composite
        symbols (e.g. ``"GeSn"``); when several distinct mixtures share the
        same composite within the cell, all of them get 1-based suffixes
        (``"GeSn1"``, ``"GeSn2"``, ...).

    """
    if cell.has_mixtures:
        raise ValueError(
            "build_mixture_cell cannot be applied to a cell that already "
            "contains mixed-species sites."
        )
    if cell.magnetic_moments is not None:
        raise ValueError(
            "build_mixture_cell does not support cells carrying magnetic moments."
        )
    if len(weights) != len(cell):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match number of atoms "
            f"({len(cell)})."
        )

    weights_arr = np.asarray(weights, dtype="double")
    scaled = cell.scaled_positions

    groups = _group_overlapping_atoms(cell, symprec)
    mixtures = build_mixtures_from_groups(cell.symbols, groups, weights_arr)
    new_scaled = [scaled[group[0]] for group in groups]

    species_table, species_ids = build_species_table_from_mixtures(
        mixtures, sort_constituents=sort_constituents
    )

    return PhonopyAtoms(
        cell=cell.cell,
        scaled_positions=np.array(new_scaled, dtype="double"),
        species_table=species_table,
        species_ids=species_ids,
    )


def apply_site_mixture(
    cell: PhonopyAtoms,
    weights: Sequence[float] | NDArray[np.double],
    symprec: float = 1e-5,
) -> PhonopyAtoms:
    """Attach per-atom concentration weights keeping atoms species-resolved.

    Returns a new cell whose species table carries the concentration of each
    atom as a weighted real species (``_Species.weight``). Unlike
    :func:`build_mixture_cell`, atoms are **not** merged: every input atom is
    preserved (``len(result) == len(cell)``) and keeps its real element symbol,
    atomic number, and mass. The weights are validated against the positional
    overlap structure of the cell: atoms at the same fractional position form a
    group whose weights must sum to 1.0, and isolated atoms must carry weight
    1.0.

    Use this function to prepare a cell with co-located atoms of different
    species (a site mixture / virtual-crystal cell). The ``weights`` list
    corresponds one-to-one with the atom order in the input cell, matching
    the VASP ``INCAR`` ``VCA`` tag order when the POSCAR lists all atoms of
    each element consecutively.

    Note: when writing displaced supercells with :func:`write_vasp`, phonopy may
    reorder atoms by symbol (``sort_positions_by_symbols``). Ensure that forces
    read from VASP are reordered to match the supercell atom order before
    setting ``phonon.forces``.

    Parameters
    ----------
    cell : PhonopyAtoms
        Input unit cell. Must not already contain mixed-species sites
        (``has_mixtures`` must be False), magnetic moments, or weighted species.
    weights : sequence of float or ndarray
        Per-atom concentration weights in the input cell order. ``len(weights)``
        must equal ``len(cell)``. Values must lie in ``(0, 1]``. Isolated atoms
        must have weight 1.0; atoms sharing a fractional position must have
        weights summing to 1.0.
    symprec : float
        Tolerance for treating two fractional positions as overlapping.

    Returns
    -------
    PhonopyAtoms
        Copy of ``cell`` with ``mixture_weights`` attached. ``natom``, symbols,
        positions, and masses are unchanged.

    """
    if cell.has_mixtures:
        raise ValueError(
            "apply_site_mixture cannot be applied to a cell that already contains "
            "mixed-species sites."
        )
    if cell.magnetic_moments is not None:
        raise ValueError(
            "apply_site_mixture does not support cells carrying magnetic moments."
        )
    if cell.has_weighted_species:
        raise ValueError(
            "apply_site_mixture cannot be applied to a cell that already has "
            "weighted species."
        )
    if len(weights) != len(cell):
        raise ValueError(
            f"Length of weights ({len(weights)}) must match number of atoms "
            f"({len(cell)})."
        )

    weights_arr = np.asarray(weights, dtype="double")
    groups = _group_overlapping_atoms(cell, symprec)

    for group in groups:
        wsum = float(weights_arr[list(group)].sum())
        if len(group) == 1:
            if not np.isclose(weights_arr[group[0]], 1.0):
                raise ValueError(
                    f"Weight of non-overlapping atom at index {group[0]} "
                    f"must be 1.0, got {weights_arr[group[0]]}."
                )
        elif not np.isclose(wsum, 1.0):
            raise ValueError(
                f"Weights of overlapping atoms at indices {list(group)} must "
                f"sum to 1.0, got {wsum}."
            )

    # Rebuild the species table with weighted real species. Atoms of
    # co-located groups receive their concentration as a species weight;
    # isolated atoms keep their original species entry (weight=None).
    # Note that the float weight enters _Species equality; weights are
    # stored once here and only copied afterwards, so bitwise comparison
    # is reliable.
    in_overlap_group = np.zeros(len(cell), dtype=bool)
    for group in groups:
        if len(group) > 1:
            in_overlap_group[list(group)] = True

    old_table = cell.species_table
    per_atom_species: list[_Species] = []
    for sid, w, is_overlapping in zip(
        cell.species_ids, weights_arr, in_overlap_group, strict=True
    ):
        base = old_table[sid]
        if is_overlapping:
            per_atom_species.append(dataclasses.replace(base, weight=float(w)))
        else:
            per_atom_species.append(base)
    new_table, new_ids = _dedup_species(per_atom_species)

    return PhonopyAtoms(
        cell=cell.cell,
        scaled_positions=cell.scaled_positions,
        species_table=new_table,
        species_ids=new_ids,
        masses=cell.masses,
    )


def _group_overlapping_atoms(
    cell: PhonopyAtoms, symprec: float = 1e-5
) -> list[list[int]]:
    """Group atom indices whose fractional positions coincide modulo lattice.

    Two atoms belong to the same group when their fractional coordinates
    agree within ``symprec`` after removing integer lattice translations.
    Every atom appears in exactly one group; non-overlapping atoms form
    singleton groups. Groups and their members are returned in ascending
    index order.

    Parameters
    ----------
    cell : PhonopyAtoms
        Cell whose atoms are grouped by positional overlap.
    symprec : float
        Tolerance for treating two fractional positions as overlapping.

    Returns
    -------
    list[list[int]]
        Groups of coinciding atom indices.

    """
    scaled = cell.scaled_positions
    natom = len(cell)

    visited = [False] * natom
    groups: list[list[int]] = []
    for i in range(natom):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, natom):
            if visited[j]:
                continue
            diff = scaled[j] - scaled[i]
            diff -= np.rint(diff)
            if np.all(np.abs(diff) < symprec):
                group.append(j)
                visited[j] = True
        groups.append(group)
    return groups


def print_cell(
    cell: PhonopyAtoms,
    mapping: NDArray[np.int64] | None = None,
    stars: Sequence[int] | None = None,
) -> None:
    """Show cell information."""
    lines = get_cell_lines(cell, mapping=mapping, stars=stars)
    print("\n".join(lines))


def get_cell_lines(
    cell: PhonopyAtoms,
    mapping: NDArray[np.int64] | None = None,
    stars: Sequence[int] | None = None,
) -> list[str]:
    """Return cell information text lines."""
    symbols = cell.symbols
    masses = cell.masses
    magmoms = cell.magnetic_moments
    lattice = cell.cell
    lines = []
    lines.append("Lattice vectors:")
    lines.append("  a %20.15f %20.15f %20.15f" % tuple(lattice[0]))
    lines.append("  b %20.15f %20.15f %20.15f" % tuple(lattice[1]))
    lines.append("  c %20.15f %20.15f %20.15f" % tuple(lattice[2]))
    lines.append("Atomic positions (fractional):")
    for i, v in enumerate(cell.scaled_positions):
        num = " "
        if stars is not None:
            if i in stars:
                num = "*"
        num += "%d" % (i + 1)
        line = "%5s %-2s%18.14f%18.14f%18.14f" % (num, symbols[i], v[0], v[1], v[2])
        if masses is not None:
            line += " %7.3f" % masses[i]
        if magmoms is not None:
            if magmoms.ndim == 1:
                line += "  %5.3f" % magmoms[i]
            else:
                line += "  %s" % magmoms[i]
        if mapping is None:
            lines.append(line)
        else:
            lines.append(line + " > %d" % (mapping[i] + 1))
    return lines


def _species_isclose(
    sp_a: _Species, sp_b: _Species, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Compare two ``_Species`` with concentration weights matched by tolerance.

    Symbol, atomic number, and mixture constituent symbols must agree exactly.
    Concentration weights (the per-atom ``weight`` and the per-constituent
    weights of a mixture) are floats that may originate from different
    construction paths, so they are compared with ``numpy.isclose`` rather
    than exact equality.

    """
    if sp_a.symbol != sp_b.symbol or sp_a.atomic_number != sp_b.atomic_number:
        return False
    if (sp_a.weight is None) != (sp_b.weight is None):
        return False
    if (
        sp_a.weight is not None
        and sp_b.weight is not None
        and not np.isclose(sp_a.weight, sp_b.weight, rtol=rtol, atol=atol)
    ):
        return False
    if (sp_a.mixture is None) != (sp_b.mixture is None):
        return False
    if sp_a.mixture is not None and sp_b.mixture is not None:
        if len(sp_a.mixture) != len(sp_b.mixture):
            return False
        for (s_a, w_a), (s_b, w_b) in zip(sp_a.mixture, sp_b.mixture, strict=True):
            if s_a != s_b or not np.isclose(w_a, w_b, rtol=rtol, atol=atol):
                return False
    return True


def get_atom_order(
    a: PhonopyAtoms,
    b: PhonopyAtoms,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[int] | None:
    """Return the atom order aligning cell-b onto cell-a, or None.

    For each atom of cell-b, the returned list gives the index of the
    matching atom of cell-a, so that ``a`` reordered by the result equals
    ``b`` (positions modulo lattice translation, species within tolerance,
    and magnetic moments). ``a.numbers[order] == b.numbers``.

    Returns None when the cells are not equivalent: different atom count or
    lattice, an atom of ``b`` with no or more than one match in ``a``, or a
    match that is not one-to-one. Matching is by position and species, so
    co-located atoms of a site mixture (e.g. Ge and Sn sharing a site) are
    resolved, which position-only matching cannot do.

    Parameters
    ----------
    a : PhonopyAtoms
        Reference cell.
    b : PhonopyAtoms
        Cell to be compared.
    rtol : float, optional
        Relative tolerance for species weights. Default is 1e-5.
    atol : float, optional
        Tolerance in Cartesian distance and for species weights.
        Default is 1e-8.

    Returns
    -------
    list[int] or None
        Atom indices of cell-a in the order of cell-b, or None when the
        cells are not equivalent.

    """
    if len(a) != len(b):
        return None
    if not np.allclose(a.cell, b.cell, rtol=rtol, atol=atol):
        return None

    a_table = a.species_table
    b_table = b.species_table
    a_species = [a_table[i] for i in a.species_ids]
    b_species = [b_table[i] for i in b.species_ids]
    indices = []
    for pos, sp in zip(b.scaled_positions, b_species, strict=True):
        diff = a.scaled_positions - pos
        diff -= np.rint(diff)
        dist = np.linalg.norm(np.dot(diff, a.cell), axis=1)
        matches = np.where(dist < atol)[0]
        if len(matches) > 1:
            matches = np.array(
                [i for i in matches if _species_isclose(a_species[i], sp, rtol, atol)],
                dtype=int,
            )
        if len(matches) != 1:
            return None
        indices.append(int(matches[0]))
    if not (np.sort(indices) == np.arange(len(indices))).all():
        return None
    if not all(
        _species_isclose(a_species[i], sp_b, rtol, atol)
        for i, sp_b in zip(indices, b_species, strict=True)
    ):
        return None
    if not _magnetic_moments_all_close(
        a.magnetic_moments, b.magnetic_moments, indices=indices, rtol=rtol, atol=atol
    ):
        return None
    return indices


def isclose(
    a: PhonopyAtoms,
    b: PhonopyAtoms,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    with_arbitrary_order: bool = False,
) -> bool:
    """Check equivalence of two cells.

    Cell-b is compared with respect to cell-a.

    Parameters
    ----------
    a : PhonopyAtoms
        Reference cell.
    b : PhonopyAtoms
        Cell to be compared.
    rtol : float, optional
        Relative tolerance in Cartesian coordinates. Default is 1e-5.
    atol : float, optional
        Tolerance in Cartesian distance. Default is 1e-8.
    with_arbitrary_order : bool, optional
        If True, atoms may appear in a different order in the two cells and
        are matched up to permutation. If False (default), atoms are
        compared index by index. Deprecated: use
        ``get_atom_order(a, b) is not None`` instead.

    Returns
    -------
    bool
        Whether two cells agree upto lattice translation of each atom.

    """
    if with_arbitrary_order:
        warnings.warn(
            "isclose(..., with_arbitrary_order=True) is deprecated. Use "
            "get_atom_order(a, b) is not None instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_atom_order(a, b, rtol=rtol, atol=atol) is not None

    if len(a) != len(b):
        return False
    if not np.allclose(a.cell, b.cell, rtol=rtol, atol=atol):
        return False
    a_species = [a.species_table[i] for i in a.species_ids]
    b_species = [b.species_table[i] for i in b.species_ids]
    if not all(
        _species_isclose(sp_a, sp_b, rtol, atol)
        for sp_a, sp_b in zip(a_species, b_species, strict=True)
    ):
        return False
    if not _magnetic_moments_all_close(
        a.magnetic_moments, b.magnetic_moments, rtol=rtol, atol=atol
    ):
        return False
    diff = a.scaled_positions - b.scaled_positions
    diff -= np.rint(diff)
    dist = np.linalg.norm(np.dot(diff, a.cell), axis=1)
    if (dist > atol).any():
        return False
    return True


def _magnetic_moments_all_close(
    a: NDArray[np.double] | None,
    b: NDArray[np.double] | None,
    indices: list[int] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    if indices is None or a is None:
        _a = a
    else:
        _a = a[indices]
    if _a is None and b is not None:
        return False
    if b is None and _a is not None:
        return False
    if _a is None and b is None:
        return True
    assert _a is not None and b is not None
    return np.allclose(_a, b, rtol=rtol, atol=atol)


def is_primitive_cell(rotations: NDArray[np.int64] | NDArray[np.int32]) -> bool:
    """Check if single identity operation exists in rotations or not.

    This is used for checking a cell is a primitive cell or not.

    """
    num_identity = 0
    identity = np.eye(3, dtype="int64")
    for r in rotations:
        if (r == identity).all():
            num_identity += 1
            if num_identity > 1:
                return False
    else:
        return True


def _trim_cell(
    relative_axes: NDArray[np.double],
    cell: PhonopyAtoms,
    check_overlap: bool = True,
    symprec: float = 1e-5,
    positions_to_reorder: NDArray[np.double] | None = None,
) -> tuple[PhonopyAtoms, NDArray[np.int64], NDArray[np.int64]]:
    """Trim overlapping atoms."""
    tcell = TrimmedCell(
        relative_axes,
        cell,
        check_overlap=check_overlap,
        symprec=symprec,
        positions_to_reorder=positions_to_reorder,
    )
    return tcell.copy(), tcell.extracted_atoms, tcell.mapping_table


#
# Delaunay and Niggli reductions
#
def get_reduced_bases(
    lattice: NDArray[np.double],
    method: Literal["niggli", "delaunay"] = "niggli",
    tolerance: float = 1e-5,
) -> NDArray[np.double]:
    """Search kinds of shortest basis vectors.

    Parameters
    ----------
    lattice : ndarray or list of list
        Basis vectors by row vectors, [a, b, c]^T
        shape=(3, 3)
    method : Literal["niggli", "delaunay"]
        delaunay: Delaunay reduction
        niggli: Niggli reduction
    tolerance : float
        Tolerance to find shortest basis vectors

    Returns
    -------
    Reduced basis as row vectors, [a_red, b_red, c_red]^T
        dtype='double'
        shape=(3, 3)
        order='C'

    """
    if method == "niggli":
        red_cell = spglib.niggli_reduce(lattice, eps=tolerance)
    else:
        red_cell = spglib.delaunay_reduce(lattice, eps=tolerance)
    if red_cell is None:
        raise RuntimeError(f"{method} reduction failed.")
    return red_cell


def get_smallest_vectors(
    supercell_bases: NDArray[np.double],
    supercell_pos: NDArray[np.double],
    primitive_pos: NDArray[np.double],
    store_dense_svecs: bool = True,
    symprec: float = 1e-5,
    lang: Literal["C", "Rust"] = "Rust",
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return shortest vectors and multiplicities.

    See the details at `ShortestPairs`.

    """
    spairs = ShortestPairs(
        supercell_bases,
        supercell_pos,
        primitive_pos,
        store_dense_svecs=store_dense_svecs,
        symprec=symprec,
        lang=lang,
    )
    return spairs.shortest_vectors, spairs.multiplicities


class ShortestPairs:
    """Find shortest atomic pair vectors.

    Attributes
    ----------
    shortest_vectors : ndarray
    multiplicities : ndarray

    Note
    ----
    Shortest vectors from an atom in primitive cell to an atom in
    supercell in the fractional coordinates of primitive cell. If an
    atom in supercell is on the border centered at an atom in
    primitive and there are multiple vectors that have the same
    distance (up to tolerance) and different directions, several
    shortest vectors are stored.
    In fact, this method is not limited to search shortest vectors between
    sueprcell atoms and primitive cell atoms, but can be used to measure
    shortest vectors between atoms in periodic supercell lattice frame.

    """

    def __init__(
        self,
        supercell_bases: NDArray[np.double],
        supercell_pos: NDArray[np.double],
        primitive_pos: NDArray[np.double],
        store_dense_svecs: bool = True,
        symprec: float = 1e-5,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        supercell_bases : array_like
            Supercell basis vectors as row vectors, (a, b, c)^T. Must be
            dtype='double', shape=(3, 3)
        supercell_pos : array_like
            Atomic positions in fractional coordinates of supercell.
            dtype='double', shape=(size_super, 3)
        primitive_pos : array_like
            Atomic positions in fractional coordinates of supercell. Note that
            not in fractional coordinates of primitive cell.  dtype='double',
            shape=(size_prim, 3)
        store_dense_svecs_: bool, optional
            ``shortest_vectors`` are stored in the dense data structure.
            Default is True.
        symprec : float, optional
            Tolerance to find equal distances of vectors. Default is 1e-5.
        lang : {"C", "Rust"}, optional
            Backend selector for the underlying ``gsv_set_smallest_vectors``
            kernel.  Default is ``"C"``.

        """
        self._supercell_bases = supercell_bases
        self._supercell_pos = supercell_pos
        self._primitive_pos = primitive_pos
        self._symprec = symprec
        self._lang: Literal["C", "Rust"] = resolve_lang(lang)
        log_dispatch(self._lang, "ShortestPairs.__init__")

        if store_dense_svecs:
            svecs, multi = self._run_dense()
            self._smallest_vectors = svecs
            self._multiplicities = multi
        else:
            svecs, multi = self._run_sparse()
            self._smallest_vectors = svecs
            self._multiplicities = multi

    @property
    def shortest_vectors(self) -> NDArray[np.double]:
        """Return shortest_vectors.

        See details in `ShortestPairs_run_sparse()` (`store_dense_svecs=True`)
        or `ShortestPairs._run_dense()` (`store_dense_svecs=False`).

        """
        return self._smallest_vectors

    @property
    def multiplicities(self) -> NDArray[np.int64]:
        """Return multiplicities.

        See details in `ShortestPairs_run_sparse()` (`store_dense_svecs=True`)
        or `ShortestPairs._run_dense()` (`store_dense_svecs=False`).

        """
        return self._multiplicities

    def _run_dense(self) -> tuple[NDArray[np.double], NDArray[np.int64]]:
        """Find shortest atomic pair vectors.

        Returns
        -------
        shortest_vectors : ndarray
            Shortest vectors in supercell coordinates.
            shape=(sum(multiplicities[:, :, 0], 3), dtype='double'
        multiplicities : ndarray
            Number of equidistance shortest vectors. The last dimension
            indicates [0] multipliticy at the pair of atoms in the supercell
            and primitive cell, and [1] integral of multiplicities to
            this pair, i.e., which indicates address used in
            `shortest_vectors`.
            shape=(size_super, size_prim, 2), dtype='int64'

        """
        (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        ) = self._transform_cell_basis()

        # Phase1 : Set multiplicity.
        # shortest_vectors is a dummy array.
        shortest_vectors = np.zeros((1, 3), dtype="double", order="C")
        multiplicity = np.zeros(
            (len(supercell_fracs), len(primitive_fracs), 2), dtype="int64", order="C"
        )
        reduced_bases_T = np.array(reduced_bases.T, dtype="double", order="C")
        trans_mat_inv_T = np.array(trans_mat_inv.T, dtype="int64", order="C")
        if self._lang == "Rust":
            import phonors  # type: ignore[import-untyped]

            phonors.gsv_set_smallest_vectors_dense(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                1,
                self._symprec,
            )
        else:
            import phonopy._phonopy as phonoc  # type: ignore

            phonoc.gsv_set_smallest_vectors_dense(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                1,
                self._symprec,
            )

        # Phase 2 : Set shortest_vectors.
        shortest_vectors = np.zeros(
            (np.sum(multiplicity[:, :, 0]), 3), dtype="double", order="C"
        )
        if self._lang == "Rust":
            phonors.gsv_set_smallest_vectors_dense(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                0,
                self._symprec,
            )
        else:
            phonoc.gsv_set_smallest_vectors_dense(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                0,
                self._symprec,
            )

        return shortest_vectors, multiplicity

    def _run_sparse(self) -> tuple[NDArray[np.double], NDArray[np.int64]]:
        """Find shortest atomic pair vectors.

        Returns
        -------
        shortest_vectors : ndarray
            Shortest vectors in supercell coordinates. The 27 in shape is the
            possible maximum number of elements.
            shape=(size_super, size_prim, 27, 3), dtype='double'
        multiplicities : ndarray
            Number of equidistance shortest vectors
            shape=(size_super, size_prim), dtype='int64'

        """
        (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        ) = self._transform_cell_basis()

        # This shortest_vectors is already used at many locations.
        # Therefore the constant number 27 = 3*3*3 can not be easily changed.
        shortest_vectors = np.zeros(
            (len(supercell_fracs), len(primitive_fracs), 27, 3),
            dtype="double",
            order="C",
        )
        multiplicity = np.zeros(
            (len(supercell_fracs), len(primitive_fracs)), dtype="int64", order="C"
        )
        reduced_bases_T = np.array(reduced_bases.T, dtype="double", order="C")
        trans_mat_inv_T = np.array(trans_mat_inv.T, dtype="int64", order="C")
        if self._lang == "Rust":
            import phonors  # type: ignore[import-untyped]

            phonors.gsv_set_smallest_vectors_sparse(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                self._symprec,
            )
        else:
            import phonopy._phonopy as phonoc  # type: ignore

            phonoc.gsv_set_smallest_vectors_sparse(
                shortest_vectors,
                multiplicity,
                supercell_fracs,
                primitive_fracs,
                lattice_points,
                reduced_bases_T,
                trans_mat_inv_T,
                self._symprec,
            )

        return shortest_vectors, multiplicity

    def _transform_cell_basis(
        self,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.double],
        NDArray[np.double],
        NDArray[np.int64],
        NDArray[np.double],
    ]:
        reduced_cell_method = "niggli"
        reduced_bases = get_reduced_bases(
            self._supercell_bases, method=reduced_cell_method, tolerance=self._symprec
        )
        trans_mat_float = np.dot(self._supercell_bases, np.linalg.inv(reduced_bases))
        trans_mat = np.rint(trans_mat_float).astype(int)
        assert (np.abs(trans_mat_float - trans_mat) < 1e-8).all()
        trans_mat_inv_float = np.linalg.inv(trans_mat)
        trans_mat_inv = np.rint(trans_mat_inv_float).astype(int)
        assert (np.abs(trans_mat_inv_float - trans_mat_inv) < 1e-8).all()

        # Reduce all positions into the cell formed by the reduced bases.
        supercell_fracs = np.dot(self._supercell_pos, trans_mat)
        supercell_fracs -= np.rint(supercell_fracs)
        supercell_fracs = np.array(supercell_fracs, dtype="double", order="C")
        primitive_fracs = np.dot(self._primitive_pos, trans_mat)
        primitive_fracs -= np.rint(primitive_fracs)
        primitive_fracs = np.array(primitive_fracs, dtype="double", order="C")

        # For each vector, we will need to consider all nearby images in the
        # reduced bases. The lattice points at which supercell images are
        # searched are composed by linear combinations of three vectors in (0,
        # a, b, c, -a-b-c, -a, -b, -c, a+b+c). There are finally 65 lattice
        # points. There is no proof that this is enough.
        lattice_1D = (-1, 0, 1)
        lattice_4D = np.array(
            [
                [i, j, k, ll]
                for i in lattice_1D
                for j in lattice_1D
                for k in lattice_1D
                for ll in lattice_1D
            ],
            dtype="int64",
            order="C",
        )
        bases = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
        lattice_points = np.dot(lattice_4D, bases)
        lattice_points = np.array(
            np.unique(lattice_points, axis=0), dtype="int64", order="C"
        )

        return (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        )


def sparse_to_dense_svecs(
    svecs: NDArray[np.double], multi: NDArray[np.int64]
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Convert sparse svecs to dense svecs."""
    dmulti = np.zeros(multi.shape + (2,), dtype="int64", order="C")
    dmulti[:, :, 0] = multi
    dsvecs = np.zeros((multi.sum(), 3), dtype="double", order="C")
    adrs = 0
    for s_i in range(multi.shape[0]):
        for p_i in range(multi.shape[1]):
            dmulti[s_i, p_i, 1] = adrs
            m = multi[s_i, p_i]
            dsvecs[adrs : (adrs + m)] = svecs[s_i, p_i, :m]
            adrs += multi[s_i, p_i]
    return dsvecs, dmulti


def dense_to_sparse_svecs(
    svecs: NDArray[np.double], multi: NDArray[np.int64]
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Convert dense svecs to sparse svecs."""
    ssvecs = np.zeros(
        (multi.shape[0], multi.shape[1], 27, 3),
        dtype="double",
        order="C",
    )
    smulti = np.zeros(multi.shape[:2], dtype="int64", order="C")
    smulti[:, :] = multi[:, :, 0]
    for s_i in range(multi.shape[0]):
        for p_i in range(multi.shape[1]):
            m = multi[s_i, p_i]
            ssvecs[s_i, p_i, : m[0]] = svecs[m[1] : m[0] + m[1]]
    return ssvecs, smulti


def compute_all_sg_permutations(
    positions: NDArray[np.double],  # scaled positions
    rotations: NDArray[np.int64],  # scaled
    translations: NDArray[np.double],  # scaled
    lattice: NDArray[np.double],  # column vectors
    symprec: float,
    types: NDArray[np.int64] | Sequence[int] | None,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Compute permutations for space group operations.

    See 'compute_permutation_for_rotation' for more info.

    Parameters
    ----------
    positions : ndarray
        Scaled positions (like PhonopyAtoms.scaled_positions) before applying
        the space group operation
    rotations : ndarray
        Matrix (rotation) parts of space group operations
        shape=(len(operations), 3, 3), dtype='int64'
    translations : ndarray
        Vector (translation) parts of space group operations
        shape=(len(operations), 3), dtype='double'
    lattice : ndarray
        Basis vectors in column vectors (like PhonopyAtoms.cell.T)
    symprec : float
        Symmetry tolerance of the distance unit
    types : array_like or None
        Per-atom integer types (required). When given, atoms are matched
        only within the same type, which disambiguates co-located atoms of
        different species (e.g. a species-resolved cell). The types must
        match the partition the operations were found from. Pass None to
        match by position alone. See ``compute_permutation_for_rotation``.
    lang : {"C", "Rust"}
        Backend for the inner permutation matcher. Default is "Rust".

    Returns
    -------
    perms : ndarray
        shape=(len(operations), len(positions)), dtype='int64', order='C'

    """
    out = []  # Finally the shape is fixed as (num_sym, num_pos_of_supercell).
    for sym, t in zip(rotations, translations, strict=True):
        rotated_positions = np.dot(positions, sym.T) + t
        out.append(
            compute_permutation_for_rotation(
                positions, rotated_positions, lattice, symprec, types, lang=lang
            )
        )
    return np.array(out, dtype="int64", order="C")


def compute_permutation_for_rotation(
    positions_a: NDArray[np.double],
    positions_b: NDArray[np.double],
    lattice: NDArray[np.double],
    symprec: float,
    types: NDArray[np.int64] | Sequence[int] | None,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Get the overall permutation such that.

        positions_a[perm[i]] == positions_b[i]   (modulo the lattice)

    or in numpy speak,

        positions_a[perm] == positions_b   (modulo the lattice)

    This version is optimized for the case where positions_a and positions_b
    are related by a rotation.

    Parameters
    ----------
    positions_a : ndarray
        Scaled positions (like PhonopyAtoms.scaled_positions) before applying
        the space group operation
    positions_b : ndarray
        Scaled positions (like PhonopyAtoms.scaled_positions) after applying
        the space group operation
    lattice : ndarray
        Basis vectors in column vectors (like PhonopyAtoms.cell.T)
    symprec : float
        Symmetry tolerance of the distance unit
    types : array_like or None
        Per-atom integer types (required). When given, atoms are matched
        only within the same type. This disambiguates co-located atoms of
        different species, which position-only matching cannot resolve
        (e.g. Ge and Sn at the same site in a species-resolved cell). The
        space group operations must map each type onto itself, which holds
        when the operations were found from the same types. Pass None to
        match by position alone.
    lang : {"C", "Rust"}
        Backend for the inner permutation matcher. Default is "Rust".

    Returns
    -------
    perm : ndarray
        A list of atomic indices that maps atoms before the space group
        operation to those after as explained above.
        shape=(len(positions), ), dtype=int

    """
    if types is None:
        return _match_positions_for_rotation(
            positions_a, positions_b, lattice, symprec, lang=lang
        )

    # Match within each type class. positions_b[i] is the rotated position
    # of atom i and shares its type, so the same index set selects a type
    # class on both sides. perm[idx] = idx[sub_perm] composes the per-class
    # result into the full permutation.
    types_arr = np.asarray(types)
    perm = np.empty(len(positions_a), dtype="int64")
    for t in np.unique(types_arr):
        idx = np.where(types_arr == t)[0]
        sub_perm = _match_positions_for_rotation(
            positions_a[idx], positions_b[idx], lattice, symprec, lang=lang
        )
        perm[idx] = idx[sub_perm]
    # Each type fills a disjoint slice of perm; together they must form a
    # bijection of all atoms.
    assert np.array_equal(np.sort(perm), np.arange(len(perm)))
    return perm


def _match_positions_for_rotation(
    positions_a: NDArray[np.double],
    positions_b: NDArray[np.double],
    lattice: NDArray[np.double],
    symprec: float,
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Return the by-position permutation for two rotation-related cells.

    perm satisfies ``positions_a[perm] == positions_b`` (modulo the
    lattice), matching atoms by position alone. This is the leaf matcher
    used by :func:`compute_permutation_for_rotation` within a single type
    class (or when no types are given).

    """

    def sort_by_lattice_distance(
        fracs: NDArray[np.double],
    ) -> tuple[NDArray[np.int64], NDArray[np.double]]:
        """Sort atoms by distance.

        Sort both sides by some measure which is likely to produce a small
        maximum value of (sorted_rotated_index - sorted_original_index).
        The C code is optimized for this case, reducing an O(n^2)
        search down to ~O(n). (for O(n log n) work overall, including the sort)

        We choose distance from the nearest bravais lattice point as our measure.

        """
        carts = np.dot(fracs - np.rint(fracs), lattice.T)
        perm = np.argsort(np.sum(carts**2, axis=1))
        sorted_fracs = np.array(fracs[perm], dtype="double", order="C")
        return perm, sorted_fracs

    perm_a, sorted_a = sort_by_lattice_distance(positions_a)
    perm_b, sorted_b = sort_by_lattice_distance(positions_b)

    perm_between = _compute_permutation(sorted_a, sorted_b, lattice, symprec, lang=lang)

    # Compose all of the permutations for the full permutation.
    #
    # Note the following properties of permutation arrays:
    #
    # 1. Inverse:         if  x[perm] == y  then  x == y[argsort(perm)]
    # 2. Associativity:   x[p][q] == x[p[q]]
    return perm_a[perm_between][np.argsort(perm_b)]


def _compute_permutation(
    positions_a: NDArray[np.double],
    positions_b: NDArray[np.double],
    lattice: NDArray[np.double],
    symprec: float,  # scaled positions  # column vectors
    lang: Literal["C", "Rust"] = "Rust",
) -> NDArray[np.int64]:
    """Return mapping defined by positions_a[perm[i]] == positions_b[i].

    Version of `_compute_permutation_for_rotation` which just directly
    calls the C/Rust function, without any conditioning of the data.
    Skipping the conditioning step makes this EXTREMELY slow on large
    structures.

    """
    permutation = np.zeros(shape=(len(positions_a),), dtype="int64")

    def permutation_error():
        raise ValueError(
            "Input forces are not enough to calculate force constants, "
            "or something wrong (e.g. crystal structure does not match)."
        )

    try:
        if lang == "Rust":
            import phonors  # type: ignore

            log_dispatch(lang, "_compute_permutation")
            kernel = phonors.compute_permutation
        else:
            import phonopy._phonopy as phonoc  # type: ignore

            log_dispatch(lang, "_compute_permutation")
            kernel = phonoc.compute_permutation

        tolerance = symprec
        for _ in range(20):
            is_found = kernel(permutation, lattice, positions_a, positions_b, tolerance)
            if is_found:
                break
            else:
                tolerance *= 1.05

        if tolerance / symprec > 1.5:
            import warnings

            msg = (
                "Crystal structure is distorted in a tricky way so that "
                "phonopy could not handle the crystal symmetry properly. "
                "It is recommended to symmetrize crystal structure well "
                "and then re-start phonon calculation from scratch."
            )
            warnings.warn(msg, stacklevel=2)

        if not is_found:
            permutation_error()

    except ImportError:
        for i, pos_b in enumerate(positions_b):
            diffs = positions_a - pos_b
            diffs -= np.rint(diffs)
            diffs = np.dot(diffs, lattice.T)

            possible_j = np.nonzero(np.sqrt(np.sum(diffs**2, axis=1)) < symprec)[0]
            if len(possible_j) != 1:
                permutation_error()

            permutation[i] = possible_j[0]

        if -1 in permutation:
            permutation_error()

    return permutation


#
# Other tiny tools
#
def get_angles(
    lattice: NDArray[np.double] | Sequence[Sequence[float]], is_radian: bool = False
) -> tuple[float, float, float]:
    """Return angles between basis vectors.

    Parameters
    ----------
    lattice : array_like
        Basis vectors given as row vectors.
    is_radian : bool
        Angles are return in radian when True. Otherwise in degree.

    Returns
    -------
    tuple[float, float, float]
        alpha, beta, gamma in either degree or radian.

    """
    a, b, c = get_cell_parameters(lattice)
    alpha = np.arccos(np.vdot(lattice[1], lattice[2]) / b / c)
    beta = np.arccos(np.vdot(lattice[2], lattice[0]) / c / a)
    gamma = np.arccos(np.vdot(lattice[0], lattice[1]) / a / b)

    if is_radian:
        return alpha, beta, gamma
    else:
        return alpha / np.pi * 180, beta / np.pi * 180, gamma / np.pi * 180


def get_cell_parameters(
    lattice: NDArray[np.double] | Sequence[Sequence[float]],
) -> NDArray[np.double]:
    """Return basis vector lengths.

    Parameters
    ----------
    lattice : array_like
        Basis vectors given as row vectors
        shape=(3, 3), dtype='double'

    Returns
    -------
    ndarray, shape=(3,), dtype='double'

    """
    return np.sqrt(np.dot(lattice, np.transpose(lattice)).diagonal())


def get_cell_matrix(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    is_radian: bool = False,
) -> NDArray[np.double]:
    """Return basis vectors in another orientation.

    Parameters
    ----------
    a, b, c : float
        Basis vector lengths.
    alpha, beta, gamm : float
        Angles between basis vectors in radian.

    Returns
    -------
    ndarray
        shape=(3, 3), dtype='double', order='C'.
        [[a_x,   0,   0],
         [b_x, b_y,   0],
         [c_x, c_y, c_z]]

    """
    if not is_radian:
        alpha *= np.pi / 180
        beta *= np.pi / 180
        gamma *= np.pi / 180
    b1 = np.cos(gamma)
    b2 = np.sin(gamma)
    b3 = 0.0
    c1 = np.cos(beta)
    c2 = (2 * np.cos(alpha) + b1**2 + b2**2 - 2 * b1 * c1 - 1) / (2 * b2)
    c3 = np.sqrt(1 - c1**2 - c2**2)
    lattice = np.zeros((3, 3), dtype="double", order="C")
    lattice[0, 0] = a
    lattice[1] = np.array([b1, b2, b3]) * b
    lattice[2] = np.array([c1, c2, c3]) * c
    return lattice


def get_cell_matrix_from_lattice(
    lattice: NDArray[np.double] | Sequence[Sequence[float]],
) -> NDArray[np.double]:
    """Return basis vectors in another orientation.

    Parameters
    ----------
    lattice : array_like
        Basis vectors given as row vectors
        shape=(3, 3), dtype='double'

    Returns
    -------
    ndarray
        shape=(3, 3), dtype='double', order='C'.
        [[a_x,   0,   0],
         [b_x, b_y,   0],
         [c_x, c_y, c_z]]

    """
    alpha, beta, gamma = get_angles(lattice, is_radian=True)
    a, b, c = get_cell_parameters(lattice)
    return get_cell_matrix(a, b, c, alpha, beta, gamma, is_radian=True)


def determinant(
    m: Sequence[Sequence[int]]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | NDArray[np.int64],
) -> float | int:
    """Compute determinant."""
    return (
        m[0][0] * m[1][1] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
        + m[0][1] * m[1][2] * m[2][0]
        - m[0][1] * m[1][0] * m[2][2]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0]
    )


class PrimitiveMatrixAutoDefaultWarning(UserWarning):
    """Issued when default ``primitive_matrix='auto'`` produces a non-identity cell.

    The phonopy v3 default for ``primitive_matrix`` was the identity
    matrix; v4 changed it to ``'auto'`` (detect the primitive cell from
    crystal symmetry).  When the auto-detected matrix is not the identity,
    the resulting q-point convention and folded-band layout differ from v3.
    This warning is emitted so users running old scripts notice the change
    instead of silently getting different numbers.  Pass
    ``primitive_matrix='P'`` (or ``--pa P`` on the command line) to restore
    the v3 behaviour.

    """


def warn_if_primitive_matrix_auto_changed_cell(
    primitive_matrix_input: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None,
    resolved_primitive_matrix: NDArray[np.double],
) -> None:
    """Emit ``PrimitiveMatrixAutoDefaultWarning`` if relevant.

    The warning fires only when the caller relied on the ``'auto'``
    default (``None`` or ``'auto'``) AND the auto-detected matrix is not
    the identity.  Explicit choices (``'P'``, an explicit matrix, etc.)
    are silent.

    """
    if primitive_matrix_input is not None and not (
        isinstance(primitive_matrix_input, str) and primitive_matrix_input == "auto"
    ):
        return
    if np.allclose(resolved_primitive_matrix, np.eye(3), atol=1e-5):
        return
    rows = "\n".join(
        "  [" + ", ".join(f"{v: .5f}" for v in row) + "]"
        for row in resolved_primitive_matrix
    )
    msg = (
        "primitive_matrix defaulted to 'auto' and was resolved to a "
        "non-identity matrix:\n"
        f"{rows}\n"
        "This differs from phonopy v3, whose default was the identity "
        "matrix. Pass primitive_matrix='P' (or --pa P on the command "
        "line) to restore the v3 behaviour."
    )
    warnings.warn(msg, PrimitiveMatrixAutoDefaultWarning, stacklevel=3)


def get_primitive_matrix_with_auto(
    unitcell: PhonopyAtoms,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None,
    symprec: float = 1e-5,
    distinguish_symbol_index: bool = False,
) -> NDArray[np.double]:
    """Return primitive matrix that supports 'auto' option.

    ``None`` is treated as ``"auto"`` so that an unspecified primitive
    matrix is resolved from crystal symmetry.
    ``distinguish_symbol_index`` is forwarded to
    ``guess_primitive_matrix``.

    """
    if primitive_matrix is None or (
        isinstance(primitive_matrix, str) and primitive_matrix == "auto"
    ):
        return guess_primitive_matrix(
            unitcell,
            symprec=symprec,
            distinguish_symbol_index=distinguish_symbol_index,
        )
    elif isinstance(primitive_matrix, str):
        return get_primitive_matrix(primitive_matrix, symprec=symprec)
    else:
        return np.array(primitive_matrix, dtype="double", order="C")


def get_primitive_matrix(
    pmat: Literal["P", "F", "I", "A", "C", "R"]
    | Sequence[Sequence[float]]
    | NDArray[np.double]
    | None = None,
    symprec: float = 1e-5,
) -> NDArray[np.double] | None:
    """Find primitive matrix from primitive cell.

    None is equivalent to "P" but None is returned.

    Parameters
    ----------
    pmat : str, np.ndarray, Sequency, or None
        symbol of centring type: "P", "F", "I", "A", "C", "R"
        3x3 matrix (can be flattened, i.e., 9 elements)
    symprec : float
        Tolerance.

    Returns
    -------
    None or 3x3 np.ndarray representing transformation matrix to primitive cell.

    """
    if isinstance(pmat, str) and pmat in ("P", "F", "I", "A", "C", "R"):
        _pmat = get_primitive_matrix_by_centring(pmat)
    elif pmat is None:
        _pmat = None
    elif len(np.ravel(pmat)) == 9:
        matrix = np.reshape(pmat, (3, 3))
        if matrix.dtype.kind in ("i", "u", "f"):
            det = np.linalg.det(matrix)
            if symprec < det and det < 1 + symprec:
                _pmat = matrix
            else:
                msg = "Determinant of primitive_matrix has to be larger than 0"
                raise RuntimeError(msg)
    else:
        msg = (
            "primitive_matrix has to be a 3x3 matrix, None, 'auto', "
            "'P', 'F', 'I', 'A', 'C', or 'R'"
        )
        raise RuntimeError(msg)

    return _pmat


def get_primitive_matrix_by_centring(centring: str) -> NDArray[np.double]:
    """Return primitive matrix corresponding to centring."""
    if centring == "P":
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double", order="C")
    elif centring == "F":
        return np.array(
            [[0.0, 1.0 / 2, 1.0 / 2], [1.0 / 2, 0, 1.0 / 2], [1.0 / 2, 1.0 / 2, 0.0]],
            dtype="double",
            order="C",
        )
    elif centring == "I":
        return np.array(
            [
                [-1.0 / 2, 1.0 / 2, 1.0 / 2],
                [1.0 / 2, -1.0 / 2, 1.0 / 2],
                [1.0 / 2, 1.0 / 2, -1.0 / 2],
            ],
            dtype="double",
        )
    elif centring == "A":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0 / 2, -1.0 / 2], [0.0, 1.0 / 2, 1.0 / 2]],
            dtype="double",
            order="C",
        )
    elif centring == "C":
        return np.array(
            [[1.0 / 2, 1.0 / 2, 0], [-1.0 / 2, 1.0 / 2, 0], [0.0, 0.0, 1.0]],
            dtype="double",
            order="C",
        )
    elif centring == "R":
        return np.array(
            [
                [2.0 / 3, -1.0 / 3, -1.0 / 3],
                [1.0 / 3, 1.0 / 3, -2.0 / 3],
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
            ],
            dtype="double",
            order="C",
        )
    else:
        msg = "centring has to be 'P', 'F', 'I', 'A', 'C', or 'R'"
        raise RuntimeError(msg)


def guess_primitive_matrix(
    unitcell: PhonopyAtoms,
    symprec: float = 1e-5,
    distinguish_symbol_index: bool = False,
) -> NDArray[np.double]:
    """Guess primitive matrix from crystal symmetry.

    For unit cells with magnetic moments, the Hall number returned by
    spglib's magnetic symmetry dataset is used. For Type-IV magnetic
    space groups, this corresponds to the XSG (the unprimed subgroup
    that does not include anti-translations), so the resulting
    primitive cell preserves the input magnetic moments and may be
    larger than the crystallographic primitive cell. In this case a
    warning is emitted. See the documentation of ``PRIMITIVE_AXES =
    AUTO`` for details.

    Parameters
    ----------
    unitcell : PhonopyAtoms
        Unit cell.
    symprec : float
        Tolerance to find symmetry operations.
    distinguish_symbol_index : bool, optional
        When True, atoms whose symbols differ only in the numeric
        suffix ("Cl" vs "Cl1") are treated as distinct species, so the
        guessed primitive cell never merges them. Must be consistent
        with the ``Symmetry`` flag of the same name. Default is False.

    """
    if unitcell.magnetic_moments is None:
        dataset = spglib.get_symmetry_dataset(
            unitcell.totuple(distinguish_symbol_index=distinguish_symbol_index),  # type: ignore
            symprec=symprec,
        )
    else:
        dataset = spglib.get_magnetic_symmetry_dataset(
            unitcell.totuple(distinguish_symbol_index=distinguish_symbol_index),  # type: ignore
            symprec=symprec,
        )
        if isinstance(dataset, SpglibMagneticDataset) and dataset.msg_type == 4:
            import warnings

            msg = (
                "The input unit cell has a magnetic ordering that breaks "
                "some of the crystallographic translational symmetry "
                "(a Type-IV magnetic space group). Phonopy chose the "
                "primitive cell to preserve this magnetic ordering, which "
                "may be larger than the crystallographic primitive cell. "
                "See the `PRIMITIVE_AXES = AUTO` documentation for details."
            )
            warnings.warn(msg, stacklevel=2)

    if isinstance(dataset, (SpglibDataset, SpglibMagneticDataset)):
        tmat = dataset.transformation_matrix
        pmat = _get_primitive_matrix_from_dataset(dataset)
        return np.array(np.dot(np.linalg.inv(tmat), pmat), dtype="double", order="C")
    else:
        return np.eye(3, dtype="double", order="C")


def shape_supercell_matrix(
    smat: NDArray[np.int64] | Sequence[int] | Sequence[Sequence[int]] | None,
) -> NDArray[np.int64]:
    """Reshape supercell matrix."""
    if smat is None:
        _smat = np.eye(3, dtype="int64", order="C")
    elif len(np.ravel(smat)) == 3:
        _smat = np.diag(smat)
    elif len(np.ravel(smat)) == 9:
        _smat = np.reshape(smat, (3, 3))
    else:
        msg = "supercell_matrix shape has to be (3,) or (3, 3)"
        raise RuntimeError(msg)
    return _smat


def estimate_supercell_matrix(
    spglib_dataset: SpglibDataset | SpglibMagneticDataset,
    max_num_atoms: int = 120,
    max_iter: int = 100,
) -> list[int]:
    """Estimate supercell matrix from conventional cell.

    Diagonal supercell matrix is estimated from basis vector lengths
    and maximum number of atoms to be accepted. Supercell is assumed
    to be made from the standardized cell and to be closest to sphere
    under keeping lattice symmetry. For triclinic, monoclinic, and
    orthorhombic cells, multiplicities for a, b, c are not constrained
    by symmetry. For tetragonal and hexagonal cells, multiplicities
    for a and b are chosen to be the same, and for cubic cell, those
    of a, b, c are the same.

    Parameters
    ----------
    spglib_dataset : tuple
        Spglib symmetry dataset
    max_num_atoms : int, optional
        Maximum number of atoms in created supercell to be tolerated.

    Returns
    -------
    list of three integer numbers
        Multiplicities for a, b, c basis vectors, respectively.

    """
    spg_type = spglib.get_spacegroup_type(spglib_dataset.hall_number)
    if spg_type is None:
        raise RuntimeError("Space group type could not be determined from hall_number.")
    spg_num = spg_type.number
    num_atoms = len(spglib_dataset.std_types)
    lengths = get_cell_parameters(spglib_dataset.std_lattice)
    if spg_num <= 74:  # Triclinic, monoclinic, and orthorhombic
        multi = _get_multiplicity_abc(
            num_atoms, lengths, max_num_atoms, max_iter=max_iter
        )
    elif spg_num <= 194:  # Tetragonal and hexagonal
        multi = _get_multiplicity_ac(
            num_atoms, lengths, max_num_atoms, max_iter=max_iter
        )
    else:  # Cubic
        multi = _get_multiplicity_a(
            num_atoms, lengths, max_num_atoms, max_iter=max_iter
        )

    return multi


def estimate_supercell_matrix_from_pointgroup(
    pointgroup_number: int,
    lattice: NDArray | Sequence[Sequence[float]],
    max_num_cells: int = 120,
    max_iter: int = 100,
) -> list[int]:
    """Estimate supercell matrix from crystallographic point group.

    Parameters
    ----------
    pointgroup_number : int
        The number representing crystallographic number from 1 to 32.
    lattice : array_like
        Basis vectors given as row vectors.
        shape=(3, 3), dtype='double'
    max_num_cells : int, optional
        Maximum number of cells in created supercell to be tolerated.

    Returns
    -------
    list of three integer numbers
        Multiplicities for a, b, c basis vectors, respectively.

    """
    abc_lengths = get_cell_parameters(lattice)

    if pointgroup_number <= 8:  # Triclinic, monoclinic, and orthorhombic
        multi = _get_multiplicity_abc(1, abc_lengths, max_num_cells, max_iter=max_iter)
    elif pointgroup_number <= 27:  # Tetragonal and hexagonal
        multi = _get_multiplicity_ac(1, abc_lengths, max_num_cells, max_iter=max_iter)
    else:  # Cubic
        multi = _get_multiplicity_a(1, abc_lengths, max_num_cells, max_iter=max_iter)

    return multi


def _get_multiplicity_abc(
    num_atoms: int,
    lengths: NDArray[np.double] | Sequence[float],
    max_num_atoms: int,
    max_iter: int = 20,
) -> list[int]:
    multi = [1, 1, 1]

    for _ in range(max_iter):
        l_super = np.multiply(multi, lengths)
        min_index = np.argmin(l_super)
        multi[min_index] += 1
        if num_atoms * np.prod(multi) > max_num_atoms:
            multi[min_index] -= 1

    return multi


def _get_multiplicity_ac(
    num_atoms: int,
    lengths: NDArray[np.double] | Sequence[float],
    max_num_atoms: int,
    max_iter: int = 20,
) -> list[int]:
    multi = [1, 1]
    a = lengths[0]
    c = lengths[2]

    for _ in range(max_iter):
        l_super = np.multiply(multi, [a, c])
        min_index = np.argmin(l_super)
        multi[min_index] += 1
        if num_atoms * multi[0] ** 2 * multi[1] > max_num_atoms:
            multi[min_index] -= 1

    return [multi[0], multi[0], multi[1]]


def _get_multiplicity_a(
    num_atoms: int,
    lengths: NDArray[np.double] | Sequence[float],
    max_num_atoms: int,
    max_iter: int = 20,
) -> list[int]:
    multi = 1
    for _ in range(max_iter):
        multi += 1
        if num_atoms * multi**3 > max_num_atoms:
            multi -= 1

    return [multi, multi, multi]
