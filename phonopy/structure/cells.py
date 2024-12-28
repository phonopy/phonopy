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

import warnings
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import spglib

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.snf import SNF3x3
from phonopy.utils import get_dot_access_dataset


class Supercell(PhonopyAtoms):
    """Build supercell from supercell matrix and unit cell.

    Attributes
    ----------
    supercell_matrix : ndarray
    s2u_map : ndarray
    u2s_map : ndarray
    u2u_map : dicst

    """

    def __init__(self, unitcell, supercell_matrix, is_old_style=True, symprec=1e-5):
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
            This swithes the algorithms. See Note.
        symprec: float, optional
            Tolerance to find overlapping atoms in supercell cell. The default
            values is 1e-5.

        """
        self._is_old_style = is_old_style
        self._s2u_map = None
        self._u2s_map = None
        self._u2u_map = None
        self._supercell_matrix = np.array(supercell_matrix, dtype="intc")
        self._create_supercell(unitcell, symprec)

    @property
    def supercell_matrix(self):
        """Return supercell_matrix.

        Returns
        -------
        ndarray
            Supercell matrix.
            shape=(3, 3), dtype='intc'

        """
        return self._supercell_matrix

    def get_supercell_matrix(self):
        """Return supercell_matrix."""
        warnings.warn(
            "Supercell.get_supercell_matrix() is deprecated."
            "Use Supercell.supercell_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.supercell_matrix

    @property
    def s2u_map(self):
        """Return atomic index mapping table from supercell to unit cell.

        Each array index and the stored value correspond to the supercell atom
        and unit cell atom in supercell atomic indices in supercell atom index.

        Returns
        -------
        ndarray
            shape=(num_atoms_in_supercell, ), dtype='long'

        """
        return self._s2u_map

    def get_supercell_to_unitcell_map(self):
        """Return atomic index mapping table from supercell to unit cell."""
        warnings.warn(
            "Supercell.get_supercell_to_unitcell_map() is deprecated."
            "Use Supercell.s2u_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.s2u_map

    @property
    def u2s_map(self):
        """Return atomic index mapping table from unit cell to supercell.

        Each array index and the stored value correspond to the unit cell atom
        and supecell atom in supercell atom index.

        Returns
        -------
        ndarray
            shape=(num_atoms_in_unitcell, ), dtype='long'

        """
        return self._u2s_map

    def get_unitcell_to_supercell_map(self):
        """Return atomic index mapping table from unit cell to supercell."""
        warnings.warn(
            "Supercell.get_unitcell_to_supercell_map() is deprecated."
            "Use Supercell.u2s_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.u2s_map

    @property
    def u2u_map(self):
        """Return atomic index mapping table from unit cell to unit cell.

        Returns
        -------
        dict
            Each key and value correspond to supercell atom index and unit cell
            atom index to represent an atom in unit cell.

        """
        return self._u2u_map

    def get_unitcell_to_unitcell_map(self):
        """Return atomic index mapping table from unit cell to unit cell."""
        warnings.warn(
            "Supercell.get_unitcell_to_unitcell_map() is deprecated."
            "Use Supercell.u2s_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.u2u_map

    def _create_supercell(self, unitcell: PhonopyAtoms, symprec):
        mat = self._supercell_matrix
        if self._is_old_style:
            P = None
            multi = self._get_surrounding_frame(mat)
            # trim_fram is to trim overlapping atoms.
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
                snf.run()
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
            print("Supercell creation failed.")
            print(
                "Probably some atoms are overwrapped. "
                "The mapping table is given below."
            )
            print(mapping_table)
            super().__init__()
        else:
            super().__init__(
                symbols=supercell.symbols,
                masses=supercell.masses,
                magnetic_moments=supercell.magnetic_moments,
                scaled_positions=supercell.scaled_positions,
                cell=supercell.cell,
            )
            self._u2s_map = np.array(np.arange(num_uatom) * N, dtype="long")
            self._u2u_map = {j: i for i, j in enumerate(self._u2s_map)}
            self._s2u_map = np.array(u2sur_map[sur2s_map] * N, dtype="long")

    def _get_simple_supercell(self, unitcell: PhonopyAtoms, multi, P):
        if self._is_old_style:
            mat = np.diag(multi)
        else:
            mat = self._supercell_matrix

        # Scaled positions within the frame, i.e., create a supercell that
        # is made simply to multiply the input cell.
        positions = unitcell.scaled_positions
        symbols = unitcell.symbols
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
        symbols_multi = [s for s in symbols for _ in range(n_l)]
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
            symbols=symbols_multi,
            masses=masses_multi,
            magnetic_moments=magmoms_multi,
            scaled_positions=positions_multi,
            cell=np.dot(mat, lattice),
        )

        return simple_supercell, atom_map

    def _get_surrounding_frame(self, supercell_matrix):
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
        primitive_matrix,
        symprec=1e-5,
        store_dense_svecs=True,
        positions_to_reorder=None,
    ):
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
        positions_to_reorder : array_like, optional
            If atomic positions in a created primitive cell is known and the
            order of atoms is expected to be sure, these positions with the
            specific order is used after position matching between this data and
            generated positions. Default is None.

        """
        self._primitive_matrix = np.array(primitive_matrix, dtype="double", order="C")
        self._symprec = symprec
        self._store_dense_svecs = store_dense_svecs
        self._p2s_map = None
        self._s2p_map = None
        self._p2p_map = None
        self._smallest_vectors = None
        self._multiplicity = None
        self._atomic_permutations = None
        self._run(supercell, positions_to_reorder=positions_to_reorder)

    @property
    def primitive_matrix(self):
        """Return primitive_matrix.

        Returns
        -------
        ndarray
            Transformation matrix from supercell to primitive cell
            dtype='double'
            shape=(3,3)

        """
        return self._primitive_matrix

    def get_primitive_matrix(self):
        """Return primitive_matrix."""
        warnings.warn(
            "Primitive.get_primitive_matrix() is deprecated."
            "Use Primitive.primitive_matrix attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.primitive_matrix

    @property
    def p2s_map(self):
        """Return mapping table of atoms from primitive cell to supercell.

        Returns
        -------
        ndarray
            Mapping table from atoms in primitive cell to those in supercell.
            Supercell atomic indices are used.
            shape=(num_atoms_in_primitive_cell,), dtype='intc'

        """
        return self._p2s_map

    def get_primitive_to_supercell_map(self):
        """Return mapping table of atoms from primitive cell to supercell."""
        warnings.warn(
            "Primitive.get_primitive_to_supercell_map() is deprecated."
            "Use Primitive.p2s_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.p2s_map

    @property
    def s2p_map(self):
        """Return mapping table of atoms from supercell to primitive cells.

        Returns
        -------
        ndarray
            Mapping table from atoms in supercell cell to those in primitive
            cell.  Supercell atomic indices are used.
            shape=(num_atoms_in_supercell, ), dtype='intc'

        """
        return self._s2p_map

    def get_supercell_to_primitive_map(self):
        """Return mapping table of atoms from supercell to primitive cells."""
        warnings.warn(
            "Primitive.get_supercell_to_primitive_map() is deprecated."
            "Use Primitive.s2p_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.s2p_map

    @property
    def p2p_map(self):
        """Return mapping table of indices in supercell and primitive cell.

        Returns
        -------
        dict
            Mapping of primitive cell atoms in supercell to those in primitive.
            cell.
            ex. {0: 0, 4: 1}

        """
        return self._p2p_map

    def get_primitive_to_primitive_map(self):
        """Return mapping table of indices in supercell and primitive cell."""
        warnings.warn(
            "Primitive.get_primitive_to_primitive_map() is deprecated."
            "Use Primitive.p2p_map attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.p2p_map

    def get_smallest_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return shortest vectors and multiplicities.

        See also the docstring of `ShortestPairs`. The older less densen format
        is deprecated. The detailed explaination is found in `ShortestPairs`
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
                dtype='long' dtype='intc', order='C'.

        """
        return self._smallest_vectors, self._multiplicity

    @property
    def atomic_permutations(self):
        """Return atomic index permutations by pure translations.

        Returns
        -------
        ndarray
            Atomic position permutation by pure translations is represented by
            changes of indices.
            dtype='intc'
            shape=(num_trans, num_atoms_in_supercell)
            ex.       supercell atomic indices
                     [[0, 1, 2, 3, 4, 5, 6, 7],
               trans  [1, 2, 3, 0, 5, 6, 7, 4],
              indices [2, 3, 0, 1, 6, 7, 4, 5],
                      [3, 0, 1, 2, 7, 4, 5, 6]]

        """
        return self._atomic_permutations

    def get_atomic_permutations(self):
        """Return atomic index permutations by pure translations."""
        warnings.warn(
            "Primitive.get_atomic_permutations() is deprecated."
            "Use Primitive.atomic_permutations attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.atomic_permutations

    @property
    def store_dense_svecs(self):
        """Return whether shortest vectors are stored in dense array or not."""
        return self._store_dense_svecs

    def _run(self, supercell: PhonopyAtoms, positions_to_reorder=None):
        self._p2s_map = self._create_primitive_cell(
            supercell, positions_to_reorder=positions_to_reorder
        )
        self._s2p_map, self._p2p_map = self._map_atomic_indices(
            supercell.scaled_positions
        )
        (self._smallest_vectors, self._multiplicity) = self._get_smallest_vectors(
            supercell
        )
        self._atomic_permutations = self._get_atomic_permutations(supercell)

    def _create_primitive_cell(
        self, supercell: PhonopyAtoms, positions_to_reorder=None
    ):
        trimmed_cell, p2s_map, mapping_table = _trim_cell(
            self._primitive_matrix,
            supercell,
            symprec=self._symprec,
            positions_to_reorder=positions_to_reorder,
        )
        mapped_symbols = [supercell.symbols[i] for i in mapping_table]
        if supercell.symbols != mapped_symbols:
            msg = [
                "Atom symbol mapping failure.",
                "Primitive cell could not be created.",
            ]
            msg.append("Primitive cell:")
            for i, s in enumerate(trimmed_cell.symbols):
                msg.append(f"  {i + 1}: {s}")
            msg.append("Original cell:")
            for i, (s1, s2) in enumerate(zip(supercell.symbols, mapped_symbols)):
                msg.append(f"  {i + 1}: {s1} -> {s2}")
            raise RuntimeError("\n".join(msg))
        super().__init__(
            symbols=trimmed_cell.symbols,
            masses=trimmed_cell.masses,
            magnetic_moments=trimmed_cell.magnetic_moments,
            scaled_positions=trimmed_cell.scaled_positions,
            cell=trimmed_cell.cell,
        )
        return p2s_map

    def _map_atomic_indices(self, s_pos_orig):
        frac_pos = np.dot(s_pos_orig, np.linalg.inv(self._primitive_matrix).T)

        p2s_positions = frac_pos[self._p2s_map]
        s2p_map = []
        for s_pos in frac_pos:
            # Compute distances from s_pos to all positions in _p2s_map.
            frac_diffs = p2s_positions - s_pos
            frac_diffs -= np.rint(frac_diffs)
            cart_diffs = np.dot(frac_diffs, self.cell)
            distances = np.sqrt((cart_diffs**2).sum(axis=1))
            indices = np.where(distances < self._symprec)[0]
            assert len(indices) == 1
            s2p_map.append(self._p2s_map[indices[0]])

        s2p_map = np.array(s2p_map, dtype="intc")
        p2p_map = dict([(j, i) for i, j in enumerate(self._p2s_map)])

        return s2p_map, p2p_map

    def _get_atomic_permutations(self, supercell: PhonopyAtoms):
        positions = supercell.scaled_positions
        diff = positions - positions[self._p2s_map[0]]
        trans = np.array(
            diff[np.where(self._s2p_map == self._p2s_map[0])[0]],
            dtype="double",
            order="C",
        )
        rotations = np.array(
            [np.eye(3, dtype="intc")] * len(trans), dtype="intc", order="C"
        )
        atomic_permutations = compute_all_sg_permutations(
            positions,
            rotations,
            trans,
            np.array(supercell.cell.T, dtype="double", order="C"),
            self._symprec,
        )

        return atomic_permutations

    def _get_smallest_vectors(
        self, supercell: PhonopyAtoms
    ) -> tuple[np.ndarray, np.ndarray]:
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
        )
        trans_mat_float = np.dot(supercell_bases, np.linalg.inv(primitive_bases))
        trans_mat = np.rint(trans_mat_float).astype(int)
        assert (np.abs(trans_mat_float - trans_mat) < 1e-8).all()
        svecs = np.array(np.dot(svecs, trans_mat), dtype="double", order="C")
        return svecs, multi


class TrimmedCell(PhonopyAtoms):
    """Trim cell.

    Trimmed cell is self.

    Attributes
    ----------
    extracted_atoms : ndarray
    mapping_table : ndarray

    """

    def __init__(
        self,
        relative_axes,
        cell: PhonopyAtoms,
        positions_to_reorder=None,
        check_overlap=True,
        symprec=1e-5,
    ):
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
        positions_to_reorder : array_like
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
    def mapping_table(self):
        """Return mappping table.

        mapping_table : ndarray
            The atomic indices of 'extracted_atom's of all atoms in the input
            cell.
            shape=(len(cell), ), dtype='intc'

        """
        return self._mapping_table

    @property
    def extracted_atoms(self):
        """Return extracted atoms.

        Returns
        -------
        extracted_atoms : ndarray
            Indices of atomic indices of input cell that are in the trimmed
            cell.
            shape=(len(trimmed_cell), ), dtype='intc'

        """
        return self._extracted_atoms

    def _run(
        self,
        cell: PhonopyAtoms,
        relative_axes,
        positions_to_reorder,
        check_overlap,
        symprec,
    ):
        trimmed_lattice = np.dot(relative_axes.T, cell.cell)
        positions_in_new_lattice = np.dot(
            cell.scaled_positions, np.linalg.inv(relative_axes).T
        )
        positions_in_new_lattice -= np.floor(positions_in_new_lattice)

        (
            trimmed_positions,
            trimmed_symbols,
            trimmed_masses,
            trimmed_magmoms,
            extracted_atoms,
            mapping_table,
        ) = self._extract(
            positions_in_new_lattice,
            trimmed_lattice,
            cell.symbols,
            cell.masses,
            cell.magnetic_moments,
            check_overlap,
            symprec,
        )

        if positions_to_reorder is not None:
            ids = self._get_reorder_indices(
                positions_to_reorder, trimmed_positions, trimmed_lattice, symprec
            )
            trimmed_positions = trimmed_positions[ids]
            trimmed_symbols = [trimmed_symbols[i] for i in ids]
            if trimmed_masses is not None:
                trimmed_masses = trimmed_masses[ids]
            if trimmed_magmoms is not None:
                trimmed_magmoms = trimmed_magmoms[ids]
            extracted_atoms = extracted_atoms[ids]

        # scale is not always to become integer.
        scale = 1.0 / np.linalg.det(relative_axes)
        if len(cell) == int(np.rint(scale * len(trimmed_symbols))):
            super().__init__(
                symbols=trimmed_symbols,
                masses=trimmed_masses,
                magnetic_moments=trimmed_magmoms,
                scaled_positions=trimmed_positions,
                cell=trimmed_lattice,
            )
            self._extracted_atoms = np.array(extracted_atoms, dtype="intc")
            self._mapping_table = mapping_table
        else:
            raise RuntimeError("Remapping of atoms by TrimmedCell failed.")

    def _extract(
        self,
        positions_in_new_lattice,
        trimmed_lattice,
        symbols,
        masses,
        magmoms,
        check_overlap,
        symprec,
    ):
        num_atoms = 0
        extracted_atoms = []
        mapping_table = np.arange(len(positions_in_new_lattice), dtype="intc")
        trimmed_positions = np.zeros_like(positions_in_new_lattice)
        trimmed_symbols = []
        if masses is None:
            trimmed_masses = None
        else:
            trimmed_masses = []
        if magmoms is None:
            trimmed_magmoms = None
        else:
            trimmed_magmoms = []

        for i, pos in enumerate(positions_in_new_lattice):
            found_overlap = False
            if check_overlap and num_atoms > 0:
                diff = trimmed_positions[:num_atoms] - pos
                diff -= np.rint(diff)
                # Older numpy doesn't support axis argument.
                distances = np.sqrt(np.sum(np.dot(diff, trimmed_lattice) ** 2, axis=1))
                overlap_indices = np.where(distances < symprec)[0]
                if len(overlap_indices) > 0:
                    assert len(overlap_indices) == 1
                    found_overlap = True
                    mapping_table[i] = extracted_atoms[overlap_indices[0]]

            if not found_overlap:
                trimmed_positions[num_atoms] = pos
                num_atoms += 1
                trimmed_symbols.append(symbols[i])
                if masses is not None:
                    trimmed_masses.append(masses[i])
                if magmoms is not None:
                    trimmed_magmoms.append(magmoms[i])
                extracted_atoms.append(i)

        if trimmed_masses is not None:
            trimmed_masses = np.array(trimmed_masses, dtype="double")
        if trimmed_magmoms is not None:
            trimmed_magmoms = np.array(trimmed_magmoms, dtype="double", order="C")

        return (
            np.array(trimmed_positions[:num_atoms], dtype="double", order="C"),
            trimmed_symbols,
            trimmed_masses,
            trimmed_magmoms,
            np.array(extracted_atoms, dtype="intc"),
            mapping_table,
        )

    def _get_reorder_indices(
        self, positions, trimmed_positions, trimmed_lattice, symprec
    ):
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
    unitcell, supercell_matrix, is_old_style=True, symprec=1e-5
) -> Supercell:
    """Create supercell."""
    return Supercell(
        unitcell, supercell_matrix, is_old_style=is_old_style, symprec=symprec
    )


def get_primitive(
    supercell: PhonopyAtoms,
    primitive_matrix: Optional[Union[str, np.ndarray, Sequence]] = None,
    symprec=1e-5,
    store_dense_svecs=True,
    positions_to_reorder=None,
):
    """Create primitive cell."""
    return Primitive(
        supercell,
        get_primitive_matrix(primitive_matrix),
        symprec=symprec,
        store_dense_svecs=store_dense_svecs,
        positions_to_reorder=positions_to_reorder,
    )


def print_cell(cell: PhonopyAtoms, mapping=None, stars=None):
    """Show cell information."""
    lines = get_cell_lines(cell, mapping=mapping, stars=stars)
    print("\n".join(lines))


def get_cell_lines(cell: PhonopyAtoms, mapping=None, stars=None):
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


def isclose(
    a: PhonopyAtoms,
    b: PhonopyAtoms,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    with_arbitrary_order: bool = False,
    return_order: bool = False,
) -> Union[bool, list]:
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
        PosDefault is False.
    return_order : bool, optional
        See ``Returns`` below. Default is False. This can be only usable with
        ``with_arbitrary_order=True``.

    Returns
    -------
    bool (``return_order=False``)
        Whether two cells agree upto lattice translation of each atom.

    or

    list (``return_order=True``).
        A list of atom indices of cell-b in the index of cell-a.
        This means ``a.numbers[indices] == b.numbers``.

    """
    if len(a) != len(b):
        return False

    if not np.allclose(a.cell, b.cell, rtol=rtol, atol=atol):
        return False

    if with_arbitrary_order:
        indices = []
        for pos in b.scaled_positions:
            diff = a.scaled_positions - pos
            diff -= np.rint(diff)
            dist = (np.dot(diff, a.cell) ** 2).sum(axis=1)
            matches = np.where(dist < atol)[0]
            if len(matches) != 1:
                return False
            indices.append(matches[0])
        if (np.sort(indices) == np.arange(len(indices))).all() and (
            a.numbers[indices] == b.numbers
        ).all():
            if return_order:
                return indices
            return True
        else:
            return False
    else:
        if (a.numbers != b.numbers).any():
            return False

        diff = a.scaled_positions - b.scaled_positions
        diff -= np.rint(diff)
        dist = np.sqrt((np.dot(diff, a.cell) ** 2).sum(axis=1))
        if (dist > atol).any():
            return False
    return True


def is_primitive_cell(rotations):
    """Check if single identity operation exists in rotations or not.

    This is used for checking a cell is a primitive cell or not.

    """
    num_identity = 0
    identity = np.eye(3, dtype="intc")
    for r in rotations:
        if (r == identity).all():
            num_identity += 1
            if num_identity > 1:
                return False
    else:
        return True


def convert_to_phonopy_primitive(
    supercell: PhonopyAtoms, primitive: PhonopyAtoms
) -> Primitive:
    """Convert PhonopyAtoms primitive cell to the Primitive instance."""
    slat = supercell.cell.T
    plat = primitive.cell.T
    pmat = np.dot(np.linalg.inv(slat), plat)
    _primitive = get_primitive(supercell, pmat)
    if not isclose(primitive, _primitive):
        msg = "Input primitive cell and generated one are inconsistent."
        raise RuntimeError(msg)
    return _primitive


def _trim_cell(
    relative_axes,
    cell: PhonopyAtoms,
    check_overlap=True,
    symprec=1e-5,
    positions_to_reorder=None,
):
    """Trim overlapping atoms."""
    tcell = TrimmedCell(
        relative_axes,
        cell,
        check_overlap=check_overlap,
        symprec=symprec,
        positions_to_reorder=positions_to_reorder,
    )
    return (tcell.copy(), tcell.extracted_atoms, tcell.mapping_table)


#
# Delaunay and Niggli reductions
#
def get_reduced_bases(lattice, method="niggli", tolerance=1e-5):
    """Search kinds of shortest basis vectors.

    Parameters
    ----------
    lattice : ndarray or list of list
        Basis vectors by row vectors, [a, b, c]^T
        shape=(3, 3)
    method : str
        delaunay: Delaunay reduction
        niggli: Niggli reduction
    tolerance : float
        Tolerance to find shortest basis vecotrs

    Returns
    -------
    Reduced basis as row vectors, [a_red, b_red, c_red]^T
        dtype='double'
        shape=(3, 3)
        order='C'

    """
    if method == "niggli":
        return spglib.niggli_reduce(lattice, eps=tolerance)
    else:
        return spglib.delaunay_reduce(lattice, eps=tolerance)


def get_smallest_vectors(
    supercell_bases, supercell_pos, primitive_pos, store_dense_svecs=False, symprec=1e-5
):
    """Return shortest vectors and multiplicities.

    See the details at `ShortestPairs`.

    """
    spairs = ShortestPairs(
        supercell_bases,
        supercell_pos,
        primitive_pos,
        store_dense_svecs=store_dense_svecs,
        symprec=symprec,
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
        supercell_bases,
        supercell_pos,
        primitive_pos,
        store_dense_svecs=True,
        symprec=1e-5,
    ):
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
            not in fractional coodinates of primitive cell.  dtype='double',
            shape=(size_prim, 3)
        store_dense_svecs_: bool, optional
            ``shortest_vectors`` are stored in the dense data structure.
            Default is True.
        symprec : float, optional
            Tolerance to find equal distances of vectors. Default is 1e-5.

        """
        self._supercell_bases = supercell_bases
        self._supercell_pos = supercell_pos
        self._primitive_pos = primitive_pos
        self._symprec = symprec

        if store_dense_svecs:
            svecs, multi = self._run_dense()
            self._smallest_vectors = svecs
            self._multiplicities = multi
        else:
            svecs, multi = self._run_sparse()
            self._smallest_vectors = svecs
            self._multiplicities = multi

    @property
    def shortest_vectors(self):
        """Return shortest_vectors.

        See details in `ShortestPairs_run_sparse()` (`store_dense_svecs=True`)
        or `ShortestPairs._run_dense()` (`store_dense_svecs=False`).

        """
        return self._smallest_vectors

    @property
    def multiplicities(self):
        """Return multiplicities.

        See details in `ShortestPairs_run_sparse()` (`store_dense_svecs=True`)
        or `ShortestPairs._run_dense()` (`store_dense_svecs=False`).

        """
        return self._multiplicities

    def _run_dense(self):
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
            shape=(size_super, size_prim, 2), dtype='long'

        """
        (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        ) = self._transform_cell_basis("long")

        # Phase1 : Set multiplicity.
        # shortest_vectors is a dummy array.
        shortest_vectors = np.zeros((1, 3), dtype="double", order="C")
        multiplicity = np.zeros(
            (len(supercell_fracs), len(primitive_fracs), 2), dtype="long", order="C"
        )
        import phonopy._phonopy as phonoc

        phonoc.gsv_set_smallest_vectors_dense(
            shortest_vectors,
            multiplicity,
            supercell_fracs,
            primitive_fracs,
            lattice_points,
            np.array(reduced_bases.T, dtype="double", order="C"),
            np.array(trans_mat_inv.T, dtype="long", order="C"),
            1,
            self._symprec,
        )

        # Phase 2 : Set shortest_vectors.
        shortest_vectors = np.zeros(
            (np.sum(multiplicity[:, :, 0]), 3), dtype="double", order="C"
        )
        phonoc.gsv_set_smallest_vectors_dense(
            shortest_vectors,
            multiplicity,
            supercell_fracs,
            primitive_fracs,
            lattice_points,
            np.array(reduced_bases.T, dtype="double", order="C"),
            np.array(trans_mat_inv.T, dtype="long", order="C"),
            0,
            self._symprec,
        )

        return shortest_vectors, multiplicity

    def _run_sparse(self):
        """Find shortest atomic pair vectors.

        Returns
        -------
        shortest_vectors : ndarray
            Shortest vectors in supercell coordinates. The 27 in shape is the
            possible maximum number of elements.
            shape=(size_super, size_prim, 27, 3), dtype='double'
        multiplicities : ndarray
            Number of equidistance shortest vectors
            shape=(size_super, size_prim), dtype='intc'

        """
        (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        ) = self._transform_cell_basis("intc")

        # This shortest_vectors is already used at many locations.
        # Therefore the constant number 27 = 3*3*3 can not be easily changed.
        shortest_vectors = np.zeros(
            (len(supercell_fracs), len(primitive_fracs), 27, 3),
            dtype="double",
            order="C",
        )
        multiplicity = np.zeros(
            (len(supercell_fracs), len(primitive_fracs)), dtype="intc", order="C"
        )
        import phonopy._phonopy as phonoc

        phonoc.gsv_set_smallest_vectors_sparse(
            shortest_vectors,
            multiplicity,
            supercell_fracs,
            primitive_fracs,
            lattice_points,
            np.array(reduced_bases.T, dtype="double", order="C"),
            np.array(trans_mat_inv.T, dtype="intc", order="C"),
            self._symprec,
        )

        return shortest_vectors, multiplicity

    def _transform_cell_basis(self, longdtype):
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
            dtype=longdtype,
            order="C",
        )
        bases = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
        lattice_points = np.dot(lattice_4D, bases)
        lattice_points = np.array(
            np.unique(lattice_points, axis=0), dtype=longdtype, order="C"
        )

        return (
            lattice_points,
            supercell_fracs,
            primitive_fracs,
            trans_mat_inv,
            reduced_bases,
        )


def sparse_to_dense_svecs(
    svecs: np.ndarray, multi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse svecs to dense svecs."""
    dmulti = np.zeros(multi.shape + (2,), dtype="long", order="C")
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
    svecs: np.ndarray, multi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert dense svecs to sparse svecs."""
    ssvecs = np.zeros(
        (multi.shape[0], multi.shape[1], 27, 3),
        dtype="double",
        order="C",
    )
    smulti = np.zeros(multi.shape[:2], dtype="intc", order="C")
    smulti[:, :] = multi[:, :, 0]
    for s_i in range(multi.shape[0]):
        for p_i in range(multi.shape[1]):
            m = multi[s_i, p_i]
            ssvecs[s_i, p_i, : m[0]] = svecs[m[1] : m[0] + m[1]]
    return ssvecs, smulti


def compute_all_sg_permutations(
    positions,  # scaled positions
    rotations,  # scaled
    translations,  # scaled
    lattice,  # column vectors
    symprec,
):
    """Compute permutations for space group operations.

    See 'compute_permutation_for_rotation' for more info.

    Parameters
    ----------
    positions : ndarray
        Scaled positions (like PhonopyAtoms.scaled_positions) before applying
        the space group operation
    rotations : ndarray
        Matrix (rotation) parts of space group operations
        shape=(len(operations), 3, 3), dtype='intc'
    translations : ndarray
        Vector (translation) parts of space group operations
        shape=(len(operations), 3), dtype='double'
    lattice : ndarray
        Basis vectors in column vectors (like PhonopyAtoms.cell.T)
    symprec : float
        Symmetry tolerance of the distance unit

    Returns
    -------
    perms : ndarray
        shape=(len(operations), len(positions)), dtype='intc', order='C'

    """
    out = []  # Finally the shape is fixed as (num_sym, num_pos_of_supercell).
    for sym, t in zip(rotations, translations):
        rotated_positions = np.dot(positions, sym.T) + t
        out.append(
            compute_permutation_for_rotation(
                positions, rotated_positions, lattice, symprec
            )
        )
    return np.array(out, dtype="intc", order="C")


def compute_permutation_for_rotation(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
    symprec: float,
):
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

    Returns
    -------
    perm : ndarray
        A list of atomic indices that maps atoms before the space group
        operation to those after as explained above.
        shape=(len(positions), ), dtype=int

    """

    def sort_by_lattice_distance(fracs):
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

    (perm_a, sorted_a) = sort_by_lattice_distance(positions_a)
    (perm_b, sorted_b) = sort_by_lattice_distance(positions_b)

    # Call the C code on our conditioned inputs.
    perm_between = _compute_permutation_c(sorted_a, sorted_b, lattice, symprec)

    # Compose all of the permutations for the full permutation.
    #
    # Note the following properties of permutation arrays:
    #
    # 1. Inverse:         if  x[perm] == y  then  x == y[argsort(perm)]
    # 2. Associativity:   x[p][q] == x[p[q]]
    return perm_a[perm_between][np.argsort(perm_b)]


def _compute_permutation_c(
    positions_a,
    positions_b,
    lattice,
    symprec,  # scaled positions  # column vectors
):
    """Return mapping defined by positions_a[perm[i]] == positions_b[i].

    Version of `_compute_permutation_for_rotation` which just directly
    calls the C function, without any conditioning of the data.
    Skipping the conditioning step makes this EXTREMELY slow on large
    structures.

    """
    permutation = np.zeros(shape=(len(positions_a),), dtype="intc")

    def permutation_error():
        raise ValueError(
            "Input forces are not enough to calculate force constants, "
            "or something wrong (e.g. crystal structure does not match)."
        )

    try:
        import phonopy._phonopy as phonoc

        tolerance = symprec
        for _ in range(20):
            is_found = phonoc.compute_permutation(
                permutation, lattice, positions_a, positions_b, tolerance
            )
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
def get_angles(lattice, is_radian: bool = False) -> tuple:
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


def get_cell_parameters(lattice) -> np.ndarray:
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


def get_cell_matrix(a, b, c, alpha, beta, gamma, is_radian=False) -> np.ndarray:
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


def get_cell_matrix_from_lattice(lattice) -> np.ndarray:
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


def determinant(m):
    """Compute determinant."""
    return (
        m[0][0] * m[1][1] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
        + m[0][1] * m[1][2] * m[2][0]
        - m[0][1] * m[1][0] * m[2][2]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0]
    )


def get_primitive_matrix(
    pmat: Optional[Union[str, np.ndarray, Sequence]] = None,
    symprec: float = 1e-5,
) -> Optional[Union[str, np.ndarray]]:
    """Find primitive matrix from primitive cell.

    None is equivalent to "P" but None is returned.

    Parameters
    ----------
    pmat : str, np.ndarray, Sequency, or None
        symbol of centring type: "P", "F", "I", "A", "C", "R"
        "auto" : estimates a centring type.
        3x3 matrix (can be flattened, i.e., 9 elements)
    symprec : float
        Tolerance.

    Returns
    -------
    None or 3x3 np.ndarray representing transformation matrix to primitive cell.

    """
    if isinstance(pmat, str) and pmat in ("P", "F", "I", "A", "C", "R", "auto"):
        if pmat == "auto":
            _pmat = pmat
        else:
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
                msg = "Determinant of primitive_matrix has to be larger " "than 0"
                raise RuntimeError(msg)
    else:
        msg = (
            "primitive_matrix has to be a 3x3 matrix, None, 'auto', "
            "'P', 'F', 'I', 'A', 'C', or 'R'"
        )
        raise RuntimeError(msg)

    return _pmat


def get_primitive_matrix_by_centring(centring) -> Optional[np.ndarray]:
    """Return primitive matrix corresponding to centring."""
    if centring == "P":
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double")
    elif centring == "F":
        return np.array(
            [[0.0, 1.0 / 2, 1.0 / 2], [1.0 / 2, 0, 1.0 / 2], [1.0 / 2, 1.0 / 2, 0.0]],
            dtype="double",
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
        )
    elif centring == "C":
        return np.array(
            [[1.0 / 2, 1.0 / 2, 0], [-1.0 / 2, 1.0 / 2, 0], [0.0, 0.0, 1.0]],
            dtype="double",
        )
    elif centring == "R":
        return np.array(
            [
                [2.0 / 3, -1.0 / 3, -1.0 / 3],
                [1.0 / 3, 1.0 / 3, -2.0 / 3],
                [1.0 / 3, 1.0 / 3, 1.0 / 3],
            ],
            dtype="double",
        )
    else:
        return None


def guess_primitive_matrix(unitcell: PhonopyAtoms, symprec: float = 1e-5):
    """Guess primitive matrix from crystal symmetry."""
    if unitcell.magnetic_moments is not None:
        msg = "Can not be used with the unit cell having magnetic moments."
        raise RuntimeError(msg)

    dataset = get_dot_access_dataset(
        spglib.get_symmetry_dataset(unitcell.totuple(), symprec=symprec)
    )
    tmat = dataset.transformation_matrix
    centring = dataset.international[0]
    pmat = get_primitive_matrix_by_centring(centring)
    return np.array(np.dot(np.linalg.inv(tmat), pmat), dtype="double", order="C")


def shape_supercell_matrix(smat: Optional[Union[Sequence, np.ndarray]]) -> np.ndarray:
    """Reshape supercell matrix."""
    if smat is None:
        _smat = np.eye(3, dtype="intc", order="C")
    elif len(np.ravel(smat)) == 3:
        _smat = np.diag(smat)
    elif len(np.ravel(smat)) == 9:
        _smat = np.reshape(smat, (3, 3))
    else:
        msg = "supercell_matrix shape has to be (3,) or (3, 3)"
        raise RuntimeError(msg)
    return _smat


def estimate_supercell_matrix(spglib_dataset, max_num_atoms=120, max_iter=100):
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
    spg_num = spglib_dataset.number
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
    pointgroup_number, lattice, max_num_cells=120, max_iter=100
):
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


def _get_multiplicity_abc(num_atoms, lengths, max_num_atoms, max_iter=20):
    multi = [1, 1, 1]

    for _ in range(max_iter):
        l_super = np.multiply(multi, lengths)
        min_index = np.argmin(l_super)
        multi[min_index] += 1
        if num_atoms * np.prod(multi) > max_num_atoms:
            multi[min_index] -= 1

    return multi


def _get_multiplicity_ac(num_atoms, lengths, max_num_atoms, max_iter=20):
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


def _get_multiplicity_a(num_atoms, lengths, max_num_atoms, max_iter=20):
    multi = 1
    for _ in range(max_iter):
        multi += 1
        if num_atoms * multi**3 > max_num_atoms:
            multi -= 1

    return [multi, multi, multi]
