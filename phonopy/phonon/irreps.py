"""Calculate irreducible representation from eigenvectors."""

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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.character_table import character_table
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import is_primitive_cell
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation


class IrReps:
    """Class to calculate irreducible representations from eigenvectors.

    Methods and terminologies used in this class may be easily found
    in textbooks such as
    - Group theory with applications in chemical physics by Patrick Jacobs
    - Symmetry and condensed matter physics by M. El-Batanouny and F. Wooten

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        qpoint: ArrayLike,
        primitive_symmetry: Symmetry,
        is_little_cogroup: bool = False,
        nac_q_direction: ArrayLike | None = None,
        factor: float | None = None,
        degeneracy_tolerance: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        self._is_little_cogroup = is_little_cogroup
        self._log_level = log_level

        self._qpoint = np.array(qpoint)
        self._primitive = dynamical_matrix.primitive
        self._symmetry_dataset = primitive_symmetry.dataset
        if not is_primitive_cell(self._symmetry_dataset.rotations):
            raise RuntimeError(
                "Non-primitve cell is used. Your unit cell may be transformed to "
                "a primitive cell by PRIMITIVE_AXIS tag."
            )
        self._symprec = primitive_symmetry.tolerance
        self._pointgroup_symbol = self._symmetry_dataset.pointgroup
        self._rotations_at_q, self._translations_at_q = self._get_rotations_at_q()

        self._freqs, self._eig_vecs = self._get_eigenvectors(
            dynamical_matrix, nac_q_direction, factor
        )
        # Degeneracy for irreps has to be determined considering character tables, too.
        # But currently only similarity of phonon frequencies is used to judge it.
        if degeneracy_tolerance is None:
            _degeneracy_tolerance = 1e-5
        else:
            _degeneracy_tolerance = degeneracy_tolerance
        self._degenerate_sets = get_degenerate_sets(
            self._freqs, cutoff=_degeneracy_tolerance
        )
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)

        self._transformation_matrix: NDArray
        self._conventional_rotations: NDArray
        self._ground_matrice: NDArray
        self._irreps: list[NDArray]
        self._characters: NDArray
        self._irrep_dims: NDArray
        self._run_irreps()

        # Only at Gamma point
        if (abs(self._qpoint) < self._symprec).all():
            irrep_labels = IrRepLabels(
                self._characters,
                self._conventional_rotations,
                self._pointgroup_symbol,
                self._log_level > 0,
            )
            self._rotation_symbols = irrep_labels.rotation_symbols
            self._ir_labels = irrep_labels.irrep_labels
        else:
            self._rotation_symbols = None
            self._ir_labels = None

    def _run_irreps(self):
        """Calculate irreps."""
        (
            self._transformation_matrix,
            self._conventional_rotations,
        ) = self._get_conventional_rotations()
        self._ground_matrices = self._get_ground_matrix()
        self._ddm.run(self._qpoint)
        self._irreps = self._get_irreps()
        self._characters, self._irrep_dims = self._get_characters()

    @property
    def band_indices(self) -> list[list[int]]:
        """Return band indices.

        Returns
        -------
        See docstring of ``degenerate_sets``.

        """
        return self._degenerate_sets

    @property
    def characters(self) -> NDArray:
        """Return characters of irreps."""
        return self._characters

    @property
    def eigenvectors(self) -> NDArray:
        """Return eigenvectors."""
        return self._eig_vecs

    @property
    def frequencies(self) -> NDArray:
        """Return frequencies in THz."""
        return self._freqs

    @property
    def irreps(self) -> list[NDArray]:
        """Return irreps."""
        return self._irreps

    @property
    def ground_matrices(self) -> NDArray:
        """Return ground matrices."""
        return self._ground_matrices

    @property
    def rotation_symbols(self) -> list[str] | None:
        """Return symbols assigned to rotation matrices."""
        return self._rotation_symbols

    @property
    def conventional_rotations(self) -> NDArray:
        """Return rotation matrices."""
        return self._conventional_rotations

    def get_projection_operators(
        self, idx_irrep: int, i: int | None = None, j: int | None = None
    ) -> NDArray:
        """Return projection operators."""
        if i is None or j is None:
            return self._get_character_projection_operators(idx_irrep)
        else:
            return self._get_projection_operators(idx_irrep, i, j)

    @property
    def qpoint(self) -> NDArray:
        """Return q-point."""
        return self._qpoint

    def show(self, show_irreps: bool = False):
        """Show irreps."""
        self._show(show_irreps)

    def write_yaml(self, show_irreps: bool = False):
        """Write irreps in yaml file."""
        self._write_yaml(show_irreps)

    def _get_eigenvectors(
        self,
        dynamical_matrix: DynamicalMatrix,
        nac_q_direction: ArrayLike | None,
        factor: float | None,
    ) -> tuple[NDArray, NDArray]:
        if factor is None:
            _factor = get_physical_units().DefaultToTHz
        else:
            _factor = factor
        dm = dynamical_matrix
        if isinstance(dm, DynamicalMatrixNAC):
            dm.run(self._qpoint, q_direction=nac_q_direction)
        else:
            dm.run(self._qpoint)
        assert dm.dynamical_matrix is not None
        eig_vals, eig_vecs = np.linalg.eigh(dm.dynamical_matrix)
        freqs = np.sqrt(abs(eig_vals)) * np.sign(eig_vals) * _factor
        return freqs, eig_vecs

    def _get_rotations_at_q(self) -> tuple[NDArray, NDArray]:
        """Return little group of q."""
        rotations_at_q = []
        trans_at_q = []
        for r, t in zip(
            self._symmetry_dataset.rotations, self._symmetry_dataset.translations
        ):
            diff = self._qpoint @ r - self._qpoint
            if (abs(diff - np.rint(diff)) < self._symprec).all():
                rotations_at_q.append(r)
                for i in range(3):
                    if np.abs(t[i] - 1) < self._symprec:
                        t[i] = 0.0
                trans_at_q.append(t)

        return np.array(rotations_at_q), np.array(trans_at_q)

    def _get_conventional_rotations(self) -> tuple[NDArray, NDArray]:
        rotations = self._rotations_at_q.copy()
        transformation_matrix = self._symmetry_dataset.transformation_matrix
        conventional_rotations = self._transform_rotations(
            transformation_matrix, rotations
        )

        return transformation_matrix, conventional_rotations

    def _transform_rotations(
        self, transformation_matrix: NDArray, rotations: NDArray
    ) -> NDArray:
        trans_rots = []

        for r in rotations:
            r_conv = similarity_transformation(transformation_matrix, r)
            trans_rots.append(np.rint(r_conv).astype(int))

        return np.array(trans_rots)

    def _get_ground_matrix(self) -> NDArray:
        matrices = []

        for r, t in zip(self._rotations_at_q, self._translations_at_q):
            lat = self._primitive.cell.T
            r_cart = similarity_transformation(lat, r)
            perm_mat = self._get_modified_permutation_matrix(r, t)
            matrices.append(np.kron(perm_mat, r_cart))

        return np.array(matrices)

    def _get_characters(self) -> tuple[NDArray, NDArray]:
        characters = []
        irrep_dims = []
        for irrep_Rs in self._irreps:
            characters.append([np.trace(rep) for rep in irrep_Rs])
            irrep_dims.append(len(irrep_Rs[0]))
        return np.array(characters), np.array(irrep_dims)

    def _get_modified_permutation_matrix(self, r: NDArray, t: NDArray) -> NDArray:
        num_atom = len(self._primitive)
        pos = self._primitive.scaled_positions
        matrix = np.zeros((num_atom, num_atom), dtype=complex)
        for i, p1 in enumerate(pos):
            p_rot = r @ p1 + t  # i -> j
            for j, p2 in enumerate(pos):
                diff = p_rot - p2  # Rx_i + t - x_j
                diff -= np.rint(diff)
                if (np.linalg.norm(diff @ self._primitive.cell) < self._symprec).all():
                    phase_factor = self._qpoint @ (np.linalg.inv(r) @ (p2 - t) - p2)
                    if self._is_little_cogroup:
                        phase_factor = t @ self._qpoint
                    matrix[j, i] = np.exp(2j * np.pi * phase_factor)
        return matrix

    def _get_irreps(self) -> list[NDArray]:
        eigvecs = self._eig_vecs.T
        irreps = []
        for band_indices in self._degenerate_sets:
            irrep_Rs = []
            for mat in self._ground_matrices:
                n_deg = len(band_indices)

                if n_deg == 1:
                    vec = eigvecs[band_indices[0]]
                    irrep_Rs.append([[np.vdot(vec, mat @ vec)]])
                    continue

                irrep_R = np.zeros((n_deg, n_deg), dtype=complex)
                for i, b_i in enumerate(band_indices):
                    vec_i = eigvecs[b_i]
                    for j, b_j in enumerate(band_indices):
                        vec_j = eigvecs[b_j]
                        irrep_R[i, j] = np.vdot(vec_i, mat @ vec_j)
                irrep_Rs.append(irrep_R)

            irreps.append(np.array(irrep_Rs))

        return irreps

    def _get_character_projection_operators(self, idx_irrep: int) -> NDArray:
        dim = self._irrep_dims[idx_irrep]
        chars = self._characters[idx_irrep]
        return (
            np.sum(
                [mat * char.conj() for mat, char in zip(self._ground_matrices, chars)],
                axis=0,
            )
            * dim
            / len(self._rotations_at_q)
        )

    def _get_projection_operators(self, idx_irrep: int, i: int, j: int) -> NDArray:
        dim = self._irrep_dims[idx_irrep]
        return (
            np.sum(
                [
                    mat * r[i, j].conj()
                    for mat, r in zip(self._ground_matrices, self._irreps[idx_irrep])
                ],
                axis=0,
            )
            * dim
            / len(self._rotations_at_q)
        )

    def _show(self, show_irreps: bool):
        print("")
        print("-------------------------------")
        print("  Irreducible representations")
        print("-------------------------------")
        print("q-point: %s" % self._qpoint)
        print("Point group: %s" % self._pointgroup_symbol)
        print("")

        if (np.abs(self._qpoint) < self._symprec).all():
            width = 6
            print("Original rotation matrices:")
            print("")
            _print_rotations(self._rotations_at_q, width=width)
        else:
            width = 4
            print("Original symmetry operations:")
            print("")
            _print_rotations(
                self._rotations_at_q, translations=self._translations_at_q, width=width
            )

        print("Transformation matrix:")
        print("")
        for v in self._transformation_matrix:
            print("%6.3f %6.3f %6.3f" % tuple(v))
        print("")
        print("Rotation matrices after transformation matrix:")
        print("")
        _print_rotations(
            self._conventional_rotations,
            rotation_symbols=self._rotation_symbols,
            width=width,
        )
        print("Character table:")
        print("")
        for i, deg_set in enumerate(self._degenerate_sets):
            text = "%3d (%8.3f): " % (deg_set[0] + 1, self._freqs[deg_set[0]])
            if self._ir_labels is None:
                print(text)
            elif self._ir_labels[i] is None:
                warning = "Not found. Try adjusting tolerance value in IRREPS."
                print("%s%s" % (text, warning))
            else:
                print("%s%s" % (text, self._ir_labels[i]))
            _print_characters(self._characters[i])
            print("")

        if show_irreps:
            self._show_irreps()

    def _show_irreps(self):
        print("IR representations:")
        print("")

        for deg_set, irrep_Rs in zip(self._degenerate_sets, self._irreps):
            print("%3d (%8.3f):" % (deg_set[0] + 1, self._freqs[deg_set[0]]))
            print("")
            for j, irrep_R in enumerate(irrep_Rs):
                for k, irrep_Rk in enumerate(irrep_R):
                    text = "     "
                    for ll, irrep_Rkl in enumerate(irrep_Rk):
                        if irrep_Rkl.real > 0:
                            sign_r = " "
                        else:
                            sign_r = "-"

                        if irrep_Rkl.imag > 0:
                            sign_i = "+"
                        else:
                            sign_i = "-"

                        if k == 0:
                            str_index = "%2d" % (j + 1)
                        else:
                            str_index = "  "

                        if ll > 0:
                            str_index = ""

                        text += "%s (%s%5.3f %s%5.3fi) " % (
                            str_index,
                            sign_r,
                            abs(irrep_Rkl.real),
                            sign_i,
                            abs(irrep_Rkl.imag),
                        )
                    print(text)
                if len(irrep_R) > 1:
                    print("")
            if len(irrep_R) == 1:
                print("")

    def _write_yaml(self, show_irreps: bool):
        lines = []
        lines.append("q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(self._qpoint))
        lines.append("point_group: %s" % self._pointgroup_symbol)
        lines.append("transformation_matrix:")
        for v in self._transformation_matrix:
            lines.append("- [ %10.7f, %10.7f, %10.7f ]" % tuple(v))
        lines.append("rotations:")
        for i, r in enumerate(self._conventional_rotations):
            lines.append("- matrix:")
            for v in r:
                lines.append("  - [ %2d, %2d, %2d ]" % tuple(v))
            if self._rotation_symbols:
                lines.append("  symbol: %s" % self._rotation_symbols[i])
        lines.append("normal_modes:")
        for i, deg_set in enumerate(self._degenerate_sets):
            text = "- band_indices: [ "
            text += "%d" % (deg_set[0] + 1)
            for bi in deg_set[1:]:
                text += ", %d" % (bi + 1)
            text += " ]"
            lines.append(text)
            lines.append("  frequency: %-15.10f" % self._freqs[deg_set[0]])
            if self._ir_labels:
                lines.append("  ir_label: %s" % self._ir_labels[i])
            text = "  characters: "
            chars = np.rint(np.abs(self._characters[i]))
            phase = (np.angle(self._characters[i]) / np.pi * 180) % 360
            if len(chars) > 1:
                text += "[ [ %2d, %5.1f ]" % (chars[0], phase[0])
                for chi, theta in zip(chars[1:], phase[1:]):
                    text += ", [ %2d, %5.1f ]" % (chi, theta)
                text += " ]"
            else:
                text += "[ [ %2d, %5.1f ] ]" % (chars[0], phase[0])
            lines.append(text)

        if show_irreps:
            self._write_yaml_irreps(lines)

        with open("irreps.yaml", "w") as w:
            print("\n".join(lines), file=w)

    def _write_yaml_irreps(self, lines: list[str]):
        lines.append("")
        lines.append("irreps:")
        for i, (deg_set, irrep_Rs) in enumerate(
            zip(self._degenerate_sets, self._irreps)
        ):
            lines.append("- # %d" % (i + 1))
            for j, irrep_R in enumerate(irrep_Rs):
                if self._rotation_symbols:
                    symbol = self._rotation_symbols[j]
                else:
                    symbol = ""
                if len(deg_set) > 1:
                    lines.append("  - # %d %s" % (j + 1, symbol))
                    for _, v in enumerate(irrep_R):
                        text = "    - [ "
                        for x in v[:-1]:
                            text += "%10.7f, %10.7f,   " % (x.real, x.imag)
                        text += "%10.7f, %10.7f ] # (" % (v[-1].real, v[-1].imag)

                        text += ("%5.0f" * len(v)) % tuple(
                            (np.angle(v) / np.pi * 180) % 360
                        )
                        text += ")"
                        lines.append(text)
                else:
                    x = irrep_R[0][0]
                    lines.append(
                        "  - [ [ %10.7f, %10.7f ] ] # (%3.0f) %d %s"
                        % (
                            x.real,
                            x.imag,
                            (np.angle(x) / np.pi * 180) % 360,
                            j + 1,
                            symbol,
                        )
                    )


def _print_characters(characters: NDArray, width: int = 6):
    text = ""
    for i, c in enumerate(characters):
        angle = np.angle(c) / np.pi * 180
        if angle < 0:
            angle += 360
        angle = np.around(angle, decimals=2) % 360
        val = abs(c)
        if val < 1e-5:
            val = 0
            angle = 0
        else:
            val = np.rint(val)
        text += "(%2d, %5.1f) " % (val, angle)
        if ((i + 1) % width == 0 and i != 0) or (len(characters) == i + 1):
            print("    " + text)
            text = ""


def _get_rotation_text(
    rotations: NDArray,
    translations: NDArray | None,
    rotation_symbols: list[str] | None,
    width: int,
    num_rest: int,
    i: int,
) -> str:
    lines = []
    if rotation_symbols is None:
        if translations is None:
            lines.append(
                ("    %2d    " * num_rest)
                % tuple(np.arange(i * width, i * width + num_rest) + 1)
            )
        else:
            lines.append(
                ("       %2d       " * num_rest)
                % tuple(np.arange(i * width, i * width + num_rest) + 1)
            )
    else:
        text = ""
        for k in range(num_rest):
            rot_symbol = rotation_symbols[i * width + k]
            if translations is None:
                if len(rot_symbol) < 3:
                    text += "    %2s    " % rot_symbol
                elif len(rot_symbol) == 3:
                    text += "    %3s   " % rot_symbol
                elif len(rot_symbol) == 4:
                    text += "   %4s   " % rot_symbol
                else:
                    text += "   %5s  " % rot_symbol
            else:
                if len(rot_symbol) < 3:
                    text += "       %2s        " % rot_symbol
                else:
                    text += "     %5s     " % rot_symbol
        lines.append(text)
    if translations is None:
        lines.append(" -------- " * num_rest)
    else:
        lines.append(" -------------- " * num_rest)

    for j in range(3):
        text = ""
        for k in range(num_rest):
            text += " %2d %2d %2d " % tuple(rotations[i * width + k][j])
            if translations is not None:
                text += "%5.2f " % translations[i * width + k][j]
        lines.append(text)
    lines.append("")

    return "\n".join(lines)


def _print_rotations(
    rotations: NDArray,
    translations: NDArray | None = None,
    rotation_symbols: list[str] | None = None,
    width: int = 6,
):
    for i in range(len(rotations) // width):
        print(
            _get_rotation_text(
                rotations, translations, rotation_symbols, width, width, i
            )
        )

    num_rest = len(rotations) % width
    if num_rest > 0:
        i = len(rotations) // width
        print(
            _get_rotation_text(
                rotations, translations, rotation_symbols, width, num_rest, i
            )
        )


class IrRepLabels:
    """Class to assign ir labels to irreps."""

    def __init__(
        self,
        characters: NDArray,
        conventional_rotations: NDArray,
        pointgroup_symbol: str,
        verbose: bool,
    ):
        self._characters = characters
        self._conventional_rotations = conventional_rotations
        self._verbose = verbose

        self._rotation_symbols, character_table_of_ptg = self._get_rotation_symbols(
            pointgroup_symbol
        )
        self._irrep_labels = self._get_irrep_labels(character_table_of_ptg)
        if self._verbose and self._irrep_labels is None:
            print("Database for this point group is not prepared.")

    @property
    def rotation_symbols(self) -> list[str]:
        """Return rotation symbols."""
        return self._rotation_symbols

    @property
    def irrep_labels(self) -> list[str | None] | None:
        """Return irrep labels."""
        return self._irrep_labels

    def _get_rotation_symbols(self, pointgroup_symbol: str) -> tuple[list[str], dict]:
        # Check availability of database
        if (
            pointgroup_symbol not in character_table.keys()
            or character_table[pointgroup_symbol] is None
        ):
            raise RuntimeError("Character table not found.")

        # Loop over possible sets of character tables for the point group
        # Among them, only one set can fit.
        for ct in character_table[pointgroup_symbol]:
            mapping_table: dict = ct["mapping_table"]
            rotation_symbols = []
            for r in self._conventional_rotations:
                rotation_symbols.append(_get_rotation_symbol(r, mapping_table))
            if None not in rotation_symbols:
                break

        if None in rotation_symbols:
            raise RuntimeError("Rotation symbols not found.")

        return rotation_symbols, ct

    def _get_irrep_labels(self, character_table_of_ptg: dict) -> list[str | None]:
        ir_labels = []
        rot_list = character_table_of_ptg["rotation_list"]
        char_table = character_table_of_ptg["character_table"]
        for chars in self._characters:
            chars_ordered = np.zeros(len(rot_list), dtype=complex)
            for rs, ch in zip(self._rotation_symbols, chars):
                chars_ordered[rot_list.index(rs)] += ch

            for i, rl in enumerate(rot_list):
                chars_ordered[i] /= len(character_table_of_ptg["mapping_table"][rl])

            found = False
            for ct_label in char_table.keys():
                if (abs(chars_ordered - np.array(char_table[ct_label])) < 1e-5).all():
                    ir_labels.append(ct_label)
                    found = True
                    break

            if not found:
                ir_labels.append(None)

            if self._verbose > 1:
                text = ""
                for v in chars_ordered:
                    text += "%5.2f " % abs(v)
                if found:
                    print("%s %s" % (text, ct_label))
                else:
                    print("%s Not found" % text)

        return ir_labels


def _get_rotation_symbol(rotation: NDArray, mapping_table: dict) -> str | None:
    for rotation_symbol in mapping_table:
        rot_mats = mapping_table[rotation_symbol]
        for r in rot_mats:
            if (r == rotation).all():
                return rotation_symbol
    return None
