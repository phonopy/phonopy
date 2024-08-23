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

import warnings
from typing import Union

import numpy as np

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.character_table import character_table
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
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
        dynamical_matrix: Union[DynamicalMatrix, DynamicalMatrixNAC],
        q,
        is_little_cogroup=False,
        nac_q_direction=None,
        factor=VaspToTHz,
        symprec=1e-5,
        degeneracy_tolerance=None,
        log_level=0,
    ):
        """Init method."""
        self._is_little_cogroup = is_little_cogroup
        self._nac_q_direction = nac_q_direction
        self._factor = factor
        self._log_level = log_level

        self._q = np.array(q)
        if degeneracy_tolerance is None:
            self._degeneracy_tolerance = 1e-5
        else:
            self._degeneracy_tolerance = degeneracy_tolerance
        self._symprec = symprec
        self._primitive = dynamical_matrix.primitive
        self._dynamical_matrix = dynamical_matrix
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._character_table = None

        self._symmetry_dataset = Symmetry(
            self._primitive, symprec=self._symprec
        ).dataset

        if not is_primitive_cell(self._symmetry_dataset.rotations):
            raise RuntimeError(
                "Non-primitve cell is used. Your unit cell may be transformed to "
                "a primitive cell by PRIMITIVE_AXIS tag."
            )

    def run(self):
        """Calculate irreps."""
        self._set_eigenvectors(self._dynamical_matrix)

        (self._rotations_at_q, self._translations_at_q) = self._get_rotations_at_q()

        self._g = len(self._rotations_at_q)

        (
            self._pointgroup_symbol,
            self._transformation_matrix,
            self._conventional_rotations,
        ) = self._get_conventional_rotations()

        self._ground_matrices = self._get_ground_matrix()
        self._degenerate_sets = self._get_degenerate_sets()
        self._irreps = self._get_irreps()
        self._characters, self._irrep_dims = self._get_characters()

        self._ir_labels = None

        if (
            self._pointgroup_symbol in character_table.keys()
            and character_table[self._pointgroup_symbol] is not None
        ):
            self._rotation_symbols = self._get_rotation_symbols()
            if (abs(self._q) < self._symprec).all() and self._rotation_symbols:
                self._ir_labels = self._get_ir_labels()
            elif (abs(self._q) < self._symprec).all():
                if self._log_level > 0:
                    print("Database for this point group is not preprared.")
            else:
                if self._log_level > 0:
                    print("Database for non-Gamma point is not prepared.")
        else:
            self._rotation_symbols = None

        return True

    def _get_degenerate_sets(self):
        deg_sets = get_degenerate_sets(self._freqs, cutoff=self._degeneracy_tolerance)
        self._ddm.run(self._q)
        return deg_sets

    @property
    def band_indices(self):
        """Return band indices.

        Returns
        -------
        See docstring of ``degenerate_sets``.

        """
        return self._degenerate_sets

    def get_band_indices(self):
        """Return band indices."""
        warnings.warn(
            "IrReps.get_band_indices() is deprecated. "
            "Use IrReps.band_indices attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.band_indices

    @property
    def characters(self):
        """Return characters of irreps."""
        return self._characters

    def get_characters(self):
        """Return characters of irreps."""
        warnings.warn(
            "IrReps.get_characters() is deprecated. "
            "Use IrReps.characters attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.characters

    @property
    def eigenvectors(self):
        """Return eigenvectors."""
        return self._eig_vecs

    def get_eigenvectors(self):
        """Return eigenvectors."""
        warnings.warn(
            "IrReps.get_eigenvectors() is deprecated. "
            "Use IrReps.eigenvectors attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eigenvectors

    @property
    def irreps(self):
        """Return irreps."""
        return self._irreps

    def get_irreps(self):
        """Return irreps."""
        warnings.warn(
            "IrReps.get_irreps() is deprecated. " "Use IrReps.irreps attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.irreps

    @property
    def ground_matrices(self):
        """Return ground matrices."""
        return self._ground_matrices

    def get_ground_matrices(self):
        """Return ground matrices."""
        warnings.warn(
            "IrReps.get_ground_matrices() is deprecated. "
            "Use IrReps.ground_matrices attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ground_matrices

    @property
    def rotation_symbols(self):
        """Return symbols assigned to rotation matrices."""
        return self._rotation_symbols

    def get_rotation_symbols(self):
        """Return symbols assigned to rotation matrices."""
        warnings.warn(
            "IrReps.get_rotation_symbols() is deprecated. "
            "Use IrReps.rotation_symbols attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rotation_symbols

    @property
    def conventional_rotations(self):
        """Return rotation matrices."""
        return self._conventional_rotations

    def get_rotations(self):
        """Return rotation matrices."""
        warnings.warn(
            "IrReps.get_rotations() is deprecated. "
            "Use IrReps.conventional_rotations attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.conventional_rotations

    def get_projection_operators(self, idx_irrep, i=None, j=None):
        """Return projection operators."""
        if i is None or j is None:
            return self._get_character_projection_operators(idx_irrep)
        else:
            return self._get_projection_operators(idx_irrep, i, j)

    @property
    def qpoint(self):
        """Return q-point."""
        return self._q

    def show(self, show_irreps=False):
        """Show irreps."""
        self._show(show_irreps)

    def write_yaml(self, show_irreps=False):
        """Write irreps in yaml file."""
        self._write_yaml(show_irreps)

    def _set_eigenvectors(self, dm):
        if self._nac_q_direction is not None and (np.abs(self._q) < 1e-5).all():
            dm.run(self._q, q_direction=self._nac_q_direction)
        else:
            dm.run(self._q)
        eig_vals, self._eig_vecs = np.linalg.eigh(dm.dynamical_matrix)
        self._freqs = np.sqrt(abs(eig_vals)) * np.sign(eig_vals) * self._factor

    def _get_rotations_at_q(self):
        rotations_at_q = []
        trans_at_q = []
        for r, t in zip(
            self._symmetry_dataset.rotations, self._symmetry_dataset.translations
        ):
            diff = np.dot(self._q, r) - self._q
            if (abs(diff - np.rint(diff)) < self._symprec).all():
                rotations_at_q.append(r)
                for i in range(3):
                    if np.abs(t[i] - 1) < self._symprec:
                        t[i] = 0.0
                trans_at_q.append(t)

        return np.array(rotations_at_q), np.array(trans_at_q)

    def _get_conventional_rotations(self):
        rotations = self._rotations_at_q.copy()
        pointgroup_symbol = self._symmetry_dataset.pointgroup
        transformation_matrix = self._symmetry_dataset.transformation_matrix
        conventional_rotations = self._transform_rotations(
            transformation_matrix, rotations
        )

        return (pointgroup_symbol, transformation_matrix, conventional_rotations)

    def _transform_rotations(self, tmat, rotations):
        trans_rots = []

        for r in rotations:
            r_conv = similarity_transformation(tmat, r)
            trans_rots.append(np.rint(r_conv).astype(int))

        return np.array(trans_rots)

    def _get_ground_matrix(self):
        matrices = []

        for r, t in zip(self._rotations_at_q, self._translations_at_q):
            lat = self._primitive.cell.T
            r_cart = similarity_transformation(lat, r)
            perm_mat = self._get_modified_permutation_matrix(r, t)
            matrices.append(np.kron(perm_mat, r_cart))

        return np.array(matrices)

    def _get_characters(self):
        characters = []
        irrep_dims = []
        for irrep_Rs in self._irreps:
            characters.append([np.trace(rep) for rep in irrep_Rs])
            irrep_dims.append(len(irrep_Rs[0]))
        return np.array(characters), np.array(irrep_dims)

    def _get_modified_permutation_matrix(self, r, t):
        num_atom = len(self._primitive)
        pos = self._primitive.scaled_positions
        matrix = np.zeros((num_atom, num_atom), dtype=complex)
        for i, p1 in enumerate(pos):
            p_rot = np.dot(r, p1) + t  # i -> j
            for j, p2 in enumerate(pos):
                diff = p_rot - p2  # Rx_i + t - x_j
                if (abs(diff - np.rint(diff)) < self._symprec).all():
                    phase_factor = np.dot(
                        self._q, np.dot(np.linalg.inv(r), p2 - t) - p2
                    )
                    if self._is_little_cogroup:
                        phase_factor = np.dot(t, self._q)
                    matrix[j, i] = np.exp(2j * np.pi * phase_factor)
        return matrix

    def _get_irreps(self):
        eigvecs = self._eig_vecs.T
        irrep = []
        for band_indices in self._degenerate_sets:
            irrep_Rs = []
            for mat in self._ground_matrices:
                n_deg = len(band_indices)

                if n_deg == 1:
                    vec = eigvecs[band_indices[0]]
                    irrep_Rs.append([[np.vdot(vec, np.dot(mat, vec))]])
                    continue

                irrep_R = np.zeros((n_deg, n_deg), dtype=complex)
                for i, b_i in enumerate(band_indices):
                    vec_i = eigvecs[b_i]
                    for j, b_j in enumerate(band_indices):
                        vec_j = eigvecs[b_j]
                        irrep_R[i, j] = np.vdot(vec_i, np.dot(mat, vec_j))
                irrep_Rs.append(irrep_R)

            irrep.append(irrep_Rs)

        return irrep

    def _get_character_projection_operators(self, idx_irrep):
        dim = self._irrep_dims[idx_irrep]
        chars = self._characters[idx_irrep]
        return (
            np.sum(
                [mat * char.conj() for mat, char in zip(self._ground_matrices, chars)],
                axis=0,
            )
            * dim
            / self._g
        )

    def _get_projection_operators(self, idx_irrep, i, j):
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
            / self._g
        )

    def _get_rotation_symbols(self):
        ptg_symbol = self._pointgroup_symbol
        for ct in character_table[ptg_symbol]:
            mapping_table = ct["mapping_table"]
            rotation_symbols = []
            for r in self._conventional_rotations:
                rotation_symbols.append(_get_rotation_symbol(r, mapping_table))
            if False in rotation_symbols:
                ret_val = None
            else:
                ret_val = rotation_symbols
            if ret_val is not None:
                self._character_table = ct
                break

        return ret_val

    def _get_ir_labels(self):
        ir_labels = []
        rot_list = self._character_table["rotation_list"]
        char_table = self._character_table["character_table"]
        for chars in self._characters:
            chars_ordered = np.zeros(len(rot_list), dtype=complex)
            for rs, ch in zip(self._rotation_symbols, chars):
                chars_ordered[rot_list.index(rs)] += ch

            for i, rl in enumerate(rot_list):
                chars_ordered[i] /= len(self._character_table["mapping_table"][rl])

            found = False
            for ct_label in char_table.keys():
                if (
                    abs(chars_ordered - np.array(char_table[ct_label])) < self._symprec
                ).all():
                    ir_labels.append(ct_label)
                    found = True
                    break

            if not found:
                ir_labels.append(None)

            if self._log_level > 1:
                text = ""
                for v in chars_ordered:
                    text += "%5.2f " % abs(v)
                if found:
                    print("%s %s" % (text, ct_label))
                else:
                    print("%s Not found" % text)

        return ir_labels

    def _show(self, show_irreps):
        print("")
        print("-------------------------------")
        print("  Irreducible representations")
        print("-------------------------------")
        print("q-point: %s" % self._q)
        print("Point group: %s" % self._pointgroup_symbol)
        print("")

        if (np.abs(self._q) < self._symprec).all():
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
        print("Rotation matrices by transformation matrix:")
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

    def _write_yaml(self, show_irreps):
        w = open("irreps.yaml", "w")
        w.write("q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(self._q))
        w.write("point_group: %s\n" % self._pointgroup_symbol)
        w.write("transformation_matrix:\n")
        for v in self._transformation_matrix:
            w.write("- [ %10.7f, %10.7f, %10.7f ]\n" % tuple(v))
        w.write("rotations:\n")
        for i, r in enumerate(self._conventional_rotations):
            w.write("- matrix:\n")
            for v in r:
                w.write("  - [ %2d, %2d, %2d ]\n" % tuple(v))
            if self._rotation_symbols:
                w.write("  symbol: %s\n" % self._rotation_symbols[i])
        w.write("normal_modes:\n")
        for i, deg_set in enumerate(self._degenerate_sets):
            w.write("- band_indices: [ ")
            w.write("%d" % (deg_set[0] + 1))
            for bi in deg_set[1:]:
                w.write(", %d" % (bi + 1))
            w.write(" ]\n")
            w.write("  frequency: %-15.10f\n" % self._freqs[deg_set[0]])
            if self._ir_labels:
                w.write("  ir_label: %s\n" % self._ir_labels[i])
            w.write("  characters: ")
            chars = np.rint(np.abs(self._characters[i]))
            phase = (np.angle(self._characters[i]) / np.pi * 180) % 360
            if len(chars) > 1:
                w.write("[ [ %2d, %5.1f ]" % (chars[0], phase[0]))
                for chi, theta in zip(chars[1:], phase[1:]):
                    w.write(", [ %2d, %5.1f ]" % (chi, theta))
                w.write(" ]\n")
            else:
                w.write("[ [ %2d, %5.1f ] ]\n" % (chars[0], phase[0]))

        if show_irreps:
            self._write_yaml_irreps(w)

        w.close()

    def _write_yaml_irreps(self, file_pointer):
        w = file_pointer
        if not self._irreps:
            self._irrep = self._get_irreps()

        w.write("\n")
        w.write("irreps:\n")
        for i, (deg_set, irrep_Rs) in enumerate(
            zip(self._degenerate_sets, self._irreps)
        ):
            w.write("- # %d\n" % (i + 1))
            for j, irrep_R in enumerate(irrep_Rs):
                if self._rotation_symbols:
                    symbol = self._rotation_symbols[j]
                else:
                    symbol = ""
                if len(deg_set) > 1:
                    w.write("  - # %d %s\n" % (j + 1, symbol))
                    for _, v in enumerate(irrep_R):
                        w.write("    - [ ")
                        for x in v[:-1]:
                            w.write("%10.7f, %10.7f,   " % (x.real, x.imag))
                        w.write("%10.7f, %10.7f ] # (" % (v[-1].real, v[-1].imag))

                        w.write(
                            ("%5.0f" * len(v))
                            % tuple((np.angle(v) / np.pi * 180) % 360)
                        )
                        w.write(")\n")
                else:
                    x = irrep_R[0][0]
                    w.write(
                        "  - [ [ %10.7f, %10.7f ] ] # (%3.0f) %d %s\n"
                        % (
                            x.real,
                            x.imag,
                            (np.angle(x) / np.pi * 180) % 360,
                            j + 1,
                            symbol,
                        )
                    )

        pass


def _get_rotation_symbol(rotation, mapping_table):
    for k in mapping_table:
        v = mapping_table[k]
        for r in v:
            if (r == rotation).all():
                return k
    return False


def _print_characters(characters, width=6):
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


def _get_rotation_text(rotations, translations, rotation_symbols, width, num_rest, i):
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


def _print_rotations(rotations, translations=None, rotation_symbols=None, width=6):
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
