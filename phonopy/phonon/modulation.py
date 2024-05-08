"""Create atomic displacements."""

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

from typing import Union

import numpy as np

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.interface.calculator import (
    get_default_cell_filename,
    write_crystal_structure,
)
from phonopy.phonon.degeneracy import get_eigenvectors
from phonopy.structure.cells import get_supercell
from phonopy.units import VaspToTHz


class Modulation:
    """Class to create atomic displacements."""

    def __init__(
        self,
        dynamical_matrix: Union[DynamicalMatrix, DynamicalMatrixNAC],
        dimension,
        phonon_modes,
        delta_q=None,
        derivative_order=None,
        nac_q_direction=None,
        factor=VaspToTHz,
    ):
        """Init method."""
        self._dm = dynamical_matrix
        self._primitive = dynamical_matrix.primitive
        self._phonon_modes = phonon_modes
        self._dimension = np.array(dimension).ravel()
        self._delta_q = delta_q  # 1st/2nd order perturbation direction
        self._nac_q_direction = nac_q_direction
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._derivative_order = derivative_order

        self._factor = factor
        dim = self._get_dimension_3x3()
        self._supercell = get_supercell(self._primitive, dim)
        complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._u = np.zeros(
            (len(self._phonon_modes), len(self._supercell), 3),
            dtype=complex_dtype,
            order="C",
        )
        self._eigvals = np.zeros(len(self._phonon_modes), dtype="double")
        self._eigvecs = np.zeros(
            (len(self._phonon_modes), len(self._primitive) * 3), dtype=complex_dtype
        )

    def run(self):
        """Calculate modulations."""
        for i, ph_mode in enumerate(self._phonon_modes):
            q, band_index, amplitude, argument = ph_mode
            eigvals, eigvecs = get_eigenvectors(
                q,
                self._dm,
                self._ddm,
                perturbation=self._delta_q,
                derivative_order=self._derivative_order,
                nac_q_direction=self._nac_q_direction,
            )
            u = self._get_displacements(eigvecs[:, band_index], q, amplitude, argument)
            self._u[i] = u
            self._eigvecs[i] = eigvecs[:, band_index]
            self._eigvals[i] = eigvals[band_index]

    def get_modulated_supercells(self):
        """Return modulations."""
        modulations = []
        for u in self._u:
            modulations.append(self._get_cell_with_modulation(u))
        return modulations

    def get_modulations_and_supercell(self):
        """Return modulations and perfect supercell."""
        return self._u, self._supercell

    def write(self, interface_mode=None, optional_structure_info=None):
        """Write supercells with modulations."""
        base_fname = get_default_cell_filename(interface_mode)

        deltas = []
        for i, u in enumerate(self._u):
            cell = self._get_cell_with_modulation(u)
            write_crystal_structure(
                f"M{base_fname}-{(i + 1):03d}",
                cell,
                interface_mode=interface_mode,
                optional_structure_info=optional_structure_info,
            )
            deltas.append(u)

        sum_of_deltas = np.sum(deltas, axis=0)
        cell = self._get_cell_with_modulation(sum_of_deltas)
        write_crystal_structure(
            f"M{base_fname}",
            cell,
            interface_mode=interface_mode,
            optional_structure_info=optional_structure_info,
        )
        no_modulations = np.zeros(sum_of_deltas.shape, dtype=complex)
        cell = self._get_cell_with_modulation(no_modulations)
        write_crystal_structure(
            f"M{base_fname}-orig",
            cell,
            interface_mode=interface_mode,
            optional_structure_info=optional_structure_info,
        )

    def write_yaml(self, filename="modulation.yaml"):
        """Write modulations to file in yaml."""
        self._write_yaml(filename=filename)

    def _get_cell_with_modulation(self, modulation):
        lattice = self._supercell.cell
        positions = self._supercell.positions
        positions += modulation.real
        scaled_positions = np.dot(positions, np.linalg.inv(lattice))
        for p in scaled_positions:
            p -= np.floor(p)
        cell = self._supercell.copy()
        cell.scaled_positions = scaled_positions

        return cell

    def _get_dimension_3x3(self):
        if len(self._dimension) == 3:
            dim = np.diag(self._dimension)
        elif len(self._dimension) == 9:
            dim = np.reshape(self._dimension, (3, 3))
        else:
            dim = np.array(self._dimension)
        if dim.shape == (3, 3):
            dim = np.array(dim, dtype="intc")
        else:
            print("Dimension is incorrectly set. Unit cell is used.")
            dim = np.eye(3, dtype="intc")

        return dim

    def _get_displacements(self, eigvec, q, amplitude, argument):
        m = self._supercell.masses
        s2u_map = self._supercell.s2u_map
        u2u_map = self._supercell.u2u_map
        s2uu_map = [u2u_map[x] for x in s2u_map]
        spos = self._supercell.scaled_positions
        dim = self._supercell.supercell_matrix
        coefs = np.exp(2j * np.pi * np.dot(np.dot(spos, dim.T), q)) / np.sqrt(m)
        u = []
        for i, coef in enumerate(coefs):
            eig_index = s2uu_map[i] * 3
            u.append(eigvec[eig_index : eig_index + 3] * coef)

        u = np.array(u) / np.sqrt(len(m))
        phase_factor = self._get_phase_factor(u, argument)
        u *= phase_factor * amplitude

        return u

    def _get_phase_factor(self, modulation, argument):
        u = np.ravel(modulation)
        index_max_elem = np.argmax(abs(u))
        max_elem = u[index_max_elem]
        phase_for_zero = max_elem / abs(max_elem)
        phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero

        return phase_factor

    def _eigvals_to_frequencies(self, eigvals):
        e = np.array(eigvals).real
        return np.sqrt(np.abs(e)) * np.sign(e) * self._factor

    def _write_yaml(self, filename="modulation.yaml"):
        w = open(filename, "w")
        primitive = self._dm.primitive
        num_atom = len(primitive)

        w.write("primitive_cell:\n")
        self._write_cell_yaml(primitive, w)
        w.write("supercell:\n")
        dim = self._get_dimension_3x3()
        w.write("  dimension:\n")
        for v in dim:
            w.write("  - [ %d, %d, %d ]\n" % tuple(v))
        self._write_cell_yaml(self._supercell, w)
        inv_lattice = np.linalg.inv(self._supercell.cell.T)

        w.write("modulations:\n")
        for u, mode in zip(self._u, self._phonon_modes):
            q = mode[0]
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            w.write("  band: %d\n" % (mode[1] + 1))
            w.write("  amplitude: %f\n" % mode[2])
            w.write("  phase: %f\n" % mode[3])
            w.write("  displacements:\n")
            for i, p in enumerate(u):
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d x (%f)\n"
                    % (p[0].real, p[0].imag, i + 1, abs(p[0]))
                )
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d y (%f)\n"
                    % (p[1].real, p[1].imag, i + 1, abs(p[1]))
                )
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d z (%f)\n"
                    % (p[2].real, p[2].imag, i + 1, abs(p[2]))
                )
            w.write("  fractional_displacements:\n")
            for i, p in enumerate(np.dot(u, inv_lattice.T)):
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d a\n" % (p[0].real, p[0].imag, i + 1)
                )
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d b\n" % (p[1].real, p[1].imag, i + 1)
                )
                w.write(
                    "  - [ %20.15f, %20.15f ] # %d c\n" % (p[2].real, p[2].imag, i + 1)
                )

        w.write("phonon:\n")
        freqs = self._eigvals_to_frequencies(self._eigvals)
        for eigvec, freq, mode in zip(self._eigvecs, freqs, self._phonon_modes):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(mode[0]))
            w.write("  band: %d\n" % (mode[1] + 1))
            w.write("  amplitude: %f\n" % mode[2])
            w.write("  phase: %f\n" % mode[3])
            w.write("  frequency: %15.10f\n" % freq)
            w.write("  eigenvector:\n")
            for j in range(num_atom):
                w.write("  - # atom %d\n" % (j + 1))
                for k in (0, 1, 2):
                    val = eigvec[j * 3 + k]
                    w.write(
                        "    - [ %17.14f, %17.14f ] # %f\n"
                        % (val.real, val.imag, np.angle(val, deg=True))
                    )

    def _write_cell_yaml(self, cell, w):
        lattice = cell.get_cell()
        positions = cell.get_scaled_positions()
        masses = cell.get_masses()
        symbols = cell.get_chemical_symbols()
        w.write("  atom_info:\n")
        for m, s in zip(masses, symbols):
            w.write("  - { name: %2s, mass: %10.5f }\n" % (s, m))

        w.write("  reciprocal_lattice:\n")
        for vec, axis in zip(np.linalg.inv(lattice), ("a*", "b*", "c*")):
            w.write("  - [ %12.8f, %12.8f, %12.8f ] # %2s\n" % (tuple(vec) + (axis,)))
        w.write("  real_lattice:\n")
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))
        w.write("  positions:\n")
        for p in positions:
            w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(p)))
