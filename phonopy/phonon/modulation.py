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

import numpy as np
import sys
from phonopy.structure.cells import get_supercell
from phonopy.interface.vasp import write_vasp
from phonopy.units import VaspToTHz
from phonopy.phonon.degeneracy import get_eigenvectors
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

class Modulation:
    def __init__(self,
                 dynamical_matrix,
                 dimension,
                 phonon_modes,
                 delta_q=None,
                 derivative_order=None,
                 nac_q_direction=None,
                 factor=VaspToTHz):

        """Class describe atomic modulations

        Atomic modulations corresponding to phonon modes are created.
        
        """
        self._dm = dynamical_matrix
        self._primitive = dynamical_matrix.get_primitive()
        self._phonon_modes = phonon_modes
        self._dimension = dimension
        self._delta_q = delta_q # 1st/2nd order perturbation direction
        self._nac_q_direction = nac_q_direction
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._derivative_order = derivative_order

        self._factor = factor
        self._delta_modulations = []
        self._eigvecs = []
        self._eigvals = []
        self._supercell = None

    def run(self):
        for ph_mode in self._phonon_modes:
            q, band_index, amplitude, argument = ph_mode
            eigvals, eigvecs = get_eigenvectors(
                q,
                self._dm,
                self._ddm,
                perturbation=self._delta_q,
                derivative_order=self._derivative_order,
                nac_q_direction=self._nac_q_direction)
            u = self._get_delta(eigvecs[:, band_index], q)
            self._eigvecs.append(eigvecs[:, band_index])
            self._eigvals.append(eigvals[band_index])
            # Set phase of modulation so that phase of the element
            # that has maximum absolute value becomes 0.
            self._set_phase_of_modulation(u, argument)
            self._delta_modulations.append(u * amplitude)

    def write(self, filename="MPOSCAR"):
        deltas = []
        for i, delta_positions in enumerate(self._delta_modulations):
            cell = self._get_cell_with_modulation(delta_positions)
            write_vasp((filename+"-%03d") % (i+1), cell, direct=True)
            deltas.append(delta_positions)
    
        sum_of_deltas = np.sum(deltas, axis=0)
        cell = self._get_cell_with_modulation(sum_of_deltas)
        write_vasp(filename, cell, direct=True)
        no_modulations = np.zeros(sum_of_deltas.shape, dtype=complex)
        cell = self._get_cell_with_modulation(no_modulations)
        write_vasp(filename+"-orig", cell, direct=True)

    def get_modulations(self):
        modulations = []
        for delta_positions in self._delta_modulations:
            modulations.append(self._get_cell_with_modulation(delta_positions))
        return modulations

    def get_delta_modulations(self):
        return self._delta_modulations, self._supercell

    def write_yaml(self):
        self._write_yaml()
                    
    def _get_cell_with_modulation(self, modulation):
        lattice = self._supercell.get_cell()
        positions = self._supercell.get_positions()
        masses = self._supercell.get_masses()
        magmoms = self._supercell.get_magnetic_moments()
        symbols = self._supercell.get_chemical_symbols()
        positions += modulation.real
        scaled_positions = np.dot(positions, np.linalg.inv(lattice))
        for p in scaled_positions:
            p -= np.floor(p)
        cell = self._supercell.copy()
        cell.set_scaled_positions(scaled_positions)

        return cell

    def _get_dimension_3x3(self):
        if len(self._dimension) == 3:
            dim = np.diag(self._dimension)
        elif len(self._dimension) == 9:
            dim = np.reshape(self._dimension, (3, 3))
        else:
            dim = np.array(self._dimension)
        if dim.shape == (3, 3):
            dim = np.array(dim, dtype='intc')
        else:
            print "Dimension is incorrectly set. Unit cell is used."
            dim = np.eye(3, dtype='intc')

        return dim
        
    def _get_delta(self, eigvec, q):
        dim = self._get_dimension_3x3()
        supercell = get_supercell(self._primitive, dim)
        m = supercell.get_masses()
        s2u_map = supercell.get_supercell_to_unitcell_map()
        u2u_map = supercell.get_unitcell_to_unitcell_map()
        s2uu_map = [u2u_map[x] for x in s2u_map]
        spos = supercell.get_scaled_positions()
        coefs = np.exp(2j * np.pi * np.dot(np.dot(spos, dim.T), q)) / np.sqrt(m)
        u = []
        for i, coef in enumerate(coefs):
            eig_index = s2uu_map[i] * 3
            u.append(eigvec[eig_index:eig_index + 3] * coef)

        self._supercell = supercell
        
        return np.array(u)
        
    def _set_phase_of_modulation(self, modulation, argument):
        u = np.ravel(modulation)
        index_max_elem = np.argmax(abs(u))
        max_elem = u[index_max_elem]
        phase_for_zero = max_elem / abs(max_elem)
        phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero
        modulation *= phase_factor

    def _check_eigvecs(self, eigvals, eigvecs, dynmat):
        modified = np.diag(np.dot(eigvecs.conj().T, np.dot(dynmat, eigvecs)))
        print self._eigvals_to_frequencies(eigvals)
        print self._eigvals_to_frequencies(modified)
        print

    def _eigvals_to_frequencies(self, eigvals):
        e = np.array(eigvals).real
        return np.sqrt(np.abs(e)) * np.sign(e) * self._factor

    def _write_yaml(self):
        w = open('modulation.yaml', 'w')
        primitive = self._dm.get_primitive()
        num_atom = primitive.get_number_of_atoms()

        w.write("primitive_cell:\n")
        self._write_cell_yaml(primitive, w)
        w.write("supercell:\n")
        dim = self._get_dimension_3x3()
        w.write("  dimension:\n")
        for v in dim:
            w.write("  - [ %d, %d, %d ]\n" % tuple(v))
        self._write_cell_yaml(self._supercell, w)

        w.write("modulations:\n")
        for deltas, mode in zip(self._delta_modulations,
                                self._phonon_modes):
            q = mode[0]
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" %
                       tuple(q))
            w.write("  band: %d\n" % (mode[1] + 1))
            w.write("  amplitude: %f\n" % mode[2])
            w.write("  phase: %f\n" % mode[3])
            w.write("  displacements:\n")
            for i, p in enumerate(deltas):
                w.write("  - [ %20.15f, %20.15f ] # %d x (%f)\n" %
                           (p[0].real, p[0].imag, i + 1, abs(p[0])))
                w.write("  - [ %20.15f, %20.15f ] # %d y (%f)\n" %
                           (p[1].real, p[1].imag, i + 1, abs(p[1])))
                w.write("  - [ %20.15f, %20.15f ] # %d z (%f)\n" %
                           (p[2].real, p[2].imag, i + 1, abs(p[2])))

        w.write("phonon:\n")
        freqs = self._eigvals_to_frequencies(self._eigvals)
        for eigvec, freq, mode in zip(self._eigvecs,
                                      freqs,
                                      self._phonon_modes):
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" %
                       tuple(mode[0]))
            w.write("  band: %d\n" % (mode[1] + 1))
            w.write("  amplitude: %f\n" % mode[2])
            w.write("  phase: %f\n" % mode[3])
            w.write("  frequency: %15.10f\n" % freq)
            w.write("  eigenvector:\n")
            for j in range(num_atom):
                w.write("  - # atom %d\n" % (j + 1))
                for k in (0, 1, 2):
                    val = eigvec[j * 3 + k]
                    w.write("    - [ %17.14f, %17.14f ] # %f\n" %
                               (val.real, val.imag, np.angle(val, deg=True)))

    def _write_cell_yaml(self, cell, w):
        lattice = cell.get_cell()
        positions = cell.get_scaled_positions()
        masses = cell.get_masses()
        symbols = cell.get_chemical_symbols()
        w.write("  atom_info:\n")
        for m, s in zip(masses, symbols):
            w.write("  - { name: %2s, mass: %10.5f }\n" % (s, m))
        
        w.write("  reciprocal_lattice:\n")
        for vec, axis in zip(np.linalg.inv(lattice), ('a*', 'b*', 'c*')):
            w.write("  - [ %12.8f, %12.8f, %12.8f ] # %2s\n" %
                    (tuple(vec) + (axis,)))
        w.write("  real_lattice:\n")
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
        w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))
        w.write("  positions:\n")
        for p in positions:
            w.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(p)))

    def _get_supercell(self):
        """Attention

        This method will be removed after
        new get_delta method is well checked.
        """
        dim = self._dimension
        scaled_positions = []
        masses = []
        magmoms_prim = self._primitive.get_magnetic_moments()
        if magmoms_prim is None:
            magmoms = None
        else:
            magmoms = []
        symbols = []
        for a, b, c in list(np.ndindex(tuple(dim))):
            for i in range(self._primitive.get_number_of_atoms()):
                p = self._primitive.get_scaled_positions()[i]
                scaled_positions.append(p + np.array([a,b,c]))
                masses.append(self._primitive.get_masses()[i])
                symbols.append(self._primitive.get_chemical_symbols()[i])
                if magmoms_prim is not None:
                    magmoms.append(magmoms_prim[i])

        lattice = np.dot(np.diag(dim), self._primitive.get_cell())
        positions = np.dot(scaled_positions, self._primitive.get_cell())

        from phonopy.structure.atoms import Atoms
        return Atoms(cell=lattice,
                     positions=positions,
                     masses=masses,
                     magmoms=magmoms,
                     symbols=symbols,
                     pbc=True)

