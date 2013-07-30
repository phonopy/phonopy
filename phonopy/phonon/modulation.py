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
from phonopy.structure.atoms import Atoms
from phonopy.interface.vasp import write_vasp
from phonopy.units import VaspToTHz
from phonopy.phonon.group_velocity import degenerate_sets, delta_dynamical_matrix
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

class Modulation:
    def __init__(self,
                 dynamical_matrix,
                 cell,
                 dimension,
                 phonon_modes,
                 delta_q=None,
                 factor=VaspToTHz):

        """Class describe atomic modulations

        Atomic modulations corresponding to phonon modes are created.
        
        """
        self._dm = dynamical_matrix
        self._cell = cell
        self._phonon_modes = phonon_modes
        self._dimension = dimension
        self._delta_q = delta_q # 1st order perturbation direction
        self._ddm = DerivativeOfDynamicalMatrix(dynamical_matrix)
        self._factor = factor
        self._delta_modulations = []
        self._eigvecs = []
        self._eigvals = []

    def run(self):
        for ph_mode in self._phonon_modes:
            q, band_index, amplitude, argument = ph_mode
            eigvals, eigvecs = self._get_eigenvectors(q)
            u = self._get_delta(eigvecs[:, band_index], q)
            self._eigvecs.append(eigvecs[:, band_index])
            self._eigvals.append(eigvals[band_index])
            # Set phase of modulation so that phase of the element
            # that has maximum absolute value becomes 0.
            self._set_phase_of_modulation(u, argument)
            self._delta_modulations.append(u.reshape(-1,3) * amplitude)

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
        return self._delta_modulations, self._get_supercell()

    def _get_cell_with_modulation(self, modulation):
        supercell = self._get_supercell()
        lattice = supercell.get_cell()
        positions = supercell.get_positions()
        positions += modulation.real
        scaled_positions = np.dot(positions, np.linalg.inv(lattice))
        for p in scaled_positions:
            p -= np.floor(p)
        supercell.set_scaled_positions(scaled_positions)
    
        return supercell
            
    def _get_delta(self, eigvec, q):
        dim = self._dimension
        m = self._cell.get_masses()
        r = self._cell.get_scaled_positions()
        u = []
        for a in range(dim[0]):
            for b in range(dim[1]):
                for c in range(dim[2]):
                    for i, e in enumerate(eigvec):
                        phase = 2j * np.pi * (
                            np.dot(r[i//3] + np.array([a,b,c]), q))
                        u.append(e / np.sqrt(m[i//3]) * np.exp(phase)) 
    
        return np.array(u)

    def _set_phase_of_modulation(self, modulation, argument):
        u = modulation
        index_max_elem = np.argmax(abs(u))
        max_elem = u[index_max_elem]
        phase_for_zero = max_elem / abs(max_elem)
        phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero
        u *= phase_factor

    def _get_supercell(self):
        dim = self._dimension
        scaled_positions = []
        masses = []
        magmoms_prim = self._cell.get_magnetic_moments()
        if magmoms_prim == None:
            magmoms = None
        else:
            magmoms = []
        symbols = []
        for a in range(dim[0]):
            for b in range(dim[1]):
                for c in range(dim[2]):
                    for i in range(self._cell.get_number_of_atoms()):
                        p = self._cell.get_scaled_positions()[i]
                        scaled_positions.append(p + np.array([a,b,c]))
                        masses.append(self._cell.get_masses()[i])
                        symbols.append(self._cell.get_chemical_symbols()[i])
                        if not magmoms_prim == None:
                            magmoms.append(magmoms_prim[i])

        lattice = np.dot(np.diag(dim), self._cell.get_cell())
        positions = np.dot(scaled_positions, self._cell.get_cell())

        return Atoms(cell=lattice,
                     positions=positions,
                     masses=masses,
                     magmoms=magmoms,
                     symbols=symbols,
                     pbc=True)

    def _get_eigenvectors(self, q):
        self._dm.set_dynamical_matrix(q)
        eigvals, eigvecs = np.linalg.eigh(self._dm.get_dynamical_matrix())
        eigvals = eigvals.real
        if self._delta_q is None:
            return eigvals, eigvecs
        else:
            deg_sets = degenerate_sets(eigvals)
            for deg in deg_sets:
                eigsets = eigvecs[:, deg].copy()
                dD = self._get_dD(q)
                p_eigvals, p_eigvecs = np.linalg.eigh(
                    np.dot(eigsets.T.conj(), np.dot(dD, eigsets)))
                eigvecs[:, deg] = np.dot(eigsets, p_eigvecs)

            return eigvals, eigvecs

    def _get_dD(self, q):
        # dD = delta_dynamical_matrix(np.array(q),
        #                             np.array(self._delta_q),
        #                             self._dm)
        self._ddm.run(q)
        ddm = self._ddm.get_derivative_of_dynamical_matrix()
        dD = np.zeros(ddm.shape[1:], dtype='complex128')
        for i in range(3):
            dD += (self._delta_q[i] / np.linalg.norm(self._delta_q)) * ddm[i] 
        return dD

    def write_yaml(self):
        file = open('modulation.yaml', 'w')
        dim = self._dimension
        factor = self._factor
        cell = self._cell
        num_atom = self._cell.get_number_of_atoms()
        modes = self._phonon_modes

        lattice = cell.get_cell()
        positions = cell.get_scaled_positions()
        masses = cell.get_masses()
        symbols = cell.get_chemical_symbols()
        file.write("unitcell:\n")
        file.write("  atom-info:\n")
        for m, s in zip( masses, symbols ):
            file.write("  - { name: %2s, mass: %10.5f }\n" % (s, m))
        
        file.write("  lattice-vectors:\n")
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))
        file.write("  positions:\n")
        for p in positions:
            file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(p)))

        supercell = self._get_supercell()
        lattice = supercell.get_cell()
        positions = supercell.get_scaled_positions()
        masses = supercell.get_masses()
        symbols = supercell.get_chemical_symbols()
        file.write("supercell:\n")
        file.write("  dimension: [ %d, %d, %d ]\n" % tuple(dim))
        file.write("  atom-info:\n")
        for m, s in zip( masses, symbols ):
            file.write("  - { name: %2s, mass: %10.5f }\n" % (s, m))
        
        file.write("  lattice-vectors:\n")
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
        file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))
        file.write("  positions:\n")
        for p in positions:
            file.write("  - [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(p)))

        file.write("modulations:\n")
        for deltas, mode in zip(self._delta_modulations,
                                self._phonon_modes):
            q = mode[0]
            file.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" %
                       tuple(q))
            file.write("  band: %d\n" % (mode[1] + 1))
            file.write("  amplitude: %f\n" % mode[2])
            file.write("  phase: %f\n" % mode[3])
            file.write("  displacements:\n")
            for i, p in enumerate(deltas):
                file.write("  - [ %20.15f, %20.15f ] # %d x (%f)\n" %
                           (p[0].real, p[0].imag, i + 1, abs(p[0])))
                file.write("  - [ %20.15f, %20.15f ] # %d y (%f)\n" %
                           (p[1].real, p[1].imag, i + 1, abs(p[1])))
                file.write("  - [ %20.15f, %20.15f ] # %d z (%f)\n" %
                           (p[2].real, p[2].imag, i + 1, abs(p[2])))

        file.write("phonon:\n")
        freqs = np.sqrt(np.abs(self._eigvals)) * np.sign(self._eigvals)
        for eigvec, freq, mode in zip(self._eigvecs,
                                      freqs,
                                      self._phonon_modes):
            file.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" %
                       tuple(mode[0]))
            file.write("  band: %d\n" % (mode[1] + 1))
            file.write("  amplitude: %f\n" % mode[2])
            file.write("  phase: %f\n" % mode[3])
            file.write("  frequency: %15.10f\n" % (freq * factor))
            file.write("  eigenvector:\n")
            for j in range(num_atom):
                file.write("  - # atom %d\n" % (j + 1))
                for k in (0, 1, 2):
                    val = eigvec[j * 3 + k]
                    file.write("    - [ %17.14f, %17.14f ] # %f\n" %
                               (val.real, val.imag, np.angle(val, deg=True)))
