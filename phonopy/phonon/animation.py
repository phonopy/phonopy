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
from phonopy.structure.cells import (get_angles, get_cell_parameters,
                                     get_cell_matrix)
from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.interface.vasp import write_vasp
from phonopy.units import VaspToTHz

class Animation(object):
    def __init__(self,
                 qpoint,
                 dynamical_matrix,
                 shift=None):
        dynamical_matrix.set_dynamical_matrix(qpoint)
        self._eigenvalues, self._eigenvectors = \
            np.linalg.eigh(dynamical_matrix.get_dynamical_matrix())
        self._qpoint = qpoint
        primitive = dynamical_matrix.get_primitive()
        self._positions = primitive.get_scaled_positions()
        self._symbols = primitive.get_chemical_symbols()
        self._masses = primitive.get_masses()
        self._lattice = primitive.get_cell()
        if shift is not None:
            self._positions = (self._positions + shift) % 1
            
    def _set_cell_oriented(self):
        # Re-oriented lattice xx, yx, yy, zx, zy, zz
        self._angles = get_angles(self._lattice)
        self._cell_params = get_cell_parameters(self._lattice)
        a, b, c = self._cell_params
        alpha, beta, gamma = self._angles
        self._lattice_oriented = get_cell_matrix(a, b, c, alpha, beta, gamma) 
        self._positions_oriented = \
            self._get_oriented_displacements(np.dot(self._positions,
                                                    self._lattice))

    # For the orientation, see get_cell_matrix
    def _get_oriented_displacements(self, vec_cartesian):
        return np.dot(np.dot(vec_cartesian, np.linalg.inv(self._lattice)),
                      self._lattice_oriented)

    def _set_displacements(self, band_index):
        u = []
        for i, e in enumerate(self._eigenvectors[:, band_index]):
            u.append(e / np.sqrt(self._masses[i // 3]))

        self._displacements = np.array(u).reshape(-1, 3)

    def write_v_sim(self,
                    amplitude=1.0,
                    factor=VaspToTHz,
                    filename="anime.ascii"):
        self._set_cell_oriented()
        lat = self._lattice_oriented
        q = self._qpoint
        text  = "# Phonopy generated file for v_sim 3.6\n"
        text += "%15.9f%15.9f%15.9f\n" % (lat[0,0], lat[1,0], lat[1,1])
        text += "%15.9f%15.9f%15.9f\n" % (lat[2,0], lat[2,1], lat[2,2])
        for s, p in zip(self._symbols, self._positions_oriented):
            text += "%15.9f%15.9f%15.9f %2s\n" % (p[0], p[1], p[2], s)

        for i, val in enumerate(self._eigenvalues):
            if val > 0:
                omega = np.sqrt(val)
            else:
                omega = -np.sqrt(-val)
            self._set_displacements(i)
            text += "#metaData: qpt=[%f;%f;%f;%f \\\n" % (
                q[0], q[1], q[2], omega * factor)
            for u in (self._get_oriented_displacements(self._displacements) *
                      amplitude):
                text += "#; %f; %f; %f; %f; %f; %f \\\n" % (
                    u[0].real, u[1].real, u[2].real,
                    u[0].imag, u[1].imag, u[2].imag)
            text += "# ]\n"
        w = open(filename, 'w')
        w.write(text)
        w.close()

    def write_arc(self,
                  band_index,
                  amplitude=1,
                  num_div=20,
                  filename="anime.arc"):
        self._set_cell_oriented()
        self._set_displacements(band_index - 1)
        displacements = self._get_oriented_displacements(self._displacements)

        a, b, c = self._cell_params
        alpha, beta, gamma = self._angles

        text = ""
        text += "!BIOSYM archive 3\n"
        text += "PBC=ON\n"
    
        for i in range(num_div):
            text += "                                                                        0.000000\n"
            text += "!DATE\n"
            text += "%-4s%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f\n" % (
                "PBC", a, b, c, alpha, beta, gamma)
            positions = (self._positions_oriented +
                         (displacements *
                          np.exp(2j * np.pi / num_div * i)).imag * amplitude)
            for j, p in enumerate(positions):
                text += "%-5s%15.9f%15.9f%15.9f CORE" % (
                    self._symbols[j], p[0], p[1], p[2])
                text += "%5s%3s%3s%9.4f%5s\n" % (
                    j + 1, self._symbols[j], self._symbols[j], 0.0, j + 1)
                
            text += "end\n"
            text += "end\n"
        
        w = open(filename, 'w')
        w.write(text)
        w.close()
            
    def write_xyz_jmol(self,
                       amplitude=10,
                       factor=VaspToTHz,
                       filename="anime.xyz_jmol"):
        self._set_cell_oriented()
        text = ""
        for i, val in enumerate(self._eigenvalues):
            if val > 0:
                freq = np.sqrt(val)
            else:
                freq = -np.sqrt(-val)
            self._set_displacements(i)
            displacements = self._get_oriented_displacements(
                self._displacements) * amplitude
            text += "%d\n" % len(self._symbols)
            text += "q %s , b %d , f %f " % (str(self._qpoint), i+1, freq * factor)
            text += "(generated by Phonopy)\n" 
            for s, p, u in zip(
                self._symbols, self._positions_oriented, displacements):
                text += "%-3s  %22.15f %22.15f %22.15f  " % (s, p[0], p[1], p[2])
                text += "%15.9f %15.9f %15.9f\n" % (u[0].real, u[1].real, u[2].real)
        w = open(filename, 'w')
        w.write(text)
        w.close()

    def write_xyz(self,
                  band_index,
                  amplitude=1,
                  num_div=20,
                  factor=VaspToTHz,
                  filename="anime.xyz"):
        self._set_cell_oriented()
        freq = self._eigenvalues[band_index - 1]
        self._set_displacements(band_index - 1)
        displacements = self._get_oriented_displacements(self._displacements)
        text = ""
        for i in range(num_div):
            text += "%d\n" % len(self._symbols)
            text += "q %s , b %d , f %f , " % (
                str(self._qpoint), band_index, freq * factor)
            text += "div %d / %d " % (i, num_div)
            text += "(generated by Phonopy)\n"
            positions = (self._positions_oriented +
                         (displacements *
                          np.exp(2j * np.pi / num_div * i)).imag * amplitude)
            for j, p in enumerate(positions):
                text += "%-3s %22.15f %22.15f %22.15f\n" % (
                    self._symbols[j], p[0], p[1], p[2])
        w = open(filename, 'w')
        w.write(text)
        w.close()


    def write_POSCAR(self,
                     band_index,
                     amplitude=1,
                     num_div=20,
                     filename="APOSCAR"):
        self._set_displacements(band_index - 1)
        for i in range(num_div):
            positions = (np.dot(self._positions, self._lattice) +
                         (self._displacements *
                          np.exp(2j * np.pi / num_div * i)).imag * amplitude)
            atoms = Atoms(cell=self._lattice,
                          positions=positions,
                          masses=self._masses,
                          symbols=self._symbols,
                          pbc=True)
            write_vasp((filename+"-%03d") % i, atoms, direct=True)


