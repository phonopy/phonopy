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
import cmath
from phonopy.units import VaspToTHz

def write_yaml(qpoints,
               cell,
               dynamical_matrix, 
               is_eigenvectors=False,
               factor=VaspToTHz):
    num_atom = cell.get_number_of_atoms()
    m = cell.get_masses()
    names = cell.get_chemical_symbols()
    positions = cell.get_scaled_positions()
    lattice = cell.get_cell()

    file = open('qpoints.yaml', 'w')
    file.write("nqpoint: %-7d\n" % len(qpoints))
    file.write("natom:   %-7d\n" % num_atom)
    file.write("atom-info:\n")
    for mass, name in zip(m, names):
        file.write("- { name: %2s, mass: %10.5f }\n" % (name, mass))
    
    file.write("real-basis:\n")
    file.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
    file.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
    file.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))

    file.write("position:\n")
    for pos in positions:
        file.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(pos)))
        

    file.write("phonon:\n")

    for q in qpoints:
        dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()

        file.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
        file.write("  q-point:\n")
            
        if is_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(dm)
        else:
            eigenvalues = np.linalg.eigvalsh(dm)

        for j, eig in enumerate(eigenvalues.real):
            if eig < 0:
                freq = -np.sqrt(-eig)
            else:
                freq = np.sqrt(eig)
            file.write("  - # %d\n" % (j+1))
            file.write("    frequency: %15.10f\n" % (freq * factor))

            if is_eigenvectors:
                file.write("    eigenvector:\n")
                for k in range(num_atom):
                    file.write("    - # atom %d\n" % (k+1))
                    for l in (0,1,2):
                        file.write("      - [ %17.14f, %17.14f ]\n" %
                                   (eigenvectors[k*3+l,j].real,
                                    eigenvectors[k*3+l,j].imag))

                file.write("    eigenvector_time_aligned:\n")
                for k in range(num_atom):
                    file.write("    - # atom %d, freq*sqrt(m) %f, [%f %f %f]\n" %
                               ((k+1, freq * factor * np.sqrt(m[k])) + tuple(positions[k])))
                    # Phase of dot(q, r) is added.
                    eig_aligned = eigenvectors[k*3:(k+1)*3, j] * np.exp(2j * np.pi * np.dot(positions[k], q))
                    for l in (0,1,2):
                        file.write("      - [ %17.14f, %17.14f ] # %7.2f, %7.4f\n" %
                                   (eig_aligned[l].real,
                                    eig_aligned[l].imag,
                                    cmath.phase(eig_aligned[l]) / np.pi * 180,
                                    abs(eig_aligned[l])))
        file.write("\n")
