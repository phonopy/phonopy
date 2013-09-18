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
               nac_q_direction=None,
               is_eigenvectors=False,
               group_velocity=None,
               write_dynamical_matrices=False,
               factor=VaspToTHz):
    num_atom = cell.get_number_of_atoms()
    m = cell.get_masses()
    names = cell.get_chemical_symbols()
    positions = cell.get_scaled_positions()
    lattice = cell.get_cell()

    if group_velocity is not None:
        group_velocity.set_q_points(qpoints, perturbation=nac_q_direction)
        group_velocities = group_velocity.get_group_velocity()

    w = open('qpoints.yaml', 'w')
    w.write("nqpoint: %-7d\n" % len(qpoints))
    w.write("natom:   %-7d\n" % num_atom)
    w.write("atom-info:\n")
    for mass, name in zip(m, names):
        w.write("- { name: %2s, mass: %10.5f }\n" % (name, mass))
    
    w.write("real-basis:\n")
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[0])))
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[1])))
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(lattice[2])))

    rec_lattice = np.linalg.inv(lattice).T
    w.write("reciprocal-basis: # q point is multiplied from rhs.\n")
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(rec_lattice[0])))
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(rec_lattice[1])))
    w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(rec_lattice[2])))

    w.write("position:\n")
    for pos in positions:
        w.write("- [ %20.15f, %20.15f, %20.15f ]\n" % (tuple(pos)))
        

    w.write("phonon:\n")

    for i, q in enumerate(qpoints):
        if nac_q_direction is not None and (np.abs(q) < 1e-5).all():
            dynamical_matrix.set_dynamical_matrix(q, q_direction=nac_q_direction)
        else:
            dynamical_matrix.set_dynamical_matrix(q)
        dm = dynamical_matrix.get_dynamical_matrix()

        w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
        if write_dynamical_matrices:
            w.write("  dynamical_matrix:\n")
            for row in dm:
                w.write("  - [ ")
                for j, elem in enumerate(row):
                    w.write("%15.10f, %15.10f" % (elem.real, elem.imag))
                    if j == len(row) - 1:
                        w.write(" ]\n")
                    else:
                        w.write(", ")
        
        w.write("  band:\n")
            
        if is_eigenvectors:
            eigenvalues, eigenvectors = np.linalg.eigh(dm)
        else:
            eigenvalues = np.linalg.eigvalsh(dm)

        for j, eig in enumerate(eigenvalues.real):
            if eig < 0:
                freq = -np.sqrt(-eig)
            else:
                freq = np.sqrt(eig)
            w.write("  - # %d\n" % (j+1))
            w.write("    frequency: %15.10f\n" % (freq * factor))

            if group_velocity is not None:
                w.write("    group_velocity: ")
                w.write("[ %13.7f, %13.7f, %13.7f ]\n" %
                        tuple(group_velocities[i, j]))


            if is_eigenvectors:
                w.write("    eigenvector:\n")
                for k in range(num_atom):
                    w.write("    - # atom %d\n" % (k+1))
                    for l in (0,1,2):
                        w.write("      - [ %17.14f, %17.14f ]\n" %
                                   (eigenvectors[k*3+l,j].real,
                                    eigenvectors[k*3+l,j].imag))

                w.write("    eigenvector_time_aligned:\n")
                for k in range(num_atom):
                    w.write("    - # atom %d, freq*sqrt(m) %f, [%f %f %f]\n" %
                            ((k+1, freq * factor * np.sqrt(m[k])) +
                             tuple(positions[k])))
                    phase_shift = np.exp(2j * np.pi * np.dot(positions[k], q))
                    eig_aligned = eigenvectors[k*3:(k+1)*3, j] * phase_shift
                    for l in (0, 1, 2):
                        w.write(
                            "      - [ %17.14f, %17.14f ] # %7.2f, %7.4f\n" %
                            (eig_aligned[l].real, eig_aligned[l].imag,
                             np.angle(eig_aligned[l], deg=True),
                             abs(eig_aligned[l])))
        w.write("\n")
