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
from phonopy.structure.symmetry import get_ir_reciprocal_mesh
from phonopy.units import VaspToTHz

def get_qpoints(mesh_numbers,
                cell,
                grid_shift=None,
                is_gamma_center=True,
                is_time_reversal=True,
                symprec=1e-5,
                is_symmetry=True):
    mesh = np.array(mesh_numbers)
    if grid_shift == None:
        shift = np.zeros(3, dtype=float)
    else:
        shift = np.array(grid_shift)

    diffby2 = np.abs(shift * 2 - (shift * 2).round())
    if (diffby2 < symprec).all() and is_symmetry: # No shift or half shift case
        diff = np.abs(shift - shift.round())
        if is_gamma_center:
            return _get_qpoint_symmetry(mesh,
                                        (diff > symprec),
                                        cell,
                                        is_time_reversal,
                                        symprec)
        else: # Monkhorst-pack
            return _get_qpoint_symmetry(mesh,
                                        np.logical_xor((diff > symprec),
                                                       (mesh % 2 == 0)),
                                        cell,
                                        is_time_reversal,
                                        symprec)
    else:
        return _get_qpoint_no_symmetry(mesh, shift)

def _get_qpoint_symmetry(mesh,
                         is_shift,
                         cell,
                         is_time_reversal,
                         symprec):
    mapping, grid = get_ir_reciprocal_mesh(mesh,
                                           cell,
                                           is_shift * 1,
                                           is_time_reversal,
                                           symprec)
    ir_list = np.unique(mapping)
    weights = np.zeros(ir_list.shape[0], dtype=int)
    qpoints = np.zeros((ir_list.shape[0], 3), dtype=float)

    for i, g in enumerate(ir_list):
        weights[i] = np.sum(mapping == g)
        qpoints[i] = (grid[g] + is_shift * 0.5) / mesh
        qpoints[i] -= (qpoints[i] > 0.5) * 1

    return qpoints, weights

def _get_qpoint_no_symmetry(mesh, shift):
    qpoints = []
    qshift = shift / mesh
    for i in (0, 1, 2):
        if mesh[i] % 2 == 0:
            qshift[i] += 0.5 / mesh[i]
            
    for i in range(mesh[2]):
        for j in range(mesh[1]):
            for k in range(mesh[0]):
                q = np.array([float(k) / mesh[0],
                              float(j) / mesh[1],
                              float(i) / mesh[2]]) + qshift
                qpoints.append(np.array([q[0] - (q[0] > 0.5),
                                         q[1] - (q[1] > 0.5),
                                         q[2] - (q[2] > 0.5)]))

    qpoints = np.array(qpoints)
    weights = np.ones(qpoints.shape[0], dtype=int)

    return qpoints, weights


class Mesh:
    def __init__(self,
                 dynamical_matrix,
                 cell,
                 mesh,
                 shift=None,
                 is_time_reversal=False,
                 is_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 group_velocity=None,
                 factor=VaspToTHz,
                 symprec=1e-5):
        self._mesh = np.array(mesh)
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor
        self._cell = cell
        self._dynamical_matrix = dynamical_matrix
        self._qpoints, self._weights = get_qpoints(self._mesh,
                                                   self._cell,
                                                   shift,
                                                   is_gamma_center,
                                                   is_time_reversal,
                                                   symprec,
                                                   is_symmetry)
        self._eigenvalues = None
        self._eigenvectors = None
        self._set_eigenvalues()

        self._group_velocities = None
        if group_velocity is not None:
            self._set_group_velocities(group_velocity)

    def get_mesh_numbers(self):
        return self._mesh
        
    def get_qpoints(self):
        return self._qpoints

    def get_weights(self):
        return self._weights

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_frequencies(self):
        return self._frequencies
    
    def get_eigenvectors(self):
        """
        Eigenvectors is a numpy array of three dimension.
        The first index runs through q-points.
        In the second and third indices, eigenvectors obtained
        using numpy.linalg.eigh are stored.
        
        The third index corresponds to the eigenvalue's index.
        The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        """
        return self._eigenvectors


    def write_yaml(self):
        f = open('mesh.yaml', 'w')
        eigenvalues = self._eigenvalues
        natom = self._cell.get_number_of_atoms()
        f.write("mesh: [ %5d, %5d, %5d ]\n" % tuple(self._mesh))
        f.write("nqpoint: %-7d\n" % self._qpoints.shape[0])
        f.write("natom:   %-7d\n" % natom)
        f.write("phonon:\n")

        for i, q in enumerate(self._qpoints):
            f.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
            f.write("  weight: %-5d\n" % self._weights[i])
            f.write("  band:\n")

            for j, eig in enumerate(eigenvalues[i]):
                f.write("  - # %d\n" % (j+1))
                if eig < 0:
                    freq = -np.sqrt(-eig)
                else:
                    freq = np.sqrt(eig)
                f.write("    frequency:  %15.10f\n" % (freq * self._factor))

                if self._group_velocities is not None:
                    f.write("    group_velocity: ")
                    f.write("[ %15.7f, %15.7f, %15.7f ]\n" %
                            tuple(self._group_velocities[i, j]))

                if self._is_eigenvectors:
                    f.write("    eigenvector:\n")
                    for k in range(natom):
                        f.write("    - # atom %d\n" % (k+1))
                        for l in (0,1,2):
                            f.write("      - [ %17.14f, %17.14f ]\n" %
                                    (self._eigenvectors[i,k*3+l,j].real,
                                     self._eigenvectors[i,k*3+l,j].imag))
            f.write("\n")

    def _set_eigenvalues(self):
        eigs = []
        vecs = []
        for q in self._qpoints:
            self._dynamical_matrix.set_dynamical_matrix(q)
            dm = self._dynamical_matrix.get_dynamical_matrix()

            if self._is_eigenvectors:
                val, vec = np.linalg.eigh(dm)
                eigs.append(val.real)
                vecs.append(vec)
            else:
                eigs.append(np.linalg.eigvalsh(dm).real)

        self._eigenvalues = np.array(eigs)
        if self._is_eigenvectors:
            self._eigenvectors = np.array(vecs)

        self._set_frequencies()
            
    def _set_frequencies(self):
        ## This expression works only python >= 2.5
        #  frequencies = []
        # for eigs in self._eigenvalues:
        #     frequencies.append(
        #         [np.sqrt(x) if x > 0 else -np.sqrt(-x) for x in eigs])
        
        self._frequencies = np.array(np.sqrt(abs(self._eigenvalues)) *
                                     np.sign(self._eigenvalues)) * self._factor

    def _set_group_velocities(self, group_velocity):
        group_velocity.set_q_points(self._qpoints)
        self._group_velocities = group_velocity.get_group_velocity()
        
        
