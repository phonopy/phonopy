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
from phonopy.structure.spglib import get_stabilized_reciprocal_mesh
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.structure.symmetry import get_lattice_vector_equivalence
from phonopy.units import VaspToTHz

def get_qpoints(mesh_numbers,
                reciprocal_lattice, # column vectors
                q_mesh_shift=None, # Monkhorst-Pack style grid shift
                is_gamma_center=True,
                is_time_reversal=True,
                fit_in_BZ=True,
                rotations=None, # Point group operations in real space
                is_symmetry=True):
    is_shift = shift2boolean(mesh_numbers,
                             q_mesh_shift=q_mesh_shift,
                             is_gamma_center=is_gamma_center)

    if is_shift and is_symmetry and has_mesh_symmetry(mesh_numbers, rotations):
        qpoints, weights = _get_ir_qpoints(mesh_numbers,
                                           is_shift,
                                           rotations,
                                           is_time_reversal)
    else:
        qpoints, weights = _get_qpoints_without_symmetry(mesh_numbers,
                                                         q_mesh_shift,
                                                         is_gamma_center)

    if fit_in_BZ:
        qpoints_in_BZ = _fit_qpoints_in_BZ(reciprocal_lattice, qpoints)
        return qpoints_in_BZ, weights
    else:
        return qpoints, weights

def shift2boolean(mesh_numbers,
                  q_mesh_shift=None,
                  is_gamma_center=False,
                  tolerance=1e-5):
    """
    Tolerance is used to judge zero/half gird shift.
    This value is not necessary to be changed usually.
    """
    mesh = np.array(mesh_numbers, dtype='intc')
    if q_mesh_shift is None:
        shift = np.zeros(3, dtype='double')
    else:
        shift = np.array(q_mesh_shift, dtype='double')

    diffby2 = np.abs(shift * 2 - np.rint(shift * 2))
    if (diffby2 < 0.01).all(): # zero/half shift
        if is_gamma_center:
            is_shift = [0, 0, 0]
        else: # Monkhorst-pack
            diff = np.abs(shift - np.rint(shift))
            is_shift = list(np.logical_xor((diff > 0.1), (mesh % 2 == 0)) * 1)
    else:
        is_shift = None

    return is_shift
    
def has_mesh_symmetry(mesh, rotations):
    if rotations is None:
        return False

    mesh_equiv = [mesh[1] == mesh[2], mesh[2] == mesh[0], mesh[0] == mesh[1]]
    lattice_equiv = get_lattice_vector_equivalence([r.T for r in rotations])
    return np.array_equal(mesh_equiv, lattice_equiv)

def _fit_qpoints_in_BZ(reciprocal_lattice, qpoints):
    # reciprocal_lattice: column vectors
    qpoint_set_in_BZ = get_qpoints_in_Brillouin_zone(reciprocal_lattice,
                                                     qpoints)
    qpoints_in_BZ = np.array([q_set[0] for q_set in qpoint_set_in_BZ],
                             dtype='double')
    return qpoints_in_BZ
    

def _get_ir_qpoints(mesh,
                    is_shift,
                    rotations,
                    is_time_reversal):
    mapping, grid = get_stabilized_reciprocal_mesh(
        mesh,
        rotations,
        is_shift=is_shift,
        is_time_reversal=is_time_reversal)
        
    ir_list = np.unique(mapping)
    weights = np.zeros_like(mapping)
    ir_qpoints = np.zeros((len(ir_list), 3), dtype='double')

    for g in mapping:
        weights[g]  += 1
    ir_weights = weights[ir_list]

    shift = np.array(is_shift, dtype='intc') * 0.5
    for i, g in enumerate(ir_list):
        ir_qpoints[i] = (grid[g] + shift) / mesh
        ir_qpoints[i] -= (ir_qpoints[i] > 0.5) * 1

    return ir_qpoints, ir_weights

def _get_qpoints_without_symmetry(mesh, shift, is_gamma_center):
    qpoints = []
    mesh_float = np.array(mesh, dtype='double')
    if is_gamma_center or (shift is None):
        qshift = [0, 0, 0]
    else:
        qshift = shift / mesh_float
        for i in (0, 1, 2):
            if mesh[i] % 2 == 0:
                qshift[i] += 0.5 / mesh[i]

    for grid_address in list(np.ndindex(tuple(mesh))):
        q = grid_address / mesh_float + qshift
        qpoints.append(q - (q > 0.5))

    qpoints = np.array(qpoints, dtype='double')
    weights = np.ones(qpoints.shape[0], dtype='intc')

    return qpoints, weights


class Mesh:
    def __init__(self,
                 dynamical_matrix,
                 mesh,
                 shift=None,
                 is_time_reversal=False,
                 is_symmetry=True,
                 is_eigenvectors=False,
                 is_gamma_center=False,
                 group_velocity=None,
                 rotations=None, # Point group operations in real space
                 factor=VaspToTHz):
        self._mesh = np.array(mesh, dtype='intc')
        self._is_eigenvectors = is_eigenvectors
        self._factor = factor
        self._cell = dynamical_matrix.get_primitive()
        self._dynamical_matrix = dynamical_matrix
        self._qpoints, self._weights = get_qpoints(
            self._mesh,
            np.linalg.inv(self._cell.get_cell()),
            q_mesh_shift=shift,
            is_gamma_center=is_gamma_center,
            is_time_reversal=is_time_reversal,
            rotations=rotations,
            is_symmetry=is_symmetry)

        self._frequencies = None
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

    def get_group_velocities(self):
        return self._group_velocities
    
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
                    f.write("[ %13.7f, %13.7f, %13.7f ]\n" %
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
