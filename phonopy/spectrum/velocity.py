# Copyright (C) 2016 Atsushi Togo
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
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.units import AMU, kb_J
from phonopy.structure.grid_points import get_qpoints

class Velocity:
    def __init__(self,
                 lattice=None, # column vectors, in Angstrom
                 positions=None, # fractional coordinates
                 timestep=None): # in femtosecond

        self._lattice = lattice
        self._positions = positions
        self._timestep = timestep
        self._velocities = None # m/s

    def run(self, skip_steps=0):
        pos = self._positions
        diff = pos[(skip_steps + 1):] - pos[skip_steps:-1]
        diff = np.where(diff > 0.5, diff - 1, diff)
        diff = np.where(diff < -0.5, diff + 1, diff)
        self._velocities = np.dot(diff, self._lattice.T * 1e5) / self._timestep

    def get_velocities(self):
        return self._velocities

    def get_timestep(self):
        return self._timestep

class VelocityQ:
    def __init__(self,
                 supercell,
                 primitive,
                 velocities, # m/s
                 symprec=1e-5):
        self._supercell = supercell
        self._primitive = primitive
        self._velocities = velocities

        (self._shortest_vectors,
         self._multiplicity) = get_smallest_vectors(supercell,
                                                    primitive,
                                                    symprec)

    def run(self, q):
        self._velocities = self._transform(q)

    def get_velocities(self):
        return self._velocities

    def _transform(self, q):
        """ exp(i q.r(i)) v(i)"""

        num_s = self._supercell.get_number_of_atoms()
        num_p = self._primitive.get_number_of_atoms()
        v = self._velocities
        v_q = np.zeros((v.shape[0], num_p, 3), dtype='complex128')

        for p_i in range(num_p):
            for s_i in range(num_s):
                pf = self._get_phase_factor(p_i, s_i, q)
                v_q[:, p_i, :] += pf * v[:, s_i, :]

        return v_q

    def _get_phase_factor(self, p_i, s_i, q):
        multi = self._multiplicity[s_i, p_i]
        pos = self._shortest_vectors[s_i, p_i, :multi]
        return np.exp(-2j * np.pi * np.dot(q, pos.T)).sum() / multi

class VelocityQpoints(VelocityQ):
    def __init__(self,
                 supercell,
                 primitive,
                 velocities, # m/s
                 symmetry=None,
                 symprec=1e-5):
        if symmetry is not None:
            symprec = symmetry.get_symmetry_tolerance()
            self._point_group_opts = symmetry.get_pointgroup_operations()
        else:
            self._point_group_opts = None
        VelocityQ.__init__(self,
                           supercell,
                           primitive,
                           velocities,
                           symprec=symprec)
        self._qpoints = None
        self._weights = None

    def run(self, verbose=False):
        num_s = self._supercell.get_number_of_atoms()
        num_p = self._primitive.get_number_of_atoms()
        N = num_s / num_p
        v = self._velocities
        v_q = np.zeros((v.shape[0], num_p, len(self._qpoints), 3),
                       dtype='complex128')
        
        for i, q in enumerate(self._qpoints):
            if verbose:
                print("%d/%d" % (i + 1, len(self._qpoints)))
            v_q[:, :, i, :] = self._transform(q)
        self._velocities = v_q

    def set_mesh(self, mesh):
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._qpoints, self._weights = get_qpoints(
            mesh,
            rec_lat,
            is_gamma_center=True,
            rotations=self._point_group_opts)

    def set_qpoints(self, qpoints):
        self._weights = np.ones(len(qpoints), dtype='intc')
        self._qpoints = qpoints

    def set_commensurate_points(self):
        supercell_matrix = np.linalg.inv(self._primitive.get_primitive_matrix())
        supercell_matrix = np.rint(supercell_matrix).astype('intc')
        self.set_qpoints(get_commensurate_points(supercell_matrix))

    def get_qpoints(self):
        return self._qpoints, self._weights

class AutoCorrelation:
    def __init__(self,
                 velocities, # in m/s
                 masses=None, # in AMU
                 temperature=None): # in K
        self._velocities = velocities
        self._masses = masses
        self._temperature = temperature

        self._vv = None
        self._n_elements = 0

    def run(self, num_frequency_points, verbose=False):
        v = self._velocities
        max_lag = num_frequency_points * 2
        n_elem = len(v) - max_lag

        if n_elem < 1:
            return False

        vv = np.zeros((max_lag,) + v.shape[1:], dtype=v.dtype, order='C')

        d = max_lag / 2
        for i in range(max_lag):
            if verbose:
                print("%d/%d" % (i + 1, max_lag))
            if np.iscomplexobj(vv):
                vv[i - d] = (v[d:(d + n_elem)] *
                             v[i:(i + n_elem)].conj()).sum(axis=0)
            else:
                vv[i - d] = (v[d:(d + n_elem)] *
                             v[i:(i + n_elem)]).sum(axis=0)

        self._vv = vv
        if self._masses is not None and self._temperature is not None:
            for i, m in enumerate(self._masses):
                self._vv[:, i] *= m * AMU / (kb_J * self._temperature)

        self._n_elements = n_elem

        return True

    def get_autocorrelation(self):
        return self._vv

    def get_number_of_elements(self):
        return self._n_elements

