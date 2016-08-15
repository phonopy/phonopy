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
from phonopy.units import AMU, kb_J

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
                 symmetry,
                 velocities): # m/s
        self._supercell = supercell
        self._primitive = primitive
        self._symmetry = symmetry
        self._velocities = velocities

        symprec = symmetry.get_symmetry_tolerance()
        (self._shortest_vectors,
         self._multiplicity) = get_smallest_vectors(supercell,
                                                    primitive,
                                                    symprec)
        self._transform([0, 0, 0])

    def run(self):
        pass

    def _transform(self, q):
        """ exp(i q.r(i)) v(i)"""

        m = self._primitive.get_masses()
        num_s = self._supercell.get_number_of_atoms()
        num_p = self._primitive.get_number_of_atoms()
        N = num_s / num_p
        v = self._velocities
        v_q = np.zeros((v.shape[0], num_p, 3), dtype='complex128')

        for p_i in range(num_p):
            for s_i in range(num_s):
                pf = np.sqrt(m[p_i]) / N * self._get_phase_factor(p_i, s_i, q)
                v_q[:, p_i, :] += pf * v[:, s_i, :]

    def _get_phase_factor(self, p_i, s_i, q):
        multi = self._multiplicity[s_i, p_i]
        pos = self._shortest_vectors[s_i, p_i, :multi]
        return np.exp(-2j * np.pi * np.dot(q, pos.T)).sum()
        

class AutoCorrelation:
    def __init__(self,
                 velocities, # in m/s
                 masses=None, # in AMU
                 temperature=None): # in K
        self._velocities = velocities
        self._masses = masses
        self._temperature = temperature

        self._vv_real = None # real space auto correlation
        self._n_elements = 0

    def run(self, num_frequency_points):
        v = self._velocities
        max_lag = num_frequency_points * 2
        n_elem = len(v) - max_lag

        if n_elem < 1:
            return False

        vv = np.zeros((max_lag,) + v.shape[1:], dtype='double', order='C')

        d = max_lag / 2
        for i in range(max_lag):
            vv[i - d] = (v[d:(d + n_elem)] * v[i:(i + n_elem)]).sum(axis=0)

        self._vv_real = vv
        if self._masses is not None and self._temperature is not None:
            for i, m in enumerate(self._masses):
                self._vv_real[:, i, :] *= m * AMU / (kb_J * self._temperature)

        self._n_elements = n_elem

        return True

    def run_at_q(self, q, num_frequency_points):
        pass

    def get_vv_real(self):
        return self._vv_real

    def get_number_of_elements(self):
        return self._n_elements

