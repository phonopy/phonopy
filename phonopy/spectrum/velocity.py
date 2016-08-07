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

class Velocity:
    def __init__(self,
                 positions=None, # fractional coordinates
                 lattice=None, # column vectors, in Angstrom
                 timestep=None): # in femtosecond

        self._positions = positions
        self._lattice = lattice
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

class AutoCorrelation:
    def __init__(self, velocities): # in m/s
        self._velocities = velocities
        self._vv_real = None # real space auto correlation
        self._n_elements = 0

    def run(self, num_frequency_points):
        v = self._velocities
        max_lag = num_frequency_points * 2
        n_elem = len(v) - max_lag

        if n_elem < 1:
            return None

        vv = np.zeros((max_lag,) + v.shape[1:], dtype='double', order='C')

        d = max_lag / 2
        for i in range(max_lag):
            vv[i - d] = (v[d:(d + n_elem)] * v[i:(i + n_elem)]).sum(axis=0)

        self._vv_real = vv
        self._n_elements = n_elem

    def run_at_q(self, q, num_frequency_points):
        pass

    def get_vv_real(self):
        return self._vv_real

    def get_number_of_elements(self):
        return self._n_elements
