# Copyright (C) 2012 Atsushi Togo
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
from phonopy.phonon.band_structure import estimate_band_connection

class Gruneisen:
    def __init__(self,
                 dynmat,
                 dynmat_plus,
                 dynmat_minus,
                 volume,
                 volume_plus,
                 volume_minus,
                 qpoints=None,
                 is_band_connection=False):
        self._dynmat = dynmat
        self._dynmat_plus = dynmat_plus
        self._dynmat_minus = dynmat_minus
        self._volume = volume
        self._dV = volume_plus - volume_minus
        self._is_band_connection = is_band_connection
        self._qpoints = qpoints

        self._gruneisen = None
        self._eigenvalues = None
        if qpoints is not None:
            self._set_gruneisen()

        if self._is_band_connection:
            self._band_order = range(self._dynmat.get_dimension())
            self._prev_eigvecs = None

    def set_qpoints(self, qpoints):
        self._qpoints = qpoints
        self._set_gruneisen()
        
    def get_gruneisen(self):
        return self._gruneisen

    def get_eigenvalues(self):
        return self._eigenvalues

    def _set_gruneisen(self):
        if self._is_band_connection:
            self._q_direction = self._qpoints[0] - self._qpoints[-1]
        dD = []
        eigvals = []
        for i, q in enumerate(self._qpoints):
            if (self._is_band_connection and
                self._dynmat.is_nac()):
                self._dynmat.set_dynamical_matrix(
                    q, q_direction=self._q_direction)
            else:
                self._dynmat.set_dynamical_matrix(q)

            dm = self._dynmat.get_dynamical_matrix()
            eigvals_at_q, eigvecs = np.linalg.eigh(dm)
            eigvals_at_q = eigvals_at_q.real
            dD_at_q = [np.vdot(eig, np.dot(self._get_dD(q), eig)).real
                       for eig in eigvecs.T]

            if self._is_band_connection:
                if self._prev_eigvecs is not None:
                    self._band_order = estimate_band_connection(
                        self._prev_eigvecs,
                        eigvecs,
                        self._band_order)
                eigvals.append([eigvals_at_q[b] for b in self._band_order])
                dD.append([dD_at_q[b] for b in self._band_order])
                self._prev_eigvecs = eigvecs
            else:
                eigvals.append(eigvals_at_q)
                dD.append(dD_at_q)

        dD = np.array(dD)
        eigvals = np.array(eigvals)
            
        self._gruneisen = -dD / eigvals * self._volume  / self._dV / 2
        self._eigenvalues = eigvals

    def _get_dD(self, q):
        if (self._is_band_connection and
            self._dynmat_plus.is_nac() and 
            self._dynmat_minus.is_nac()):
            self._dynmat_plus.set_dynamical_matrix(
                q, q_direction=self._q_direction)
            self._dynmat_minus.set_dynamical_matrix(
                q, q_direction=self._q_direction)
        else:
            self._dynmat_plus.set_dynamical_matrix(q)
            self._dynmat_minus.set_dynamical_matrix(q)
        dm_plus = self._dynmat_plus.get_dynamical_matrix()
        dm_minus = self._dynmat_minus.get_dynamical_matrix()
        return dm_plus - dm_minus


