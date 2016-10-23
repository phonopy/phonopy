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
from phonopy.phonon.degeneracy import rotate_eigenvectors

class Gruneisen(object):
    def __init__(self,
                 dynmat,
                 dynmat_plus,
                 dynmat_minus,
                 qpoints=None,
                 is_band_connection=False):
        self._dynmat = dynmat
        self._dynmat_plus = dynmat_plus
        self._dynmat_minus = dynmat_minus
        self._volume = dynmat.get_primitive().get_volume()
        self._volume_plus = dynmat_plus.get_primitive().get_volume()
        self._volume_minus = dynmat_minus.get_primitive().get_volume()
        self._is_band_connection = is_band_connection
        self._qpoints = qpoints

        self._gruneisen = None
        self._gamma_prime = None
        self._eigenvalues = None
        if qpoints is not None:
            self._set_gruneisen()


    def set_qpoints(self, qpoints):
        self._qpoints = qpoints
        self._set_gruneisen()

    def get_gruneisen(self):
        return self._gruneisen

    def get_gamma_prime(self):
        return self._gamma_prime

    def get_eigenvalues(self):
        return self._eigenvalues

    def _set_gruneisen(self):
        dV = self._volume_plus - self._volume_minus

        if self._is_band_connection:
            self._q_direction = self._qpoints[0] - self._qpoints[-1]
            band_order = range(self._dynmat.get_dimension())
            prev_eigvecs = None

        edDe = [] # <e|dD|e>
        eigvals = []
        for i, q in enumerate(self._qpoints):
            if self._is_band_connection and self._dynmat.is_nac():
                self._dynmat.set_dynamical_matrix(
                    q, q_direction=self._q_direction)
            else:
                self._dynmat.set_dynamical_matrix(q)

            dm = self._dynmat.get_dynamical_matrix()
            evals, evecs = np.linalg.eigh(dm)
            evals_at_q = evals.real
            dD = self._get_dD(q, self._dynmat_minus, self._dynmat_plus)
            evecs_at_q, edDe_at_q = rotate_eigenvectors(evals_at_q, evecs, dD)

            if self._is_band_connection:
                if prev_eigvecs is not None:
                    band_order = estimate_band_connection(
                        prev_eigvecs,
                        evecs_at_q,
                        band_order)
                eigvals.append([evals_at_q[b] for b in band_order])
                edDe.append([edDe_at_q[b] for b in band_order])
                prev_eigvecs = evecs_at_q
            else:
                eigvals.append(evals_at_q)
                edDe.append(edDe_at_q)

        edDe = np.array(edDe, dtype='double', order='C')
        self._eigenvalues = np.array(eigvals, dtype='double', order='C')
        self._gruneisen = -edDe / dV / self._eigenvalues * self._volume / 2


    def _get_dD(self, q, d_a, d_b):
        if (self._is_band_connection and d_a.is_nac() and d_b.is_nac()):
            d_a.set_dynamical_matrix(q, q_direction=self._q_direction)
            d_b.set_dynamical_matrix(q, q_direction=self._q_direction)
        else:
            d_a.set_dynamical_matrix(q)
            d_b.set_dynamical_matrix(q)
        dm_a = d_a.get_dynamical_matrix()
        dm_b = d_b.get_dynamical_matrix()
        return (dm_b - dm_a)
