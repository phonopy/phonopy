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
from phonopy.units import THzToEv, Kb


# D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
# f(Q) = \sum_i a_i \exp((-b_i Q^2) + c
# Q is in angstron^-1
# a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
#
# Examples:
#  {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
#          0.767888, 0.070139, 0.995612, 14.1226457,
#          0.968249, 0.217037, 0.045300],  # 1+
#   'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
#          6.524271, 19.467656, 2.355626, 60.320301,
#          35.829404, 0.000436, -34.916604],  # 1-
#   'Si': [5.275329, 2.631338, 3.191038, 33.730728,
#          1.511514, 0.081119, 1.356849, 86.288640,
#          2.519114, 1.170087, 0.145073]}  # neutral

def atomic_form_factor(Q, f_x):
    a, b = np.array(f_x[:10]).reshape(-1, 2).T
    return (a * np.exp(-b * Q ** 2)).sum() + f_x[10]


class DynamicStructureFactor(object):
    """Class to generate irreducible grid points on uniform mesh grids

    Attributes
    ----------
    qpoints: ndarray
       q-points in reduced coordinates of reciprocal lattice with G shifted.
       dtype='double'
       shape=(qpoints, 3)
    S: ndarray
       Dynamic structure factor
       dtype='double'
       shape=(qpoints, phonon bands)

    """

    def __init__(self,
                 phonon,
                 q_points,
                 f_params,
                 T,
                 G=None,
                 cutoff_frequency=1e-3):
        self._phonon = phonon
        self._G = np.array(G)  # reciprocal lattice points
        self._q_points = np.array(q_points)  # (n_q, 3) array
        self._f_params = f_params
        self._T = T
        self._cutoff_frequency = cutoff_frequency

        self._primitive = phonon.get_primitive()
        self._rec_lat = np.linalg.inv(self._primitive.get_cell())

        self._freqs = None
        self._eigvecs = None
        self._set_phonon()

        self.qpoints = None
        self.S = None

    def run(self):
        self.qpoints = self._q_points + self._G
        self.S = []
        num_atom = self._primitive.get_number_of_atoms()
        for Q, freqs, eigvecs in zip(self.qpoints, self._freqs, self._eigvecs):
            temps, disps = self._get_thermal_displacements(Q)
            DW = np.exp(-0.5 * np.linalg.norm(Q) * disps[0])
            S = np.zeros(num_atom * 3, dtype='double')
            for i in range(num_atom * 3):
                if freqs[i] > self._cutoff_frequency:
                    F = self._phonon_structure_factor(Q, DW, freqs[i],
                                                      eigvecs[:, i])
                    n = 1.0 / (np.exp(freqs[i] * THzToEv / (Kb * self._T)) - 1)
                    S[i] = abs(F) ** 2 * (n + 1)
            self.S.append(S)
        self.S = np.array(self.S, dtype='double', order='C')

    def _set_phonon(self):
        self._phonon.set_qpoints_phonon(self._q_points, is_eigenvectors=True)
        self._freqs, self._eigvecs = self._phonon.get_qpoints_phonon()

    def _get_thermal_displacements(self, Q):
        self._phonon.set_thermal_displacements(temperatures=[self._T],
                                               direction=Q,
                                               freq_min=1e-3)
        return self._phonon.get_thermal_displacements()

    def _phonon_structure_factor(self, Q, DW, freq, eigvec):
        num_atom = self._primitive.get_number_of_atoms()
        pos = self._primitive.get_scaled_positions()
        symbols = self._primitive.get_chemical_symbols()
        masses = self._primitive.get_masses()
        W = eigvec.reshape(-1, 3)
        val = 0
        Q_cart = np.dot(self._rec_lat, Q)
        for i in range(num_atom):
            m = masses[i]
            f = atomic_form_factor(np.linalg.norm(Q_cart),
                                   self._f_params[symbols[i]])
            phase = np.exp(2j * np.pi * np.dot(Q, pos[i]))
            QW = np.dot(Q_cart, W[i])
            val += f / np.sqrt(2 * m) * DW[i] * QW * phase
        val /= np.sqrt(freq)
        return val
