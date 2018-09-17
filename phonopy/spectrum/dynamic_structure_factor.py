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
from phonopy.units import THzToEv, Kb, AMU, THz
from phonopy.phonon.qpoints import QpointsPhonon
from phonopy.phonon.thermal_displacement import ThermalDisplacements


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
#
#
# Neutron scattering length
# https://www.ncnr.nist.gov/resources/n-lengths/
# Exmple: {'Na': 3.63,
#          'Cl': 9.5770}

def atomic_form_factor_WK1995(Q, f_x):
    a, b = np.array(f_x[:10]).reshape(-1, 2).T
    return (a * np.exp(-b * Q ** 2)).sum() + f_x[10]


class DynamicStructureFactor(object):
    """Calculate dynamic structure factor

    Result is given in m^2/J with setting k'/k * N = 1 when b is given
    in Angstron.

    Note
    ----
    In computation, the heaviest part is the calculation of thermal
    displacements that is used in Deby-Waller factor. The heavy part
    of the thermal displacement is computed many times for the same
    values. Therefore It is possible to improve the performance of
    dynamic structure factor, but it requires to make
    ThermalDisplacements keep Q2 values in its instance.

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
                 mesh_phonon,
                 qpoints,
                 G,
                 T,
                 func_atomic_form_factor=None,
                 scattering_lengths=None,
                 freq_min=None,
                 freq_max=None):
        """

        Parameters
        ----------
        mesh_phonon: Mesh or IterMesh
            Mesh phonon instance that is ready to get frequencies and
            eigenvectors.
        qpoints: array_like
            q-points measured from G-point.
            dtype='double'
            shape=(qpoints, 3)
        G: array_like
            G-point.
            dtype='double'
            shape=(3, )
        T: float
            Temperature in K.
        func_atomic_form_factor: Function object
            Function that returns atomic form factor (``func`` below):

                f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                                   0.767888, 0.070139, 0.995612, 14.1226457,
                                   0.968249, 0.217037, 0.045300],
                            'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                                   6.524271, 19.467656, 2.355626, 60.320301,
                                   35.829404, 0.000436, -34.916604],b|

                def get_func_AFF(f_params):
                    def func(symbol, Q):
                        return atomic_form_factor_WK1995(Q, f_params[symbol])
                    return func

        scattering_lengths: dictionary
            Coherent scattering lengths averaged over isotopes and spins.
            Supposed for INS. For example, {'Na': 3.63, 'Cl': 9.5770}.
        freq_min: float
            Minimum phonon frequency to determine wheather include or not.
        freq_max: float
            Maximum phonon frequency to determine wheather include or not. Only
            for Debye-Waller factor.

        """

        self._mesh_phonon = mesh_phonon
        self._dynamical_matrix = mesh_phonon.dynamical_matrix
        self._primitive = self._dynamical_matrix.primitive
        self._qpoints = np.array(qpoints)  # (n_q, 3) array
        self._G = G
        self._func_AFF = func_atomic_form_factor
        self._b = scattering_lengths
        self._T = T
        if freq_min is None:
            self._fmin = 0
        else:
            self._fmin = freq_min
        if freq_max is None:
            self._fmax = None
        else:
            self._fmax = freq_max

        self._rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._freqs = None
        self._eigvecs = None
        self._set_phonon()
        self._q_count = 0
        self._unit_convertion_factor = 1.0 / (AMU * (2 * np.pi * THz) ** 2)

        self.Qpoints = self._qpoints + np.array(G)  # reciprocal lattice points
        self.S = np.zeros(self._freqs.shape, dtype='double', order='C')

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._q_count == len(self._qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            S = self._run_at_Q()
            self.S[self._q_count] = S
            self._q_count += 1
            return S

    def run(self):
        for S in self:
            pass

    def _run_at_Q(self):
        freqs = self._freqs[self._q_count]
        eigvecs = self._eigvecs[self._q_count]
        Q_cart = np.dot(self._rec_lat, self.Qpoints[self._q_count])
        Q_length = np.linalg.norm(Q_cart)
        if Q_length < 1e-8:
            DW = np.zeros(len(self._primitive.get_number_of_atoms()),
                          dtype='double')
        else:
            _, disps = self._get_thermal_displacements(Q_cart)
            DW = np.exp(-0.5 * Q_length ** 2 * disps[0])
        S = np.zeros(len(freqs), dtype='double')
        for i, f in enumerate(freqs):
            if self._fmin < f:
                F = self._phonon_structure_factor(Q_cart, DW, f,
                                                  eigvecs[:, i])
                n = 1.0 / (np.exp(f * THzToEv / (Kb * self._T)) - 1)
                S[i] = abs(F) ** 2 * (n + 1)
        return S * self._unit_convertion_factor

    def _set_phonon(self):
        qpoints_phonon = QpointsPhonon(self._qpoints,
                                       self._dynamical_matrix,
                                       is_eigenvectors=True)
        self._freqs = qpoints_phonon.frequencies
        self._eigvecs = qpoints_phonon.eigenvectors

    def _get_thermal_displacements(self, proj_dir):
        td = ThermalDisplacements(self._mesh_phonon,
                                  projection_direction=proj_dir,
                                  freq_min=self._fmin,
                                  freq_max=self._fmax)
        td.set_temperatures([self._T])
        td.run()
        return td.get_thermal_displacements()

    def _phonon_structure_factor(self, Q_cart, DW, freq, eigvec):
        symbols = self._primitive.get_chemical_symbols()
        masses = self._primitive.get_masses()
        pos = self._primitive.get_scaled_positions()
        phase = np.exp(-2j * np.pi * np.dot(pos, self._G))
        W = eigvec.reshape(-1, 3)
        val = 0
        for i, m in enumerate(masses):
            if self._func_AFF is not None:
                f = self._func_AFF(symbols[i], np.linalg.norm(Q_cart))
            elif self._b is not None:
                f = self._b[symbols[i]]
            else:
                raise RuntimeError
            QW = np.dot(Q_cart, W[i])
            val += f / np.sqrt(2 * m) * DW[i] * QW * phase[i]
        val /= np.sqrt(freq)
        return val
