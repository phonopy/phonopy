"""Calculate dynamic structure factor at harmonic level."""

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

from typing import Union

import numpy as np

from phonopy.phonon.mesh import IterMesh, Mesh
from phonopy.phonon.qpoints import QpointsPhonon
from phonopy.phonon.random_displacements import bose_einstein_dist
from phonopy.phonon.thermal_displacement import ThermalDisplacements
from phonopy.structure.brillouin_zone import get_qpoints_in_Brillouin_zone
from phonopy.units import AMU, THz


class DynamicStructureFactor:
    r"""Calculate dynamic structure factor at harmonic level.

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

    Atomic form factor
    ------------------
    D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
    f(s) = \sum_i a_i \exp((-b_i s^2) + c
    a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
    s is defined by |Q|/2 in angstron^-1 where Q without 2pi.

    Examples
    --------
     {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
             0.767888, 0.070139, 0.995612, 14.1226457,
             0.968249, 0.217037, 0.045300],  # 1+
      'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
             6.524271, 19.467656, 2.355626, 60.320301,
             35.829404, 0.000436, -34.916604],  # 1-
      'Si': [5.275329, 2.631338, 3.191038, 33.730728,
             1.511514, 0.081119, 1.356849, 86.288640,
             2.519114, 1.170087, 0.145073]}  # neutral

    Neutron scattering length
    -------------------------
    https://www.ncnr.nist.gov/resources/n-lengths/
    Exmple: {'Na': 3.63,
             'Cl': 9.5770}

    Attributes
    ----------
    qpoints: ndarray
       q-points in reduced coordinates measured from nearest G point.
       dtype='double'
       shape=(qpoints, 3)
    dynamic_structure_factors: ndarray
       Dynamic structure factors.
       dtype='double'
       shape=(qpoints, phonon bands)

    """

    def __init__(
        self,
        mesh_phonon: Union[Mesh, IterMesh],
        Qpoints,
        T,
        atomic_form_factor_func=None,
        scattering_lengths=None,
        freq_min=None,
        freq_max=None,
    ):
        """Init method.

        Parameters
        ----------
        mesh_phonon: Mesh or IterMesh
            Mesh phonon instance that is ready to get frequencies and
            eigenvectors.
        Qpoints: array_like
            Q-points in any Brillouin zone.
            dtype='double'
            shape=(qpoints, 3)
        T: float
            Temperature in K.
        atomic_form_factor_func: Function object
            Function that returns atomic form factor (``func`` below):

                f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                                   0.767888, 0.070139, 0.995612, 14.1226457,
                                   0.968249, 0.217037, 0.045300],
                            'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                                   6.524271, 19.467656, 2.355626, 60.320301,
                                   35.829404, 0.000436, -34.916604],b|

                def get_func_AFF(f_params):
                    def func(symbol, s):
                        return atomic_form_factor_WK1995(s, f_params[symbol])
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
        self._Qpoints = np.array(Qpoints)  # (n_q, 3) array

        self._func_AFF = atomic_form_factor_func
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

        self._rec_lat = np.linalg.inv(self._primitive.cell)
        self.qpoints = None
        self._set_qpoints()  # self.qpoints needed in self._set_phonon()
        self.frequencies = None
        self._eigvecs = None
        self._set_phonon()

        self._q_count = 0
        self._unit_convertion_factor = 1.0 / (AMU * (2 * np.pi * THz) ** 2)

        self.dynamic_structure_factors = np.zeros(
            self.frequencies.shape, dtype="double", order="C"
        )

    def __iter__(self):
        """Define iterator of calculation over q-points."""
        return self

    def __next__(self):
        """Calculate at next q-point."""
        if self._q_count == len(self._Qpoints):
            self._q_count = 0
            raise StopIteration
        else:
            S = self._run_at_Q()
            self.dynamic_structure_factors[self._q_count] = S
            self._q_count += 1
            return S

    def run(self):
        """Calculate at all q-points."""
        for _ in self:
            pass

    def _run_at_Q(self):
        freqs = self.frequencies[self._q_count]
        eigvecs = self._eigvecs[self._q_count]
        Q_cart = np.dot(self._rec_lat, self._Qpoints[self._q_count])
        G_vector = self._Qpoints[self._q_count] - self.qpoints[self._q_count]
        Q_length = np.linalg.norm(Q_cart)
        if Q_length < 1e-8:
            debye_waller = np.zeros(len(self._primitive), dtype="double")
        else:
            _, disps = self._get_thermal_displacements(Q_cart)
            debye_waller = np.exp(-0.5 * (2 * np.pi * Q_length) ** 2 * disps[0])
        S = np.zeros(len(freqs), dtype="double")
        for i, f in enumerate(freqs):
            if self._fmin < f:
                F = self._phonon_structure_factor(
                    Q_cart,
                    G_vector,
                    debye_waller,
                    f,
                    eigvecs[:, i],
                )
                n = bose_einstein_dist(f, self._T)
                S[i] = abs(F) ** 2 * (n + 1)
        return S * self._unit_convertion_factor

    def _set_phonon(self):
        qpoints_phonon = QpointsPhonon(
            self.qpoints, self._dynamical_matrix, with_eigenvectors=True
        )
        self.frequencies = qpoints_phonon.frequencies
        self._eigvecs = qpoints_phonon.eigenvectors

    def _get_thermal_displacements(self, proj_dir):
        td = ThermalDisplacements(
            self._mesh_phonon,
            projection_direction=proj_dir,
            freq_min=self._fmin,
            freq_max=self._fmax,
        )
        td.temperatures = [self._T]
        td.run()
        return td.temperatures, td.thermal_displacements

    def _phonon_structure_factor(self, Q_cart, G_vector, DW, freq, eigvec):
        """Return F(Q, q nu).

        The phase factor is different by exp(iq.r) from that of the book
        "Thermal neutron scattering" because of different difinition of
        dynamical matrix.

        """
        symbols = self._primitive.symbols
        masses = self._primitive.masses
        pos = self._primitive.scaled_positions
        phase = np.exp(2j * np.pi * np.dot(pos, G_vector))
        eigvec_atoms = eigvec.reshape(-1, 3)
        val = 0
        for i, m in enumerate(masses):
            if self._func_AFF is not None:
                f = self._func_AFF(symbols[i], np.linalg.norm(Q_cart) / 2)
            elif self._b is not None:
                f = self._b[symbols[i]]
            else:
                raise RuntimeError
            QW = np.dot(Q_cart, eigvec_atoms[i]) * 2 * np.pi
            val += f / np.sqrt(2 * m) * DW[i] * QW * phase[i]
        val /= np.sqrt(freq)
        return val

    def _set_qpoints(self):
        qpoints = get_qpoints_in_Brillouin_zone(self._rec_lat, self._Qpoints)
        self.qpoints = np.array([q[0] for q in qpoints], dtype="double", order="C")


def atomic_form_factor_WK1995(s, f_x):
    """Return atomic form factor of WK1995.

    D. Waasmaier and A. Kirfel, Acta Cryst. (1995). A51, 416-431

    s = sin(theta)/lambda = |k' - k|/2 = |Q|/2

    where k, k', Q are given without 2pi.

    f_x = [a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c]

    """
    a, b = np.array(f_x[:10]).reshape(-1, 2).T
    return (a * np.exp(-b * s**2)).sum() + f_x[10]


def atomic_form_factor_ITC(s, f_x):
    """Return atomic form factor of international tables for crystallography C.

    ITC table 6.1.1.4.

    s = sin(theta)/lambda = |k' - k|/2 = |Q|/2

    where k, k', Q are given without 2pi.

    f_x = [a1, b1, a2, b2, a3, b3, a4, b4, c]

    """
    a, b = np.array(f_x[:8]).reshape(-1, 2).T
    return (a * np.exp(-b * s**2)).sum() + f_x[8]
