# Copyright (C) 2018 Atsushi Togo
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
from phonopy.units import Kb

class ElectronFreeEnergy(object):
    """Fixed density-of-states approximation for energy and entropy of electrons

    This is supposed to be used for metals, i.e., chemical potential is not
    in band gap.

    Entropy
    -------

    .. math::

       S_i = -gk_{\mathrm{B}}\Sigma_i\[f_i \ln f_i + (1-f_i)\ln (1-f_i)\]

    .. math::

       f_i = \left[1+\exp\left(\frac{\epsilon_i - \mu}{T}\right)\right\]^{-1}

    where :math:`g` is 1 for non-spin polarized systems and 2 for spin
    polarized systems.

    Energy
    ------

    .. math::

       E_i = f_i \epsilon_i

    Attributes
    ----------
    chemical_potential:
        Chemical potential


    """

    def __init__(self, eigenvalues, weights, n_electrons, initial_efermi=0.0):
        """

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues in eV.
            dtype='double'
            shape=(spin, kpoints, bands)
        weights: ndarray
            Geometric k-point weights (number of arms of k-star in BZ).
            dtype='intc'
            shape=(irreducible_kpoints,)
        n_electrons: float
            Number of electrons in unit cell.
        efermi: float
            Initial Fermi energy

        """

        # shape=(kpoints, spin, bands)
        self._eigenvalues = np.array(eigenvalues.swapaxes(0, 1),
                                     dtype='double', order='C')
        self._weights = weights
        self._n_electrons = n_electrons
        self._initial_efermi = initial_efermi

        self.chemical_potential = None

        if self._eigenvalues.shape[1] == 1:
            self._g = 2
        elif self._eigenvalues.shape[1] == 2:
            self._g = 1
        else:
            raise RuntimeError

        self._T = None
        self.mu = None
        self.entropy = None
        self.energy = None

    def run(self, T):
        """

        Parameters
        ----------
        T: float
            Temperature in K

        """

        self._T = T * Kb
        self.mu = self._chemical_potential()
        self.f = self._f(self._eigenvalues, self.mu)
        self.entropy = self._entropy()
        self.energy = self._energy()

    def _entropy(self):
        S = 0
        for f_k, w in zip(self.f.reshape(len(self._weights), -1),
                          self._weights):
            _f = np.extract((f_k > 1e-12) * (f_k < 1 - 1e-12), f_k)
            S -= (_f * np.log(_f) + (1 - _f) * np.log(1 - _f)).sum() * w
        return S * self._g * self._T / self._weights.sum()

    def _energy(self):
        occ_eigvals = self.f * self._eigenvalues
        return np.dot(occ_eigvals.reshape(len(self._weights), -1).sum(axis=1),
                      self._weights) * self._g / self._weights.sum()

    def _chemical_potential(self):
        emin = np.min(self._eigenvalues)
        emax = np.max(self._eigenvalues)
        mu = (emin + emax) / 2

        for i in range(1000):
            n = self._number_of_electrons(mu)
            if abs(n - self._n_electrons) < 1e-10:
                break
            elif (n < self._n_electrons):
                emin = mu
            else:
                emax = mu
            mu = (emin + emax) / 2

        return mu

    def _number_of_electrons(self, mu):
        eigvals = self._eigenvalues.reshape(len(self._weights), -1)
        n = np.dot(self._f(eigvals, mu).sum(axis=1),
                   self._weights) * self._g / self._weights.sum()
        return n

    def _f(self, e, mu):
        return 1.0 / (1 + np.exp((e - mu) / self._T))
