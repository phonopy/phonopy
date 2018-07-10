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

    def __init__(self, eigenvalues, weights, n_electrons):
        """

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues.
            dtype='double'
            shape=(spin, kpoints, bands)
        weights: ndarray
            Geometric k-point weights (number of arms of k-star in BZ).
            dtype='intc'
            shape=(irreducible_kpoints,)
        n_electrons: float
            Number of electrons in unit cell.

        """

        self._energies = energies
        self._weights = weights
        self._n_electrons = n_electrons

        self._n_electrons = None
        self.chemical_potential = None

    def run(self, T):
        pass

    def _entropy(self):
        mu = self.chemical_potential
        g = 3 - len(energies)
        S = 0
        for energies_spin in energies:
            for E, w in zip(np.array(energies_spin), weights):
                f = 1.0 / (1 + np.exp((E - mu) / T))
                f = np.extract((f > 1e-10) * (f < 1 - 1e-10), f)
                S += - np.sum(f * np.log(f) + (1 - f) * np.log(1 - f)) * w
        return S * g

    def _chemical_potential(self):
        emax = np.max(energies)
        emin = np.min(energies)

        for i in range(100):
            mu = (emax + emin) / 2
            n = self._number_of_electrons(energies, weights, mu, T)
            if abs(n - num_electrons) < 1e-8:
                break
            elif (n < num_electrons):
                emin = mu
            else:
                emax = mu

        return mu

    def _number_of_electrons(self, mu):
        g = 3 - self._energies.shape[0] # 2 or 1
        mu = chemical_potential
        n = 0
        for energies_spin in energies:
            for E, w in zip(np.array(energies_spin), weights):
                n += np.sum(1.0 / (1 + np.exp((E - mu) / T))) * w
        return n * g
