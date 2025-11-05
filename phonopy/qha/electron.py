"""Calculation of free energy of one-electronic states."""

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

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units


def get_free_energy_at_T(
    tmin: float,
    tmax: float,
    tstep: float,
    eigenvalues: NDArray,
    weights: NDArray,
    n_electrons: float | None,
) -> tuple[NDArray, NDArray]:
    """Return free energies at given temperatures."""
    free_energies = []
    efe = ElectronFreeEnergy(eigenvalues, weights, n_electrons)
    temperatures = np.arange(tmin, tmax + 1e-8, tstep)
    for temp in temperatures:
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
    return temperatures, np.array(free_energies)


class ElectronFreeEnergy:
    r"""Class to calculate free energy of one-electronic states.

    Fixed density-of-states approximation for energy and entropy of electrons.

    This is supposed to be used for metals, i.e., chemical potential is not
    in band gap.

    Entropy
    -------

    .. math::

       S_\text{el}(V) = -gk_{\mathrm{B}}\Sigma_i \{ f_i(V) \ln f_i(V) +
       [1-f_i(V)]\ln [1-f_i(V)] \}

    .. math::

       f_i(V) = \left\{ 1 + \exp\left[\frac{\epsilon_i(V) - \mu(V)}{T}\right]
       \right\}^{-1}

    where :math:`g` is 1 for non-spin polarized systems and 2 for spin
    polarized systems.

    Energy
    ------

    .. math::

       E_\text{el}(V) = g\sum_i f_i(V) \epsilon_i(V)

    Attributes
    ----------
    entropy: float
        Entropy in eV (T * S).
    energy: float
        Energy in eV.
    free_energy: float
        energy - entropy in eV.
    mu: float
        Chemical potential in eV.

    """

    def __init__(self, eigenvalues, weights, n_electrons):
        """Init method.

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues in eV.
            dtype='double'
            shape=(spin, kpoints, bands)
        weights: ndarray
            Geometric k-point weights (number of arms of k-star in BZ).
            dtype='int_'
            shape=(irreducible_kpoints,)
        n_electrons: float
            Number of electrons in unit cell.
        efermi: float
            Initial Fermi energy

        """
        # shape=(kpoints, spin, bands)
        self._eigenvalues = np.array(
            eigenvalues.swapaxes(0, 1), dtype="double", order="C"
        )
        self._weights = weights
        self._n_electrons = n_electrons

        if self._eigenvalues.shape[1] == 1:
            self._g = 2
        elif self._eigenvalues.shape[1] == 2:
            self._g = 1
        else:
            raise RuntimeError

        self._T: float
        self._f: NDArray
        self._mu = None
        self._entropy = None
        self._energy = None

    def run(self, temp: float):
        """Calculate free energies.

        Parameters
        ----------
        temp: float
            Temperature in K

        """
        if temp < 1e-10:
            self._T = 1e-10
        else:
            self._T = temp * get_physical_units().KB
        self._mu = self._chemical_potential()
        self._f = self._occupation_number(self._eigenvalues, self._mu)
        self._entropy = self._get_entropy()
        self._energy = self._get_energy()

    @property
    def free_energy(self) -> float:
        """Return free energies."""
        return self.energy - self.entropy

    @property
    def energy(self) -> float:
        """Return energies."""
        if self._energy is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._energy

    @property
    def entropy(self) -> float:
        """Return entropies."""
        if self._entropy is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._entropy

    @property
    def mu(self) -> float:
        """Return chemical potential."""
        if self._mu is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._mu

    def _get_entropy(self) -> float:
        entropy = 0.0
        for f_k, w in zip(self._f.reshape(len(self._weights), -1), self._weights):
            _f = np.extract((f_k > 1e-12) * (f_k < 1 - 1e-12), f_k)
            entropy -= (_f * np.log(_f) + (1 - _f) * np.log(1 - _f)).sum() * w
        return float(entropy * self._g * self._T / self._weights.sum())

    def _get_energy(self) -> float:
        occ_eigvals = self._f * self._eigenvalues
        return float(
            np.dot(
                occ_eigvals.reshape(len(self._weights), -1).sum(axis=1), self._weights
            )
            * self._g
            / self._weights.sum()
        )

    def _chemical_potential(self) -> float:
        emin = np.min(self._eigenvalues)
        emax = np.max(self._eigenvalues)
        mu = (emin + emax) / 2

        for _ in range(1000):
            n = self._number_of_electrons(mu)
            if abs(n - self._n_electrons) < 1e-10:
                break
            elif n < self._n_electrons:
                emin = mu
            else:
                emax = mu
            mu = (emin + emax) / 2

        return float(mu)

    def _number_of_electrons(self, mu: float) -> float:
        eigvals = self._eigenvalues.reshape(len(self._weights), -1)
        n = (
            np.dot(self._occupation_number(eigvals, mu).sum(axis=1), self._weights)
            * self._g
            / self._weights.sum()
        )
        return float(n)

    def _occupation_number(self, e: NDArray, mu: float) -> NDArray:
        de = (e - mu) / self._T
        de = np.where(de < 100, de, 100.0)  # To avoid overflow
        de = np.where(de > -100, de, -100.0)  # To avoid underflow
        return 1.0 / (1 + np.exp(de))
