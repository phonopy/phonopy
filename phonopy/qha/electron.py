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

import dataclasses
import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units


@dataclasses.dataclass(frozen=True)
class ElectronicStates:
    """Electronic states at a volume point.

    Input container for computing electronic free energies with
    ElectronFreeEnergy.

    Attributes
    ----------
    eigenvalues : ndarray
        Eigenvalues in eV. shape=(spin, kpoints, bands). The spin axis has
        length 1 for non-spin-polarized and 2 for spin-polarized systems.
    weights : ndarray
        Relative k-point weights (e.g., number of arms of the k-star).
        shape=(kpoints,)
    n_electrons : float
        Number of electrons in the unit cell.
    volume : float, optional
        Unit cell volume in angstrom^3. Used only for consistency checks
        against the unit cells the states belong to.

    """

    eigenvalues: NDArray[np.double]
    weights: NDArray[np.int64] | NDArray[np.double]
    n_electrons: float
    volume: float | None = None

    def __post_init__(self) -> None:
        """Validate shapes."""
        if self.eigenvalues.ndim != 3:
            raise ValueError(
                "eigenvalues must have shape (spin, kpoints, bands), not "
                f"{self.eigenvalues.shape}."
            )
        if self.eigenvalues.shape[0] not in (1, 2):
            raise ValueError(
                "The spin axis of eigenvalues must have length 1 or 2, not "
                f"{self.eigenvalues.shape[0]}."
            )
        if self.weights.ndim != 1 or len(self.weights) != self.eigenvalues.shape[1]:
            raise ValueError("weights must have one value per k-point of eigenvalues.")


def compute_free_energy_and_entropy(
    electronic_states: ElectronicStates,
    temperatures: Sequence[float] | NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return band free energies and entropies at temperatures.

    Parameters
    ----------
    electronic_states : ElectronicStates
        Electronic states at a volume point.
    temperatures : array_like
        Temperatures in K. shape=(temperatures,)

    Returns
    -------
    tuple of ndarray
        Band free energies in eV and entropies S_el in eV/K at the given
        temperatures. shape=(temperatures,) each.

    """
    efe = ElectronFreeEnergy(
        electronic_states.eigenvalues,
        electronic_states.weights,
        electronic_states.n_electrons,
    )
    free_energies = []
    entropies = []
    for temp in np.array(temperatures, dtype="double"):
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
        # ElectronFreeEnergy.entropy returns T * S in eV.
        if temp > 1e-10:
            entropies.append(efe.entropy / temp)
        else:
            entropies.append(0.0)
    return (
        np.array(free_energies, dtype="double"),
        np.array(entropies, dtype="double"),
    )


def write_electronic_states_hdf5(
    electronic_structures: Sequence[ElectronicStates],
    volumes: Sequence[float] | NDArray[np.double],
    energies: Sequence[float] | NDArray[np.double],
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> None:
    """Write electronic states with volumes and static energies in hdf5.

    The file contains one group "volume-XXX" per volume point with the
    datasets eigenvalues ((spin, kpoints, bands), eV), weights
    ((kpoints,)), n_electrons, volume (angstrom^3), and energy (eV,
    static internal energy). The number of volume points is stored in
    the root attribute "n_volumes".

    """
    import h5py

    if not (len(electronic_structures) == len(volumes) == len(energies)):
        raise ValueError(
            "electronic_structures, volumes, and energies must have the same length."
        )
    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["n_volumes"] = len(electronic_structures)
        for i, (electronic_states, volume, energy) in enumerate(
            zip(electronic_structures, volumes, energies, strict=True)
        ):
            group = w.create_group(f"volume-{i:03d}")
            group.create_dataset(
                "eigenvalues",
                data=electronic_states.eigenvalues,
                compression="gzip",
            )
            group.create_dataset("weights", data=electronic_states.weights)
            group.create_dataset(
                "n_electrons", data=float(electronic_states.n_electrons)
            )
            group.create_dataset("volume", data=float(volume))
            group.create_dataset("energy", data=float(energy))


def read_electronic_states_hdf5(
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> tuple[NDArray[np.double], NDArray[np.double], list[ElectronicStates]]:
    """Read electronic states with volumes and static energies from hdf5.

    Returns
    -------
    tuple
        Volumes in angstrom^3 with shape (volumes,), static internal
        energies in eV with shape (volumes,), and a list of
        ElectronicStates, in the file order. The latter two are the
        internal_energies and electronic_structures parameters of
        run_qha, respectively.

    """
    import h5py

    volumes = []
    energies = []
    electronic_structures = []
    with h5py.File(filename, "r") as f:
        n_volumes = int(f.attrs["n_volumes"])
        for i in range(n_volumes):
            group = f[f"volume-{i:03d}"]
            volume = float(group["volume"][()])
            electronic_structures.append(
                ElectronicStates(
                    eigenvalues=group["eigenvalues"][:],
                    weights=group["weights"][:],
                    n_electrons=float(group["n_electrons"][()]),
                    volume=volume,
                )
            )
            volumes.append(volume)
            energies.append(float(group["energy"][()]))
    return (
        np.array(volumes, dtype="double"),
        np.array(energies, dtype="double"),
        electronic_structures,
    )


def get_free_energy_at_T(
    tmin: float,
    tmax: float,
    tstep: float,
    eigenvalues: NDArray[np.double],
    weights: NDArray[np.int64] | NDArray[np.double],
    n_electrons: float,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return free energies at given temperatures."""
    free_energies = []
    efe = ElectronFreeEnergy(eigenvalues, weights, n_electrons)
    temperatures = np.arange(tmin, tmax + 1e-8, tstep, dtype="double")
    for temp in temperatures:
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
    return temperatures, np.array(free_energies, dtype="double")


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

    def __init__(
        self,
        eigenvalues: NDArray[np.double],
        weights: NDArray[np.int64] | NDArray[np.double],
        n_electrons: float,
    ) -> None:
        """Init method.

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues in eV.
            dtype='double'
            shape=(spin, kpoints, bands)
        weights: ndarray
            Relative k-point weights, e.g., geometric k-point weights
            (number of arms of k-star in BZ) or normalized weights.
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
        self._f: NDArray[np.double]  # occupation numbers, shape=(kpoints, spin, bands)
        self._mu = None
        self._entropy = None
        self._energy = None

    def run(self, temp: float) -> None:
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
        # f: shape=(kpoints, spin*bands), row i holds all (spin, band)
        # occupation numbers at the i-th irreducible k-point.
        f = self._f.reshape(len(self._weights), -1)
        mask = (f > 1e-12) & (f < 1 - 1e-12)
        f_safe = np.where(mask, f, 0.5)  # avoid log(0); masked out below anyway
        terms = np.where(
            mask, f_safe * np.log(f_safe) + (1 - f_safe) * np.log(1 - f_safe), 0.0
        )
        entropy = -(terms.sum(axis=1) * self._weights).sum()
        return float(entropy * self._g * self._T / self._weights.sum())

    def _get_energy(self) -> float:
        # occ_eigvals: shape=(kpoints, spin, bands), same as self._eigenvalues.
        occ_eigvals = self._f * self._eigenvalues
        # reshape to (kpoints, spin*bands), sum over spin*bands leaves
        # shape=(kpoints,), one value per irreducible k-point, matching
        # self._weights for the np.dot below.
        return float(
            np.dot(
                occ_eigvals.reshape(len(self._weights), -1).sum(axis=1), self._weights
            )
            * self._g
            / self._weights.sum()
        )

    def _chemical_potential(self) -> float:
        try:
            from scipy.optimize import brentq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        emin = np.min(self._eigenvalues)
        emax = np.max(self._eigenvalues)
        # brentq's default xtol (2e-12) is too loose here: near T -> 0 the
        # occupation number is a near step function, so n(mu) can change by
        # O(1e-3) for an O(1e-10) change in mu. A tight xtol is needed to
        # match the occupation numbers (and hence energy/entropy) to the
        # precision expected by callers.
        mu = brentq(
            lambda mu: self._number_of_electrons(mu) - self._n_electrons,
            emin,
            emax,
            xtol=1e-14,
        )
        return float(mu)

    def _number_of_electrons(self, mu: float) -> float:
        # eigvals: shape=(kpoints, spin*bands); occupation_number keeps the
        # same shape, and summing over spin*bands leaves shape=(kpoints,),
        # matching self._weights for the np.dot below.
        eigvals = self._eigenvalues.reshape(len(self._weights), -1)
        n = (
            np.dot(self._occupation_number(eigvals, mu).sum(axis=1), self._weights)
            * self._g
            / self._weights.sum()
        )
        return float(n)

    def _occupation_number(
        self, e: NDArray[np.double], mu: float
    ) -> NDArray[np.double]:
        """Return occupation numbers, same shape as `e`."""
        de = (e - mu) / self._T
        de = np.where(de < 100, de, 100.0)  # To avoid overflow
        de = np.where(de > -100, de, -100.0)  # To avoid underflow
        return 1.0 / (1 + np.exp(de))
