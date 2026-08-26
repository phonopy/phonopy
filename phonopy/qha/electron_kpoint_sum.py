# SPDX-License-Identifier: BSD-3-Clause
"""Electronic free energy by the sum over irreducible k-points.

The route that needs nothing beyond the eigenvalues, the k-point weights and
the electron count, so it runs on states carrying no sampling grid: an
explicit k-point list has none to build, and neither has an
electronic_states.hdf5 written before the grid was stored in it.

It integrates a delta-function density of states and converges far more
slowly than the linear tetrahedron method of electron.py, which is what runs
wherever the grid is there: reaching the same answer needs many more
irreducible k-points.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units
from phonopy.qha.electron_states import (
    ElectronicStates,
    entropy_terms,
    fermi_dirac_occupation,
)

# Temperature in K standing in for 0 K, where k_B T is a division by zero.
_ZERO_TEMPERATURE = 1e-10
# k_B T in eV standing in for 0 K here, which is k_B times a temperature five
# orders larger than the above. The two are not interchangeable: sharpening
# this step changes which states at the Fermi level are partially occupied,
# and with it the free energy at 0 K by a meV.
_ZERO_KT = 1e-10


def compute_free_energy_by_kpoint_sum(
    electronic_states: ElectronicStates,
    temperatures: Sequence[float] | NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return band free energies and entropies at temperatures.

    Each value is the whole band sum at that temperature, E - TS over the
    occupied states, on the energy zero the eigenvalues carry. It is not a
    difference from 0 K, which is what compute_free_energy_by_tetrahedron
    returns instead.

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
        spin_degeneracy=electronic_states.spin_degeneracy,
    )
    free_energies = []
    entropies = []
    for temp in np.array(temperatures, dtype="double"):
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
        # ElectronFreeEnergy.entropy returns T * S in eV.
        if temp > _ZERO_TEMPERATURE:
            entropies.append(efe.entropy / temp)
        else:
            entropies.append(0.0)
    return (
        np.array(free_energies, dtype="double"),
        np.array(entropies, dtype="double"),
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

    where :math:`g` is 2 for non-spin polarized systems and 1 for spin
    polarized systems.

    Energy
    ------

    .. math::

       E_\text{el}(V) = g\sum_i f_i(V) \epsilon_i(V)

    Checking mu
    -----------

    :math:`\mu` is the root of the electron-count equation, solved by
    _chemical_potential():

    .. math::

       N = \frac{g}{\sum_k w_k} \sum_k w_k \sum_i f_{ki}(V)

    where :math:`N` is the number of electrons in the unit cell and
    :math:`w_k` the k-point weights. Only the eigenvalues, the weights,
    :math:`N` and :math:`g` enter, so comparing mu against the Fermi energy
    reported by the electronic structure code that produced them checks all
    four.

    The comparison is meaningful only between quantities at the same
    temperature, so pair them by how that code broadened the occupations:

    - Schemes that reproduce the zero-broadening answer -- tetrahedron methods,
      Methfessel-Paxton, Gaussian -- against mu at 0 K. Their width is a
      numerical device rather than a temperature; treating it as one makes the
      agreement worse, not better.
    - Fermi-Dirac (thermal) smearing of width sigma against mu at
      T = sigma / k_B.

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
        spin_degeneracy: int | None = None,
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
        spin_degeneracy: int, optional
            Number of electrons one eigenvalue can hold, i.e., g above. With
            None (default) it is inferred from the length of the spin axis of
            eigenvalues: 2 for length 1 and 1 for length 2. This inference is
            wrong for non-collinear calculations, whose spin axis has length
            1 while each spinor state holds one electron, so g = 1 must be
            given explicitly there.

        """
        # shape=(kpoints, spin, bands)
        self._eigenvalues = np.array(
            eigenvalues.swapaxes(0, 1), dtype="double", order="C"
        )
        self._weights = weights
        self._n_electrons = n_electrons

        if spin_degeneracy is None:
            if self._eigenvalues.shape[1] == 1:
                self._g = 2
            elif self._eigenvalues.shape[1] == 2:
                self._g = 1
            else:
                raise RuntimeError(
                    "The spin axis of eigenvalues must have length 1 or 2, not "
                    f"{self._eigenvalues.shape[1]}."
                )
        elif spin_degeneracy in (1, 2):
            self._g = spin_degeneracy
        else:
            raise ValueError(f"spin_degeneracy must be 1 or 2, not {spin_degeneracy}.")

        self._T: float
        self._f: NDArray[np.double]  # occupation numbers, shape=(kpoints, spin, bands)
        self._mu: float | None = None
        self._entropy: float | None = None
        self._energy: float | None = None

    def run(self, temp: float) -> None:
        """Calculate free energies.

        Parameters
        ----------
        temp: float
            Temperature in K

        """
        if temp < _ZERO_TEMPERATURE:
            self._T = _ZERO_KT
        else:
            self._T = temp * get_physical_units().KB
        mu = self._chemical_potential()
        self._mu = mu
        self._f = self._occupation_number(self._eigenvalues, mu)
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
        entropy = -(entropy_terms(f).sum(axis=1) * self._weights).sum()
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
        return fermi_dirac_occupation(e, mu, self._T)


# The name this function carried while it was the only route in electron.py.
compute_free_energy_and_entropy = compute_free_energy_by_kpoint_sum
