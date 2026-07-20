# SPDX-License-Identifier: BSD-3-Clause
"""Fit of lattice parameters as functions of cell volume.

This module contains pure math routines with no I/O and no plotting.

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class LatticeParametersFit:
    """Fit of lattice parameters vs volume for fixed-angle crystals.

    The cell volume is modeled as

        V = k * a * b * c = k * a^3 * r_b(V) * r_c(V)

    where a, b, c are the lattice-vector lengths, r_b = b / a and
    r_c = c / a are axial ratios fitted as polynomials of V, and k is a
    geometric constant containing the cell-angle factor. k is determined
    from the input data as mean(V_i / (a_i b_i c_i)) and must be constant
    over all volume points, which holds if and only if the cell angles do
    not depend on volume. Lattice parameters are recovered as

        a(V) = (V / (k * r_b(V) * r_c(V)))^(1/3)
        b(V) = r_b(V) * a(V)
        c(V) = r_c(V) * a(V)

    so that k * a(V) * b(V) * c(V) = V holds exactly at any evaluated
    volume. No crystal-system flag is needed: cubic cells give constant
    ratios r_b = r_c = 1, hexagonal cells give r_b = 1, etc.

    """

    def __init__(
        self,
        volumes: Sequence[float] | NDArray[np.double],
        lattice_parameters: Sequence[Sequence[float]] | NDArray[np.double],
        degree: int = 2,
        k_tol: float = 1e-4,
    ) -> None:
        """Init method.

        Parameters
        ----------
        volumes : array_like
            Unit cell volumes (V) in angstrom^3. shape=(volumes,)
        lattice_parameters : array_like
            Lattice-vector lengths (a, b, c) at each volume in angstrom.
            shape=(volumes, 3)
        degree : int, optional
            Degree of the polynomials fitted to the axial ratios vs V.
        k_tol : float, optional
            Maximum allowed relative deviation of V_i / (a_i b_i c_i)
            from its mean.

        """
        self._volumes = np.array(volumes, dtype="double")
        self._lattice_parameters = np.array(lattice_parameters, dtype="double")
        self._degree = degree

        if self._volumes.ndim != 1:
            raise ValueError("volumes must be a 1D array.")
        if self._lattice_parameters.shape != (len(self._volumes), 3):
            raise ValueError(
                "lattice_parameters must have shape (len(volumes), 3), "
                f"not {self._lattice_parameters.shape}."
            )
        if not (self._lattice_parameters > 0).all():
            raise ValueError("Lattice parameters must be positive.")
        if len(self._volumes) < degree + 1:
            raise RuntimeError(
                f"At least {degree + 1} volume points are needed for "
                f"lattice parameter fitting with polynomials of degree {degree}."
            )

        k_points = self._volumes / self._lattice_parameters.prod(axis=1)
        self._k = float(k_points.mean())
        if np.abs(k_points / self._k - 1).max() >= k_tol:
            raise RuntimeError(
                "Volumes are not consistent with V = k * a * b * c with a "
                "constant k. Cell angles must not depend on volume."
            )

        a = self._lattice_parameters[:, 0]
        self._ratio_coefficients = np.array(
            [
                np.polyfit(self._volumes, self._lattice_parameters[:, i] / a, degree)
                for i in (1, 2)
            ]
        )

    @property
    def k(self) -> float:
        """Return the geometric constant k = V / (a b c)."""
        return self._k

    @property
    def degree(self) -> int:
        """Return the degree of the axial-ratio polynomials."""
        return self._degree

    @property
    def ratio_coefficients(self) -> NDArray[np.double]:
        """Return polynomial coefficients of the axial ratios vs V.

        Rows correspond to b/a and c/a; columns are in np.polyfit order
        (highest degree first). shape=(2, degree + 1)

        """
        return self._ratio_coefficients

    def evaluate(
        self, volumes: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return lattice parameters (a, b, c) at volumes.

        Volumes outside the fitted range are extrapolated with a warning.

        Parameters
        ----------
        volumes : array_like
            Unit cell volumes in angstrom^3. shape=(n,)

        Returns
        -------
        ndarray
            Lattice parameters in angstrom. shape=(n, 3)

        """
        v = np.array(volumes, dtype="double")
        if v.min() < self._volumes.min() or v.max() > self._volumes.max():
            warnings.warn(
                "Lattice parameters are extrapolated outside the fitted volume range.",
                UserWarning,
                stacklevel=2,
            )
        r_b = np.polyval(self._ratio_coefficients[0], v)
        r_c = np.polyval(self._ratio_coefficients[1], v)
        a = (v / (self._k * r_b * r_c)) ** (1.0 / 3)
        return np.array([a, r_b * a, r_c * a]).T


def compute_axial_thermal_expansion(
    temperatures: NDArray[np.double],
    lattice_parameters: NDArray[np.double],
) -> NDArray[np.double]:
    """Compute linear thermal expansion coefficients along lattice vectors.

    alpha_x = (1/x) dx/dT for x = a, b, c by central differences. The
    returned array has length len(temperatures) - 1 with a leading row of
    zeros, mirroring the volumetric thermal expansion convention.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(num_elems,)
    lattice_parameters : ndarray
        Lattice-vector lengths (a, b, c) at temperatures in angstrom.
        shape=(num_elems, 3)

    """
    alpha = [np.zeros(3)]
    for i in range(1, len(temperatures) - 1):
        dt = temperatures[i + 1] - temperatures[i - 1]
        dl = lattice_parameters[i + 1] - lattice_parameters[i - 1]
        alpha.append(dl / dt / lattice_parameters[i])

    return np.array(alpha, dtype="double")
