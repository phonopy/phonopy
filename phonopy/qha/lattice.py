# SPDX-License-Identifier: BSD-3-Clause
"""Fit of lattice parameters as functions of cell volume.

This module contains pure math routines with no I/O and no plotting.

"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phonopy.qha.thermal import freeze_ndarray_fields

PRIMITIVE_VOLUME_ABC_RATIO_TOL = 1e-4
"""Relative spread of V / (a b c) over the input cells that is still accepted.

Above it the cells do not share one shape, so no single ratio describes
them and the lattice parameters cannot be recovered from a volume.

"""


def primitive_volume_abc_ratio(
    primitive_volumes: NDArray[np.double], lattice_lengths: NDArray[np.double]
) -> float:
    """Return V / (a b c), the constant shared by cells of one shape.

    V is the primitive cell volume and a, b, c are the lengths of the
    conventional unit cell, so the ratio carries both the cell-angle factor
    and the conventional-to-primitive one. Neither depends on the lengths,
    so every cell gives the same ratio and the mean only averages
    round-off.

    Parameters
    ----------
    primitive_volumes : ndarray
        Primitive cell volume of each cell in angstrom^3. shape=(n_cells,)
    lattice_lengths : ndarray
        Lattice-vector lengths (a, b, c) of the conventional unit cell of
        each cell in angstrom. shape=(n_cells, 3)

    Returns
    -------
    float
        The ratio, averaged over the cells.

    Raises
    ------
    RuntimeError
        When the ratio varies over the cells by more than
        PRIMITIVE_VOLUME_ABC_RATIO_TOL, i.e. the cells differ in more than
        their lengths.

    """
    ratios = primitive_volumes / lattice_lengths.prod(axis=-1)
    ratio = float(ratios.mean())
    if np.abs(ratios / ratio - 1).max() >= PRIMITIVE_VOLUME_ABC_RATIO_TOL:
        raise RuntimeError(
            "Volumes are not consistent with V = k * a * b * c with a "
            "constant k. Cell angles must not depend on volume."
        )
    return ratio


@dataclass(frozen=True)
class LatticeGrid:
    """The sample cells an anisotropic QHA is run over.

    What the input cells fix, as opposed to what a temperature fixes: the
    basis vectors that were sampled and the primitive cell they are
    reduced to. The lengths, the volumes and the volume of any other cell
    of the same shape all follow from those two.

    Attributes
    ----------
    lattices : ndarray
        Basis vectors (a, b, c as row vectors) of the conventional unit
        cell at each sample point in angstrom, as PhonopyAtoms.cell gives
        them. shape=(n_points, 3, 3)
    primitive_matrix : ndarray
        Transformation matrix to the primitive cell from the conventional
        unit cell, shared by every sample cell. shape=(3, 3)

    """

    lattices: NDArray[np.double]
    primitive_matrix: NDArray[np.double]

    def __post_init__(self) -> None:
        """Make ndarray fields read-only and refuse cells of differing shape."""
        freeze_ndarray_fields(self)
        # Raises unless one ratio describes every sample cell.
        primitive_volume_abc_ratio(self.primitive_volumes, self.lattice_lengths)

    @property
    def n_points(self) -> int:
        """Return the number of sample cells."""
        return self.lattices.shape[0]

    @property
    def lattice_lengths(self) -> NDArray[np.double]:
        """Return the conventional unit cell's (a, b, c) at each sample point.

        In angstrom. shape=(n_points, 3)

        """
        return np.linalg.norm(self.lattices, axis=2)

    @property
    def primitive_volumes(self) -> NDArray[np.double]:
        """Return the primitive cell volume of each sample cell.

        In angstrom^3. shape=(n_points,)

        """
        return np.abs(np.linalg.det(self.lattices)) * np.linalg.det(
            self.primitive_matrix
        )

    @property
    def primitive_volume_abc_ratio(self) -> float:
        """Return V / (a b c), the constant shared by the sample cells."""
        return primitive_volume_abc_ratio(self.primitive_volumes, self.lattice_lengths)

    def abc_to_primitive_volume(
        self, lattice_lengths: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the primitive cell volume of cells with these lengths.

        Parameters
        ----------
        lattice_lengths : ndarray
            Lattice-vector lengths (a, b, c) of the conventional unit cell
            in angstrom. shape=(n_cells, 3), or (3,) for one cell.

        Returns
        -------
        ndarray
            Primitive-cell volumes in angstrom^3. shape=(n_cells,), or a
            scalar for one cell.

        """
        return self.primitive_volume_abc_ratio * lattice_lengths.prod(axis=-1)


class LatticeParametersFit:
    """Fit of lattice parameters vs volume for fixed-angle crystals.

    The primitive cell volume is modeled as

        V = k * a * b * c = k * a^3 * r_b(V) * r_c(V)

    where a, b, c are the lattice-vector lengths of the conventional unit
    cell, r_b = b / a and r_c = c / a are axial ratios fitted as
    polynomials of V, and k is a geometric constant containing the
    cell-angle factor and the conventional-to-primitive one. k is determined
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
    ) -> None:
        """Init method.

        Parameters
        ----------
        volumes : array_like
            Primitive cell volumes (V) in angstrom^3. shape=(volumes,)
        lattice_parameters : array_like
            Lattice-vector lengths (a, b, c) of the conventional unit cell
            at each volume in angstrom. shape=(volumes, 3)
        degree : int, optional
            Degree of the polynomials fitted to the axial ratios vs V.
            The spread of V_i / (a_i b_i c_i) that is still accepted is
            PRIMITIVE_VOLUME_ABC_RATIO_TOL.

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

        self._primitive_volume_abc_ratio = primitive_volume_abc_ratio(
            self._volumes, self._lattice_parameters
        )

        a = self._lattice_parameters[:, 0]
        self._ratio_coefficients = np.array(
            [
                np.polyfit(self._volumes, self._lattice_parameters[:, i] / a, degree)
                for i in (1, 2)
            ]
        )

    @property
    def primitive_volume_abc_ratio(self) -> float:
        """Return the geometric constant k = V / (a b c).

        V is the primitive cell volume and a, b, c are the lengths of the
        conventional unit cell, so it carries both the cell-angle factor
        and the conventional-to-primitive one.

        """
        return self._primitive_volume_abc_ratio

    @property
    def degree(self) -> int:
        """Return the degree of the axial-ratio polynomials."""
        return self._degree

    def evaluate(
        self, volumes: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the conventional unit cell's (a, b, c) at volumes.

        Volumes outside the fitted range are extrapolated with a warning.

        Parameters
        ----------
        volumes : array_like
            Primitive cell volumes in angstrom^3. shape=(n,)

        Returns
        -------
        ndarray
            Lattice parameters (a, b, c) of the conventional unit cell in
            angstrom. shape=(n, 3)

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
        a = (v / (self._primitive_volume_abc_ratio * r_b * r_c)) ** (1.0 / 3)
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
        Lattice-vector lengths (a, b, c) of the conventional unit cell at
        temperatures in angstrom. shape=(num_elems, 3)

    """
    alpha = [np.zeros(3)]
    for i in range(1, len(temperatures) - 1):
        dt = temperatures[i + 1] - temperatures[i - 1]
        dl = lattice_parameters[i + 1] - lattice_parameters[i - 1]
        alpha.append(dl / dt / lattice_parameters[i])

    return np.array(alpha, dtype="double")
