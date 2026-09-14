# SPDX-License-Identifier: BSD-3-Clause
"""Pure computation routines shared by QHA implementations.

Functions in this module operate on plain arrays and have no I/O,
plotting, or object state.

One unit system runs through this module: eV, angstrom and K. Energies
are in eV, volumes in angstrom^3, entropies and heat capacities in eV/K,
and bulk moduli in eV/angstrom^3. Nothing here converts units, so the
formulas are the textbook ones. A caller reporting J/K/mol or GPa
converts at its own boundary.

"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class CpPolyfitArrays(NamedTuple):
    """Results of C_P computation via polynomial fits of Cv(V) and S(V).

    cp is in eV/K and dsdv in eV/K/angstrom^3. Both have the same length
    as the input temperatures with a leading 0.0 element. The parameter
    lists have two fewer elements, corresponding to temperatures[1:-1].

    """

    cp: NDArray[np.double]
    dsdv: NDArray[np.double]
    volume_cv_parameters: list[NDArray[np.double]]
    volume_entropy_parameters: list[NDArray[np.double]]


def compute_volumetric_thermal_expansion(
    temperatures: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
) -> NDArray[np.double]:
    """Compute volumetric thermal expansion coefficients beta.

    beta = (1/V) dV/dT by central differences. The returned array has
    length len(temperatures) - 1 with a leading 0.0 element.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(num_elems,)
    equilibrium_volumes : ndarray
        Equilibrium volumes at temperatures in angstrom^3.
        shape=(num_elems,)

    """
    beta = [0.0]
    for i in range(1, len(temperatures) - 1):
        dt = temperatures[i + 1] - temperatures[i - 1]
        dv = equilibrium_volumes[i + 1] - equilibrium_volumes[i - 1]
        beta.append(dv / dt / equilibrium_volumes[i])

    return np.array(beta, dtype="double")


def compute_heat_capacity_p_numerical(
    temperatures: NDArray[np.double],
    gibbs_free_energies: NDArray[np.double],
) -> NDArray[np.double]:
    """Compute C_P as numerical second derivative of Gibbs free energy.

    C_P = -T d^2G/dT^2 with the second derivative obtained from a local
    quadratic fit over three neighboring temperature points. The returned
    array has length len(temperatures) - 1 with a leading 0.0 element.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(num_elems,)
    gibbs_free_energies : ndarray
        Gibbs free energies at temperatures in eV. shape=(num_elems,)

    Returns
    -------
    ndarray
        Heat capacities at constant pressure in eV/K. shape=(num_elems - 1,)

    """
    cp = []
    g = np.array(gibbs_free_energies)
    cp.append(0.0)

    for i in range(1, len(temperatures) - 1):
        t = temperatures[i]
        parameters = np.polyfit(temperatures[i - 1 : i + 2], g[i - 1 : i + 2], 2)
        cp.append(-(2 * parameters[0]) * t)

    return np.array(cp, dtype="double")


class EntropyEnthalpyArrays(NamedTuple):
    """System entropy and enthalpy at constant pressure.

    Both arrays have the same length as the input temperatures. Entropy is
    in eV/K and enthalpy is in eV, so that G = H - T S holds directly.

    """

    entropy: NDArray[np.double]
    enthalpy: NDArray[np.double]


def compute_entropy_enthalpy_temperature(
    temperatures: NDArray[np.double],
    volumes: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    entropy: NDArray[np.double],
    gibbs_free_energies: NDArray[np.double],
) -> EntropyEnthalpyArrays:
    """Evaluate S(T, p) and H(T, p) from S(V) fits at V_eq.

    S(T, p) = S(T, V_eq(T, p)) where S(V) is a degree-4 polynomial at each
    temperature, the same interpolation used for heat_capacity_P_polyfit.
    G(T) is not differentiated. H = G + T S.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(num_elems,)
    volumes : ndarray
        Unit cell volumes of the input volume grid in angstrom^3.
        shape=(volumes,)
    equilibrium_volumes : ndarray
        Equilibrium volumes at temperatures in angstrom^3.
        shape=(num_elems,)
    entropy : ndarray
        Entropies at constant volume in eV/K, indexed consistently
        with temperatures. shape=(>=num_elems, volumes)
    gibbs_free_energies : ndarray
        Gibbs free energies at temperatures in eV. shape=(num_elems,)

    """
    if len(volumes) < 5:
        raise RuntimeError(
            "At least 5 volume points are needed to fit S(V) to a "
            "polynomial of degree 4."
        )

    n = len(temperatures)
    entropies = np.empty(n, dtype="double")
    for i in range(n):
        try:
            parameters = np.polyfit(volumes, entropy[i], 4)
        except np.lib.polynomial.RankWarning as exc:  # type: ignore
            msg = ["Failed to fit entropies to polynomial of degree 4."]
            msg += ["At least 5 volume points are needed for the fitting."]
            raise RuntimeError("\n".join(msg)) from exc
        entropies[i] = float(np.polyval(parameters, equilibrium_volumes[i]))

    enthalpies = gibbs_free_energies + temperatures * entropies
    return EntropyEnthalpyArrays(
        entropy=np.array(entropies, dtype="double"),
        enthalpy=np.array(enthalpies, dtype="double"),
    )


def compute_heat_capacity_p_polyfit(
    temperatures: NDArray[np.double],
    volumes: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    cv: NDArray[np.double],
    entropy: NDArray[np.double],
) -> CpPolyfitArrays:
    """Compute C_P via polynomial fits of Cv(V) and S(V).

    C_P = Cv(V(T)) + T (dV/dT) (dS/dV) where Cv(V) and S(V) are fitted to
    degree-4 polynomials of volume at each temperature and dV/dT is
    obtained from a local quadratic fit of V(T).

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(num_elems,)
    volumes : ndarray
        Unit cell volumes of the input volume grid in angstrom^3.
        shape=(volumes,)
    equilibrium_volumes : ndarray
        Equilibrium volumes at temperatures in angstrom^3.
        shape=(num_elems,)
    cv : ndarray
        Heat capacities at constant volume in eV/K, indexed
        consistently with temperatures. shape=(>=num_elems, volumes)
    entropy : ndarray
        Entropies at constant volume in eV/K, indexed consistently
        with temperatures. shape=(>=num_elems, volumes)

    """
    cp = [0.0]
    dsdv = [0.0]
    volume_cv_parameters = []
    volume_entropy_parameters = []

    for j in range(1, len(temperatures) - 1):
        t = temperatures[j]
        x = equilibrium_volumes[j]

        try:
            parameters = np.polyfit(volumes, cv[j], 4)
        except np.lib.polynomial.RankWarning as exc:  # type: ignore
            msg = ["Failed to fit heat capacities to polynomial of degree 4."]
            if len(volumes) < 5:
                msg += ["At least 5 volume points are needed for the fitting."]
            raise RuntimeError("\n".join(msg)) from exc

        cv_p = np.dot(parameters, np.array([x**4, x**3, x**2, x, 1]))
        volume_cv_parameters.append(parameters)

        try:
            parameters = np.polyfit(volumes, entropy[j], 4)
        except np.lib.polynomial.RankWarning as exc:  # type: ignore
            msg = ["Failed to fit entropies to polynomial of degree 4."]
            if len(volumes) < 5:
                msg += ["At least 5 volume points are needed for the fitting."]
            raise RuntimeError("\n".join(msg)) from exc

        dsdv_t = np.dot(parameters[:4], np.array([4 * x**3, 3 * x**2, 2 * x, 1]))
        volume_entropy_parameters.append(parameters)

        try:
            parameters = np.polyfit(
                temperatures[j - 1 : j + 2],
                equilibrium_volumes[j - 1 : j + 2],
                2,
            )
        except np.lib.polynomial.RankWarning as exc:  # type: ignore
            raise RuntimeError(
                "Failed to fit equilibrium volumes vs T to polynomial of degree 2."
            ) from exc
        dvdt = parameters[0] * 2 * t + parameters[1]

        cp.append(cv_p + t * dvdt * dsdv_t)
        dsdv.append(dsdv_t)

    return CpPolyfitArrays(
        cp=np.array(cp, dtype="double"),
        dsdv=np.array(dsdv, dtype="double"),
        volume_cv_parameters=volume_cv_parameters,
        volume_entropy_parameters=volume_entropy_parameters,
    )


def compute_gruneisen_parameters(
    volumes: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    bulk_moduli: NDArray[np.double],
    thermal_expansions: NDArray[np.double],
    cv: NDArray[np.double],
) -> NDArray[np.double]:
    """Compute thermodynamic Gruneisen parameters.

    gamma = beta B V / Cv with Cv(V) evaluated from a degree-4 polynomial
    fit of the heat capacities over the input volume grid. The returned
    array has length len(equilibrium_volumes) - 1 with a leading 0.0
    element.

    Parameters
    ----------
    volumes : ndarray
        Unit cell volumes of the input volume grid in angstrom^3.
        shape=(volumes,)
    equilibrium_volumes : ndarray
        Equilibrium volumes at temperatures in angstrom^3.
        shape=(num_elems,)
    bulk_moduli : ndarray
        Bulk moduli at temperatures in eV/angstrom^3. shape=(num_elems,)
    thermal_expansions : ndarray
        Volumetric thermal expansion coefficients at temperatures in 1/K.
        shape=(num_elems - 1,)
    cv : ndarray
        Heat capacities at constant volume in eV/K, indexed
        consistently with temperatures. shape=(>=num_elems, volumes)

    """
    gamma = [0.0]
    for i in range(1, len(equilibrium_volumes) - 1):
        v = equilibrium_volumes[i]
        kt = bulk_moduli[i]
        beta = thermal_expansions[i]
        try:
            parameters = np.polyfit(volumes, cv[i], 4)
        except np.lib.polynomial.RankWarning as exc:  # type: ignore
            msg = ["Failed to fit heat capacities to polynomial of degree 4."]
            if len(volumes) < 5:
                msg += ["At least 5 volume points are needed for the fitting."]
            raise RuntimeError("\n".join(msg)) from exc
        cv_v = np.dot(parameters, [v**4, v**3, v**2, v, 1]) / v
        # Cv vanishes as T -> 0, where gamma is undefined.
        if cv_v < 1e-12:
            gamma.append(0.0)
        else:
            gamma.append(beta * kt / cv_v)

    return np.array(gamma, dtype="double")


def _compositions(total: int, ndim: int) -> list[tuple[int, ...]]:
    """Return all non-negative integer tuples of length ndim summing to total."""
    if ndim == 1:
        return [(total,)]
    result: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, ndim - 1):
            result.append((first, *rest))
    return result


def generate_total_degree_exponents(ndim: int, degree: int) -> NDArray[np.int64]:
    """Return exponent tuples of all monomials up to a total degree.

    A total-degree polynomial in ndim variables of degree `degree`
    consists of the monomials prod_k x_k^e_k with sum_k e_k <= degree.
    The returned rows are these exponent tuples (e_0, ..., e_{ndim-1}),
    ordered by increasing total degree, so the first row is the constant
    term (all zeros). The number of terms is C(ndim + degree, degree),
    which is far fewer than the (degree + 1)^ndim of a tensor-product
    basis (e.g. 10 vs 16 for ndim=2, degree=3).

    Parameters
    ----------
    ndim : int
        Number of variables (1 to 3 for lattice-length DOF).
    degree : int
        Maximum total degree.

    Returns
    -------
    ndarray
        Exponent tuples. shape=(n_terms, ndim)

    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    exponents: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        exponents.extend(_compositions(total, ndim))
    return np.array(exponents, dtype="int64")


def polynomial_design_matrix(
    points: NDArray[np.double], exponents: NDArray[np.int64]
) -> NDArray[np.double]:
    """Build the design matrix of monomials at sample points.

    Column t is the monomial prod_k x_k^exponents[t, k] evaluated at each
    point. With non-dimensionalized points the matrix is well conditioned
    for least-squares polynomial fitting.

    Parameters
    ----------
    points : ndarray
        Sample points. shape=(n_points, ndim)
    exponents : ndarray
        Monomial exponent tuples. shape=(n_terms, ndim)

    Returns
    -------
    ndarray
        Design matrix. shape=(n_points, n_terms)

    """
    pts = np.asarray(points, dtype="double")
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n_points, ndim).")
    if exponents.shape[1] != pts.shape[1]:
        raise ValueError("exponents and points must have the same ndim.")
    design = np.ones((pts.shape[0], exponents.shape[0]), dtype="double")
    for d in range(pts.shape[1]):
        design *= pts[:, d : d + 1] ** exponents[:, d][None, :]
    return design
