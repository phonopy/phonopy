# SPDX-License-Identifier: BSD-3-Clause
"""Smooth lattice parameters along temperature before differentiating them.

The axial thermal expansions are central differences of a(T), b(T), c(T), so
whatever scatter those carry reaches them amplified. Lattice parameters from a
sampled method -- SSCHA, or any other route whose free energy is a Monte Carlo
average -- carry the scatter of that sampling, since every temperature is
minimized on its own.

The form fitted is a **sum of Einstein terms of opposite sign**, the usual way
to model a lattice parameter that contracts at low temperature and expands at
high temperature. Each term is theta / (exp(theta/T) - 1), which is zero at
T = 0 with zero slope, so the fitted expansion vanishes at 0 K as the third
law requires.

A smoothing spline is not offered. It assumes only smoothness, and has no way
of knowing that the expansion must vanish at 0 K: fitted to the same data it
returns a finite expansion there.

The Einstein fit runs every starting guess and keeps the one with the smallest
residual among those that pass explicit checks, rather than the first that
converges. A fit that converges to a qualitatively wrong curve -- a monotone
one where the data contracts, or a contraction several times deeper than the
data shows -- is otherwise accepted silently, which is the failure this guards
against. When no candidate passes, EinsteinFitFailure is raised rather than
absorbed by a fallback.

"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

# Einstein temperatures are drawn from this ladder. It spans the range a
# lattice parameter can carry structure over: below its lowest entry the
# occupation of any mode is frozen on a typical temperature grid, and above
# its highest a term is indistinguishable from a straight line.
DEFAULT_THETA_POOL = (80.0, 150.0, 250.0, 380.0, 500.0)

# Largest argument exp() can take in double precision, less one for margin.
# theta/T beyond it leaves both the Einstein factor and its derivative zero
# to any precision the type carries, so clipping there changes no value that
# can be represented, and it keeps exp() from overflowing to inf.
_MAX_EXP_ARG = float(np.log(np.finfo("double").max)) - 1.0

# A negative Einstein temperature gives a term that is finite as T -> 0 but
# zero at T = 0 exactly, i.e. a step, which is not a lattice parameter.
THETA_MINIMUM = 1.0

# How far the fitted curve's steepest contraction may stray from the raw
# curve's before the fit is rejected, as (low, high) factors. Wide on purpose:
# a backstop against a qualitatively wrong answer, not a goodness-of-fit
# measure.
DEFAULT_DIP_FACTORS = (0.3, 2.0)


class EinsteinFitFailure(RuntimeError):
    """No Einstein fit of the requested form passed the checks.

    Raised rather than absorbed. A caller that can tolerate an occasional bad
    realization should catch this and count it, so that the rate stays
    visible; silently substituting another route is what makes such failures
    invisible.

    """


@dataclass
class _EinsteinFit:
    """A fitted sum of Einstein terms, with what it took to get there.

    Attributes
    ----------
    model : ndarray
        The fitted curve on the input temperatures. shape=(temperatures,)
    y0 : float
        The T = 0 value of the fit.
    amplitudes : ndarray
        Term amplitudes, ordered by Einstein temperature.
    thetas : ndarray
        Einstein temperatures in K, ascending.
    rms : float
        Root mean square residual against the input values.
    n_converged : int
        Number of starting guesses that converged.
    n_accepted : int
        Number of converged fits that passed the checks.

    """

    model: NDArray[np.double]
    y0: float
    amplitudes: NDArray[np.double]
    thetas: NDArray[np.double]
    rms: float
    n_converged: int
    n_accepted: int

    def slope(self, temperatures: NDArray[np.double]) -> NDArray[np.double]:
        """Return dy/dT of the fitted model, analytically.

        The model is differentiable in closed form, so a thermal expansion
        need not go through finite differences at all. Whether it should is a
        separate question: finite differences are what a measured a(T), c(T)
        would be differentiated by.

        """
        out = np.zeros_like(temperatures, dtype="double")
        for amp, theta in zip(self.amplitudes, self.thetas, strict=True):
            out = out + amp * _einstein_term_derivative(temperatures, theta)
        return out

    def describe(self) -> str:
        """Return a one-line summary for a log."""
        terms = " ".join(
            f"{a * 1e3:+.3f}@{th:.0f}K"
            for a, th in zip(self.amplitudes, self.thetas, strict=True)
        )
        return (
            f"rms {self.rms * 1e3:.4f} mA, {self.n_accepted}/{self.n_converged} "
            f"accepted, {terms}"
        )


def _einstein_term(
    temperatures: NDArray[np.double], theta: float
) -> NDArray[np.double]:
    """Return theta / (exp(theta/T) - 1) on the given temperatures.

    Zero at T = 0 with zero slope, so a lattice parameter built from these
    terms satisfies the third law: the thermal expansion vanishes at 0 K.

    """
    out = np.zeros_like(temperatures, dtype="double")
    if not np.isfinite(theta) or abs(theta) < 1e-6:
        # A vanishing Einstein temperature carries no temperature dependence,
        # and the expression would divide by zero. The optimizer does wander
        # there, so it is answered rather than avoided.
        return out
    warm = temperatures > 1e-8
    x = np.clip(theta / temperatures[warm], -_MAX_EXP_ARG, _MAX_EXP_ARG)
    out[warm] = theta / np.expm1(x)
    return out


def _einstein_term_derivative(
    temperatures: NDArray[np.double], theta: float
) -> NDArray[np.double]:
    """Return d/dT of _einstein_term, as (x/2)^2 / sinh^2(x/2) with x = theta/T.

    Writing it with sinh rather than as theta^2 exp(x) / (T (exp(x) - 1))^2
    avoids the ratio of two overflowing exponentials at low temperature, where
    the factor has to go to zero and the naive form goes to inf / inf.

    """
    out = np.zeros_like(temperatures, dtype="double")
    if not np.isfinite(theta) or abs(theta) < 1e-6:
        return out
    warm = temperatures > 1e-8
    half = np.clip(0.5 * theta / temperatures[warm], -_MAX_EXP_ARG, _MAX_EXP_ARG)
    out[warm] = np.square(half / np.sinh(half))
    return out


def _n_einstein(
    temperatures: NDArray[np.double], y0: float, *params: float
) -> NDArray[np.double]:
    """Return a lattice parameter modelled by a sum of Einstein terms.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(temperatures,)
    y0 : float
        The value at T = 0, to which the terms are added.
    *params : float
        Amplitude and Einstein temperature of each term, flattened in pairs:
        amp1, theta1, amp2, theta2, and so on. One function therefore serves
        any number of terms, and the count comes from the length of the
        starting guess handed to curve_fit, which calls this as f(x, *p).

    """
    out = np.full_like(temperatures, float(y0), dtype="double")
    for amp, theta in zip(params[0::2], params[1::2], strict=True):
        out = out + amp * _einstein_term(temperatures, theta)
    return out


def _central_slope(
    temperatures: NDArray[np.double], values: NDArray[np.double]
) -> NDArray[np.double]:
    """Return dy/dT by the central differences the expansions are taken with.

    The checks below have to be measured on the same quantity the caller
    differentiates, or they would reject fits on a difference of convention.

    """
    return (values[2:] - values[:-2]) / (temperatures[2:] - temperatures[:-2])


def _starting_guesses(
    values: NDArray[np.double], n_terms: int, theta_pool: Sequence[float]
) -> list[tuple[float, ...]]:
    """Return the starting guesses for an n-term fit.

    Einstein temperatures are every combination of n values from the pool,
    which spreads them rather than committing to one arrangement, and the
    amplitudes take every sign pattern with at least one term of each sign.
    All-positive and all-negative patterns are left out: their sum is monotone
    and cannot describe a contraction followed by an expansion.

    """
    span = float(values.max() - values.min())
    if span <= 0.0:
        span = max(abs(float(values[0])), 1.0) * 1e-6
    patterns = [
        signs
        for signs in itertools.product((-1.0, 1.0), repeat=n_terms)
        if len(set(signs)) > 1
    ]
    guesses = []
    for thetas in itertools.combinations(theta_pool, n_terms):
        for signs in patterns:
            guess: list[float] = [float(values[0])]
            for sign, theta in zip(signs, thetas, strict=True):
                guess += [sign * span, theta]
            guesses.append(tuple(guess))
    return guesses


def _contracts(values: NDArray[np.double]) -> bool:
    """Return whether the series falls measurably below its starting value.

    Two of the checks below encode "contracts, then expands" and are wrong for
    a lattice parameter that only expands. Noise alone puts a slightly
    negative point into dy/dT near 0 K, where the slope has to vanish, so the
    test cannot be on the sign of the slope. It is on the depth instead.

    """
    span = float(values.max() - values.min())
    return span > 0.0 and float(values.min()) < float(values[0]) - 0.02 * span


def _rejection(
    temperatures: NDArray[np.double],
    values: NDArray[np.double],
    popt: NDArray[np.double],
    dip_factors: tuple[float, float],
) -> str | None:
    """Return why this fit is unacceptable, or None if it is acceptable.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K the fit was made on. shape=(temperatures,)
    values : ndarray
        The lattice parameter the fit was made to. shape=(temperatures,)
    popt : ndarray
        Fitted parameters as curve_fit returns them, in the layout
        _n_einstein takes: y0, amp1, theta1, amp2, theta2, and so on.
    dip_factors : tuple of float
        (low, high) bounds on the fitted steepest contraction relative to the
        raw one.

    """
    if not np.all(np.isfinite(popt)):
        return "non-finite parameters"
    amplitudes = popt[1::2]
    thetas = popt[2::2]
    model = _n_einstein(temperatures, popt[0], *popt[1:])
    if not np.all(np.isfinite(model)):
        return "non-finite model"
    if np.any(thetas < THETA_MINIMUM):
        return "non-positive Einstein temperature"
    contracts = _contracts(values)
    if contracts and (np.all(amplitudes >= 0.0) or np.all(amplitudes <= 0.0)):
        # A sum of same-sign Einstein terms is monotone in T, so it cannot
        # represent a lattice parameter that contracts and then expands. For
        # one that only expands, same-sign terms are the right model.
        return "same-sign amplitudes"
    raw_dip = float(_central_slope(temperatures, values)[1:].min())
    if raw_dip < 0.0:
        fitted_dip = float(_central_slope(temperatures, model)[1:].min())
        low, high = dip_factors
        # The lower bound -- the fit must not invent a contraction several
        # times deeper than the data shows -- applies either way. The upper
        # bound asks the fit to keep a contraction that is there, so it
        # applies only when there is one: on a monotone series the raw dip is
        # a noise artifact at the low-temperature end, and a fitted dip of
        # zero is the correct answer rather than a failure.
        if fitted_dip < high * raw_dip:
            return f"dip {fitted_dip / raw_dip:.2f} times the raw one"
        if contracts and fitted_dip > low * raw_dip:
            return f"dip {fitted_dip / raw_dip:.2f} times the raw one"
    return None


def _fit_einstein(
    temperatures: NDArray[np.double],
    values: NDArray[np.double],
    sigma: NDArray[np.double] | None = None,
    n_terms: int = 2,
    theta_pool: Sequence[float] = DEFAULT_THETA_POOL,
    dip_factors: tuple[float, float] = DEFAULT_DIP_FACTORS,
) -> _EinsteinFit:
    """Return the best acceptable n-term Einstein fit of values(temperatures).

    Every starting guess is run and the one with the smallest residual among
    those that pass the checks is returned.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K, ascending. shape=(temperatures,)
    values : ndarray
        The lattice parameter to fit, in angstrom. shape=(temperatures,)
    sigma : ndarray, optional
        Per-point uncertainties, passed to the least-squares fit.
    n_terms : int, optional
        Number of Einstein terms, at least 2. Default is 2. Three terms track
        a curve more closely than two, at the cost of more freedom.
    theta_pool : sequence of float, optional
        Einstein temperatures the starting guesses are drawn from.
    dip_factors : tuple of float, optional
        (low, high) bounds on the fitted steepest contraction relative to the
        raw one.

    Raises
    ------
    EinsteinFitFailure
        When no candidate fit passes the checks.

    """
    # scipy is not a declared dependency of phonopy, so it is imported where
    # it is used, as elsewhere in phonopy.qha.
    from scipy.optimize import curve_fit

    if n_terms < 2:
        raise ValueError("an Einstein fit of a lattice parameter needs two terms")
    # best holds the smallest residual seen and the popt that produced it.
    best: tuple[float, NDArray[np.double]] | None = None
    n_converged = 0
    n_accepted = 0
    reasons: dict[str, int] = {}
    for guess in _starting_guesses(values, n_terms, theta_pool):
        try:
            popt, _ = curve_fit(
                _n_einstein,
                temperatures,
                values,
                p0=guess,
                sigma=sigma,
                absolute_sigma=sigma is not None,
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        n_converged += 1
        reason = _rejection(temperatures, values, popt, dip_factors)
        if reason is not None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        n_accepted += 1
        model = _n_einstein(temperatures, popt[0], *popt[1:])
        rms = float(np.sqrt(np.mean(np.square(model - values))))
        if best is None or rms < best[0]:
            best = (rms, popt)
    if best is None:
        summary = ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
        raise EinsteinFitFailure(
            f"no acceptable {n_terms}-term Einstein fit out of {n_converged} "
            f"converged ({summary or 'none converged'})"
        )
    rms, popt = best
    order = np.argsort(popt[2::2])
    return _EinsteinFit(
        model=_n_einstein(temperatures, popt[0], *popt[1:]),
        y0=float(popt[0]),
        amplitudes=np.asarray(popt[1::2])[order],
        thetas=np.asarray(popt[2::2])[order],
        rms=rms,
        n_converged=n_converged,
        n_accepted=n_accepted,
    )


SmoothingMethod = Literal["none", "einstein"]
SMOOTHING_METHODS = get_args(SmoothingMethod)


def smooth_lattice_parameters(
    temperatures: NDArray[np.double],
    lattice_parameters: NDArray[np.double],
    method: SmoothingMethod = "einstein",
    n_terms: int = 2,
    sigma: NDArray[np.double] | None = None,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return smoothed lattice parameters and their temperature derivatives.

    Each of a, b and c is smoothed on its own. A column that does not vary is
    returned unchanged with a zero derivative, so a fixed lattice length is
    never fitted.

    The derivatives are those of the fitted model, in closed form. Having
    committed to a model there is no reason to approximate its slope by finite
    differences again: on a 10 K grid the difference is largest where the
    curvature is, which is where a lattice parameter turns from contraction to
    expansion.

    Parameters
    ----------
    temperatures : ndarray
        Temperatures in K, ascending. shape=(temperatures,)
    lattice_parameters : ndarray
        (a, b, c) at those temperatures in angstrom.
        shape=(temperatures, 3)
    method : Literal["none", "einstein"], optional
        "einstein" to fit a sum of Einstein terms (the default), or "none" to
        return the lattice parameters unchanged with zero derivatives.
    n_terms : int, optional
        Number of Einstein terms, at least 2. Default is 2. More terms follow
        a curve more closely, at the cost of more freedom to follow its noise.
    sigma : ndarray, optional
        Per-point uncertainties of each column, shape=(temperatures, 3).

    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(f"method must be one of {SMOOTHING_METHODS}, not {method!r}.")
    values = np.array(lattice_parameters, dtype="double")
    if method == "none":
        return values, np.zeros_like(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"lattice_parameters must have shape (temperatures, 3), not {values.shape}."
        )

    out = values.copy()
    slopes = np.zeros_like(values)
    for column in range(3):
        series = values[:, column]
        if np.ptp(series) == 0.0:
            continue
        errors = None if sigma is None else np.asarray(sigma)[:, column]
        fit = _fit_einstein(temperatures, series, errors, n_terms=n_terms)
        out[:, column] = fit.model
        slopes[:, column] = fit.slope(temperatures)
    return out, slopes
