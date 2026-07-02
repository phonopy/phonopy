"""Tests for lattice-parameter fitting in phonopy.qha.lattice."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from phonopy.qha.lattice import LatticeParametersFit, compute_axial_thermal_expansion

volumes_ref = np.linspace(140.0, 190.0, 11)


def _make_lattice_parameters(
    volumes: NDArray[np.double],
    k: float,
    r_b: NDArray[np.double],
    r_c: NDArray[np.double],
) -> NDArray[np.double]:
    """Build exact (a, b, c) data satisfying V = k * a * b * c."""
    a = (volumes / (k * r_b * r_c)) ** (1.0 / 3)
    return np.array([a, r_b * a, r_c * a]).T


def _quadratic(
    volumes: NDArray[np.double], c0: float, c1: float, c2: float
) -> NDArray[np.double]:
    dv = volumes - 165.0
    return c0 + c1 * dv + c2 * dv**2


@pytest.mark.parametrize(
    "k",
    [np.sqrt(3) / 2, 1.0],  # hexagonal-like and tetragonal-like angle factors
)
def test_round_trip_uniaxial(k: float) -> None:
    """Fit of exact uniaxial (b = a) quadratic-ratio data is a round trip."""
    r_b = np.ones(len(volumes_ref))
    r_c = _quadratic(volumes_ref, 1.60, 1e-3, 2e-5)
    lattice_parameters = _make_lattice_parameters(volumes_ref, k, r_b, r_c)

    fit = LatticeParametersFit(volumes_ref, lattice_parameters)

    np.testing.assert_allclose(fit.k, k, rtol=1e-12)
    np.testing.assert_allclose(fit.evaluate(volumes_ref), lattice_parameters, rtol=1e-8)


def test_round_trip_orthorhombic() -> None:
    """Fit of exact orthorhombic data with two varying ratios is a round trip."""
    r_b = _quadratic(volumes_ref, 1.10, 5e-4, 0.0)
    r_c = _quadratic(volumes_ref, 1.35, -8e-4, 1e-5)
    lattice_parameters = _make_lattice_parameters(volumes_ref, 1.0, r_b, r_c)

    fit = LatticeParametersFit(volumes_ref, lattice_parameters)

    np.testing.assert_allclose(fit.evaluate(volumes_ref), lattice_parameters, rtol=1e-8)


def test_volume_consistency() -> None:
    """Evaluation volumes are reproduced exactly by k * a * b * c."""
    r_b = _quadratic(volumes_ref, 1.10, 5e-4, 0.0)
    r_c = _quadratic(volumes_ref, 1.35, -8e-4, 1e-5)
    lattice_parameters = _make_lattice_parameters(volumes_ref, 1.0, r_b, r_c)
    fit = LatticeParametersFit(volumes_ref, lattice_parameters)

    v = np.linspace(volumes_ref[0], volumes_ref[-1], 23)
    abc = fit.evaluate(v)
    np.testing.assert_allclose(fit.k * abc.prod(axis=1), v, rtol=1e-13)


def test_isotropic() -> None:
    """Constant unit ratios give a = b = c = (V / k)^(1/3)."""
    k = 0.9
    ones = np.ones(len(volumes_ref))
    lattice_parameters = _make_lattice_parameters(volumes_ref, k, ones, ones)
    fit = LatticeParametersFit(volumes_ref, lattice_parameters)

    v = np.linspace(volumes_ref[0], volumes_ref[-1], 7)
    abc = fit.evaluate(v)
    for i in range(3):
        np.testing.assert_allclose(abc[:, i], (v / k) ** (1.0 / 3), rtol=1e-10)


def test_k_inconsistent() -> None:
    """Perturbing one length breaks the constancy of k."""
    r_b = np.ones(len(volumes_ref))
    r_c = _quadratic(volumes_ref, 1.60, 1e-3, 2e-5)
    lattice_parameters = _make_lattice_parameters(volumes_ref, 1.0, r_b, r_c)
    lattice_parameters[3, 2] *= 1.001

    with pytest.raises(RuntimeError):
        LatticeParametersFit(volumes_ref, lattice_parameters)


def test_too_few_points() -> None:
    """Fewer volume points than degree + 1 raise RuntimeError."""
    volumes = volumes_ref[:2]
    ones = np.ones(2)
    lattice_parameters = _make_lattice_parameters(volumes, 1.0, ones, ones)

    with pytest.raises(RuntimeError):
        LatticeParametersFit(volumes, lattice_parameters, degree=2)


def test_invalid_shapes() -> None:
    """Malformed inputs raise ValueError."""
    ones = np.ones(len(volumes_ref))
    lattice_parameters = _make_lattice_parameters(volumes_ref, 1.0, ones, ones)

    with pytest.raises(ValueError):
        LatticeParametersFit(volumes_ref, lattice_parameters[:, :2])
    with pytest.raises(ValueError):
        LatticeParametersFit(volumes_ref, -lattice_parameters)


def test_extrapolation_warning() -> None:
    """Evaluation outside the fitted volume range warns but returns values."""
    ones = np.ones(len(volumes_ref))
    lattice_parameters = _make_lattice_parameters(volumes_ref, 1.0, ones, ones)
    fit = LatticeParametersFit(volumes_ref, lattice_parameters)

    with pytest.warns(UserWarning):
        abc = fit.evaluate([volumes_ref[0] - 5.0])
    assert np.isfinite(abc).all()


def test_axial_thermal_expansion() -> None:
    """Central differences reproduce the analytic expansion of linear data."""
    temperatures = np.linspace(0.0, 1000.0, 11)
    x0 = np.array([3.0, 3.3, 5.1])
    alpha = np.array([5e-6, 7e-6, 9e-6])
    lattice_parameters = x0 * (1.0 + alpha * temperatures[:, None])

    result = compute_axial_thermal_expansion(temperatures, lattice_parameters)

    assert result.shape == (len(temperatures) - 1, 3)
    np.testing.assert_allclose(result[0], np.zeros(3), atol=1e-30)
    expected = alpha / (1.0 + alpha * temperatures[1:-1, None])
    np.testing.assert_allclose(result[1:], expected, rtol=1e-10)
