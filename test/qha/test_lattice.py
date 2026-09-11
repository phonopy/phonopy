# SPDX-License-Identifier: BSD-3-Clause
"""Tests for lattice-parameter fitting in phonopy.qha.lattice."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from phonopy.qha.lattice import (
    LatticeGrid,
    LatticeParametersFit,
    compute_axial_thermal_expansion,
)

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

    np.testing.assert_allclose(fit.primitive_volume_abc_ratio, k, rtol=1e-12)
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
    np.testing.assert_allclose(
        fit.primitive_volume_abc_ratio * abc.prod(axis=1), v, rtol=1e-13
    )


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


def test_lattice_grid_refuses_cells_of_differing_shape() -> None:
    """A grid whose cells differ in more than their lengths is refused.

    One ratio V / (a b c) then describes none of them, and the volume of
    an interpolated cell would be wrong at every temperature.

    """
    lengths = np.array([[3.0, 3.0, 5.0], [3.1, 3.1, 5.1], [3.2, 3.2, 5.2]])
    lattices = np.array([np.diag(row) for row in lengths])
    LatticeGrid(lattices, np.eye(3))  # right angles throughout: accepted

    lattices[1, 2, 0] = 0.5  # that cell alone is no longer orthogonal
    with pytest.raises(RuntimeError, match="constant k"):
        LatticeGrid(lattices, np.eye(3))


def _grid_of(lengths: NDArray[np.double]) -> LatticeGrid:
    """Return a grid of orthogonal cells with these lengths."""
    return LatticeGrid(np.array([np.diag(row) for row in lengths]), np.eye(3))


def test_detect_dof_hexagonal() -> None:
    """A and b tied and c independent give two DOF with a mapped to b."""
    a = np.array([3.0, 3.1, 3.2])
    c = np.array([5.0, 4.9, 5.1])
    grid = _grid_of(np.stack([a, a, c], axis=1))
    assert grid.column_map == (0, 0, 2)
    np.testing.assert_array_equal(grid.free_axis_indices, [0, 2])


def test_detect_dof_orthorhombic() -> None:
    """Three independently varying lengths give three DOF."""
    grid = _grid_of(np.array([[3.0, 4.0, 5.0], [3.1, 4.1, 4.9], [2.9, 3.9, 5.1]]))
    assert grid.column_map == (0, 1, 2)


def test_detect_dof_cubic() -> None:
    """A = b = c collapse to a single DOF shared by all three columns."""
    a = np.array([3.0, 3.1, 3.2])
    grid = _grid_of(np.stack([a, a, a], axis=1))
    assert grid.column_map == (0, 0, 0)
    np.testing.assert_array_equal(grid.free_axis_indices, [0])


def test_detect_dof_unsampled_column() -> None:
    """A length that never varies is refused, not carried as a constant."""
    a = np.array([3.0, 3.1, 3.2])
    b = np.full(3, 4.0)
    c = np.array([5.0, 5.1, 4.9])
    with pytest.raises(ValueError, match="Lattice length b is the same"):
        _grid_of(np.stack([a, b, c], axis=1))


def test_detect_dof_no_variation() -> None:
    """Cells with no varying lattice length raise ValueError."""
    with pytest.raises(ValueError):
        _grid_of(np.tile([3.0, 4.0, 5.0], (4, 1)))


def test_spread_puts_one_value_per_dof_on_three_lengths() -> None:
    """Every length reads the free DOF its representative column names."""
    a = np.array([3.0, 3.1, 3.2])
    c = np.array([5.0, 4.9, 5.1])
    grid = _grid_of(np.stack([a, a, c], axis=1))

    np.testing.assert_allclose(grid.spread(np.array([3.2, 5.1])), [3.2, 3.2, 5.1])
    series = np.array([[3.0, 5.0], [3.1, 5.1]])
    np.testing.assert_allclose(grid.spread(series), [[3.0, 3.0, 5.0], [3.1, 3.1, 5.1]])


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
