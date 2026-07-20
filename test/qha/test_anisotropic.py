# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the anisotropic QHA free-energy surface fit."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from numpy.typing import NDArray
from qha_utils import MESH, internal_energies, scaled_phonopy

from phonopy import Phonopy
from phonopy.qha.anisotropic import (
    FreeEnergySurfaceFit,
    _detect_lattice_dof,
    _reconstruct_lattice_parameters,
    run_anisotropic_qha,
)
from phonopy.qha.calc import (
    generate_total_degree_exponents,
    polynomial_design_matrix,
)

TEMPERATURES = np.arange(0.0, 1001.0, 200.0)


def _grid(centers: list[float], half_widths: list[float], n: int) -> NDArray[np.double]:
    """Build a regular tensor-product grid of points around centers."""
    axes = [
        np.linspace(c - h, c + h, n) for c, h in zip(centers, half_widths, strict=True)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


@pytest.mark.parametrize(
    ("ndim", "degree"),
    [(1, 3), (2, 2), (2, 4), (3, 2), (3, 3)],
)
def test_exponent_count(ndim: int, degree: int) -> None:
    """The number of total-degree monomials is C(ndim + degree, degree)."""
    exponents = generate_total_degree_exponents(ndim, degree)
    assert exponents.shape == (comb(ndim + degree, degree), ndim)
    assert (exponents.sum(axis=1) <= degree).all()
    # First row is the constant term.
    np.testing.assert_array_equal(exponents[0], np.zeros(ndim, dtype="int64"))


def test_design_matrix_values() -> None:
    """Design matrix columns reproduce the intended monomials."""
    exponents = generate_total_degree_exponents(2, 2)  # 1, x, y, x^2, xy, y^2
    points = np.array([[2.0, 3.0], [1.0, 5.0]])
    design = polynomial_design_matrix(points, exponents)
    for row, (x, y) in zip(design, points, strict=True):
        expected = np.array([x**e0 * y**e1 for e0, e1 in exponents])
        np.testing.assert_allclose(row, expected)


def _quadratic_2d(points: NDArray[np.double]) -> NDArray[np.double]:
    """F(a, c) = 1 + 2 (a - 3)^2 + 3 (c - 5)^2 + 0.5 (a - 3)(c - 5)."""
    da = points[:, 0] - 3.0
    dc = points[:, 1] - 5.0
    return 1.0 + 2.0 * da**2 + 3.0 * dc**2 + 0.5 * da * dc


def test_fit_round_trip_2d() -> None:
    """A degree-2 fit reproduces an exact quadratic surface."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 5)
    values = _quadratic_2d(points)
    fit = FreeEnergySurfaceFit(points, values, degree=2)

    probe = _grid([3.0, 5.0], [0.15, 0.15], 4)
    np.testing.assert_allclose(fit.evaluate(probe), _quadratic_2d(probe), atol=1e-10)


def test_minimum_2d() -> None:
    """The fitted minimum of the coupled quadratic is at (3, 5)."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 5)
    values = _quadratic_2d(points)
    fit = FreeEnergySurfaceFit(points, values, degree=2)

    x_min = fit.minimize()
    np.testing.assert_allclose(x_min, [3.0, 5.0], atol=1e-6)


def test_gradient_matches_finite_difference() -> None:
    """Analytic gradient agrees with a central finite difference."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 6)
    values = _quadratic_2d(points) + 0.1 * (points[:, 0] - 3.0) ** 3
    fit = FreeEnergySurfaceFit(points, values, degree=3)

    x = np.array([[3.05, 4.95]])
    analytic = fit.gradient(x)[0]
    eps = 1e-6
    numeric = np.empty(2)
    for j in range(2):
        step = np.zeros((1, 2))
        step[0, j] = eps
        numeric[j] = (fit.evaluate(x + step)[0] - fit.evaluate(x - step)[0]) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_minimum_1d_and_3d() -> None:
    """Minima are recovered for cubic (1D) and orthorhombic (3D) DOF."""
    pts_1d = np.linspace(3.8, 4.2, 7)[:, None]
    fit_1d = FreeEnergySurfaceFit(pts_1d, 5.0 * (pts_1d[:, 0] - 4.0) ** 2, degree=2)
    np.testing.assert_allclose(fit_1d.minimize(), [4.0], atol=1e-6)

    pts_3d = _grid([3.0, 4.0, 5.0], [0.2, 0.2, 0.2], 4)
    da = pts_3d[:, 0] - 3.0
    db = pts_3d[:, 1] - 4.0
    dc = pts_3d[:, 2] - 5.0
    values = 2.0 * da**2 + 3.0 * db**2 + 4.0 * dc**2 + 0.3 * da * db - 0.2 * db * dc
    fit_3d = FreeEnergySurfaceFit(pts_3d, values, degree=2)
    np.testing.assert_allclose(fit_3d.minimize(), [3.0, 4.0, 5.0], atol=1e-6)


def test_extrapolation_warning() -> None:
    """A minimum outside the sampled box warns."""
    # Minimum at a = 5 lies above the sampled range [3.8, 4.2].
    pts = np.linspace(3.8, 4.2, 7)[:, None]
    values = (pts[:, 0] - 5.0) ** 2
    fit = FreeEnergySurfaceFit(pts, values, degree=2)
    with pytest.warns(UserWarning):
        fit.minimize()
    assert fit.minimum_extrapolated is True


def test_fit_diagnostics() -> None:
    """The fit exposes RMS residual, rank, and minimize state."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 5)
    values = _quadratic_2d(points)
    fit = FreeEnergySurfaceFit(points, values, degree=2)

    assert fit.n_terms == 6
    assert fit.rank == 6
    assert not fit.is_rank_deficient
    assert fit.rms_residual == pytest.approx(0.0, abs=1e-10)

    # Convergence flags are unset until minimize runs.
    assert fit.minimize_converged is None
    assert fit.minimum_extrapolated is None
    fit.minimize()
    assert fit.minimize_converged
    assert fit.minimum_extrapolated is False


def test_fit_rank_deficient() -> None:
    """Collinear sample points give a rank-deficient design matrix."""
    t = np.linspace(-1.0, 1.0, 8)
    points = np.stack([3.0 + 0.1 * t, 5.0 + 0.2 * t], axis=1)  # points on a line
    values = t**2
    fit = FreeEnergySurfaceFit(points, values, degree=2)
    assert fit.is_rank_deficient
    assert fit.rank < fit.n_terms


def test_too_few_points() -> None:
    """Fewer sample points than polynomial terms raise RuntimeError."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 2)  # 4 points
    values = _quadratic_2d(points)
    with pytest.raises(RuntimeError):
        FreeEnergySurfaceFit(points, values, degree=3)  # needs 10 terms


def test_invalid_shapes() -> None:
    """Malformed inputs raise ValueError."""
    points = _grid([3.0, 5.0], [0.2, 0.3], 4)
    values = _quadratic_2d(points)
    with pytest.raises(ValueError):
        FreeEnergySurfaceFit(points, values[:-1])
    with pytest.raises(ValueError):
        FreeEnergySurfaceFit(points[:, 0], values)  # 1D points array


# --- Free lattice DOF detection --------------------------------------------


def test_detect_dof_hexagonal() -> None:
    """A and b tied and c independent give two DOF with a mapped to b."""
    a = np.array([3.0, 3.1, 3.2])
    c = np.array([5.0, 4.9, 5.1])
    lengths = np.stack([a, a, c], axis=1)
    free_indices, column_map, _ = _detect_lattice_dof(lengths)
    assert free_indices == [0, 2]
    np.testing.assert_array_equal(column_map, [0, 0, 1])


def test_detect_dof_orthorhombic() -> None:
    """Three independently varying lengths give three DOF."""
    lengths = np.array([[3.0, 4.0, 5.0], [3.1, 4.1, 4.9], [2.9, 3.9, 5.1]])
    free_indices, column_map, _ = _detect_lattice_dof(lengths)
    assert free_indices == [0, 1, 2]
    np.testing.assert_array_equal(column_map, [0, 1, 2])


def test_detect_dof_cubic() -> None:
    """A = b = c collapse to a single DOF shared by all three columns."""
    a = np.array([3.0, 3.1, 3.2])
    lengths = np.stack([a, a, a], axis=1)
    free_indices, column_map, _ = _detect_lattice_dof(lengths)
    assert free_indices == [0]
    np.testing.assert_array_equal(column_map, [0, 0, 0])


def test_detect_dof_fixed_column() -> None:
    """A constant column is fixed (mapped to -1), not a degree of freedom."""
    a = np.array([3.0, 3.1, 3.2])
    b = np.full(3, 4.0)
    c = np.array([5.0, 5.1, 4.9])
    lengths = np.stack([a, b, c], axis=1)
    free_indices, column_map, fixed_values = _detect_lattice_dof(lengths)
    assert free_indices == [0, 2]
    np.testing.assert_array_equal(column_map, [0, -1, 1])
    assert fixed_values[1] == pytest.approx(4.0)


def test_detect_dof_no_variation() -> None:
    """Cells with no varying lattice length raise ValueError."""
    lengths = np.tile([3.0, 4.0, 5.0], (4, 1))
    with pytest.raises(ValueError):
        _detect_lattice_dof(lengths)


def test_reconstruct_lattice_parameters() -> None:
    """Free DOF fill their columns; fixed columns keep their stored value."""
    column_map = np.array([0, 0, 1])  # hexagonal: a = b = x[0], c = x[1]
    fixed_values = np.array([0.0, 0.0, 0.0])
    abc = _reconstruct_lattice_parameters(
        np.array([3.2, 5.1]), column_map, fixed_values
    )
    np.testing.assert_allclose(abc, [3.2, 3.2, 5.1])


# --- End-to-end driver over Phonopy instances ------------------------------


def _tetragonal_phonopys(ph_nacl: Phonopy) -> list[Phonopy]:
    """Build a 3x3 (a, c) grid of tetragonal cells from cubic NaCl."""
    phonopys = []
    for sa in (0.98, 1.00, 1.02):
        for sc in (0.98, 1.00, 1.02):
            phonopys.append(scaled_phonopy(ph_nacl, np.array([sa, sa, sc])))
    return phonopys


def _tetragonal_internal_energies(phonopys: list[Phonopy]) -> NDArray[np.double]:
    """Return a stiff 2D parabola U(a, c) with its minimum inside the grid."""
    lengths = np.array([np.linalg.norm(ph.unitcell.cell, axis=1) for ph in phonopys])
    a = lengths[:, 0]
    c = lengths[:, 2]
    a0 = a.mean() * 0.998
    c0 = c.mean() * 1.004
    return 3.0 * (a - a0) ** 2 + 2.0 * (c - c0) ** 2 - 40.0


def test_run_anisotropic_tetragonal(ph_nacl: Phonopy) -> None:
    """End-to-end anisotropic QHA over a 2D (a, c) tetragonal grid."""
    phonopys = _tetragonal_phonopys(ph_nacl)
    energies = _tetragonal_internal_energies(phonopys)

    result = run_anisotropic_qha(
        phonopys, TEMPERATURES, internal_energies=energies, mesh=MESH, surface_degree=2
    )

    n = len(TEMPERATURES) - 1
    assert result.temperatures.shape == (n,)
    assert result.equilibrium_lattice_parameters.shape == (n, 3)
    np.testing.assert_array_equal(result.free_lattice_indices, [0, 2])

    elp = result.equilibrium_lattice_parameters
    # b is tied to a for a tetragonal cell.
    np.testing.assert_allclose(elp[:, 0], elp[:, 1], rtol=1e-12)

    # The equilibrium stays inside the sampled grid (no extrapolation).
    low = result.lattice_lengths.min(axis=0)
    high = result.lattice_lengths.max(axis=0)
    assert (elp >= low - 1e-9).all() and (elp <= high + 1e-9).all()

    # Fit diagnostics: full rank, finite residuals, no extrapolation.
    assert result.surface_fit_rms.shape == (n,)
    assert (result.surface_fit_rms >= 0.0).all()
    assert result.minimum_extrapolated.shape == (n,)
    assert not result.minimum_extrapolated.any()
    assert result.surface_n_terms == 6  # C(2 + 2, 2)
    assert result.surface_fit_rank == result.surface_n_terms

    # Volume consistency: k * a * b * c == equilibrium volume.
    volumes = np.array([ph.primitive.volume for ph in phonopys])
    lengths = np.array([np.linalg.norm(ph.unitcell.cell, axis=1) for ph in phonopys])
    k = (volumes / lengths.prod(axis=1)).mean()
    np.testing.assert_allclose(
        k * elp.prod(axis=1), result.equilibrium_volumes, rtol=1e-12
    )

    # Axial expansions sum to the volumetric beta (exact in the continuum).
    alpha_sum = result.axial_thermal_expansions.sum(axis=1)
    np.testing.assert_allclose(alpha_sum[0], 0.0, atol=1e-30)
    np.testing.assert_allclose(alpha_sum[1:], result.thermal_expansion[1:], rtol=1e-2)


def test_run_anisotropic_cubic_one_dof(ph_nacl: Phonopy) -> None:
    """A cubic series collapses to one DOF with a = b = c at equilibrium."""
    phonopys = [
        scaled_phonopy(ph_nacl, np.array([s, s, s])) for s in np.linspace(0.98, 1.03, 6)
    ]
    volumes = np.array([ph.primitive.volume for ph in phonopys])

    result = run_anisotropic_qha(
        phonopys,
        TEMPERATURES,
        internal_energies=internal_energies(volumes),
        mesh=MESH,
        surface_degree=2,
    )

    np.testing.assert_array_equal(result.free_lattice_indices, [0])
    elp = result.equilibrium_lattice_parameters
    np.testing.assert_allclose(elp[:, 0], elp[:, 1], rtol=1e-12)
    np.testing.assert_allclose(elp[:, 0], elp[:, 2], rtol=1e-12)

    alpha = result.axial_thermal_expansions
    np.testing.assert_allclose(
        3.0 * alpha[1:, 0], result.thermal_expansion[1:], rtol=1e-2
    )


def test_run_anisotropic_requires_force_constants(ph_nacl: Phonopy) -> None:
    """A Phonopy without force constants raises RuntimeError."""
    phonopys = _tetragonal_phonopys(ph_nacl)
    bare = Phonopy(
        phonopys[0].unitcell,
        supercell_matrix=phonopys[0].supercell_matrix,
        primitive_matrix=phonopys[0].primitive_matrix,
        log_level=0,
    )
    phonopys[0] = bare
    with pytest.raises(RuntimeError):
        run_anisotropic_qha(
            phonopys,
            TEMPERATURES,
            internal_energies=np.zeros(len(phonopys)),
            mesh=MESH,
            surface_degree=2,
        )
