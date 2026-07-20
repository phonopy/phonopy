# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.qha.anisotropic_output and anisotropic_plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phonopy.qha import anisotropic_output as aniso_output
from phonopy.qha import anisotropic_plot as aniso_plot
from phonopy.qha.anisotropic import AnisotropicQHAResult


def _synthetic_result(n: int = 6) -> AnisotropicQHAResult:
    """Build a small hexagonal-like result without any phonon calculation."""
    temperatures = np.linspace(0.0, 500.0, n)
    a = 3.0 + 1e-4 * temperatures
    c = 5.0 + 2e-4 * temperatures
    elp = np.stack([a, a, c], axis=1)
    volumes = 0.8 * a * a * c
    beta = np.gradient(volumes, temperatures) / volumes
    axial = np.zeros((n, 3))
    axial[1:, 0] = 1e-4 / a[1:]
    axial[1:, 1] = 1e-4 / a[1:]
    axial[1:, 2] = 2e-4 / c[1:]
    n_points = 9
    return AnisotropicQHAResult(
        temperatures=temperatures,
        lattice_lengths=np.tile([3.0, 3.0, 5.0], (n_points, 1)),
        free_lattice_indices=np.array([0, 2], dtype="int64"),
        surface_degree=2,
        helmholtz_lattice=np.zeros((n, n_points)),
        equilibrium_lattice_parameters=elp,
        equilibrium_volumes=volumes,
        gibbs_free_energies=-40.0 + 1e-3 * temperatures,
        thermal_expansion=beta,
        axial_thermal_expansions=axial,
        surface_fit_rms=np.zeros(n),
        surface_fit_rank=6,
        surface_n_terms=6,
        minimum_extrapolated=np.zeros(n, dtype=bool),
    )


@pytest.fixture
def result() -> AnisotropicQHAResult:
    """Return a synthetic anisotropic QHA result."""
    return _synthetic_result()


def _data_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]


def test_write_lattice_parameters(result: AnisotropicQHAResult, tmp_path: Path) -> None:
    """Lattice-parameter writer emits one row per temperature with a, b, c."""
    fn = tmp_path / "lp.dat"
    aniso_output.write_lattice_parameters_temperature(result, filename=fn)
    lines = _data_lines(fn)
    assert len(lines) == len(result.temperatures)
    t, a, b, c = (float(x) for x in lines[-1].split())
    np.testing.assert_allclose(
        [t, a, b, c],
        [result.temperatures[-1], *result.equilibrium_lattice_parameters[-1]],
        rtol=1e-12,
    )


def test_write_axial_thermal_expansion(
    result: AnisotropicQHAResult, tmp_path: Path
) -> None:
    """Axial-expansion writer emits alpha_a, alpha_b, alpha_c and their sum."""
    fn = tmp_path / "ax.dat"
    aniso_output.write_axial_thermal_expansion(result, filename=fn)
    lines = _data_lines(fn)
    assert len(lines) == len(result.temperatures)
    values = [float(x) for x in lines[-1].split()]
    np.testing.assert_allclose(
        values[1:4], result.axial_thermal_expansions[-1], rtol=1e-8
    )
    np.testing.assert_allclose(
        values[4], result.axial_thermal_expansions[-1].sum(), rtol=1e-8
    )


def test_write_volume_and_free_energy(
    result: AnisotropicQHAResult, tmp_path: Path
) -> None:
    """Volume and free-energy writers emit one row per temperature."""
    fn_v = tmp_path / "v.dat"
    fn_f = tmp_path / "f.dat"
    aniso_output.write_volume_temperature(result, filename=fn_v)
    aniso_output.write_free_energy_temperature(result, filename=fn_f)
    assert len(_data_lines(fn_v)) == len(result.temperatures)
    assert len(_data_lines(fn_f)) == len(result.temperatures)


def test_plot_functions(result: AnisotropicQHAResult) -> None:
    """All plot functions build a figure without error."""
    import matplotlib

    matplotlib.use("Agg")

    for func in (
        aniso_plot.plot_anisotropic_qha,
        aniso_plot.plot_lattice_parameters,
        aniso_plot.plot_volume_temperature,
        aniso_plot.plot_axial_thermal_expansion,
        aniso_plot.plot_free_energy_temperature,
    ):
        plt = func(result)
        plt.close("all")


def test_public_api_exports() -> None:
    """run_anisotropic_qha and AnisotropicQHAResult are exposed at top level."""
    import phonopy

    assert phonopy.run_anisotropic_qha is not None
    assert phonopy.AnisotropicQHAResult is AnisotropicQHAResult
