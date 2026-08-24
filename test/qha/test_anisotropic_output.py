# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.qha.anisotropic_output and anisotropic_plot."""

from __future__ import annotations

import dataclasses
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


def test_provenance_header(result: AnisotropicQHAResult, tmp_path: Path) -> None:
    """The header records the settings that produced the numbers."""
    tagged = dataclasses.replace(result, mesh=200.0, with_electronic=True, pressure=1.5)
    fn = tmp_path / "ax.dat"
    aniso_output.write_axial_thermal_expansion(
        tagged, filename=fn, provenance=["dataset=d.hdf5"]
    )
    lines = fn.read_text().splitlines()
    for item in (
        "mesh=200",
        "surface_degree=2",
        "F_el=on",
        "pressure=1.5 GPa",
        "grid_points=9",
        "temperatures=0-500 K step 100",
        "dataset=d.hdf5",
    ):
        assert item in lines[0]
    assert lines[1].startswith("# temperature (K), alpha_a")


def test_provenance_omits_unset_settings(result: AnisotropicQHAResult) -> None:
    """An unrecorded mesh and no pressure are left out; F_el reads off."""
    line = aniso_output.format_provenance(result)
    assert "mesh=" not in line
    assert "pressure" not in line
    assert "F_el=off" in line


def test_provenance_explicit_mesh(result: AnisotropicQHAResult) -> None:
    """An explicit mesh triple is rendered as given rather than as a length."""
    line = aniso_output.format_provenance(
        dataclasses.replace(result, mesh=[20, 20, 14])
    )
    assert "mesh=20 20 14" in line


def test_plot_functions(result: AnisotropicQHAResult) -> None:
    """The single-quantity plots build a figure and return the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")

    for func in (
        aniso_plot.plot_lattice_parameters,
        aniso_plot.plot_volume_temperature,
        aniso_plot.plot_axial_thermal_expansion,
        aniso_plot.plot_free_energy_temperature,
    ):
        plt = func(result)
        plt.close("all")


def test_plot_anisotropic_qha_returns_figure(result: AnisotropicQHAResult) -> None:
    """The summary plot returns a Figure and puts a and c on separate axes.

    The two lattice parameters differ by more than their temperature variation,
    so a single y axis shows two flat parallel lines. The left panel therefore
    carries a twin axis, and both must be populated.

    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = aniso_plot.plot_anisotropic_qha(result)
    assert isinstance(fig, plt.Figure)
    lattice_panel = fig.axes[0]
    twin = [
        ax
        for ax in fig.axes
        if ax is not lattice_panel and ax.bbox.bounds == lattice_panel.bbox.bounds
    ]
    assert twin, "the lattice-parameter panel has no twin axis"
    assert lattice_panel.lines and twin[0].lines
    plt.close(fig)


def test_contour_plots_write_one_file_per_temperature(
    result: AnisotropicQHAResult, tmp_path, monkeypatch
) -> None:
    """The contour plots are part of the plotting module, not of the script.

    They are the diagnostics that show where the free-energy valley sits and
    what each term contributes to it, so an API caller has to be able to reach
    them without importing the command-line script.

    """
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)

    # The shared fixture puts every cell at the same lattice, which the output
    # writers do not mind but a surface refit does. Give it a real 3 x 3 grid.
    a, c = np.meshgrid([2.90, 2.95, 3.00], [4.90, 5.00, 5.10], indexing="ij")
    a, c = a.ravel(), c.ravel()
    lattice_lengths = np.stack([a, a, c], axis=1)
    bowl = 3.0 * (a - 2.96) ** 2 + 2.0 * (c - 4.98) ** 2
    result = dataclasses.replace(
        result,
        lattice_lengths=lattice_lengths,
        helmholtz_lattice=np.tile(bowl, (len(result.temperatures), 1)),
    )

    temperatures = [0.0, 300.0]
    written = aniso_plot.plot_F_contours(result, temperatures)
    assert len(written) == len(temperatures)
    assert all((tmp_path / name).exists() for name in written)

    internal_energies = bowl * 0.5
    written = aniso_plot.plot_component_contours(
        result, internal_energies, None, temperatures
    )
    assert len(written) == len(temperatures)
    assert all((tmp_path / name).exists() for name in written)


def test_public_api_exports() -> None:
    """run_anisotropic_qha and AnisotropicQHAResult are exposed at top level."""
    import phonopy

    assert phonopy.run_anisotropic_qha is not None
    assert phonopy.AnisotropicQHAResult is AnisotropicQHAResult
