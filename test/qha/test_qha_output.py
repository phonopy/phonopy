# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.qha.output and phonopy.qha.plot."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from phonopy import PhonopyQHA
from phonopy.qha import output as qha_output
from phonopy.qha import plot as qha_plot
from phonopy.qha.qha import QHAResult

SIMPLE_WRITER_NAMES = [
    "write_helmholtz_volume",
    "write_volume_temperature",
    "write_thermal_expansion",
    "write_gibbs_temperature",
    "write_bulk_modulus_temperature",
    "write_gruneisen_temperature",
]


@pytest.mark.parametrize("writer_name", SIMPLE_WRITER_NAMES)
def test_writers_identical_to_legacy(
    qha_result_nacl: QHAResult,
    qha_ref_nacl: PhonopyQHA,
    tmp_path: Path,
    writer_name: str,
) -> None:
    """New writers produce byte-identical files to the legacy writers."""
    fn_old = tmp_path / "old.dat"
    fn_new = tmp_path / "new.dat"
    getattr(qha_ref_nacl, writer_name)(filename=fn_old)
    getattr(qha_output, writer_name)(qha_result_nacl, filename=fn_new)
    assert fn_new.read_bytes() == fn_old.read_bytes()


def test_write_helmholtz_volume_fitted_identical(
    qha_result_nacl: QHAResult, qha_ref_nacl: PhonopyQHA, tmp_path: Path
) -> None:
    """Fitted Helmholtz writer produces a byte-identical file."""
    fn_old = tmp_path / "old.dat"
    fn_new = tmp_path / "new.dat"
    qha_ref_nacl.write_helmholtz_volume_fitted(10, filename=fn_old)
    qha_output.write_helmholtz_volume_fitted(qha_result_nacl, 10, filename=fn_new)
    assert fn_new.read_bytes() == fn_old.read_bytes()


def test_write_heat_capacity_P_identical(
    qha_result_nacl: QHAResult, qha_ref_nacl: PhonopyQHA, tmp_path: Path
) -> None:
    """C_P writer produces byte-identical files to the legacy polyfit writer."""
    old_names = [tmp_path / f"old-{i}.dat" for i in range(4)]
    new_names = [tmp_path / f"new-{i}.dat" for i in range(4)]
    qha_ref_nacl.write_heat_capacity_P_polyfit(
        filename=old_names[0],
        filename_ev=old_names[1],
        filename_cvv=old_names[2],
        filename_dsdvt=old_names[3],
    )
    qha_output.write_heat_capacity_P(
        qha_result_nacl,
        filename=new_names[0],
        filename_ev=new_names[1],
        filename_cvv=new_names[2],
        filename_dsdvt=new_names[3],
    )
    for fn_new, fn_old in zip(new_names, old_names, strict=True):
        assert fn_new.read_bytes() == fn_old.read_bytes()


def test_write_lattice_parameters_temperature(
    qha_result_nacl: QHAResult, tmp_path: Path
) -> None:
    """Lattice parameters file round-trips through np.loadtxt."""
    fn = tmp_path / "lattice_parameters-temperature.dat"
    qha_output.write_lattice_parameters_temperature(qha_result_nacl, filename=fn)

    assert fn.read_text().startswith("#")
    data = np.loadtxt(fn)
    assert qha_result_nacl.lattice is not None
    np.testing.assert_allclose(data[:, 0], qha_result_nacl.temperatures, atol=1e-10)
    np.testing.assert_allclose(
        data[:, 1:], qha_result_nacl.lattice.lattice_parameters, atol=1e-10
    )


def test_write_axial_thermal_expansion(
    qha_result_nacl: QHAResult, tmp_path: Path
) -> None:
    """Axial thermal expansion file round-trips through np.loadtxt."""
    fn = tmp_path / "axial_thermal_expansion.dat"
    qha_output.write_axial_thermal_expansion(qha_result_nacl, filename=fn)

    assert fn.read_text().startswith("#")
    data = np.loadtxt(fn)
    assert qha_result_nacl.lattice is not None
    alpha = qha_result_nacl.lattice.axial_thermal_expansions
    np.testing.assert_allclose(data[:, 0], qha_result_nacl.temperatures, atol=1e-10)
    np.testing.assert_allclose(data[:, 1:4], alpha, atol=1e-16)
    np.testing.assert_allclose(data[:, 4], alpha.sum(axis=1), atol=1e-16)


def test_writers_raise_without_data(qha_result_nacl: QHAResult, tmp_path: Path) -> None:
    """Writers raise RuntimeError when the required data are absent."""
    no_lattice = dataclasses.replace(qha_result_nacl, lattice=None)
    with pytest.raises(RuntimeError):
        qha_output.write_lattice_parameters_temperature(
            no_lattice, filename=tmp_path / "a.dat"
        )
    with pytest.raises(RuntimeError):
        qha_output.write_axial_thermal_expansion(
            no_lattice, filename=tmp_path / "b.dat"
        )


def test_plot_functions(qha_result_nacl: QHAResult) -> None:
    """All plot functions build a figure without error."""
    import matplotlib

    matplotlib.use("Agg")

    for func in (
        qha_plot.plot_qha,
        qha_plot.plot_helmholtz_volume,
        qha_plot.plot_volume_temperature,
        qha_plot.plot_thermal_expansion,
        qha_plot.plot_gibbs_temperature,
        qha_plot.plot_bulk_modulus_temperature,
        qha_plot.plot_heat_capacity_P,
        qha_plot.plot_gruneisen_temperature,
        qha_plot.plot_lattice_parameters,
        qha_plot.plot_axial_thermal_expansion,
    ):
        plt = func(qha_result_nacl)
        plt.close("all")


def test_plots_raise_without_data(qha_result_nacl: QHAResult) -> None:
    """Plot functions raise RuntimeError when the required data are absent."""
    import matplotlib

    matplotlib.use("Agg")

    no_lattice = dataclasses.replace(qha_result_nacl, lattice=None)
    with pytest.raises(RuntimeError):
        qha_plot.plot_lattice_parameters(no_lattice)
    with pytest.raises(RuntimeError):
        qha_plot.plot_axial_thermal_expansion(no_lattice)
