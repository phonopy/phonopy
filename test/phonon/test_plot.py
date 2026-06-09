"""Smoke tests for figure-level plotting of phonon results."""

from __future__ import annotations

import matplotlib.pyplot as plt

from phonopy import Phonopy

_BAND_PATHS = [[[0, 0, 0], [0.25, 0, 0.25], [0.5, 0, 0.5]]]


def _run_total_dos(ph: Phonopy) -> None:
    ph.run_mesh([5, 5, 5])
    ph.run_total_dos()


def _run_projected_dos(ph: Phonopy) -> None:
    ph.run_mesh([5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False)
    ph.run_projected_dos()


def test_plot_band_structure(ph_nacl: Phonopy):
    """Test Phonopy.plot_band_structure (ImageGrid branch)."""
    ph_nacl.run_band_structure(_BAND_PATHS)
    assert ph_nacl.plot_band_structure() is plt
    plt.close("all")


def test_plot_band_structure_legacy(ph_nacl: Phonopy):
    """Test Phonopy.plot_band_structure (legacy single-axes branch)."""
    ph_nacl.run_band_structure(_BAND_PATHS, is_legacy_plot=True)
    assert ph_nacl.plot_band_structure() is plt
    plt.close("all")


def test_plot_band_structure_and_dos(ph_nacl: Phonopy):
    """Test Phonopy.plot_band_structure_and_dos with total DOS."""
    ph_nacl.run_band_structure(_BAND_PATHS)
    _run_total_dos(ph_nacl)
    assert ph_nacl.plot_band_structure_and_dos() is plt
    plt.close("all")


def test_plot_band_structure_and_dos_legacy(ph_nacl: Phonopy):
    """Test Phonopy.plot_band_structure_and_dos (legacy branch)."""
    ph_nacl.run_band_structure(_BAND_PATHS, is_legacy_plot=True)
    _run_total_dos(ph_nacl)
    assert ph_nacl.plot_band_structure_and_dos() is plt
    plt.close("all")


def test_plot_band_structure_and_pdos(ph_nacl: Phonopy):
    """Test Phonopy.plot_band_structure_and_dos with projected DOS."""
    ph_nacl.run_band_structure(_BAND_PATHS)
    _run_projected_dos(ph_nacl)
    assert ph_nacl.plot_band_structure_and_dos(pdos_indices=[[0], [1]]) is plt
    plt.close("all")


def test_plot_total_dos(ph_nacl: Phonopy):
    """Test Phonopy.plot_total_dos."""
    _run_total_dos(ph_nacl)
    assert ph_nacl.plot_total_dos() is plt
    plt.close("all")


def test_plot_total_dos_tight_range(ph_nacl: Phonopy):
    """Test Phonopy.plot_total_dos with tight frequency range."""
    _run_total_dos(ph_nacl)
    assert ph_nacl.plot_total_dos(with_tight_frequency_range=True) is plt
    plt.close("all")


def test_plot_projected_dos(ph_nacl: Phonopy):
    """Test Phonopy.plot_projected_dos."""
    _run_projected_dos(ph_nacl)
    assert ph_nacl.plot_projected_dos() is plt
    plt.close("all")


def test_plot_projected_dos_options(ph_nacl: Phonopy):
    """Test Phonopy.plot_projected_dos with indices, legend, tight range."""
    _run_projected_dos(ph_nacl)
    returned = ph_nacl.plot_projected_dos(
        pdos_indices=[[0], [1]],
        legend=["Na", "Cl"],
        with_tight_frequency_range=True,
    )
    assert returned is plt
    plt.close("all")


def test_plot_thermal_properties(ph_nacl: Phonopy):
    """Test Phonopy.plot_thermal_properties."""
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(t_min=0, t_max=100, t_step=50)
    assert ph_nacl.plot_thermal_properties() is plt
    plt.close("all")


def test_plot_thermal_properties_leaves_rcparams_untouched(ph_nacl: Phonopy):
    """plot_thermal_properties must not mutate global matplotlib rcParams."""
    ph_nacl.run_mesh([5, 5, 5])
    ph_nacl.run_thermal_properties(t_min=0, t_max=100, t_step=50)
    pdf_fonttype = plt.rcParams["pdf.fonttype"]
    font_family = list(plt.rcParams["font.family"])
    ph_nacl.plot_thermal_properties()
    assert plt.rcParams["pdf.fonttype"] == pdf_fonttype
    assert list(plt.rcParams["font.family"]) == font_family
    plt.close("all")


def test_plot_thermal_displacements(ph_nacl: Phonopy):
    """Test Phonopy.plot_thermal_displacements."""
    ph_nacl.run_mesh([5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False)
    ph_nacl.run_thermal_displacements(t_min=0, t_max=100, t_step=50)
    assert ph_nacl.plot_thermal_displacements(is_legend=True) is plt
    plt.close("all")


def test_thermal_displacements_plot_takes_ax(ph_nacl: Phonopy):
    """ThermalDisplacements.plot takes a matplotlib Axes like other results."""
    ph_nacl.run_mesh([5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False)
    td = ph_nacl.run_thermal_displacements(t_min=0, t_max=100, t_step=50)
    _, ax = plt.subplots()
    td.plot(ax, is_legend=True)
    assert len(ax.lines) > 0
    plt.close("all")
