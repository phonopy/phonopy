# SPDX-License-Identifier: BSD-3-Clause
"""Plotting functions for anisotropic QHA results.

All functions take an AnisotropicQHAResult as the first argument and return
the matplotlib.pyplot module with the created figure active. matplotlib is
imported inside the functions and no global rcParams are modified.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phonopy.qha.anisotropic import AnisotropicQHAResult


def plot_anisotropic_qha(result: AnisotropicQHAResult) -> Any:
    """Return pyplot with lattice parameters, V(T) and axial expansion."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    _draw_lattice_parameters(axs[0], result)
    _draw_volume_temperature(axs[1], result)
    _draw_axial_thermal_expansion(axs[2], result)
    fig.tight_layout()
    return plt


def plot_lattice_parameters(result: AnisotropicQHAResult) -> Any:
    """Return pyplot of equilibrium lattice parameters vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_lattice_parameters(ax, result)
    return plt


def plot_volume_temperature(result: AnisotropicQHAResult) -> Any:
    """Return pyplot of equilibrium volume vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_volume_temperature(ax, result)
    return plt


def plot_axial_thermal_expansion(result: AnisotropicQHAResult) -> Any:
    """Return pyplot of axial thermal expansion coefficients vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_axial_thermal_expansion(ax, result)
    return plt


def plot_free_energy_temperature(
    result: AnisotropicQHAResult,
    xlabel: str = "Temperature (K)",
    ylabel: str = "Free energy (eV)",
) -> Any:
    """Return pyplot of the minimized free energy vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(result.temperatures, result.gibbs_free_energies, "r-")
    ax.set_xlim(result.temperatures[0], result.temperatures[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return plt


def _draw_lattice_parameters(ax: Any, result: AnisotropicQHAResult) -> None:
    temperatures = result.temperatures
    for i, label in enumerate(("$a$", "$b$", "$c$")):
        ax.plot(temperatures, result.equilibrium_lattice_parameters[:, i], label=label)
    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Lattice parameters $(\AA)$")
    ax.legend()


def _draw_volume_temperature(ax: Any, result: AnisotropicQHAResult) -> None:
    temperatures = result.temperatures
    ax.plot(temperatures, result.equilibrium_volumes, "r-")
    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Volume $(\AA^3)$")


def _draw_axial_thermal_expansion(ax: Any, result: AnisotropicQHAResult) -> None:
    temperatures = result.temperatures
    labels = (r"$\alpha_a$", r"$\alpha_b$", r"$\alpha_c$")
    for i, label in enumerate(labels):
        ax.plot(temperatures, result.axial_thermal_expansions[:, i], label=label)
    ax.plot(temperatures, result.thermal_expansion, "k--", label=r"$\beta$")
    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal expansion $(\mathrm{K}^{-1})$")
    ax.legend()
