"""Plotting functions for QHA results.

All functions take a QHAResult as the first argument and return the
matplotlib.pyplot module with the created figure active. matplotlib is
imported inside the functions and no global rcParams are modified.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from phonopy.qha.eos import get_eos

if TYPE_CHECKING:
    from phonopy.qha.qha import QHAResult


def plot_qha(
    result: QHAResult,
    thin_number: int = 10,
    volume_temp_exp: tuple[Any, Any] | None = None,
    energy_plot_factor: float | None = None,
) -> Any:
    """Return pyplot with Helmholtz-volume, V(T) and thermal expansion."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    _draw_helmholtz_volume(
        axs[0], result, thin_number=thin_number, energy_plot_factor=energy_plot_factor
    )
    _draw_volume_temperature(axs[1], result, exp_data=volume_temp_exp)
    _draw_thermal_expansion(axs[2], result)
    fig.tight_layout()
    return plt


def plot_helmholtz_volume(
    result: QHAResult,
    thin_number: int = 10,
    energy_plot_factor: float | None = None,
    xlabel: str = r"Volume $(\AA^3)$",
    ylabel: str = "Free energy",
) -> Any:
    """Return pyplot of Helmholtz free energies vs volume at temperatures."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_helmholtz_volume(
        ax,
        result,
        thin_number=thin_number,
        energy_plot_factor=energy_plot_factor,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return plt


def plot_volume_temperature(
    result: QHAResult, exp_data: tuple[Any, Any] | None = None
) -> Any:
    """Return pyplot of volume vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_volume_temperature(ax, result, exp_data=exp_data)
    return plt


def plot_thermal_expansion(result: QHAResult) -> Any:
    """Return pyplot of thermal expansion vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_thermal_expansion(ax, result)
    return plt


def plot_gibbs_temperature(
    result: QHAResult,
    xlabel: str = "Temperature (K)",
    ylabel: str = "Gibbs free energy (eV)",
) -> Any:
    """Return pyplot of Gibbs free energy vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_temperature_curve(
        ax, result.temperatures, result.gibbs_free_energies, xlabel, ylabel
    )
    return plt


def plot_bulk_modulus_temperature(
    result: QHAResult,
    xlabel: str = "Temperature (K)",
    ylabel: str = "Bulk modulus (GPa)",
) -> Any:
    """Return pyplot of bulk modulus vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_temperature_curve(ax, result.temperatures, result.bulk_moduli, xlabel, ylabel)
    return plt


def plot_heat_capacity_P(
    result: QHAResult,
    Z: int = 1,
    exp_data: tuple[Any, Any] | None = None,
) -> Any:
    """Return pyplot of C_P vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_temperature_curve(
        ax,
        result.temperatures,
        result.heat_capacity_P.heat_capacities / Z,
        "Temperature (K)",
        r"$C\mathrm{_P}$ $\mathrm{(J/mol\cdot K)}$",
    )
    if exp_data:
        ax.plot(exp_data[0], exp_data[1], "ro")
    return plt


def plot_gruneisen_temperature(result: QHAResult) -> Any:
    """Return pyplot of Gruneisen parameter vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_temperature_curve(
        ax,
        result.temperatures,
        result.gruneisen_parameters,
        "Temperature (K)",
        "Gruneisen parameter",
    )
    return plt


def plot_lattice_parameters(result: QHAResult) -> Any:
    """Return pyplot of lattice parameters vs temperature."""
    import matplotlib.pyplot as plt

    if result.lattice is None:
        raise RuntimeError("Lattice parameters are not available.")
    fig, ax = plt.subplots()
    temperatures = result.temperatures
    for i, label in enumerate(("$a$", "$b$", "$c$")):
        ax.plot(temperatures, result.lattice.lattice_parameters[:, i], label=label)
    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Lattice parameters $(\AA)$")
    ax.legend()
    return plt


def plot_axial_thermal_expansion(result: QHAResult) -> Any:
    """Return pyplot of axial thermal expansion coefficients vs temperature."""
    import matplotlib.pyplot as plt

    if result.lattice is None:
        raise RuntimeError("Axial thermal expansions are not available.")
    fig, ax = plt.subplots()
    temperatures = result.temperatures
    labels = (r"$\alpha_a$", r"$\alpha_b$", r"$\alpha_c$")
    for i, label in enumerate(labels):
        ax.plot(
            temperatures, result.lattice.axial_thermal_expansions[:, i], label=label
        )
    ax.plot(temperatures, result.thermal_expansion, "k--", label=r"$\beta$")
    ax.set_xlim(temperatures[0], temperatures[-1])
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal expansion $(\mathrm{K}^{-1})$")
    ax.legend()
    return plt


def _draw_helmholtz_volume(
    ax: Any,
    result: QHAResult,
    thin_number: int = 10,
    energy_plot_factor: float | None = None,
    xlabel: str = r"Volume $(\AA^3)$",
    ylabel: str = "Free energy",
) -> None:
    if energy_plot_factor is None:
        _energy_plot_factor = 1.0
        _ylabel = ylabel + " (eV)"
    else:
        _energy_plot_factor = energy_plot_factor
        _ylabel = ylabel

    eos = get_eos(result.eos_name)
    temperatures = result.temperatures
    equiv_volumes = result.equilibrium_volumes
    equiv_energies = result.gibbs_free_energies

    volume_points = np.linspace(min(result.volumes), max(result.volumes), 201)
    selected_volumes = []
    selected_energies = []

    thin_index = 0
    for i, _ in enumerate(temperatures):
        if i % thin_number == 0:
            selected_volumes.append(equiv_volumes[i])
            selected_energies.append(equiv_energies[i])

    e0 = 0.0
    for i, t in enumerate(temperatures):
        if t >= 298:
            if i > 0:
                de = equiv_energies[i] - equiv_energies[i - 1]
                dt = t - temperatures[i - 1]
                e0 = (298 - temperatures[i - 1]) / dt * de + equiv_energies[i - 1]
            else:
                e0 = 0.0
            break
    e0 *= _energy_plot_factor

    for i, _ in enumerate(temperatures):
        if i % thin_number == 0:
            ax.plot(
                result.volumes,
                np.array(result.helmholtz_volume[i]) * _energy_plot_factor - e0,
                "bo",
                markeredgecolor="b",
                markersize=3,
            )
            ax.plot(
                volume_points,
                eos(volume_points, result.eos_parameters[i]) * _energy_plot_factor - e0,
                "b-",
            )
            thin_index = i

    for i, j in enumerate((0, thin_index)):
        ax.text(
            result.volumes[-2],
            (result.helmholtz_volume[j, -1] + (1 - i * 2) * 0.1 - 0.05)
            * _energy_plot_factor
            - e0,
            "%dK" % int(temperatures[j]),
            fontsize=8,
        )

    ax.plot(
        selected_volumes,
        np.array(selected_energies) * _energy_plot_factor - e0,
        "ro-",
        markeredgecolor="r",
        markersize=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(_ylabel)


def _draw_volume_temperature(
    ax: Any,
    result: QHAResult,
    exp_data: tuple[Any, Any] | None = None,
    xlabel: str = "Temperature (K)",
    ylabel: str = r"Volume $(\AA^3)$",
) -> None:
    _draw_temperature_curve(
        ax, result.temperatures, result.equilibrium_volumes, xlabel, ylabel
    )
    if exp_data:
        ax.plot(exp_data[0], exp_data[1], "ro")


def _draw_thermal_expansion(
    ax: Any,
    result: QHAResult,
    xlabel: str = "Temperature (K)",
    ylabel: str = r"Thermal expansion $(\mathrm{K}^{-1})$",
) -> None:
    from matplotlib.ticker import ScalarFormatter

    class FixedScaledFormatter(ScalarFormatter):
        def __init__(self) -> None:
            super().__init__(useMathText=True)

        def _set_orderOfMagnitude(self, range: float) -> None:
            self.orderOfMagnitude = -6

    ax.yaxis.set_major_formatter(FixedScaledFormatter())
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    _draw_temperature_curve(
        ax, result.temperatures, result.thermal_expansion, xlabel, ylabel
    )


def _draw_temperature_curve(
    ax: Any,
    temperatures: Any,
    values: Any,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(temperatures, values, "r-")
    ax.set_xlim(temperatures[0], temperatures[-1])
