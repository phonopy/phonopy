# SPDX-License-Identifier: BSD-3-Clause
"""Plotting functions for anisotropic QHA results.

All functions take an AnisotropicQHAResult as the first argument. The
single-quantity plots return the matplotlib.pyplot module with the created
figure active; plot_anisotropic_qha returns the Figure, and the contour
functions write one file per temperature and return the names written. No
global rcParams are modified.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from phonopy.qha.anisotropic import AnisotropicQHAResult, FreeEnergySurfaceFit
from phonopy.qha.thermal import compute_electronic_contributions_from_states

# Free energies are handled in eV throughout and converted only for plotting.
_EV_TO_MEV = 1000.0


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


def _evaluate_surface(result: AnisotropicQHAResult, temperature: float, n: int) -> dict:
    """Rebuild the fitted F surface at the nearest temperature and evaluate it.

    Returns the sample cells, the dense n x n evaluation mesh, and F offset by
    its own minimum (F - F_min) in eV, so that only the surface shape remains.

    """
    fi = result.free_lattice_indices
    i = int(np.argmin(np.abs(result.temperatures - temperature)))
    free_points = result.lattice_lengths[:, fi]
    fit = FreeEnergySurfaceFit(
        free_points, result.helmholtz_lattice[i], degree=result.surface_degree
    )

    lo0, lo1 = free_points.min(axis=0)
    hi0, hi1 = free_points.max(axis=0)
    grid0, grid1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    mesh = np.column_stack([grid0.ravel(), grid1.ravel()])
    fe = fit.evaluate(mesh).reshape(grid0.shape)
    fe = fe - fe.min()
    return {
        "i": i,
        "t": float(result.temperatures[i]),
        "free_points": free_points,
        "grid0": grid0,
        "grid1": grid1,
        "fe": fe,
    }


def plot_F_contours(
    result: AnisotropicQHAResult,
    temperatures: Sequence[float],
    n: int = 200,
) -> list[str]:
    """Save contour maps of F - F_min over the 2 free lattice DOF.

    One map per requested temperature (snapped to the nearest computed
    temperature), all sharing one color scale so valley depth and curvature are
    comparable. Overlays the sample cells and the located minimum. Returns the
    written filenames, empty unless there are exactly 2 free lattice DOF.

    """
    fi = result.free_lattice_indices
    if len(fi) != 2:
        print(f"Skip contour map: {len(fi)} free lattice DOF (need 2).")
        return []

    data = [_evaluate_surface(result, t, n) for t in temperatures]
    vmax = max(float(d["fe"].max()) for d in data) * _EV_TO_MEV
    levels = np.linspace(0.0, vmax, 41)

    axis = ("a", "b", "c")
    written = []
    for d in data:
        i = d["i"]
        fe = d["fe"] * _EV_TO_MEV
        fig, ax = plt.subplots()
        filled = ax.contourf(d["grid0"], d["grid1"], fe, levels=levels, extend="max")
        ax.contour(
            d["grid0"],
            d["grid1"],
            fe,
            levels=levels[::2],
            colors="k",
            linewidths=0.4,
        )
        fig.colorbar(filled, label="F - F_min (meV)")

        ax.plot(
            d["free_points"][:, 0],
            d["free_points"][:, 1],
            "wo",
            ms=3,
            label="samples",
        )
        eq = result.equilibrium_lattice_parameters[i]
        extrapolated = bool(result.minimum_extrapolated[i])
        ax.plot(
            eq[fi[0]],
            eq[fi[1]],
            "rX" if extrapolated else "r*",
            ms=14,
            label="minimum (extrapolated)" if extrapolated else "minimum",
        )

        ax.set_xlabel(f"{axis[fi[0]]} (A)")
        ax.set_ylabel(f"{axis[fi[1]]} (A)")
        ax.set_title(f"Free energy surface at T = {d['t']:.1f} K")
        ax.legend()

        filename = f"F_contour_{int(round(d['t']))}K.png"
        fig.savefig(filename)
        plt.close(fig)
        written.append(filename)
    return written


def _fit_and_grid(
    free_points: NDArray[np.double],
    values: NDArray[np.double],
    degree: int,
    n: int,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Fit a total-degree polynomial to values and evaluate it on a mesh.

    Returns (grid0, grid1, fe) with fe offset by its own minimum, so only the
    surface shape and tilt remain (any additive constant drops out).

    """
    fit = FreeEnergySurfaceFit(free_points, values, degree=degree)
    lo0, lo1 = free_points.min(axis=0)
    hi0, hi1 = free_points.max(axis=0)
    grid0, grid1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    mesh = np.column_stack([grid0.ravel(), grid1.ravel()])
    fe = fit.evaluate(mesh).reshape(grid0.shape)
    return grid0, grid1, fe - fe.min()


def plot_component_contours(
    result: AnisotropicQHAResult,
    internal_energies: Sequence[float],
    electronic_structures: Sequence | None,
    temperatures: Sequence[float],
    n: int = 200,
    electronic_free_energies: (
        Sequence[Sequence[float]] | NDArray[np.double] | None
    ) = None,
) -> list[str]:
    """Split the F(a, c) contour into its static, phonon and electronic parts.

    Draws U, F_ph, optionally F_el and the total on the same (a, c) domain so
    the valley shape can be attributed: U sets the static shape, while the
    near-linear F_ph (+ F_el) ramps carry the temperature-driven shift. Each
    panel is offset by its own minimum and shares one color scale across the
    requested temperatures. One figure per temperature. Returns the written
    filenames, empty unless exactly 2 free lattice DOF.

    The electronic term comes either from electronic_structures, which are
    integrated here, or ready-made as electronic_free_energies with one row
    per temperature of the result; without either, the F_el panel is left out.

    """
    fi = result.free_lattice_indices
    if len(fi) != 2:
        print(f"Skip component contours: {len(fi)} free lattice DOF (need 2).")
        return []

    free_points = result.lattice_lengths[:, fi]
    u_static = np.asarray(internal_energies, dtype="double")
    if electronic_free_energies is not None:
        fe_el_rel = np.asarray(electronic_free_energies, dtype="double")
    elif electronic_structures is not None:
        fe_el_rel, _ = compute_electronic_contributions_from_states(
            electronic_structures, result.temperatures
        )
    else:
        fe_el_rel = None

    axis = ("a", "b", "c")
    degree = result.surface_degree

    frames: list[dict[str, Any]] = []
    for t in temperatures:
        i = int(np.argmin(np.abs(result.temperatures - t)))
        total = result.helmholtz_lattice[i]
        f_el = fe_el_rel[i] if fe_el_rel is not None else np.zeros_like(u_static)
        f_ph = total - u_static - f_el
        panels = [("U (static)", u_static), ("F_ph", f_ph)]
        if fe_el_rel is not None:
            panels.append(("F_el", f_el))
        panels.append(("F total", total))
        frames.append({"i": i, "t": float(result.temperatures[i]), "panels": panels})

    n_panels = len(frames[0]["panels"])
    fitted = []
    panel_vmax = [0.0] * n_panels
    for fr in frames:
        row = []
        for p, (_, values) in enumerate(fr["panels"]):
            g0, g1, fe = _fit_and_grid(free_points, values, degree, n)
            fe = fe * _EV_TO_MEV
            row.append((g0, g1, fe))
            panel_vmax[p] = max(panel_vmax[p], float(fe.max()))
        fitted.append(row)
    panel_levels = [
        np.linspace(0.0, vmax if vmax > 0.0 else 1.0, 41) for vmax in panel_vmax
    ]

    written = []
    for fr, row in zip(frames, fitted, strict=True):
        eq = result.equilibrium_lattice_parameters[fr["i"]]
        fig, axes = plt.subplots(
            1, n_panels, figsize=(4.2 * n_panels, 4.0), squeeze=False
        )
        for ax, (name, _), (g0, g1, fe), levels in zip(
            axes[0], fr["panels"], row, panel_levels, strict=True
        ):
            filled = ax.contourf(g0, g1, fe, levels=levels, extend="max")
            ax.contour(g0, g1, fe, levels=levels[::2], colors="k", linewidths=0.3)
            fig.colorbar(filled, ax=ax, label=f"{name} - min (meV)")
            ax.plot(free_points[:, 0], free_points[:, 1], "wo", ms=2)
            ax.plot(eq[fi[0]], eq[fi[1]], "r*", ms=12)
            ax.set_xlabel(f"{axis[fi[0]]} (A)")
            ax.set_ylabel(f"{axis[fi[1]]} (A)")
            ax.set_title(name)
        fig.suptitle(f"Free energy decomposition at T = {fr['t']:.1f} K")
        fig.tight_layout()
        filename = f"F_decompose_{int(round(fr['t']))}K.png"
        fig.savefig(filename)
        plt.close(fig)
        written.append(filename)
    return written


def plot_anisotropic_qha(result: AnisotropicQHAResult) -> Any:
    """Three-panel QHA summary with a dual-scale lattice-parameter panel.

    Lattice parameters, V(T) and axial thermal expansion, but the leftmost
    panel puts a (and b, if it differs) on the left y-axis and c on the right
    y-axis, so the small a and c changes are both visible despite the large a-c
    offset. Returns the Figure.

    """
    t = result.temperatures
    lat = result.equilibrium_lattice_parameters
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))

    ax_a = axs[0]
    ax_c = ax_a.twinx()
    (la,) = ax_a.plot(t, lat[:, 0], color="C0", label="$a$")
    handles = [la]
    if not np.allclose(lat[:, 1], lat[:, 0]):
        (lb,) = ax_a.plot(t, lat[:, 1], color="C2", label="$b$")
        handles.append(lb)
    (lc,) = ax_c.plot(t, lat[:, 2], color="C1", label="$c$")
    handles.append(lc)
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_xlabel("Temperature (K)")
    ax_a.set_ylabel(r"$a$ $(\AA)$", color="C0")
    ax_c.set_ylabel(r"$c$ $(\AA)$", color="C1")
    ax_a.tick_params(axis="y", labelcolor="C0")
    ax_c.tick_params(axis="y", labelcolor="C1")
    ax_a.legend(handles, [h.get_label() for h in handles], loc="best")

    axs[1].plot(t, result.equilibrium_volumes, "r-")
    axs[1].set_xlim(t[0], t[-1])
    axs[1].set_xlabel("Temperature (K)")
    axs[1].set_ylabel(r"Volume $(\AA^3)$")
    axs[1].tick_params(axis="y", which="both", right=True, labelright=False)

    labels = (r"$\alpha_a$", r"$\alpha_b$", r"$\alpha_c$")
    for i, label in enumerate(labels):
        axs[2].plot(t, result.axial_thermal_expansions[:, i], label=label)
    axs[2].plot(t, result.thermal_expansion, "k--", label=r"$\beta$")
    axs[2].set_xlim(t[0], t[-1])
    axs[2].set_xlabel("Temperature (K)")
    axs[2].set_ylabel(r"Thermal expansion $(\mathrm{K}^{-1})$")
    axs[2].tick_params(axis="y", which="both", right=True, labelright=False)
    axs[2].axhline(0.0, color="0.6", lw=0.7, ls=":", zorder=0)
    axs[2].legend()

    fig.tight_layout()
    return fig
