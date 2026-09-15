# SPDX-License-Identifier: BSD-3-Clause
"""Plotting functions for anisotropic QHA results.

All functions take an AnisotropicQHAResult as the first argument. The
single-quantity plots return the matplotlib.pyplot module with the created
figure active; plot_anisotropic_qha returns the Figure, and the contour
functions write one file per temperature and return the names written. No
global rcParams are modified.

"""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from phonopy.qha.anisotropic import AnisotropicQHAResult, FreeEnergySurfaceFit
from phonopy.qha.thermal import compute_electronic_contributions_from_states

# Free energies are handled in eV throughout and converted only for plotting.
_EV_TO_MEV = 1000.0

# One hue, light to dark. A contour map of F - F_min carries a magnitude and
# nothing else, so a sequential map is the honest encoding; a multi-hue map
# spends hue on a quantity that has no categories and collides with anything
# drawn on top of it.
_SEQUENTIAL = "Purples"

# The marker for the located minimum. Warm against the cool surface, but not
# a saturated red: the point is to be found, not to shout.
_MINIMUM = "#eb6834"


# What the rest of phonopy sets before it draws (qha/core.py, the plot
# scripts): a serif family and Type 42 so the text stays text in a PDF. Set
# inside a context rather than on the global rcParams, so that a figure drawn
# from someone else's script does not silently change their style.
def _styled(func):
    """Draw inside phonopy's figure style, without leaking it.

    The style is applied for the call and restored afterwards, so a figure
    drawn from someone else's script does not silently change the style of
    the figures around it.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_STYLE):
            return func(*args, **kwargs)

    return wrapper


_STYLE = {
    "pdf.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "mathtext.cal": "serif:italic",
}


def _band_colours(n_bands: int) -> list:
    """Colours for the bands between contour levels, light to dark.

    Taken from a part of the map rather than all of it: the palest end is
    indistinguishable from the page, and the darkest hides the markers drawn
    on top.

    """
    return [plt.get_cmap(_SEQUENTIAL)(v) for v in np.linspace(0.06, 0.62, n_bands)]


def _write_surface_dat(
    filename: str,
    header: str,
    blocks: Sequence[tuple[str, FreeEnergySurfaceFit, NDArray[np.double]]],
    free_axis_lengths: NDArray[np.double],
    axis_names: tuple[str, str],
    minimum: NDArray[np.double] | None = None,
    reference: NDArray[np.double] | None = None,
) -> str:
    """Write the fits behind a contour figure, so they can be replotted.

    A contour map is a picture of a polynomial, and the polynomial is ten
    numbers. Writing those, with the non-dimensionalization they are defined
    against and the cells they were fitted to, lets anyone draw the same
    surface at their own resolution and in their own style, which a bitmap
    does not.

    One block per panel: the coefficients, then the sampled cells with the
    value that was fitted at each. Energies are in meV, offset by the minimum
    of the fit over the sampled box, as in the figure.

    """
    lines = [f"# {line}" for line in header.splitlines()]
    lines += [
        "#",
        "# The fitted polynomial of each block is",
        "#     P(x) = sum_p c_p u_1^p_1 u_2^p_2,   u = (x - origin) / scale",
        f"# with x = ({axis_names[0]}, {axis_names[1]}) in angstrom and P in meV.",
        "# 'offset' is what was subtracted so that the drawn minimum is zero.",
        "#",
    ]
    if reference is not None:
        lines.append("# strain reference, the origin of the figure axes, which is not")
        lines.append("# the same as the origin each fit is non-dimensionalized about:")
        lines.append(f"#     {reference[0]:.8f} {reference[1]:.8f}")
        lines.append("#")
    if minimum is not None:
        lines.append(f"# located minimum   {minimum[0]:.6f} {minimum[1]:.6f}")
        lines.append("#")

    for name, fit, values in blocks:
        offset = float(np.min(values))
        lines += [
            f"# --- {name} ---",
            f"# degree  {fit.degree}",
            f"# origin  {fit.origin[0]:.8f} {fit.origin[1]:.8f}",
            f"# scale   {fit.scale[0]:.8f} {fit.scale[1]:.8f}",
            f"# offset  {offset * _EV_TO_MEV:.8f}",
            f"# rms residual of the fit  {fit.rms_residual * _EV_TO_MEV:.6f} meV",
            "# coefficients:  p_1  p_2  c_p (meV)",
        ]
        for (p1, p2), c in zip(fit.exponents, fit.coefficients, strict=True):
            lines.append(f"  {int(p1):3d} {int(p2):3d}  {c * _EV_TO_MEV: .10e}")
        lines.append(
            f"# sampled cells:  {axis_names[0]} (A)  {axis_names[1]} (A)"
            "  value - offset (meV)"
        )
        for (x0, x1), value in zip(free_axis_lengths, values, strict=True):
            lines.append(f"  {x0:.8f} {x1:.8f}  {(value - offset) * _EV_TO_MEV: .10e}")
        lines.append("#")

    pathlib.Path(filename).write_text("\n".join(lines) + "\n")
    return filename


def _rounded_levels(vmax: float, count: int = 6) -> NDArray[np.double]:
    """Contour levels spaced geometrically and rounded to 1, 2 or 5.

    A free-energy valley is quadratic near its minimum, so levels spaced
    linearly crowd together at the edge of the box and leave the middle empty;
    geometric levels are about evenly spaced in distance from the minimum
    instead. Rounding each to the nearest 1, 2 or 5 times a power of ten is
    what makes them readable: 0.2, 0.5, 1, 2, 5 rather than 0.847, 1.694.

    """
    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.array([1.0])
    raw = vmax * 2.0 ** -np.arange(count - 1, -1, -1, dtype="double")
    nice = []
    for value in raw:
        power = 10.0 ** np.floor(np.log10(value))
        nice.append(
            power * min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(m * power - value))
        )
    # Rounding can collide; keep the distinct ones in order.
    return np.array(sorted(set(nice)))


def _strain(
    free_axis_lengths: NDArray[np.double],
    internal_energies: Sequence[float] | NDArray[np.double] | None = None,
    degree: int = 3,
) -> NDArray[np.double]:
    """Return the reference the strain axes are measured from.

    It is the lattice that minimizes the static energy U, when U is given. That
    point is a property of the electronic-structure calculation and of nothing
    else: the lattice located at T = 0 already carries the zero-point term and
    therefore moves when the vibrational model does, which makes it the worse
    origin to draw two calculations against.

    Without U, or if its minimum falls outside the sampled cells, the mean of
    the sampled cells is used instead. That is also the centre the surface
    polynomial is non-dimensionalized about.

    """
    lengths = np.asarray(free_axis_lengths, dtype="double")
    centre = lengths.mean(axis=0)
    if internal_energies is None:
        return centre
    try:
        fit = FreeEnergySurfaceFit(
            lengths, np.asarray(internal_energies, dtype="double"), degree=degree
        )
        minimum = np.asarray(fit.minimize(), dtype="double")
    except (RuntimeError, ValueError):
        return centre
    inside = np.all(minimum >= lengths.min(axis=0)) and np.all(
        minimum <= lengths.max(axis=0)
    )
    return minimum if inside else centre


def _window(
    free_axis_lengths: NDArray[np.double],
    reference: NDArray[np.double],
    margin: float,
) -> NDArray[np.double]:
    """Half-width of the drawn window along each free lattice DOF.

    Half the range of the sampled cells, times margin: with margin = 1.5 the
    window is one and a half times as wide as the sampled cells are. The size
    is fixed by the grid alone and not by where the reference sits, so two
    calculations drawn with the same margin come out at the same scale, which
    is the whole point of drawing them in strain. A reference far from the
    centre of the grid can therefore push some sampled cells outside the
    window; the surface near the minimum is what the figure is for.

    A margin above one draws the fitted polynomial outside the cells it was
    fitted to, where nothing constrains it. The sampled cells are drawn on top
    so that it is plain how far the surface is being taken on trust.

    """
    lengths = np.asarray(free_axis_lengths, dtype="double")
    _ = reference  # the size does not depend on it, only the centring does
    return 0.5 * margin * (lengths.max(axis=0) - lengths.min(axis=0))


def _as_strain(values: NDArray[np.double], reference: float) -> NDArray[np.double]:
    """Lengths as a percentage strain from the reference."""
    return (np.asarray(values, dtype="double") / reference - 1.0) * 100.0


def plot_lattice_parameters(result: AnisotropicQHAResult) -> Any:
    """Return pyplot of equilibrium lattice parameters vs temperature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _draw_lattice_parameters(ax, result)
    return plt


def plot_lattice_smoothing(result: AnisotropicQHAResult) -> Any:
    """Return a Figure showing the smoothing against the minima it was fitted to.

    One column per free lattice DOF, since those are the lengths the surface
    minimization moves. The upper row is the fit as a line over the minima as
    dots, and the lower row is what is left over, fit minus minimum, in
    1e-4 angstrom. The residuals are what the fit is judged on: sampling
    scatter shows as a band around zero of the width the scatter has, and a
    fit of the wrong shape shows as an excursion over a range of temperature.

    Raises ValueError for a result that was not smoothed, which has no minima
    of its own to compare against.

    """
    import matplotlib.pyplot as plt

    smoothing_fit = result.lattice_smoothing_fit
    if smoothing_fit is None:
        raise ValueError(
            "The result was not smoothed, so its lattice parameters are the "
            "surface minima themselves and there is nothing to compare."
        )
    unsmoothed = result.unsmoothed_lattice_parameters

    t = result.temperatures
    columns = [int(i) for i in result.lattice_grid.free_axis_indices]
    names = ("a", "b", "c")
    fig, axs = plt.subplots(
        2, len(columns), figsize=(4.0 * len(columns), 6.0), sharex=True, squeeze=False
    )
    for k, i in enumerate(columns):
        name = names[i]
        fitted = result.equilibrium_lattice_parameters[:, i]
        ax = axs[0][k]
        ax.plot(t, unsmoothed[:, i], ".", color="0.4", label="surface minima")
        ax.plot(t, fitted, "-", color=f"C{k}")
        ax.set_ylabel(rf"${name}$ $(\AA)$")
        ax.set_title(
            f"{smoothing_fit.method}, {smoothing_fit.n_terms} terms" if k == 0 else ""
        )
        ax.legend(loc="best")

        residual = (fitted - unsmoothed[:, i]) * 1e4
        ax = axs[1][k]
        ax.plot(t, residual, ".-", color=f"C{k}")
        ax.axhline(0.0, color="0.6", lw=0.7, ls=":", zorder=0)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(rf"${name}$ fit $-$ minimum $(10^{{-4}} \AA)$")

    fig.tight_layout()
    return fig


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


def _evaluate_surface(
    result: AnisotropicQHAResult,
    temperature: float,
    n: int,
    lo: NDArray[np.double] | None = None,
    hi: NDArray[np.double] | None = None,
) -> dict:
    """Rebuild the fitted F surface at the nearest temperature and evaluate it.

    Returns the sample cells, the dense n x n evaluation mesh, and F offset by
    its own minimum (F - F_min) in eV, so that only the surface shape remains.

    """
    fi = result.lattice_grid.free_axis_indices
    i = int(np.argmin(np.abs(result.temperatures - temperature)))
    free_axis_lengths = result.lattice_grid.lattice_lengths[:, fi]
    fit = FreeEnergySurfaceFit(
        free_axis_lengths, result.helmholtz_lattice[i], degree=result.polynomial_degree
    )

    lo0, lo1 = free_axis_lengths.min(axis=0) if lo is None else lo
    hi0, hi1 = free_axis_lengths.max(axis=0) if hi is None else hi
    grid0, grid1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    mesh = np.column_stack([grid0.ravel(), grid1.ravel()])
    fe = fit.evaluate(mesh).reshape(grid0.shape)
    fe = fe - fe.min()
    return {
        "i": i,
        "t": float(result.temperatures[i]),
        "fit": fit,
        "values": np.asarray(result.helmholtz_lattice[i], dtype="double"),
        "free_axis_lengths": free_axis_lengths,
        "grid0": grid0,
        "grid1": grid1,
        "fe": fe,
    }


@_styled
def plot_F_contours(
    result: AnisotropicQHAResult,
    temperatures: Sequence[float],
    n: int = 200,
    image_format: str = "png",
    internal_energies: Sequence[float] | NDArray[np.double] | None = None,
    margin: float = 1.5,
) -> list[str]:
    """Save contour maps of F - F_min over the 2 free lattice DOF.

    One map per requested temperature (snapped to the nearest computed
    temperature), all sharing one color scale so valley depth and curvature are
    comparable. Overlays the sample cells and the located minimum. Returns the
    written filenames, empty unless there are exactly 2 free lattice DOF.

    The axes are the strain from a fixed reference rather than the lattice
    parameters themselves, which is what lets two calculations be put side by
    side. The reference is the minimum of ``internal_energies`` when they are
    given and the centre of the sampled cells otherwise; see _strain.
    ``margin`` sets how far past the sampled cells the window reaches, as a
    multiple of the distance to the furthest of them, so that the located
    minimum is visible even when it sits near the edge. ``image_format`` is
    the extension to write, png or pdf.

    """
    fi = result.lattice_grid.free_axis_indices
    if len(fi) != 2:
        print(f"Skip contour map: {len(fi)} free lattice DOF (need 2).")
        return []

    free_axis_lengths = result.lattice_grid.lattice_lengths[:, fi]
    reference = _strain(free_axis_lengths, internal_energies, result.polynomial_degree)
    half = _window(free_axis_lengths, reference, margin)
    data = [
        _evaluate_surface(result, t, n, reference - half, reference + half)
        for t in temperatures
    ]
    vmax = max(float(d["fe"].max()) for d in data) * _EV_TO_MEV
    levels = _rounded_levels(vmax)

    axis = ("a", "b", "c")
    written = []
    for d in data:
        i = d["i"]
        fe = d["fe"] * _EV_TO_MEV
        grid0 = _as_strain(d["grid0"], reference[0])
        grid1 = _as_strain(d["grid1"], reference[1])
        fig, ax = plt.subplots()
        ax.contourf(
            grid0,
            grid1,
            fe,
            levels=levels,
            colors=_band_colours(len(levels) + 1),
            extend="both",
        )
        # The levels are labelled on the contours themselves. A colorbar would
        # say the same thing in a fifth of the width, and with geometric levels
        # it reads badly whichever way it is spaced.
        lines = ax.contour(grid0, grid1, fe, levels=levels, colors="w", linewidths=0.6)
        ax.clabel(lines, fmt="%g", fontsize=7, inline_spacing=2, colors="k")

        # A cross is quieter than a filled marker and does not hide the
        # surface under it.
        ax.plot(
            _as_strain(d["free_axis_lengths"][:, 0], reference[0]),
            _as_strain(d["free_axis_lengths"][:, 1], reference[1]),
            ls="none",
            marker="+",
            ms=4.5,
            mew=0.7,
            color="k",
            alpha=0.6,
            label="samples",
        )
        eq = result.equilibrium_lattice_parameters[i]
        extrapolated = bool(result.minimum_extrapolated[i])
        # A circle, not a star: what matters is where its centre is. Open when
        # the minimum falls outside the sampled cells and is therefore read
        # off the fit where nothing constrains it.
        ax.plot(
            _as_strain(eq[fi[0]], reference[0]),
            _as_strain(eq[fi[1]], reference[1]),
            ls="none",
            marker="o",
            ms=7,
            mfc="none" if extrapolated else _MINIMUM,
            mec=_MINIMUM,
            mew=1.4,
            label="minimum (extrapolated)" if extrapolated else "minimum",
        )

        first, second = axis[fi[0]], axis[fi[1]]
        ax.set_xlabel(
            f"$({first} - {first}_0)/{first}_0$ (%),"
            f"  ${first}_0$ = {reference[0]:.4f} $\\AA$"
        )
        ax.set_ylabel(
            f"$({second} - {second}_0)/{second}_0$ (%),"
            f"  ${second}_0$ = {reference[1]:.4f} $\\AA$"
        )
        ax.set_title(f"F - F_min (meV) at T = {d['t']:.1f} K")
        ax.legend()

        filename = f"F_contour_{int(round(d['t']))}K.{image_format}"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        written.append(filename)

        written.append(
            _write_surface_dat(
                f"F_contour_{int(round(d['t']))}K.dat",
                f"Free energy surface at T = {d['t']:.1f} K",
                [("F - F_min", d["fit"], d["values"])],
                d["free_axis_lengths"],
                (first, second),
                minimum=np.array([eq[fi[0]], eq[fi[1]]]),
                reference=reference,
            )
        )
    return written


def _fit_and_grid(
    free_axis_lengths: NDArray[np.double],
    values: NDArray[np.double],
    degree: int,
    n: int,
    lo: NDArray[np.double] | None = None,
    hi: NDArray[np.double] | None = None,
) -> tuple[
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    FreeEnergySurfaceFit,
]:
    """Fit a total-degree polynomial to values and evaluate it on a mesh.

    Returns (grid0, grid1, fe, fit) with fe offset by its own minimum, so only
    the surface shape and tilt remain (any additive constant drops out). The
    fit itself comes back so that it can be written out beside the figure.

    """
    fit = FreeEnergySurfaceFit(free_axis_lengths, values, degree=degree)
    lo0, lo1 = free_axis_lengths.min(axis=0) if lo is None else lo
    hi0, hi1 = free_axis_lengths.max(axis=0) if hi is None else hi
    grid0, grid1 = np.meshgrid(np.linspace(lo0, hi0, n), np.linspace(lo1, hi1, n))
    mesh = np.column_stack([grid0.ravel(), grid1.ravel()])
    fe = fit.evaluate(mesh).reshape(grid0.shape)
    return grid0, grid1, fe - fe.min(), fit


@_styled
def plot_component_contours(
    result: AnisotropicQHAResult,
    internal_energies: Sequence[float] | NDArray[np.double],
    electronic_structures: Sequence | None,
    temperatures: Sequence[float],
    n: int = 200,
    electronic_free_energies: (
        Sequence[Sequence[float]] | NDArray[np.double] | None
    ) = None,
    image_format: str = "png",
    margin: float = 1.5,
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
    fi = result.lattice_grid.free_axis_indices
    if len(fi) != 2:
        print(f"Skip component contours: {len(fi)} free lattice DOF (need 2).")
        return []

    free_axis_lengths = result.lattice_grid.lattice_lengths[:, fi]
    u_static = np.asarray(internal_energies, dtype="double")
    if electronic_free_energies is not None:
        fe_el_rel = np.asarray(electronic_free_energies, dtype="double")
    elif electronic_structures is not None:
        fe_el_rel, _ = compute_electronic_contributions_from_states(
            electronic_structures,
            result.temperatures,
            primitive_volumes=result.lattice_grid.primitive_volumes,
        )
    else:
        fe_el_rel = None

    axis = ("a", "b", "c")
    degree = result.polynomial_degree
    reference = _strain(free_axis_lengths, u_static, degree)
    half = _window(free_axis_lengths, reference, margin)

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
            g0, g1, fe, fit = _fit_and_grid(
                free_axis_lengths,
                values,
                degree,
                n,
                reference - half,
                reference + half,
            )
            fe = fe * _EV_TO_MEV
            row.append((g0, g1, fe, fit))
            panel_vmax[p] = max(panel_vmax[p], float(fe.max()))
        fitted.append(row)
    # Each panel keeps its own scale: F_el is often a hundred times smaller
    # than U, and one scale over all of them would leave it blank.
    panel_levels = [_rounded_levels(vmax) for vmax in panel_vmax]
    samples = (
        _as_strain(free_axis_lengths[:, 0], reference[0]),
        _as_strain(free_axis_lengths[:, 1], reference[1]),
    )
    first, second = axis[fi[0]], axis[fi[1]]

    written = []
    for fr, row in zip(frames, fitted, strict=True):
        eq = result.equilibrium_lattice_parameters[fr["i"]]
        fig, axes = plt.subplots(
            1, n_panels, figsize=(3.1 * n_panels, 3.4), squeeze=False
        )
        for ax, (name, _), (g0, g1, fe, _fit), levels in zip(
            axes[0], fr["panels"], row, panel_levels, strict=True
        ):
            g0 = _as_strain(g0, reference[0])
            g1 = _as_strain(g1, reference[1])
            ax.contourf(
                g0,
                g1,
                fe,
                levels=levels,
                colors=_band_colours(len(levels) + 1),
                extend="both",
            )
            lines = ax.contour(g0, g1, fe, levels=levels, colors="w", linewidths=0.5)
            ax.clabel(lines, fmt="%g", fontsize=6, inline_spacing=2, colors="k")
            ax.plot(
                *samples, ls="none", marker="+", ms=4, mew=0.6, color="k", alpha=0.6
            )
            ax.plot(
                _as_strain(eq[fi[0]], reference[0]),
                _as_strain(eq[fi[1]], reference[1]),
                ls="none",
                marker="o",
                ms=6,
                mfc=_MINIMUM,
                mec=_MINIMUM,
                mew=1.2,
            )
            ax.set_xlabel(f"$({first} - {first}_0)/{first}_0$ (%)")
            ax.set_ylabel(f"$({second} - {second}_0)/{second}_0$ (%)")
            ax.set_title(f"{name} - min (meV)")
        fig.suptitle(
            f"Free energy decomposition at T = {fr['t']:.1f} K"
            f"   (${first}_0$ = {reference[0]:.4f}, "
            f"${second}_0$ = {reference[1]:.4f} $\\AA$)"
        )
        fig.tight_layout()
        filename = f"F_decompose_{int(round(fr['t']))}K.{image_format}"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        written.append(filename)

        written.append(
            _write_surface_dat(
                f"F_decompose_{int(round(fr['t']))}K.dat",
                f"Free energy decomposition at T = {fr['t']:.1f} K",
                [
                    (name, fit, np.asarray(values, dtype="double"))
                    for (name, values), (_, _, _, fit) in zip(
                        fr["panels"], row, strict=True
                    )
                ],
                free_axis_lengths,
                (first, second),
                minimum=np.array([eq[fi[0]], eq[fi[1]]]),
                reference=reference,
            )
        )
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
