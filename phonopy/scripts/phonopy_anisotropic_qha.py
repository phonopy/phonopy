# SPDX-License-Identifier: BSD-3-Clause
"""Run the anisotropic QHA from an intermediate dataset.

Reads aniso_qha_dataset.hdf5 (built by phonopy-anisotropic-qha-dataset), rebuilds one
Phonopy per grid point from the stored displacements and forces, runs
run_anisotropic_qha and writes the lattice parameters, axial thermal expansion
and volume versus temperature, plus optional free-energy surface diagnostics.
The dataset is read the same way whether the forces came from DFT or an MLP.

Usage::

    phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
        --contour-temp 0 500 1000 --compare-vinet

"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from phonopy import run_anisotropic_qha, run_qha
from phonopy.qha import anisotropic_output
from phonopy.qha.anisotropic import AnisotropicQHAResult, FreeEnergySurfaceFit
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.thermal import compute_electronic_contributions_from_states


def _evaluate_surface(result: AnisotropicQHAResult, temperature: float, n: int) -> dict:
    """Rebuild the fitted F surface at the nearest temperature and evaluate it.

    Returns the sample cells, the dense n x n evaluation mesh, and F offset by
    its own minimum (F - F_min) so that only the surface shape remains.

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
    vmax = max(float(d["fe"].max()) for d in data)
    levels = np.linspace(0.0, vmax, 41)

    axis = ("a", "b", "c")
    written = []
    for d in data:
        i = d["i"]
        fig, ax = plt.subplots()
        filled = ax.contourf(
            d["grid0"], d["grid1"], d["fe"], levels=levels, extend="max"
        )
        ax.contour(
            d["grid0"],
            d["grid1"],
            d["fe"],
            levels=levels[::2],
            colors="k",
            linewidths=0.4,
        )
        fig.colorbar(filled, label="F - F_min (eV)")

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
) -> list[str]:
    """Split the F(a, c) contour into its static, phonon and electronic parts.

    Draws U, F_ph, optionally F_el and the total on the same (a, c) domain so
    the valley shape can be attributed: U sets the static shape, while the
    near-linear F_ph (+ F_el) ramps carry the temperature-driven shift. Each
    panel is offset by its own minimum and shares one color scale across the
    requested temperatures. One figure per temperature. Returns the written
    filenames, empty unless exactly 2 free lattice DOF.

    """
    fi = result.free_lattice_indices
    if len(fi) != 2:
        print(f"Skip component contours: {len(fi)} free lattice DOF (need 2).")
        return []

    free_points = result.lattice_lengths[:, fi]
    u_static = np.asarray(internal_energies, dtype="double")
    if electronic_structures is not None:
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
            fig.colorbar(filled, ax=ax, label=f"{name} - min (eV)")
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


def plot_anisotropic_qha_dualscale(result: AnisotropicQHAResult) -> Any:
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


def main_diagonal_positions(result: AnisotropicQHAResult) -> NDArray[np.int64]:
    """Return positions of the main-diagonal cells of a tensor lattice grid.

    On a regular N x ... x N grid the main diagonal is the set of cells with the
    same rank along every free axis. These span the volume range monotonically
    with one shape per volume, so they form a clean 1D volume path a Vinet EOS
    can fit stably. Ordered by increasing volume proxy.

    """
    free = result.lattice_lengths[:, result.free_lattice_indices]
    ranks = np.empty(free.shape, dtype=int)
    for j in range(free.shape[1]):
        unique = np.unique(np.round(free[:, j], 6))
        ranks[:, j] = np.searchsorted(unique, np.round(free[:, j], 6))
    on_diagonal = np.all(ranks == ranks[:, :1], axis=1)
    positions = np.where(on_diagonal)[0]
    order = np.argsort(free[positions].prod(axis=1))
    return positions[order]


def compare_thermal_expansion_vinet(
    result: AnisotropicQHAResult,
    phonopys: Sequence,
    temperatures: NDArray[np.double],
    internal_energies: Sequence[float],
    electronic_structures: Sequence | None,
    mesh: float,
    positions: Sequence[int] | None = None,
    verbose: bool = False,
) -> None:
    """Compare thermal expansion: anisotropic 2D fit vs Vinet volume-path QHA.

    The Vinet path is run on a 1D subset (default the main diagonal). The
    difference in alpha_a vs alpha_c between the two methods is the anisotropy
    the fixed-shape volume path cannot capture. Writes
    thermal_expansion_compare.dat and .png and prints the max and mean absolute
    differences.

    """
    if positions is None:
        selected = list(main_diagonal_positions(result))
    else:
        selected = list(positions)
    if len(selected) < 5:
        print(
            f"Only {len(selected)} cells selected for the Vinet path; "
            f"run_qha needs at least 5. Skipping the comparison."
        )
        return

    sub_phonopys = [phonopys[k] for k in selected]
    sub_energies = [internal_energies[k] for k in selected]
    sub_electronic = (
        None
        if electronic_structures is None
        else [electronic_structures[k] for k in selected]
    )

    print(f"# Vinet volume path over {len(selected)} diagonal cells")
    for k in selected:
        a, b, c = result.lattice_lengths[k]
        print(f"  pos {k:3d}  a={a:.4f} c={c:.4f} c/a={c / a:.4f}")

    qha = run_qha(
        sub_phonopys,
        temperatures,
        internal_energies=sub_energies,
        electronic_structures=sub_electronic,
        mesh=mesh,
        eos="vinet",
        verbose=verbose,
    )

    t = result.temperatures
    beta_a = result.thermal_expansion
    alpha_a_a = result.axial_thermal_expansions[:, 0]
    alpha_c_a = result.axial_thermal_expansions[:, 2]

    beta_v = np.interp(t, qha.temperatures, qha.thermal_expansion)
    if qha.lattice is not None:
        axial_v = qha.lattice.axial_thermal_expansions
        alpha_a_v = np.interp(t, qha.temperatures, axial_v[:, 0])
        alpha_c_v = np.interp(t, qha.temperatures, axial_v[:, 2])
    else:
        print("Vinet QHA returned no lattice data; axial comparison skipped.")
        alpha_a_v = np.full_like(t, np.nan)
        alpha_c_v = np.full_like(t, np.nan)

    labels = ("beta (volumetric)", "alpha_a", "alpha_c")
    aniso = (beta_a, alpha_a_a, alpha_c_a)
    vinet = (beta_v, alpha_a_v, alpha_c_v)

    header = (
        "T(K)  beta_aniso  beta_vinet  alpha_a_aniso  alpha_a_vinet  "
        "alpha_c_aniso  alpha_c_vinet  (all 1/K)"
    )
    table = np.column_stack(
        [t, beta_a, beta_v, alpha_a_a, alpha_a_v, alpha_c_a, alpha_c_v]
    )
    np.savetxt("thermal_expansion_compare.dat", table, header=header)

    print("# Thermal expansion: anisotropic 2D fit vs Vinet volume-path QHA")
    for name, ya, yv in zip(labels, aniso, vinet, strict=True):
        diff = ya - yv
        print(
            f"  {name:18s} max|diff| = {np.nanmax(np.abs(diff)):.3e} /K, "
            f"mean|diff| = {np.nanmean(np.abs(diff)):.3e} /K"
        )

    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    for ax, name, ya, yv in zip(axes, labels, aniso, vinet, strict=True):
        ax.plot(t, ya * 1e6, "-", label="anisotropic 2D")
        ax.plot(t, yv * 1e6, "--", label="Vinet diagonal path")
        ax.set_ylabel(f"{name} (1e-6/K)")
        ax.legend()
    axes[-1].set_xlabel("Temperature (K)")
    fig.tight_layout()
    fig.savefig("thermal_expansion_compare.png")
    plt.close(fig)
    print("Wrote thermal_expansion_compare.dat and thermal_expansion_compare.png")


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description="Run the anisotropic QHA from an intermediate dataset."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="aniso_qha_dataset.hdf5",
        help="intermediate dataset (default: aniso_qha_dataset.hdf5)",
    )
    parser.add_argument("--tmax", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--mesh", type=float, default=100.0)
    parser.add_argument(
        "--fc-calculator",
        default="symfc",
        help="force-constant calculator (default: symfc)",
    )
    parser.add_argument(
        "--surface-degree",
        type=int,
        default=3,
        help="total degree of the F(a, c) surface polynomial (default: 3)",
    )
    parser.add_argument(
        "--electronic",
        action="store_true",
        help="add the electronic free energy F_el from the electronic states "
        "stored in the dataset (default: ignore them)",
    )
    parser.add_argument(
        "--contour-temp",
        type=float,
        nargs="*",
        help="temperatures (K) for F(a, c) contour maps (2 DOF only); default: tmax",
    )
    parser.add_argument(
        "--decompose-contours",
        action="store_true",
        help="also write U / F_ph / F_el / total contour panels",
    )
    parser.add_argument(
        "--compare-vinet",
        action="store_true",
        help="also run a Vinet volume-path QHA on the main diagonal and "
        "compare the thermal expansion",
    )
    parser.add_argument(
        "--vinet-index",
        type=int,
        nargs="*",
        help="grid indices for the Vinet volume path; default: main diagonal",
    )
    return parser.parse_args()


def run() -> None:
    """Run the phonopy-anisotropic-qha command."""
    args = get_options()

    dataset = read_aniso_qha_dataset(args.filename)
    indices = [point.index for point in dataset.grid_points]

    phonopys = []
    internal_energies = []
    electronic_structures: list | None = [] if args.electronic else None
    for point in dataset.grid_points:
        phonopys.append(point.to_phonopy(fc_calculator=args.fc_calculator))
        internal_energies.append(point.internal_energy)
        if electronic_structures is not None:
            if point.electronic_states is None:
                electronic_structures = None
            else:
                electronic_structures.append(point.electronic_states)
    if args.electronic and electronic_structures is None:
        print("  requested --electronic but the dataset has no electronic states")
    print(
        f"Loaded {len(phonopys)} grid point(s) from {args.filename} "
        f"(electronic F_el: {'on' if electronic_structures is not None else 'off'})"
    )

    temperatures = np.arange(0.0, args.tmax + args.dt, args.dt)
    result = run_anisotropic_qha(
        phonopys,
        temperatures,
        internal_energies=internal_energies,
        electronic_structures=electronic_structures,
        mesh=args.mesh,
        surface_degree=args.surface_degree,
        verbose=True,
    )

    anisotropic_output.write_lattice_parameters_temperature(result)
    anisotropic_output.write_axial_thermal_expansion(result)
    anisotropic_output.write_volume_temperature(result)
    fig = plot_anisotropic_qha_dualscale(result)
    fig.savefig("anisotropic_qha.png")
    plt.close(fig)
    print(
        "Wrote lattice_parameters-temperature.dat, axial_thermal_expansion.dat, "
        "volume-temperature.dat and anisotropic_qha.png"
    )

    contour_temps = args.contour_temp if args.contour_temp else [args.tmax]
    written = plot_F_contours(result, contour_temps)
    if written:
        print("Wrote " + ", ".join(written))

    if args.decompose_contours:
        written = plot_component_contours(
            result, internal_energies, electronic_structures, contour_temps
        )
        if written:
            print("Wrote " + ", ".join(written))

    if args.compare_vinet:
        positions = None
        if args.vinet_index:
            positions = [indices.index(i) for i in args.vinet_index]
        compare_thermal_expansion_vinet(
            result,
            phonopys,
            temperatures,
            internal_energies,
            electronic_structures,
            args.mesh,
            positions=positions,
            verbose=True,
        )


if __name__ == "__main__":
    run()
