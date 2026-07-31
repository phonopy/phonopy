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

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from phonopy import run_anisotropic_qha, run_qha
from phonopy.qha import anisotropic_output, anisotropic_plot
from phonopy.qha.anisotropic import AnisotropicQHAResult
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset

# Free energies are handled in eV throughout and converted only for plotting.
_EV_TO_MEV = 1000.0


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

    # savetxt comments every header line, so the second needs no marker here.
    header = (
        anisotropic_output.format_provenance(result)
        + "\nT(K)  beta_aniso  beta_vinet  alpha_a_aniso  alpha_a_vinet  "
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
    parser.add_argument(
        "--mesh",
        type=float,
        default=200.0,
        help="phonon sampling mesh (default: 200). The axial split needs a "
        "denser mesh than the volumetric expansion: 100 leaves alpha_c off by "
        "~20% while beta is already converged",
    )
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
    if any(point.dataset is None for point in dataset.grid_points):
        raise SystemExit(
            f"{args.filename} carries no displacements or forces, so the "
            f"phonon free energy cannot be computed from it.\n"
            f"Such a dataset is built from the static grid alone (no --phonon "
            f"given to phonopy-anisotropic-qha-dataset) and is meant for a "
            f"method whose force constants depend on temperature. Compute the "
            f"free energies with that method and pass them to "
            f"run_anisotropic_qha through its phonon_free_energies argument; "
            f"see the anisotropic QHA documentation."
        )
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

    provenance = [
        f"dataset={args.filename}",
        f"fc_calculator={args.fc_calculator}",
    ]
    anisotropic_output.write_lattice_parameters_temperature(
        result, provenance=provenance
    )
    anisotropic_output.write_axial_thermal_expansion(result, provenance=provenance)
    anisotropic_output.write_volume_temperature(result, provenance=provenance)
    fig = anisotropic_plot.plot_anisotropic_qha(result)
    fig.savefig("anisotropic_qha.png")
    plt.close(fig)
    print(
        "Wrote lattice_parameters-temperature.dat, axial_thermal_expansion.dat, "
        "volume-temperature.dat and anisotropic_qha.png"
    )

    contour_temps = args.contour_temp if args.contour_temp else [args.tmax]
    written = anisotropic_plot.plot_F_contours(result, contour_temps)
    if written:
        print("Wrote " + ", ".join(written))

    if args.decompose_contours:
        written = anisotropic_plot.plot_component_contours(
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
