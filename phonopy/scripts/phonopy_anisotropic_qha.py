# SPDX-License-Identifier: BSD-3-Clause
"""Run the anisotropic QHA from an intermediate dataset.

Reads aniso_qha_dataset.hdf5 (built by phonopy-anisotropic-qha-dataset),
rebuilds one Phonopy per grid point from the stored displacements and forces,
runs run_anisotropic_qha and writes the lattice parameters, axial thermal
expansion and volume versus temperature, plus optional free-energy surface
diagnostics. The dataset is read the same way whether the forces came from DFT
or an MLP.

Usage::

    phonopy-anisotropic-qha aniso_qha_dataset.hdf5 --tmax 1000 --dt 10 \
        --contour-temp 0 500 1000 --compare-eos

"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from phonopy import Phonopy, run_anisotropic_qha, run_qha
from phonopy.qha import anisotropic_output, anisotropic_plot
from phonopy.qha.anisotropic import AnisotropicQHAResult
from phonopy.qha.anisotropic_dataset import read_aniso_qha_dataset
from phonopy.qha.free_energy_io import (
    FreeEnergyKind,
    check_free_energies,
    read_free_energies_hdf5,
    write_free_energies_hdf5,
)
from phonopy.qha.thermal import compute_electronic_contributions_from_states


def main_diagonal_positions(grid_shape: Sequence[int]) -> NDArray[np.int64]:
    """Return the positions of the main-diagonal cells of a tensor grid.

    The main diagonal is the set of cells with the same index along every free
    axis. These span the volume range monotonically with one shape per volume,
    so they form a clean 1D volume path a Vinet EOS can fit stably.

    The grid points are stored in row-major order over the free DOF, so the
    position of the cell (k, k, ...) is k times the sum of the axis strides.
    A grid whose axes have unequal counts has as many diagonal cells as its
    shortest axis.

    """
    strides = [int(np.prod(grid_shape[j + 1 :])) for j in range(len(grid_shape))]
    return np.arange(min(grid_shape), dtype="int64") * sum(strides)


def suggest_eos_cells(result: AnisotropicQHAResult, indices: Sequence[int]) -> None:
    """Print the cells of the grid, and a volume path to compare along.

    Called when the dataset records no grid shape and the user named no cells,
    so that --eos-index can be filled in from what is actually there. The
    cells are listed by volume, and cells that share a c/a are pointed out:
    those have one shape, which is what a volume-path EOS fit assumes. A grid
    sampled over equal fractional ranges has its main diagonal among them.

    """
    lengths = result.lattice_lengths
    order = np.argsort(lengths.prod(axis=1))
    ratios = np.round(lengths[:, 2] / lengths[:, 0], 4)

    print("# Cells by volume. --eos-index takes the indices of this column.")
    print(f"  {'index':>6} {'a':>9} {'c':>9} {'c/a':>8}")
    for k in order:
        a, _, c = lengths[k]
        # Grid points are numbered from 1 here and on the command line; the
        # stored index is 0-origin.
        print(f"  {indices[k] + 1:6d} {a:9.4f} {c:9.4f} {ratios[k]:8.4f}")

    # The largest set of one shape, in volume order. Ties keep the first.
    values, counts = np.unique(ratios[order], return_counts=True)
    best = values[np.argmax(counts)]
    same_shape = [indices[k] + 1 for k in order if ratios[k] == best]
    if len(same_shape) >= 5:
        print(
            f"# {len(same_shape)} cells share c/a = {best:.4f}, "
            f"a constant-shape volume path:"
        )
        print("#   --eos-index " + " ".join(str(i) for i in same_shape))
    else:
        print("# No five cells share a c/a, so no constant-shape path is")
        print("# available. Naming cells of varying shape still runs, but the")
        print("# comparison is then between two different paths.")


def _read_free_energies(
    filename: str | None,
    kind: FreeEnergyKind,
    temperatures: NDArray[np.double],
    lattice_lengths: NDArray[np.double],
) -> NDArray[np.double] | None:
    """Read a ready-made free energy and check it against this run.

    The file is typically computed on another machine, so nothing ties it to
    the dataset read here: check_free_energies compares its kind, temperature
    grid and grid points before the values are used.

    """
    if filename is None:
        return None

    values = check_free_energies(
        read_free_energies_hdf5(filename),
        kind,
        temperatures,
        lattice_lengths,
        filename=filename,
    )
    print(f"{kind.capitalize()} free energy read from {filename}")
    return values


def compare_thermal_expansion_eos(
    result: AnisotropicQHAResult,
    phonopys: Sequence,
    temperatures: NDArray[np.double],
    internal_energies: Sequence[float],
    electronic_structures: Sequence | None,
    mesh: float,
    positions: Sequence[int],
    verbose: bool = False,
) -> None:
    """Compare thermal expansion: anisotropic 2D fit vs Vinet volume-path QHA.

    The Vinet path is run on the 1D subset of cells given by ``positions``,
    which the caller picks: the main diagonal of the grid, or a set named by
    hand with --eos-index. The
    difference in alpha_a vs alpha_c between the two methods is the anisotropy
    the fixed-shape volume path cannot capture. Writes
    thermal_expansion_compare.dat and .png and prints the max and mean absolute
    differences.

    """
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
    # run_qha withholds lattice data for triclinic and monoclinic crystals,
    # whose cell angles may depend on volume. The dataset this script reads is
    # built by phonopy-anisotropic-qha-dataset, which rejects those crystal
    # systems outright, so the comparison never meets one. When angle degrees
    # of freedom are supported, this is where to decide what the axial
    # comparison against a volume path should mean for them.
    assert qha.lattice is not None
    axial_v = qha.lattice.axial_thermal_expansions
    alpha_a_v = np.interp(t, qha.temperatures, axial_v[:, 0])
    alpha_c_v = np.interp(t, qha.temperatures, axial_v[:, 2])

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
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="highest temperature in K (default: 1000, or the grid of "
        "--phonon-free-energies when neither --tmax nor --dt is given)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="temperature step in K (default: 10, or the grid of "
        "--phonon-free-energies when neither --tmax nor --dt is given)",
    )
    parser.add_argument(
        "--mesh",
        type=float,
        default=200.0,
        # A literal percent has to be escaped: argparse expands help strings
        # with the % operator, and Python 3.14 validates that at parser
        # construction rather than only when --help is formatted.
        help="phonon sampling mesh (default: 200). The axial split needs a "
        "denser mesh than the volumetric expansion: 100 leaves alpha_c off by "
        "~20%% while beta is already converged",
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
        "--no-electronic",
        dest="electronic",
        action="store_false",
        default=True,
        help="leave out the electronic free energy F_el, which is otherwise "
        "integrated from the electronic states stored in the dataset",
    )
    parser.add_argument(
        "--electronic-window",
        type=float,
        default=None,
        metavar="EV",
        help="half-width of the energy window F_el is integrated over "
        "(default: 12 k_B T of the highest temperature, at least 0.5 eV)",
    )
    parser.add_argument(
        "--electronic-spacing",
        type=float,
        default=0.0005,
        metavar="EV",
        help="spacing of the energy grid inside that window (default: 0.0005)",
    )
    parser.add_argument(
        "--electronic-free-energies",
        metavar="FILE",
        help="add F_el(T) - F_el(0) read from an hdf5 written by "
        "write_free_energies_hdf5, instead of integrating the stored "
        "electronic states here",
    )
    parser.add_argument(
        "--smooth-lattice",
        default=None,
        choices=("none", "einstein"),
        help="smooth a(T), b(T), c(T) along temperature before differentiating "
        "them. Default: einstein with --phonon-free-energies, whose scatter "
        "the differences amplify, and none otherwise",
    )
    parser.add_argument(
        "--smooth-terms",
        type=int,
        default=2,
        metavar="N",
        help="number of Einstein terms --smooth-lattice fits, at least 2 (default: 2)",
    )
    parser.add_argument(
        "--phonon-free-energies",
        metavar="FILE",
        help="take F_ph(T) from an hdf5 written by write_free_energies_hdf5 "
        "instead of computing it from the stored force constants; this is "
        "the way in for a method whose force constants depend on temperature",
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
        "--compare-eos",
        action="store_true",
        help="also run a Vinet volume-path QHA on the main diagonal and "
        "compare the thermal expansion",
    )
    parser.add_argument(
        "--eos-index",
        type=int,
        nargs="*",
        help="grid points for the Vinet volume path, numbered from 1; "
        "default: main diagonal",
    )
    return parser.parse_args()


def run() -> None:
    """Run the phonopy-anisotropic-qha command."""
    args = get_options()

    dataset = read_aniso_qha_dataset(args.filename)
    if (
        any(point.dataset is None for point in dataset.grid_points)
        and not args.phonon_free_energies
    ):
        raise SystemExit(
            f"{args.filename} carries no displacements or forces, so the "
            f"phonon free energy cannot be computed from it.\n"
            f"Such a dataset is built from the static grid alone (no --phonon "
            f"given to phonopy-anisotropic-qha-dataset) and is meant for a "
            f"method whose force constants depend on temperature. Compute the "
            f"free energies with that method and give them with "
            f"--phonon-free-energies; see the anisotropic QHA documentation."
        )
    indices = [point.index for point in dataset.grid_points]

    # Read before the force constants are built, so a mismatched file is
    # reported in a second rather than after the solver has run.
    if args.tmax is None and args.dt is None and args.phonon_free_energies:
        # The file carries the grid it was computed on, so asking for it again
        # on the command line would only be a way of getting it wrong.
        temperatures = read_free_energies_hdf5(args.phonon_free_energies).temperatures
        print(
            f"Temperature grid taken from {args.phonon_free_energies}: "
            f"{len(temperatures)} points, {temperatures[0]} to "
            f"{temperatures[-1]} K"
        )
    else:
        tmax = 1000.0 if args.tmax is None else args.tmax
        dt = 10.0 if args.dt is None else args.dt
        temperatures = np.arange(0.0, tmax + dt, dt)
    lattice_lengths = np.array(
        [np.linalg.norm(point.cell.cell, axis=1) for point in dataset.grid_points],
        dtype="double",
    )
    electronic_free_energies = _read_free_energies(
        args.electronic_free_energies, "electronic", temperatures, lattice_lengths
    )
    phonon_free_energies = _read_free_energies(
        args.phonon_free_energies, "phonon", temperatures, lattice_lengths
    )
    phonopys = []
    internal_energies = []
    read_states = args.electronic and not args.electronic_free_energies
    electronic_structures: list | None = [] if read_states else None
    for point in dataset.grid_points:
        if phonon_free_energies is None:
            phonopys.append(point.to_phonopy(fc_calculator=args.fc_calculator))
        else:
            # The free energies replace the mesh sampling, so the force
            # constants are never read and need not be built.
            phonopys.append(
                Phonopy(
                    point.cell,
                    supercell_matrix=point.supercell_matrix,
                    primitive_matrix=point.primitive_matrix,
                    log_level=0,
                )
            )
        internal_energies.append(point.internal_energy)
        if electronic_structures is not None:
            if point.electronic_states is None:
                electronic_structures = None
            else:
                electronic_structures.append(point.electronic_states)
    if read_states and electronic_structures is None:
        print("  the dataset has no electronic states, so F_el is left out")
    with_electronic = (
        electronic_structures is not None or args.electronic_free_energies is not None
    )
    print(
        f"Loaded {len(phonopys)} grid point(s) from {args.filename} "
        f"(electronic F_el: {'on' if with_electronic else 'off'})"
    )

    if electronic_structures is not None:
        electronic_free_energies, _ = compute_electronic_contributions_from_states(
            electronic_structures,
            temperatures,
            window=args.electronic_window,
            energy_spacing=args.electronic_spacing,
            require_tetrahedron=True,
        )
        write_free_energies_hdf5(
            temperatures,
            electronic_free_energies,
            "fel.hdf5",
            kind="electronic",
            lattice_lengths=lattice_lengths,
        )
        print("Wrote fel.hdf5, which --electronic-free-energies takes back")

    result = run_anisotropic_qha(
        phonopys,
        temperatures,
        internal_energies=internal_energies,
        electronic_free_energies=electronic_free_energies,
        phonon_free_energies=phonon_free_energies,
        mesh=args.mesh,
        surface_degree=args.surface_degree,
        lattice_smoothing=args.smooth_lattice,
        smoothing_terms=args.smooth_terms,
        verbose=True,
    )
    if args.smooth_lattice is None and result.lattice_smoothing != "none":
        print(
            f"Lattice parameters smoothed along temperature "
            f"(--smooth-lattice {result.lattice_smoothing})."
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

    # The highest temperature of the run, which --tmax need not have set.
    contour_temps = (
        args.contour_temp if args.contour_temp else [float(temperatures[-1])]
    )
    written = anisotropic_plot.plot_F_contours(result, contour_temps)
    if written:
        print("Wrote " + ", ".join(written))

    if args.decompose_contours:
        written = anisotropic_plot.plot_component_contours(
            result,
            internal_energies,
            None,  # F_el is passed as free energies below, states or not.
            contour_temps,
            electronic_free_energies=(
                None
                if electronic_free_energies is None
                else electronic_free_energies[: len(result.temperatures)]
            ),
        )
        if written:
            print("Wrote " + ", ".join(written))

    if args.compare_eos and phonon_free_energies is not None:
        print("Skip the EOS cross-check: the volume-path driver computes")
        print("F_ph from force constants, which this run does not have.")
    elif args.compare_eos and args.electronic_free_energies:
        print("Skip the EOS cross-check: the volume-path driver takes the")
        print("electronic term as states, and F_el was given ready-made, so")
        print("the two paths would not carry the same physics. Run with")
        print("--electronic instead to compare them.")
    elif args.compare_eos:
        positions: list[int] | None = None
        if args.eos_index:
            positions = [indices.index(i - 1) for i in args.eos_index]
        elif dataset.grid_shape is not None:
            positions = list(main_diagonal_positions(dataset.grid_shape))
        else:
            print("The dataset records no grid shape, so it has no main")
            print("diagonal to take: its cells were either sampled randomly,")
            print("or gathered in an order the builder could not read as a")
            print("grid. Name the cells of a volume path with --eos-index.")
            suggest_eos_cells(result, indices)
        if positions is not None:
            compare_thermal_expansion_eos(
                result,
                phonopys,
                temperatures,
                internal_energies,
                electronic_structures,
                args.mesh,
                positions,
                verbose=True,
            )


if __name__ == "__main__":
    run()
