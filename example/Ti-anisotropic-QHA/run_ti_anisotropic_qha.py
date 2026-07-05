"""Run the anisotropic QHA for HCP Ti from a trained pypolymlp.

Builds a regular grid of unit cells over the free lattice DOF (a, c for
hexagonal Ti) around the equilibrium cell, computes force constants for each
grid cell from the machine-learning potential (polymlp.yaml) via random
displacements + symfc, and runs run_anisotropic_qha to obtain a(T), c(T) and
the axial thermal expansions alpha_a, alpha_c.

The equilibrium cell, supercell matrix and primitive matrix are read from the
input phonopy(_disp).yaml so every grid cell shares the same conventions.
HCP Ti has no free internal coordinate, so no per-cell relaxation is needed.

Usage::

    python run_ti_anisotropic_qha.py phonopy_disp.yaml --mlp polymlp.yaml

"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

import phonopy
from phonopy import Phonopy, run_anisotropic_qha
from phonopy.qha import anisotropic_output, anisotropic_plot
from phonopy.qha.lattice_sampling import LatticeDOF, get_free_lattice_dof
from phonopy.structure.atoms import PhonopyAtoms


def build_lattice_grid(
    cell: PhonopyAtoms,
    dof: LatticeDOF,
    ranges: dict[str, tuple[float, float]],
    counts: dict[str, int],
) -> list[PhonopyAtoms]:
    """Return unit cells on a regular grid over the free lattice DOF.

    Each free-DOF label is swept on a linspace over its (min, max) range,
    and the grid is the Cartesian product of these axes. Scaling the lattice
    vectors preserves cell angles and fractional atomic positions.

    Parameters
    ----------
    cell : PhonopyAtoms
        Equilibrium unit cell.
    dof : LatticeDOF
        Free lattice DOF from get_free_lattice_dof.
    ranges : dict
        (min, max) length range per free-DOF label.
    counts : dict
        Number of grid points per free-DOF label.

    Returns
    -------
    list of PhonopyAtoms

    """
    axes = {
        label: np.linspace(ranges[label][0], ranges[label][1], counts[label])
        for label in dof.labels
    }
    cells = []
    for combo in itertools.product(*(axes[label] for label in dof.labels)):
        lattice = cell.cell.copy()
        for label, target in zip(dof.labels, combo, strict=True):
            scale = target / dof.current_lengths[label]
            for row in dof.rows[label]:
                lattice[row] *= scale
        cells.append(
            PhonopyAtoms(
                symbols=cell.symbols,
                cell=lattice,
                scaled_positions=cell.scaled_positions,
                masses=cell.masses,
            )
        )
    return cells


def get_options() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filename",
        nargs="?",
        default="phonopy_disp.yaml",
        help="phonopy(_disp).yaml giving the equilibrium cell and matrices",
    )
    parser.add_argument("--mlp", default="polymlp.yaml", help="trained MLP file")
    for label in ("a", "b", "c"):
        parser.add_argument(
            f"--{label}",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            help=f"grid range of lattice parameter {label} "
            "(default: current value +/- --margin percent)",
        )
    parser.add_argument(
        "--num", type=int, default=5, help="grid points per free DOF (default: 5)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="default half-width of each range in percent (default: 1.0)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.03,
        help="random displacement distance in angstrom (default: 0.03)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=4,
        help="random-displacement supercells per grid cell (default: 4)",
    )
    parser.add_argument(
        "--tmax", type=float, default=1000.0, help="maximum temperature (default: 1000)"
    )
    parser.add_argument(
        "--dt", type=float, default=10.0, help="temperature step (default: 10)"
    )
    parser.add_argument(
        "--mesh", type=float, default=100.0, help="phonon sampling mesh (default: 100)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the Ti anisotropic QHA."""
    args = get_options()

    eq = phonopy.load(args.filename, produce_fc=False, is_nac=False, log_level=0)
    cell = eq.unitcell
    dof = get_free_lattice_dof(cell)

    provided = {
        label: tuple(value)
        for label in ("a", "b", "c")
        if (value := getattr(args, label)) is not None
    }
    ranges = {}
    counts = {}
    for label in dof.labels:
        if label in provided:
            ranges[label] = provided[label]
        else:
            length = dof.current_lengths[label]
            ranges[label] = (
                length * (1 - args.margin / 100),
                length * (1 + args.margin / 100),
            )
        counts[label] = args.num

    grid_cells = build_lattice_grid(cell, dof, ranges, counts)
    print(
        f"Free lattice DOF: {list(dof.labels)}; "
        f"grid of {len(grid_cells)} cells ({args.num} per DOF)."
    )

    phonopys = []
    internal_energies = []
    for i, grid_cell in enumerate(grid_cells):
        ph = Phonopy(
            grid_cell,
            supercell_matrix=eq.supercell_matrix,
            primitive_matrix=eq.primitive_matrix,
            log_level=0,
        )
        ph.load_mlp(args.mlp)
        ph.generate_displacements(
            distance=args.distance, number_of_snapshots=args.snapshots
        )
        ph.evaluate_mlp()
        ph.produce_force_constants(fc_calculator="symfc")
        phonopys.append(ph)

        # Static energy U per primitive cell (perfect supercell energy from
        # the MLP, normalized to the primitive cell like the phonon props).
        energies, _, _ = ph.mlp.evaluate([ph.supercell])
        n_ratio = len(ph.supercell) / len(ph.primitive)
        internal_energies.append(energies[0] / n_ratio)
        print(f"  cell {i + 1}/{len(grid_cells)} done")

    # One extra temperature point is consumed by the finite differences.
    temperatures = np.arange(0.0, args.tmax + args.dt, args.dt)
    result = run_anisotropic_qha(
        phonopys,
        temperatures,
        internal_energies=internal_energies,
        mesh=args.mesh,
    )

    anisotropic_output.write_lattice_parameters_temperature(result)
    anisotropic_output.write_axial_thermal_expansion(result)
    anisotropic_output.write_volume_temperature(result)
    plt = anisotropic_plot.plot_anisotropic_qha(result)
    plt.savefig("anisotropic_qha.png")
    print(
        "Wrote lattice_parameters-temperature.dat, axial_thermal_expansion.dat, "
        "volume-temperature.dat and anisotropic_qha.png"
    )


if __name__ == "__main__":
    main()
