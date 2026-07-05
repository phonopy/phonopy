"""Command to generate cells with randomly sampled lattice parameters.

The free lattice-length degrees of freedom are determined from the
symmetry of the cell in the input phonopy(_disp).yaml. Run without ranges
to inspect the free DOF, then give a range per free parameter to sample
cells. With --rd, random-displacement supercells are produced directly for
machine-learning-potential training. All lengths are in the native length
unit of the input cell.

"""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

import phonopy
from phonopy.interface.calculator import write_crystal_structure
from phonopy.physical_units import get_calculator_physical_units
from phonopy.qha.lattice_sampling import (
    LatticeDOF,
    build_random_displacement_supercells,
    get_free_lattice_dof,
    sample_strained_cells,
)


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description=(
            "Generate cells with randomly sampled lattice parameters, "
            "preserving cell symmetry."
        )
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="phonopy_disp.yaml",
        help="phonopy(_disp).yaml providing the cell, supercell matrix and calculator",
    )
    for label in ("a", "b", "c"):
        parser.add_argument(
            f"--{label}",
            nargs=2,
            type=float,
            metavar=("MIN", "MAX"),
            help=f"range of lattice parameter {label}",
        )
    parser.add_argument(
        "-n", "--num", type=int, default=10, help="number of cells (default: 10)"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--rd",
        type=float,
        default=None,
        metavar="DISTANCE",
        help="generate random-displacement supercells with this displacement distance",
    )
    parser.add_argument(
        "--symprec", type=float, default=1e-5, help="symmetry tolerance (default: 1e-5)"
    )
    return parser.parse_args()


def _print_dof(dof: LatticeDOF, calculator: str, length_unit: str) -> None:
    """Print the free lattice DOF and how to specify their ranges."""
    print(f"Calculator: {calculator}   Length unit: {length_unit}")
    print(
        f"Crystal system: {dof.crystal_system} "
        f"(space group No. {dof.spacegroup_number})"
    )
    tie = f"   ({dof.tie_description})" if dof.tie_description else ""
    print(f"Free lattice parameter(s): {', '.join(dof.labels)}{tie}")
    current = "   ".join(
        f"{label} = {dof.current_lengths[label]:.6f}" for label in dof.labels
    )
    print(f"  current: {current} {length_unit}")
    example = "  ".join(
        f"--{label} {dof.current_lengths[label] * 0.98:.4f} "
        f"{dof.current_lengths[label] * 1.02:.4f}"
        for label in dof.labels
    )
    print(f"Give a range per free parameter, e.g.  {example}")


def run() -> None:
    """Run the phonopy-strain-cells command."""
    args = get_options()

    phonon = phonopy.load(args.filename, produce_fc=False, is_nac=False, log_level=0)
    cell = phonon.unitcell
    calculator = phonon.calculator or "vasp"
    length_unit = get_calculator_physical_units(calculator).length_unit

    try:
        dof = get_free_lattice_dof(cell, symprec=args.symprec)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    provided = {
        label: (value[0], value[1])
        for label in ("a", "b", "c")
        if (value := getattr(args, label)) is not None
    }

    if not provided:
        _print_dof(dof, calculator, length_unit)
        return

    extra = [label for label in provided if label not in dof.labels]
    if extra:
        sys.exit(
            f"Error: {extra} are not free lattice parameters; "
            f"free DOF are {list(dof.labels)}."
        )
    missing = [label for label in dof.labels if label not in provided]
    if missing:
        sys.exit(f"Error: ranges are required for free DOF {missing}.")

    ranges = {label: provided[label] for label in dof.labels}
    unitcells = sample_strained_cells(cell, dof, ranges, num=args.num, seed=args.seed)

    if args.rd is not None:
        cells = build_random_displacement_supercells(
            unitcells, phonon.supercell_matrix, distance=args.rd, seed=args.seed
        )
        prefix = "supercell"
        kind = "random-displacement supercell"
    else:
        cells = unitcells
        prefix = "unitcell"
        kind = "strained unit cell"

    for i, structure in enumerate(cells):
        write_crystal_structure(
            f"{prefix}-{i + 1:05d}", structure, interface_mode=calculator
        )
    print(
        f"Wrote {len(cells)} {kind}(s) as "
        f"{prefix}-00001 .. {prefix}-{len(cells):05d} in {calculator} format."
    )


if __name__ == "__main__":
    run()
