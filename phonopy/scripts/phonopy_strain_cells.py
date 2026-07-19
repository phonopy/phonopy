"""Command to generate cells with sampled lattice parameters.

The free lattice-length degrees of freedom are determined from the
symmetry of the cell in the input phonopy(_disp).yaml. Run without ranges
to inspect the free DOF, then give a range per free parameter to sample
cells: randomly by default, or on a regular tensor grid with --grid (whose
main diagonal is the isotropic volume path for the Vinet cross-check). With
--rd, random-displacement supercells are produced directly for
machine-learning-potential training. All lengths are in the native length
unit of the input cell.

"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Literal

import numpy as np

import phonopy
from phonopy.interface.calculator import write_crystal_structure
from phonopy.physical_units import get_calculator_physical_units
from phonopy.qha.lattice_sampling import (
    LatticeDOF,
    build_random_displacement_supercells,
    build_strain_cells_manifest,
    get_free_lattice_dof,
    grid_strained_cells,
    sample_strained_cells,
    write_strain_cells,
    write_strain_cells_manifest,
)

MANIFEST_FILENAME = "strain_cells.yaml"
STRUCTURE_FILENAME = "strain_cells.hdf5"


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description=(
            "Generate cells with sampled lattice parameters (random or, with "
            "--grid, a regular tensor grid), preserving cell symmetry."
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
        "-n",
        "--num",
        type=int,
        default=10,
        help="number of randomly sampled cells (ignored with --grid) (default: 10)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="sample the free lattice lengths on a regular tensor grid instead "
        "of randomly: one N (same count on every free axis) or one N per free "
        "DOF; the main diagonal is the isotropic volume path for "
        "phonopy-anisotropic-qha --compare-vinet when ranges are symmetric",
    )
    parser.add_argument(
        "--amplitude",
        "--amin",
        dest="displacement_distance",
        type=float,
        default=None,
        metavar="DISTANCE",
        help="random displacement distance, and its floor when --amax is "
        "given (default: phonopy's default)",
    )
    parser.add_argument(
        "--amax",
        dest="displacement_distance_max",
        type=float,
        default=None,
        metavar="DISTANCE",
        help="maximum random displacement distance; one distance per supercell "
        "is then drawn uniformly from [0, --amax) and raised to --amin when "
        "smaller, which reserves a share of wholly near-equilibrium "
        "supercells, spanning the large-amplitude region needed to train an "
        "MLP for SSCHA (see --amax-per-atom for the per-atom draw)",
    )
    parser.add_argument(
        "--amax-per-atom",
        dest="displacement_distance_per_atom",
        action="store_true",
        help="with --amax, draw the distance per atom instead of per supercell, "
        "uniformly over [--amin, --amax), so every supercell spans the whole "
        "range; this decorrelates the displacement amplitude from the lattice "
        "strain, which matters when few supercells are generated per strained "
        "cell. --amin is a sampling bound here rather than a floor",
    )
    parser.add_argument(
        "--rd",
        "--random-displacements",
        dest="random_displacements",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="generate N random-displacement supercells per cell (default N: 1)",
    )
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--symprec", type=float, default=1e-5, help="symmetry tolerance (default: 1e-5)"
    )
    return parser.parse_args()


def _print_dof(
    dof: LatticeDOF, volume: float, calculator: str, length_unit: str
) -> None:
    """Print the free lattice DOF and how to specify their ranges.

    For a few reference strains (+/-1, 2, 3 percent) the ready-to-copy
    ranges and the spanned min/max cell volume are shown. Because every
    lattice vector is scaled, the extreme volumes are the all-min and
    all-max corners, i.e. volume * (1 -/+ p)^3.

    """
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
    print(f"  current: {current} {length_unit}   volume {volume:.4f} {length_unit}^3")
    print("Give a range per free parameter, e.g.")
    for percent in (1, 2, 3):
        factor = percent / 100
        ranges = "   ".join(
            f"--{label} {dof.current_lengths[label] * (1 - factor):.4f} "
            f"{dof.current_lengths[label] * (1 + factor):.4f}"
            for label in dof.labels
        )
        v_lo = volume * (1 - factor) ** 3
        v_hi = volume * (1 + factor) ** 3
        print(f"  +/-{percent}%   {ranges}   (volume {v_lo:.4f} .. {v_hi:.4f})")


def _resolve_grid_counts(values: list[int], dof: LatticeDOF) -> dict[str, int]:
    """Map the --grid values to a per-free-DOF point count.

    One value is broadcast to every free axis (a square grid); one value per
    free DOF sets each axis independently. Any other count is an error.

    """
    if len(values) == 1:
        return {label: values[0] for label in dof.labels}
    if len(values) == len(dof.labels):
        return dict(zip(dof.labels, values, strict=True))
    sys.exit(
        f"Error: --grid takes one value or one per free DOF {list(dof.labels)}, "
        f"but got {len(values)} values."
    )


def _print_diagonal_path(
    dof: LatticeDOF, ranges: dict[str, tuple[float, float]], counts: dict[str, int]
) -> None:
    """Print the grid main diagonal -- the volume path --compare-vinet uses.

    The diagonal is the cell taken at the same rank on every free axis, so it
    has ``min(counts)`` cells ordered by increasing volume. The free lengths and
    their ratios to the first axis are listed so the cell shape along the path
    (constant c/a or not) is visible: a constant-shape path needs equal counts
    and equal fractional ranges, but a varying-shape path is still a valid, if
    less clean, input to the Vinet cross-check.

    """
    axes = {label: np.linspace(*ranges[label], counts[label]) for label in dof.labels}
    n_diag = min(counts[label] for label in dof.labels)
    ref, others = dof.labels[0], dof.labels[1:]
    ratio_header = "  ".join(f"{label}/{ref}" for label in others)
    print(f"  Main diagonal ({n_diag} cells), the --compare-vinet volume path:")
    print(f"    {'  '.join(dof.labels)}   {ratio_header}".rstrip())
    for i in range(n_diag):
        lengths = "  ".join(f"{axes[label][i]:.4f}" for label in dof.labels)
        ratios = "  ".join(f"{axes[label][i] / axes[ref][i]:.4f}" for label in others)
        print(f"    {lengths}   {ratios}".rstrip())
    if n_diag < 5:
        print("    (fewer than 5 cells; --compare-vinet needs at least 5.)")


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
        _print_dof(dof, cell.volume, calculator, length_unit)
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

    # A concrete seed is resolved only when randomness is actually used
    # (random length sampling, or random-displacement supercells), so a
    # deterministic grid run without --rd records no seed and can be replayed
    # from the manifest verbatim.
    with_rd = args.random_displacements is not None
    if args.displacement_distance_max is not None:
        if not with_rd:
            sys.exit("Error: --amax applies to random displacements; add --rd.")
        if (
            args.displacement_distance is not None
            and args.displacement_distance_max <= args.displacement_distance
        ):
            sys.exit("Error: --amax must be larger than --amin.")
    elif args.displacement_distance_per_atom:
        sys.exit("Error: --amax-per-atom applies to --amax; add --amax.")
    distance_sampling: Literal["supercell", "atom"] = (
        "atom" if args.displacement_distance_per_atom else "supercell"
    )
    needs_random = args.grid is None or with_rd
    if args.random_seed is not None:
        seed = args.random_seed
    elif needs_random:
        seed = int(np.random.default_rng().integers(2**32))
    else:
        seed = None

    grid_counts: dict[str, int] | None = None
    if args.grid is not None:
        grid_counts = _resolve_grid_counts(args.grid, dof)
        unitcells = grid_strained_cells(cell, dof, ranges, num=grid_counts)
        sampling = "grid"
    else:
        unitcells = sample_strained_cells(cell, dof, ranges, num=args.num, seed=seed)
        sampling = "random"

    displacements = None
    if with_rd:
        cells, displacements = build_random_displacement_supercells(
            unitcells,
            phonon.supercell_matrix,
            distance=args.displacement_distance,
            max_distance=args.displacement_distance_max,
            distance_sampling=distance_sampling,
            count=args.random_displacements,
            seed=seed,
        )
        # Each strained cell yields args.random_displacements supercells; repeat
        # its lengths per file so the manifest stays aligned with the outputs.
        manifest_cells = [
            uc for uc in unitcells for _ in range(args.random_displacements)
        ]
        prefix = "supercell"
        kind = "random-displacement supercell"
    else:
        cells = unitcells
        manifest_cells = list(unitcells)
        prefix = "unitcell"
        kind = "strained unit cell"

    filenames = [f"{prefix}-{i + 1:05d}" for i in range(len(cells))]
    for filename, structure in zip(filenames, cells, strict=True):
        write_crystal_structure(filename, structure, interface_mode=calculator)

    if grid_counts is None:
        grid_shape = None
    else:
        grid_shape = [grid_counts[label] for label in dof.labels]

    manifest = build_strain_cells_manifest(
        phonopy_version=phonopy.__version__,
        calculator=calculator,
        length_unit=length_unit,
        source=args.filename,
        dof=dof,
        command_line=" ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        ranges=ranges,
        num=None if grid_counts is not None else args.num,
        grid_shape=grid_shape,
        displacement_distance=args.displacement_distance,
        displacement_distance_max=args.displacement_distance_max,
        displacement_distance_sampling=distance_sampling,
        random_displacements=args.random_displacements,
        symprec=args.symprec,
        seed=seed,
        sampling=sampling,
        prefix=prefix,
        kind=kind,
        unitcells=manifest_cells,
        filenames=filenames,
    )
    write_strain_cells_manifest(MANIFEST_FILENAME, manifest)

    if displacements is not None:
        write_strain_cells(
            STRUCTURE_FILENAME,
            unitcells=unitcells,
            supercells=cells,
            displacements=displacements,
            supercell_matrix=phonon.supercell_matrix,
            phonopy_version=phonopy.__version__,
            calculator=calculator,
            length_unit=length_unit,
            displacement_distance=args.displacement_distance,
            displacement_distance_max=args.displacement_distance_max,
            displacement_distance_sampling=distance_sampling,
        )

    print(
        f"Wrote {len(cells)} {kind}(s) as "
        f"{prefix}-00001 .. {prefix}-{len(cells):05d} in {calculator} format."
    )
    if grid_counts is not None:
        shape = " x ".join(str(grid_counts[label]) for label in dof.labels)
        print(f"Grid sampling: {shape} over ({', '.join(dof.labels)}).")
        _print_diagonal_path(dof, ranges, grid_counts)
    if seed is not None:
        print(f"Random seed: {seed}")
    print(f"Provenance written to {MANIFEST_FILENAME}")
    if displacements is not None:
        print(f"Structures and displacements written to {STRUCTURE_FILENAME}")


if __name__ == "__main__":
    run()
