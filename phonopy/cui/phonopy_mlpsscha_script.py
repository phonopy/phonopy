# SPDX-License-Identifier: BSD-3-Clause
"""Command to run SSCHA at one temperature with a machine-learning potential.

One call is one run at one temperature, and it writes what every iteration
sampled to its own hdf5 file. Which iterations to average over is chosen
afterwards, from the listing -v prints, rather than here.

The cell and the supercell matrix come from the phonopy.yaml-like file. It
may also carry the force constants the run starts from, or the
displacements and forces they are fitted from; carrying neither is allowed,
and the run then starts from force constants fitted to displacements drawn
at --distance and evaluated by the potential. The potential is a pypolymlp
file, given with --mlp.

The supercells themselves are not kept, since the hdf5 file holds what was
computed from them. --save-dataset writes the last iteration's, with the
forces the potential gave them, as a compressed phonopy.yaml-like file.

"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

import phonopy
from phonopy import Phonopy
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import MLPSSCHA
from phonopy.sscha.run import write_sscha_run_hdf5

# Phonopy.save appends the ".xz" of the compressed file it writes.
DATASET_FILENAME = "phonopy_mlpsscha_dataset.yaml"


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description=(
            "Run SSCHA at one temperature with a machine-learning potential "
            "and write what each iteration sampled to an hdf5 file."
        )
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="phonopy_params.yaml",
        help="phonopy.yaml-like file giving the cell and the starting force "
        "constants (default: %(default)s)",
    )
    parser.add_argument(
        "--mlp",
        default="polymlp.yaml",
        help="pypolymlp file to evaluate the supercells with (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=300.0,
        help="temperature in K (default: %(default)s)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=1000,
        help="supercells each iteration draws (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="iterations to run (default: %(default)s)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.01,
        help="displacement distance of the initialization step, used only "
        "when the input file carries no force constants (default: %(default)s)",
    )
    parser.add_argument(
        "--mesh",
        type=float,
        default=100.0,
        help="mesh the harmonic part of the free energy is sampled on "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="seed of the whole run; each iteration derives its own from it, "
        "so that the run is reproducible and its iterations stay independent",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mlpsscha.hdf5",
        help="file the run is written to (default: %(default)s)",
    )
    parser.add_argument(
        "--transient",
        type=int,
        default=1,
        help="how many iterations at the start of the run the listing marks "
        "as its transient (default: %(default)s)",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="write the displacements and forces of the last iteration to "
        f'"{DATASET_FILENAME}.xz"',
    )
    parser.add_argument(
        "--all-force-constants",
        action="store_true",
        help="also write the force constants of every iteration, so that "
        "they can be averaged over a transient afterwards; the refit made "
        "after the last iteration is written either way",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="log each iteration and list them at the end; "
        "-vv adds the force-constant fit",
    )
    return parser.parse_args()


def write_dataset(ph: Phonopy, filename: str = DATASET_FILENAME) -> None:
    """Write the displacements and forces the last iteration sampled.

    One structure per snapshot makes the file large, so it is compressed.

    """
    written = ph.save(
        filename,
        settings={"force_sets": True, "displacements": True},
        compression="xz",
    )
    print(f"Wrote {written}")


def main() -> None:
    """Run the phonopy-mlpsscha command."""
    args = get_options()

    ph = phonopy.load(args.filename, log_level=args.verbose)
    if args.verbose:
        if ph.nac_params is None:
            print("NAC parameters are not used.")
        else:
            print("NAC parameters are used.")

    sscha = MLPSSCHA(
        ph,
        PhonopyMLP().load(args.mlp),
        temperature=args.temperature,
        number_of_snapshots=args.snapshots,
        max_iterations=args.iterations,
        distance=args.distance,
        mesh=args.mesh,
        random_seed=args.random_seed,
        log_level=args.verbose,
    )
    sscha_run = sscha.run().to_sscha_run(args.all_force_constants)
    if args.verbose:
        sscha_run.report(args.transient)

    write_sscha_run_hdf5(sscha_run, args.output)
    print(f"Wrote {args.output}")

    if args.save_dataset:
        write_dataset(sscha.phonopy)


if __name__ == "__main__":
    main()
