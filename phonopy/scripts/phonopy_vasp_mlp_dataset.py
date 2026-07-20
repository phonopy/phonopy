# SPDX-License-Identifier: BSD-3-Clause
"""Command to build a pypolymlp training dataset from VASP output files.

Extracts the final structure, total energy, forces and stress from each
VASP vaspout.h5 or vasprun.xml and writes them together into a single HDF5
dataset that can be read back to train a pypolymlp machine-learning
potential. Reading vaspout.h5 is preferable when it is available because it
carries the full precision of the calculation, whereas vasprun.xml is
written with six digits.

"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from phonopy.interface.pypolymlp import (
    read_vasprun_dataset,
    write_pypolymlp_structure_dataset,
)


def get_options() -> Namespace:
    """Parse command-line options."""
    parser = ArgumentParser(
        description=(
            "Build a pypolymlp training dataset (structures, energies, forces, "
            "stresses) from VASP vaspout.h5 or vasprun.xml files."
        )
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help=(
            "vaspout.h5 files, or vasprun.xml files (optionally lzma/gzip/bz2 "
            "compressed); vaspout.h5 is read at full precision"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="polymlp_dataset.hdf5",
        help="output HDF5 file (default: polymlp_dataset.hdf5)",
    )
    return parser.parse_args()


def run() -> None:
    """Run the phonopy-vasp-mlp-dataset command."""
    args = get_options()
    data = read_vasprun_dataset(args.filenames)
    write_pypolymlp_structure_dataset(data, filename=args.output)
    has_stress = "yes" if data.stresses is not None else "no"
    print(
        f"Wrote {len(data.structures)} structures to {args.output} "
        f"(stress: {has_stress})."
    )


if __name__ == "__main__":
    run()
