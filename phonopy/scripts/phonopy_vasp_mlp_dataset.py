"""Command to build a pypolymlp training dataset from vasprun.xml files.

Extracts the final structure, total energy, forces and stress from each
VASP vasprun.xml and writes them together into a single HDF5 dataset that
can be read back to train a pypolymlp machine-learning potential.

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
            "stresses) from VASP vasprun.xml files."
        )
    )
    parser.add_argument(
        "vaspruns",
        nargs="+",
        help="vasprun.xml files (optionally lzma/gzip/bz2 compressed)",
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
    data = read_vasprun_dataset(args.vaspruns)
    write_pypolymlp_structure_dataset(data, filename=args.output)
    has_stress = "yes" if data.stresses is not None else "no"
    print(
        f"Wrote {len(data.structures)} structures to {args.output} "
        f"(stress: {has_stress})."
    )


if __name__ == "__main__":
    run()
