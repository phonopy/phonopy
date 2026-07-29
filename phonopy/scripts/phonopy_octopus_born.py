# SPDX-License-Identifier: BSD-3-Clause
"""phonopy-octopus-born command-line tool."""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

from phonopy.interface.octopus import get_born_octopus


def fracval(frac: str) -> float:
    """Convert fractional value string to float."""
    if frac.find("/") == -1:
        return float(frac)
    x = frac.split("/")
    return float(x[0]) / float(x[1])


def get_options() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "phonopy-octopus-born: assemble a phonopy BORN file from an Octopus "
            "em_resp calculation (dielectric tensor and Born effective charges)."
        )
    )
    parser.set_defaults(
        primitive_axes=None,
        supercell_matrix=None,
        symmetrize_tensors=True,
        symprec=1e-5,
    )
    parser.add_argument(
        "--dim", dest="supercell_matrix", help="Same behavior as DIM tag"
    )
    parser.add_argument(
        "--pa",
        "--primitive-axis",
        "--primitive-axes",
        dest="primitive_axes",
        help="Same as PRIMITIVE_AXES tags",
    )
    parser.add_argument(
        "--nost",
        "--no-symmetrize-tensors",
        dest="symmetrize_tensors",
        action="store_false",
        help="Prevent from symmetrizing tensors",
    )
    parser.add_argument(
        "--tolerance", dest="symprec", type=float, help="Symmetry tolerance to search"
    )
    parser.add_argument(
        "cell_filename",
        help=(
            "Unit cell: an Octopus geometry include file (numeric format, e.g. as "
            "produced by phonopy-calc-convert) or a VASP-style POSCAR. Prefer the "
            "geometry file included in the em_resp run so the atom order matches."
        ),
    )
    parser.add_argument(
        "em_resp_dir",
        nargs="?",
        default="em_resp/freq_0.0000",
        help=(
            "Directory holding the em_resp results 'epsilon' and 'born_charges' "
            "(default: em_resp/freq_0.0000)"
        ),
    )
    return parser.parse_args()


def run() -> None:
    """Run phonopy-octopus-born."""
    args = get_options()

    if args.primitive_axes:
        vals = [fracval(x) for x in args.primitive_axes.split()]
        if len(vals) == 9:
            primitive_axes = np.array(vals).reshape(3, 3)
        else:
            print("Primitive axes are incorrectly set.")
            sys.exit(1)
    else:
        primitive_axes = np.eye(3)

    if args.supercell_matrix:
        vals = [int(x) for x in args.supercell_matrix.split()]
        if len(vals) == 9:
            supercell_matrix = np.reshape(np.array(vals, dtype="int64"), (3, 3))
        elif len(vals) == 3:
            supercell_matrix = np.diag(np.array(vals, dtype="int64"))
        else:
            print("Supercell matrix is incorrectly set.")
            sys.exit(1)
    else:
        supercell_matrix = np.eye(3, dtype="int64")

    epsilon_filename = os.path.join(args.em_resp_dir, "epsilon")
    born_filename = os.path.join(args.em_resp_dir, "born_charges")

    try:
        with warnings.catch_warnings():
            # symmetrize_borns_and_epsilon reports broken symmetry via
            # warnings.warn; raise it so it can be caught below.
            warnings.simplefilter("error", UserWarning)
            borns, epsilon, atom_indices = get_born_octopus(
                args.cell_filename,
                epsilon_filename,
                born_filename,
                primitive_matrix=primitive_axes,
                supercell_matrix=supercell_matrix,
                symmetrize_tensors=args.symmetrize_tensors,
                symprec=args.symprec,
            )
    except UserWarning:
        print("# Symmetry broken")
        sys.exit(0)

    text = "# epsilon and Z* of atoms "
    text += " ".join(["%d" % n for n in atom_indices + 1])
    lines = [text]
    lines.append(("%13.8f " * 9) % tuple(epsilon.flatten()))
    for z in borns:
        lines.append(("%13.8f " * 9) % tuple(z.flatten()))
    print("\n".join(lines))
    sys.exit(0)
