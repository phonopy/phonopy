# Copyright (C) 2012 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys
import warnings

import numpy as np

from phonopy.interface.vasp import get_born_OUTCAR, get_born_vasprunxml


def fracval(frac):
    """Convert fractional value string to float."""
    if frac.find("/") == -1:
        return float(frac)
    else:
        x = frac.split("/")
        return float(x[0]) / float(x[1])


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy vasp-born command-line-tool")
    parser.set_defaults(
        num_atoms=None,
        primitive_axes=None,
        supercell_matrix=None,
        symmetrize_tensors=True,
        read_outcar=False,
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
        "--outcar",
        dest="read_outcar",
        action="store_true",
        help=(
            "Read OUTCAR instead of vasprun.xml. " "POSCAR is necessary in this case."
        ),
    )
    parser.add_argument(
        "filenames", nargs="*", help="Filenames: vasprun.xml or OUTCAR and POSCAR"
    )
    args = parser.parse_args()
    return args


def run():
    """Rurn phonopy-vasp-born."""
    args = get_options()

    if args.filenames:
        outcar_filename = args.filenames[0]
    else:
        if args.read_outcar:
            outcar_filename = "OUTCAR"
        else:
            outcar_filename = "vasprun.xml"

    if len(args.filenames) > 1:
        poscar_filename = args.filenames[1]
    else:
        poscar_filename = "POSCAR"

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
            supercell_matrix = np.reshape(np.array(vals, dtype="intc"), (3, 3))
        elif len(vals) == 3:
            supercell_matrix = np.diag(np.array(vals, dtype="intc"))
        else:
            print("Supercell matrix is incorrectly set.")
            sys.exit(1)
    else:
        supercell_matrix = np.eye(3, dtype="intc")

    with warnings.catch_warnings():
        # To catch warnings as error, set warnings.simplefilter("error")
        warnings.simplefilter("always")

        try:
            if args.read_outcar:
                borns, epsilon, atom_indices = get_born_OUTCAR(
                    poscar_filename=poscar_filename,
                    outcar_filename=outcar_filename,
                    primitive_matrix=primitive_axes,
                    supercell_matrix=supercell_matrix,
                    symmetrize_tensors=args.symmetrize_tensors,
                    symprec=args.symprec,
                )
            else:
                borns, epsilon, atom_indices = get_born_vasprunxml(
                    filename=outcar_filename,
                    primitive_matrix=primitive_axes,
                    supercell_matrix=supercell_matrix,
                    symmetrize_tensors=args.symmetrize_tensors,
                    symprec=args.symprec,
                )
        except UserWarning:
            text = "# Symmetry broken"
            lines = [text]
            print("\n".join(lines))
            sys.exit(0)

        try:
            text = "# epsilon and Z* of atoms "
            text += " ".join(["%d" % n for n in atom_indices + 1])
            lines = [text]
            lines.append(("%13.8f " * 9) % tuple(epsilon.flatten()))
            for z in borns:
                lines.append(("%13.8f " * 9) % tuple(z.flatten()))
        except Exception:
            sys.exit(1)
        else:
            print("\n".join(lines))
            sys.exit(0)

    sys.exit(1)
