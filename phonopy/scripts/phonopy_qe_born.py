# Copyright (C) 2024 Florian Knoop
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

import numpy as np

from phonopy.interface.qe import read_pwscf
from phonopy.structure.symmetry import elaborate_borns_and_epsilon


def parse_ph_out(file: str, natoms: int) -> dict:
    """Parse BEC and dielectric tensor from QE ph.x output w/ a minor sanity check."""
    hook_eps = "Dielectric constant in cartesian axis"
    hook_bec = "Effective charges (d Force / dE) in cartesian axis without acoustic"

    def _strip(xx: str) -> str:
        return xx.strip().strip("(").strip(")")

    def parse_epsilon(fp) -> np.ndarray:
        """Parse the dielectric tensor from ph.x output."""
        next(fp)  # skip 1 line
        lines = [_strip(next(fp)) for _ in range(3)]
        return np.array([np.fromstring(x, sep=" ") for x in lines])

    def parse_bec(fp, natoms) -> np.ndarray:
        """Parse the BEC from ph.x output."""
        next(fp)  # skip 1 line
        bec = []
        for _ in range(natoms):
            next(fp)
            bec.extend(next(fp).split()[2:5] for _ in range(3))

        return np.reshape(bec, (natoms, 3, 3)).astype(float)

    epsilon, bec = None, None
    with open(file) as f:
        for line in f:
            if "number of atoms/cell      =" in line:
                _natoms = int(line.split()[4])
                assert natoms == _natoms, (natoms, _natoms)
            if hook_eps in line:
                epsilon = parse_epsilon(f)
            if hook_bec in line:
                bec = parse_bec(f, natoms=natoms)

    return epsilon, bec


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse phonopy BORN file from QE output"
    )
    parser.set_defaults(
        num_atoms=None,
        # primitive_axes=None,
        # supercell_matrix=None,
        symmetrize_tensors=True,
        symprec=1e-5,
    )

    parser.add_argument("file_pw", help="input for pw.x")
    parser.add_argument("file_ph", help="output of ph.x")
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-qe-born."""
    args = get_options()

    try:
        borns, epsilon, atom_indices = get_born_qe_ph(
            file_pw=args.file_pw,
            file_ph=args.file_ph,
            # primitive_matrix=primitive_axes,
            # supercell_matrix=supercell_matrix,
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


def get_born_qe_ph(
    file_pw=None,
    file_ph=None,
    # primitive_matrix=None,
    # supercell_matrix=None,
    is_symmetry=True,
    symmetrize_tensors=True,
    symprec=1e-5,
):
    """Parse ph.out to get NAC parameters.

    Returns
    -------
    See elaborate_borns_and_epsilon.

    """
    ucell, _ = read_pwscf(file_pw)
    epsilon, borns = parse_ph_out(file_ph, ucell.get_number_of_atoms())
    if len(borns) == 0 or len(epsilon) == 0:
        return None
    else:
        return elaborate_borns_and_epsilon(
            ucell,
            borns,
            epsilon,
            # primitive_matrix=primitive_matrix,
            # supercell_matrix=supercell_matrix,
            is_symmetry=is_symmetry,
            symmetrize_tensors=symmetrize_tensors,
            symprec=symprec,
        )
