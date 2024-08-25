"""PWmat calculator interface."""

# Copyright (C) 2014 Atsushi Togo
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

from phonopy.file_IO import iter_collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms, atom_data


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "force (eV/A)"
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        pwmat_forces = iter_collect_forces(
            filename,
            num_atoms,
            hook,
            [1, 2, 3],
        )
        pwmat_forces = [[-x for x in sublist] for sublist in pwmat_forces]
        if check_forces(pwmat_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                pwmat_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(pwmat_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_atom_config(filename):
    """Read structure information from atom.config."""
    with open(filename) as f:
        lines = f.read().splitlines()

    element_index = []
    lattice_vectors = []
    atom_position = []
    read_lattice = read_position = False

    for line in lines:
        line = line.strip()

        # lattice constant
        if line.lower().startswith("lattice"):
            read_lattice = True
        elif read_lattice:
            try:
                parts = [float(part) for part in line.split()[:3]]
                lattice_vectors.append(parts[0:3])
            except ValueError:
                read_lattice = False

        # atom position
        if line.lower().startswith("position"):
            read_position = True
        elif read_position:
            try:
                parts = [float(part) for part in line.split()]
                element_index.append(int(parts[0]))
                atom_position.append(parts[1:4])
            except ValueError:
                read_position = False

    cell_args = {
        "symbols": [atom_data[n][1] for n in element_index],
        "cell": lattice_vectors,
        "scaled_positions": atom_position,
    }

    return PhonopyAtoms(**cell_args)


def write_atom_config(filename, cell):
    """Write atom.config to file."""
    with open(filename, "w") as f:
        f.write(get_pwmat_structure(cell))


def get_pwmat_structure(cell):
    """Return PWmat structure in text."""
    lattice = cell.cell
    positions = cell.scaled_positions
    numbers = cell.numbers
    mag_mom = cell.magnetic_moments

    line = []
    line.append(f" {len(positions)}")
    line.append("Lattice vector (Angstrom)")
    for row in lattice:
        line.append(" ".join(f" {val:.16f}" for val in row))
    line.append("Position, move_x, move_y, move_z")
    for number, position in zip(numbers, positions):
        line.append(
            f" {number}  "
            + "   ".join(f"{val:.16f}" for val in position)
            + "  1   1   1"
        )
    if mag_mom is not None and len(mag_mom) == len(cell.numbers):
        if np.size(mag_mom[0]) == 1:
            line.append("magnetic")
            for number, mag in zip(numbers, mag_mom):
                line.append(f" {number}  {mag:.16f}")

        if np.size(mag_mom[0]) == 3:
            line.append("magnetic_xyz")
            for number, mag in zip(numbers, mag_mom):
                line.append(f" {number}  {mag[0]:.6f} {mag[1]:.6f} {mag[2]:.6f}")
    return "\n".join(line)


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pre_filename="supercell",
    width=3,
):
    """Write supercells with displacements to files."""
    write_atom_config(
        "%s.config" % pre_filename,
        supercell,
    )
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.config".format(
            i, pre_filename=pre_filename, width=width
        )
        write_atom_config(filename, cell)
