"""DFTB+ calculator interface."""

# Copyright (C) 2015 Atsushi Togo
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

from phonopy.file_IO import collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import Atoms
from phonopy.units import dftbpToBohr


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "forces              :real:2:"
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        f = open(filename)
        dftbp_forces = collect_forces(f, num_atoms, hook, [0, 1, 2])
        if check_forces(dftbp_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                dftbp_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(dftbp_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


#
# read dftbp-files
#


def read_dftbp(filename):
    """Read DFTB+ structure files in gen format.

    Parameters
    ----------
    filename: name of the gen-file to be read

    Returns
    -------
    atoms: an object of the phonopy.Atoms class, representing the structure
    found in filename

    """
    infile = open(filename, "r")

    lines = infile.readlines()

    # remove any comments
    for ss in lines:
        if ss.strip().startswith("#"):
            lines.remove(ss)

    natoms = int(lines[0].split()[0])
    symbols = lines[1].split()

    if lines[0].split()[1].lower() == "f":
        is_scaled = True
        scale_pos = 1
        scale_latvecs = dftbpToBohr
    else:
        is_scaled = False
        scale_pos = dftbpToBohr
        scale_latvecs = dftbpToBohr

    # assign positions and expanded symbols
    positions = []
    expaned_symbols = []

    for ii in range(2, natoms + 2):
        lsplit = lines[ii].split()

        expaned_symbols.append(symbols[int(lsplit[1]) - 1])
        positions.append([float(ss) * scale_pos for ss in lsplit[2:5]])

    # origin is ignored, may be used in future
    # origin = [float(ss) for ss in lines[natoms + 2].split()]

    # assign coords of unitcell
    cell = []

    for ii in range(natoms + 3, natoms + 6):
        lsplit = lines[ii].split()

        cell.append([float(ss) * scale_latvecs for ss in lsplit[:3]])
    cell = np.array(cell)

    if is_scaled:
        atoms = Atoms(symbols=expaned_symbols, cell=cell, scaled_positions=positions)
    else:
        atoms = Atoms(symbols=expaned_symbols, cell=cell, positions=positions)

    return atoms


#
# write dftb+ .gen-file
#
def get_reduced_symbols(symbols):
    """Reduce expanded list of symbols.

    Parameters
    ----------
    symbols:
        list containing any chemical symbols as often as
        the atom appears in the structure.

    Returns
    -------
    reduced_symbols: any symbols appears only once.

    """
    reduced_symbols = []

    for ss in symbols:
        if ss not in reduced_symbols:
            reduced_symbols.append(ss)

    return reduced_symbols


def write_dftbp(filename, atoms):
    """Write DFTB+ readable, gen-formatted structure files.

    Parameters
    ----------
    filename: name of the gen-file to be written
    atoms: object containing information about structure

    """
    scale_pos = dftbpToBohr

    lines = ""

    # 1. line, use absolute positions
    natoms = atoms.get_number_of_atoms()
    lines += str(natoms)
    lines += " S \n"

    # 2. line
    expaned_symbols = atoms.get_chemical_symbols()
    symbols = get_reduced_symbols(expaned_symbols)
    lines += " ".join(symbols) + "\n"

    atom_numbers = []
    for ss in expaned_symbols:
        atom_numbers.append(symbols.index(ss) + 1)

    positions = atoms.get_positions() / scale_pos

    for ii in range(natoms):
        pos = positions[ii]
        pos_str = "{:3d} {:3d} {:20.15f} {:20.15f} {:20.15f}\n".format(
            ii + 1, atom_numbers[ii], pos[0], pos[1], pos[2]
        )
        lines += pos_str

    # origin arbitrary
    lines += "0.0 0.0 0.0\n"

    cell = atoms.get_cell() / scale_pos

    for ii in range(3):
        cell_str = "{:20.15f} {:20.15f} {:20.15f}\n".format(
            cell[ii][0], cell[ii][1], cell[ii][2]
        )
        lines += cell_str

    outfile = open(filename, "w")
    outfile.write(lines)


def write_supercells_with_displacements(
    supercell, cells_with_disps, ids, pre_filename="geo.gen", width=3
):
    """Write perfect supercell and supercells with displacements.

    Parameters
    ----------
    supercell: perfect supercell
    cells_with_disps: supercells with displaced atoms
    filename: root-filename

    """
    # original cell
    write_dftbp(pre_filename + "S", supercell)

    # displaced cells
    for i, cell in zip(ids, cells_with_disps):
        filename = "{pre_filename}S-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )
        write_dftbp(filename, cell)
