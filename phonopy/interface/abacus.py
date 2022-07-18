"""ABACUS calculator interface."""
# Copyright (C) 2022 Yuyang Ji
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

import re
import string
import sys
from collections import Counter

import numpy as np

from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms, atom_data, symbol_map
from phonopy.units import Bohr


#
# read ABACUS STRU
#
def read_abacus(filename, elements=[]):
    """Parse ABACUS structure, distance in unit au (bohr)."""
    pps = []
    orbitals = None
    cell = []
    magmoms = []
    numbers = []
    positions = []
    with open(filename, "r") as file:
        if _search_sentence(file, "ATOMIC_SPECIES"):
            for it, elem in enumerate(elements):
                line = _skip_notes(file.readline())
                _, _, pseudo = line.split()
                pps.append(pseudo)

        if _search_sentence(file, "NUMERICAL_ORBITAL"):
            orbitals = []
            for elem in elements:
                orbitals.append(_skip_notes(file.readline()))

        if _search_sentence(file, "LATTICE_CONSTANT"):
            lat0 = float(_skip_notes(file.readline()).split()[0])

        if _search_sentence(file, "LATTICE_VECTORS"):
            for i in range(3):
                cell.append(_list_elem_2float(_skip_notes(file.readline()).split()))
        cell = np.array(cell) * lat0

        if _search_sentence(file, "ATOMIC_POSITIONS"):
            ctype = _skip_notes(file.readline())

        for elem in elements:
            if _search_sentence(file, elem):
                magmoms.append(float(_skip_notes(file.readline()).split()[0]))
                na = int(_skip_notes(file.readline()).split()[0])
                numbers.append(na)
                for i in range(na):
                    line = _skip_notes(file.readline())
                    positions.append(_list_elem_2float(line.split()[:3]))

    expanded_symbols = _expand(numbers, elements)
    magnetic_moments = _expand(numbers, magmoms)
    if ctype == "Direct":
        atoms = PhonopyAtoms(
            symbols=expanded_symbols,
            cell=cell,
            scaled_positions=positions,
            magnetic_moments=magnetic_moments,
        )
    elif ctype == "Cartesian":
        atoms = PhonopyAtoms(
            symbols=expanded_symbols,
            cell=cell,
            positions=positions,
            magnetic_moments=magnetic_moments,
        )
    elif ctype == "Cartesian_angstrom":
        atoms = PhonopyAtoms(
            symbols=expanded_symbols,
            cell=cell,
            positions=np.array(positions) / Bohr,
            magnetic_moments=magnetic_moments,
        )

    return atoms, pps, orbitals


#
# write ABACUS STRU
#
def write_abacus(filename, atoms, pps, orbitals=None):
    """Write structure to file."""
    with open(filename, "w") as f:
        f.write(get_abacus_structure(atoms, pps, orbitals))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pps,
    orbitals,
    pre_filename="STRU",
    width=3,
):
    """Write supercells with displacements to files."""
    write_abacus("%s.in" % pre_filename, supercell, pps)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )
        write_abacus(filename, cell, pps, orbitals)


def get_abacus_structure(atoms, pps, orbitals=None):
    """Return ABACUS structure in text."""
    empty_line = ""
    line = []
    line.append("ATOMIC_SPECIES")
    elements = list(Counter(atoms.symbols).keys())
    numbers = list(Counter(atoms.symbols).values())

    for i, elem in enumerate(elements):
        line.append(f"{elem}\t{atom_data[symbol_map[elem]][3]}\t{pps[i]}")
    line.append(empty_line)

    if orbitals:
        line.append("NUMERICAL_ORBITAL")
        for i in range(len(elements)):
            line.append(f"{orbitals[i]}")
        line.append(empty_line)

    line.append("LATTICE_CONSTANT")
    line.append(str(1.0))
    line.append(empty_line)

    line.append("LATTICE_VECTORS")
    for i in range(3):
        line.append(" ".join(_list_elem2str(atoms.cell[i])))
    line.append(empty_line)

    line.append("ATOMIC_POSITIONS")
    line.append("Direct")
    line.append(empty_line)
    index = 0
    for i, elem in enumerate(elements):
        line.append(f"{elem}\n{atoms.magnetic_moments[index]}\n{numbers[i]}")
        for j in range(index, index + numbers[i]):
            line.append(
                " ".join(_list_elem2str(atoms.scaled_positions[j])) + " " + "1 1 1"
            )
        line.append(empty_line)
        index += numbers[i]

    return "\n".join(line)


#
# set Force
#
def read_abacus_output(filename):
    """Read ABACUS forces from last self-consistency iteration."""
    with open(filename, "r") as file:
        for line in file:
            if re.search(r"TOTAL ATOM NUMBER = [0-9]+", line):
                natom = int(re.search("[0-9]+", line).group())
                force = np.zeros((natom, 3))
            if re.search(r"TOTAL-FORCE \(eV/Angstrom\)", line):
                for i in range(4):
                    file.readline()
                for i in range(natom):
                    _, fx, fy, fz = file.readline().split()
                    force[i] = (float(fx), float(fy), float(fz))

    return force


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        abacus_forces = read_abacus_output(filename)
        if check_forces(abacus_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                abacus_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(abacus_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


#
# tools
#
def _expand(num_atoms, attr):
    expanded_attr = []
    for s, num in zip(attr, num_atoms):
        expanded_attr += [s] * num
    return expanded_attr


def _search_sentence(file, sentence):
    """Search sentence in file."""
    if isinstance(sentence, str):
        sentence = sentence.strip()
        for line in file:
            line = _skip_notes(line).strip()
            if line == sentence:
                return line
    elif isinstance(sentence, list):
        sentence = _list_elem2strip(sentence)
        for line in file:
            line = _skip_notes(line).strip()
            if line in sentence:
                return line

    file.seek(0, 0)
    return False


def _skip_notes(line):
    """Delete comments lines with '#' or '//'."""
    line = re.compile(r"#.*").sub("", line)
    line = re.compile(r"//.*").sub("", line)
    line = line.strip()
    return line


def _list_elem2strip(a, ds=string.whitespace):
    """Strip element of list with `str` type."""

    def list_strip(s):
        return s.strip(ds)

    return list(map(list_strip, a))


def _list_elem_2float(a):
    """Convert type of list element to float."""
    return list(map(float, a))


def _list_elem2str(a):
    """Convert type of list element to str."""
    return list(map(str, a))
