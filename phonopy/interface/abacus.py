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

_re_float = r"[-+]?\d+\.*\d*(?:[Ee][-+]\d+)?"

#
# read ABACUS STRU
#


def read_abacus(filename):
    """Read structure information from abacus structure file, distance in unit au (bohr)."""
    fd = open(filename, "r")
    contents = fd.read()
    title_str = r"(?:LATTICE_CONSTANT|NUMERICAL_ORBITAL|ABFS_ORBITAL|LATTICE_VECTORS|LATTICE_PARAMETERS|ATOMIC_POSITIONS)"

    # remove comments and empty lines
    contents = re.compile(r"#.*|//.*").sub("", contents)
    contents = re.compile(r"\n{2,}").sub("\n", contents)

    # specie, mass, pps
    specie_pattern = re.compile(rf"ATOMIC_SPECIES\s*\n([\s\S]+?)\s*\n{title_str}")
    specie_lines = np.array(
        [line.split() for line in specie_pattern.search(contents).group(1).split("\n")]
    )
    symbols = specie_lines[:, 0]
    ntype = len(symbols)
    mass = specie_lines[:, 1].astype(float)
    try:
        potential = dict(zip(symbols, specie_lines[:, 2].tolist()))
    except IndexError:
        potential = None

    # basis
    aim_title = "NUMERICAL_ORBITAL"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    orb_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    orb_lines = orb_pattern.search(contents)
    if orb_lines:
        basis = dict(zip(symbols, orb_lines.group(1).split("\n")))
    else:
        basis = None

    # ABFs basis
    aim_title = "ABFS_ORBITAL"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    abf_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    abf_lines = abf_pattern.search(contents)
    if abf_lines:
        offsite_basis = dict(zip(symbols, abf_lines.group(1).split("\n")))
    else:
        offsite_basis = None

    # lattice constant
    aim_title = "LATTICE_CONSTANT"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    a0_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    a0_lines = a0_pattern.search(contents)
    atom_lattice_scale = float(a0_lines.group(1))

    # lattice vector
    aim_title = "LATTICE_VECTORS"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    vec_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    vec_lines = vec_pattern.search(contents)
    if vec_lines:
        atom_lattice = np.array(
            [line.split() for line in vec_pattern.search(contents).group(1).split("\n")]
        ).astype(float)
    atom_lattice = atom_lattice * atom_lattice_scale

    aim_title = "ATOMIC_POSITIONS"
    type_pattern = re.compile(rf"{aim_title}\s*\n(\w+)\s*\n")
    # type of coordinates
    atom_pos_type = type_pattern.search(contents).group(1)
    assert atom_pos_type in [
        "Direct",
        "Cartesian",
    ], "Only two type of atomic coordinates are supported: 'Direct' or 'Cartesian'."

    block_pattern = re.compile(rf"{atom_pos_type}\s*\n([\s\S]+)")
    block = block_pattern.search(contents).group()
    if block[-1] != "\n":
        block += "\n"
    atom_magnetism = []
    atom_symbol = []
    atom_potential = []
    atom_basis = []
    atom_offsite_basis = []
    # atom_mass = []
    atom_block = []
    for i, symbol in enumerate(symbols):
        pattern = re.compile(rf"{symbol}\s*\n({_re_float})\s*\n(\d+)")
        sub_block = pattern.search(block)
        number = int(sub_block.group(2))

        # symbols, magnetism
        sym = [symbol] * number
        atom_mags = [float(sub_block.group(1))] * number
        for j in range(number):
            atom_symbol.append(sym[j])
            atom_potential.append(potential[symbol])
            if basis:
                atom_basis.append(basis[symbol])
            if offsite_basis:
                atom_offsite_basis.append(offsite_basis[symbol])
            # atom_mass.append(masses[j])
            atom_magnetism.append(atom_mags[j])

        if i == ntype - 1:
            lines_pattern = re.compile(
                rf"{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+)\s*\n"
            )
        else:
            lines_pattern = re.compile(
                rf"{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+?)\s*\n\w+\s*\n{_re_float}"
            )
        lines = lines_pattern.search(block)
        for j in [line.split() for line in lines.group(1).split("\n")]:
            atom_block.append(j)
    atom_block = np.array(atom_block)
    atom_magnetism = np.array(atom_magnetism)

    # position
    atom_positions = atom_block[:, 0:3].astype(float)
    natoms = len(atom_positions)

    def _get_index(labels, num):
        index = None
        res = []
        for l in labels:
            if l in atom_block:
                index = np.where(atom_block == l)[-1][0]
        if index is not None:
            res = atom_block[:, index + 1 : index + 1 + num].astype(float)

        return res, index

    # velocity
    v_labels = ["v", "vel", "velocity"]
    atom_vel, v_index = _get_index(v_labels, 3)

    # magnetism
    m_labels = ["mag", "magmom"]
    if "angle1" in atom_block or "angle2" in atom_block:
        import warnings

        warnings.warn(
            "Non-colinear angle-settings are not yet supported for this interface."
        )
    mags, m_index = _get_index(m_labels, 1)
    try:  # non-colinear
        if m_index:
            atom_magnetism = atom_block[:, m_index + 1 : m_index + 4].astype(float)
    except IndexError:  # colinear
        if m_index:
            atom_magnetism = mags

    # to ase
    if atom_pos_type == "Direct":
        atoms = PhonopyAtoms(
            symbols=atom_symbol,
            cell=atom_lattice,
            scaled_positions=atom_positions,
            magnetic_moments=atom_magnetism,
        )
    elif atom_pos_type == "Cartesian":
        atoms = PhonopyAtoms(
            symbols=atom_symbol,
            cell=atom_lattice,
            positions=atom_positions * atom_lattice_scale,
            magnetic_moments=atom_magnetism,
        )

    fd.close()
    return atoms, atom_potential, atom_basis, atom_offsite_basis


# READ ABACUS STRU -END-


#
# write ABACUS STRU
#


def write_abacus(filename, atoms, pps, orbitals=None, abfs=None):
    """Write structure to file."""
    with open(filename, "w") as f:
        f.write(get_abacus_structure(atoms, pps, orbitals, abfs))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pps,
    orbitals,
    abfs=None,
    pre_filename="STRU",
    width=3,
):
    """Write supercells with displacements to files."""
    write_abacus("%s.in" % pre_filename, supercell, pps, orbitals, abfs)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )
        write_abacus(filename, cell, pps, orbitals, abfs)


def get_abacus_structure(atoms, pps, orbitals=None, abfs=None):
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

    if abfs:
        line.append("ABFS_ORBITAL")
        for i in range(len(elements)):
            line.append(f"{abfs[i]}")
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
        line.append(f"{elem}\n{0}\n{numbers[i]}")
        for j in range(index, index + numbers[i]):
            line.append(
                " ".join(_list_elem2str(atoms.scaled_positions[j]))
                + " "
                + "1 1 1"
                + " mag "
                + f"{atoms.magnetic_moments[j]}"
            )
        line.append(empty_line)
        index += numbers[i]

    return "\n".join(line)


#
# set Force
#
def read_abacus_output(filename):
    """Read ABACUS forces from last self-consistency iteration."""
    force = None
    with open(filename, "r") as file:
        for line in file:
            if re.search(r"TOTAL ATOM NUMBER = [0-9]+", line):
                natom = int(re.search("[0-9]+", line).group())
            if re.search(r"TOTAL-FORCE \(eV/Angstrom\)", line):
                force = np.zeros((natom, 3))
                for i in range(4):
                    file.readline()
                for i in range(natom):
                    _, fx, fy, fz = file.readline().split()
                    force[i] = (float(fx), float(fy), float(fz))
    if force is None:
        raise ValueError("Force data not found.")

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
