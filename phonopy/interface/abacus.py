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
    """Read structure information, distance in unit au (bohr)."""
    fd = open(filename, "r")
    contents = fd.read()
    title_str = (
        r"(?:LATTICE_CONSTANT|NUMERICAL_ORBITAL|ABFS_ORBITAL|"
        + r"LATTICE_VECTORS|LATTICE_PARAMETERS|ATOMIC_POSITIONS)"
    )

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
    try:
        atom_potential = dict(zip(symbols, specie_lines[:, 2].tolist()))
    except IndexError:
        atom_potential = None

    # basis
    aim_title = "NUMERICAL_ORBITAL"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    orb_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    orb_lines = orb_pattern.search(contents)
    if orb_lines:
        atom_basis = dict(zip(symbols, orb_lines.group(1).split("\n")))
    else:
        atom_basis = None

    # ABFs basis
    aim_title = "ABFS_ORBITAL"
    aim_title_sub = title_str.replace("|" + aim_title, "")
    abf_pattern = re.compile(rf"{aim_title}\s*\n([\s\S]+?)\s*\n{aim_title_sub}")
    abf_lines = abf_pattern.search(contents)
    if abf_lines:
        atom_offsite_basis = dict(zip(symbols, abf_lines.group(1).split("\n")))
    else:
        atom_offsite_basis = None

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
            # atom_mass.append(masses[j])
            atom_magnetism.append(atom_mags[j])

        if i == ntype - 1:
            lines_pattern = re.compile(
                rf"{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+)\s*\n"
            )
        else:
            lines_pattern = re.compile(
                rf"{symbol}\s*\n{_re_float}\s*\n\d+\s*\n([\s\S]+?)"
                + rf"\s*\n\w+\s*\n{_re_float}"
            )
        lines = lines_pattern.search(block)
        for j in [line.split() for line in lines.group(1).split("\n")]:
            atom_block.append(j)
    atom_block = np.array(atom_block)
    atom_magnetism = np.array(atom_magnetism)

    # position
    atom_positions = atom_block[:, 0:3].astype(float)

    def _get_index(labels, num):
        index = None
        res = []
        for label in labels:
            if label in atom_block:
                index = np.where(atom_block == label)[-1][0]
        if index is not None:
            res = atom_block[:, index + 1 : index + 1 + num].astype(float)

        return res, index

    # magnetism
    m_labels = ["mag", "magmom"]
    if "angle1" in atom_block or "angle2" in atom_block:
        import warnings

        warnings.warn(
            "Non-colinear angle-settings are not yet supported for this interface.",
            stacklevel=2,
        )
    mags, m_index = _get_index(m_labels, 1)
    try:  # non-colinear
        if m_index:
            atom_magnetism = atom_block[:, m_index + 1 : m_index + 4].astype(float)
    except IndexError:  # colinear
        if m_index:
            atom_magnetism = mags

    magnetic_moments = atom_magnetism if atom_magnetism.flatten().any() else None

    # to ase
    if atom_pos_type == "Direct":
        atoms = PhonopyAtoms(
            symbols=atom_symbol,
            cell=atom_lattice,
            scaled_positions=atom_positions,
            magnetic_moments=magnetic_moments,
        )
    elif atom_pos_type == "Cartesian":
        atoms = PhonopyAtoms(
            symbols=atom_symbol,
            cell=atom_lattice,
            positions=atom_positions * atom_lattice_scale,
            magnetic_moments=magnetic_moments,
        )
    else:
        warnings.warn("Only 'Direct' and 'Cartesian' are supported.", stacklevel=2)

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
    orbitals=None,
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

    for _, elem in enumerate(elements):
        line.append(f"{elem}\t{atom_data[symbol_map[elem]][3]}\t{pps[elem]}")
    line.append(empty_line)

    if orbitals:
        line.append("NUMERICAL_ORBITAL")
        for _, elem in enumerate(elements):
            line.append(f"{orbitals[elem]}")
        line.append(empty_line)

    if abfs:
        line.append("ABFS_ORBITAL")
        for _, elem in enumerate(elements):
            line.append(f"{abfs[elem]}")
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
            if atoms.magnetic_moments is not None:
                line_part = (
                    " ".join(_list_elem2str(atoms.scaled_positions[j])) + " 1 1 1"
                )
                # Add the magnetic moments part
                mag_mom = atoms.magnetic_moments[j]
                if isinstance(mag_mom, (list, np.ndarray)):
                    if (
                        len(mag_mom) == 3
                    ):  # Check the three components only if it is a list or numpy array
                        line_part += " mag " + " ".join(map(str, mag_mom))
                else:  # If single value (float), add directly
                    line_part += f" mag {mag_mom}"
                # Append the completed line
                line.append(line_part)
            else:
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
    force = None
    with open(filename, "r") as file:
        for line in file:
            if re.search(r"TOTAL ATOM NUMBER = [0-9]+", line):
                natom = int(re.search("[0-9]+", line).group())
            if re.search(r"TOTAL-FORCE \(eV/Angstrom\)", line):
                force = np.zeros((natom, 3))
                _match_pattern = r"^(\s*)([A-Za-z]*[0-9]+)((\s*[+-]?"
                _match_pattern += r"[0-9]+\.[0-9]+(e[+-][0-9]{2})?){3})(.*)$"
                _match = re.match(_match_pattern, line)
                while not _match:
                    line = file.readline()
                    _match = re.match(_match_pattern, line)
                iatom = 0
                while _match:
                    # print(_match.group(3).split())
                    fx, fy, fz = (_match.group(3).split()[i].strip() for i in range(3))
                    force[iatom] = (float(fx), float(fy), float(fz))
                    iatom += 1
                    if iatom == natom:
                        break
                    else:
                        line = file.readline()
                        _match = re.match(_match_pattern, line)

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


def _list_elem2str(a):
    """Convert type of list element to str."""

    def f_str(x):
        return f"{x:0<12f}"

    return list(map(f_str, a))
