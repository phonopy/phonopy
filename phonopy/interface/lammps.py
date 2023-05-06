"""LAMMPS calculator interface."""
# Copyright (C) 2023 Atsushi Togo
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
from __future__ import annotations

import io
import os
import re
from typing import Union

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_cell_matrix_from_lattice


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    pre_filename="supercell",
    width=3,
):
    """Write supercells with displacements to files."""
    write_lammps(pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements):
        filename = f"{pre_filename}-{i:0{width}d}"
        write_lammps(filename, cell)


def write_lammps(filename, cell):
    """Write LAMMPS structure to file."""
    with open(filename, "w") as w:
        w.write("\n".join(LammpsStructureDumper(cell).get_lines()))


class LammpsStructureDumper:
    """Class to create LAMMPS input structure file.

    LAMMPS requires the basis vectors to be defined in the following way

    a = (a_x 0 0)
    b = (b_x b_y 0)
    c = (c_x c_y c_z)

    So the ``cell.cell`` has to be rotated to match this style. The coordinates
    of atoms have to be given in Cartesian coordinates with respect to this
    set of basis vectors.

    """

    def __init__(self, cell: PhonopyAtoms):
        """Init method."""
        lattice = get_cell_matrix_from_lattice(cell.cell)
        lmps_cell = PhonopyAtoms(
            cell=lattice, scaled_positions=cell.scaled_positions, numbers=cell.numbers
        )
        self._run(lmps_cell)

    def get_lines(self) -> list[str]:
        """Return LAMMPS structure str lines."""
        return self._lines

    def _run(self, cell: PhonopyAtoms):
        unums, uids = np.unique(cell.numbers, return_index=True)
        usyms = [cell.symbols[i] for i in uids]
        num_map = {n: i for i, n in enumerate(unums)}

        lines = ["#", ""]
        lines.append(f"{len(cell.numbers)} atoms")
        lines.append(f"{len(np.unique(cell.numbers))} atom types")
        lines.append("")
        lines.append(f"0.0 {cell.cell[0, 0]} xlo xhi")
        lines.append(f"0.0 {cell.cell[1, 1]} ylo yhi")
        lines.append(f"0.0 {cell.cell[2, 2]} zlo zhi")
        lines.append("")
        lines.append(f"{cell.cell[1, 0]} {cell.cell[2, 0]} {cell.cell[2, 1]} xy xz yz")
        lines.append("")
        lines.append("")
        lines.append("Atom Type Labels")
        lines.append("")
        for i, symbol in enumerate(usyms):
            lines.append(f"{i + 1} {symbol}")
        lines.append("")
        lines.append("")
        lines.append("Atoms")
        lines.append("")
        for i, (position, number) in enumerate(zip(cell.positions, cell.numbers)):
            pos_str = f"{position[0]} {position[1]} {position[2]}"
            lines.append(f"{i + 1} {usyms[num_map[number]]} {pos_str}")
        self._lines = lines


class LammpsStructureLoader:
    """Class to load LAMMPS input structure file.

    lmps = LammpsStructureLoader()
    lmps.parse(lines)
    cell: PhonopyAtoms = lmps.cell

    """

    _header_hooks = {
        "xlo_xhi": (re.compile(r"xlo\s+xhi"), "_set_xlo_xhi"),
        "ylo_yhi": (re.compile(r"ylo\s+yhi"), "_set_xlo_xhi"),
        "zlo_zhi": (re.compile(r"zlo\s+zhi"), "_set_xlo_xhi"),
        "xy_xz_yz": (re.compile(r"xy\s+xz\s+yz"), "_set_xlo_xhi"),
        "atoms": (re.compile("atoms"), "_set_number_of_atoms"),
        "atom_types": (re.compile(r"atom\s+types"), "_set_number_of_atoms"),
    }

    def __init__(self):
        """Init method."""
        self._header_tags = {}
        self._atom_type_labels = {}
        self._atom_ids = None
        self._atom_labels = None
        self._atom_positions = None
        self._cell = None

    @property
    def cell(self) -> PhonopyAtoms:
        """Return parsed cell."""
        return self._cell

    def load(self, fp: Union[str, bytes, os.PathLike, io.IOBase]):
        """Load and parse LAMMPS structure file.

        Parameters
        ----------
        fp : filename or stream
            filename or stream.

        """
        if isinstance(fp, io.IOBase):
            self._parse(fp.readlines())
        else:
            with open(fp) as f:
                self._parse(f.readlines())
        return self

    def _parse(self, lines):
        """Parse LAMMPS structure file."""
        re_ATL = re.compile(r"Atom\s+Type\s+Labels")
        for line in lines:
            for key, (regex, method_name) in self._header_hooks.items():
                if regex.search(line):
                    getattr(self, method_name)(key, regex.sub("", line))
                    break

        i = 0
        while True:
            if i > len(lines):
                break
            line = lines[i]
            # Hook "Atom Type Labels"
            if re_ATL.search(line):
                i += self._parse_AtomTypeLabels(lines[(i + 1) :])
                continue
            # Hook "Atoms", this must be after "Atom Type Labels".
            if "Atoms" == line.split("#")[0].strip():
                self._parse_Atoms(lines[(i + 1) :])
                break
            i += 1

        lattice = np.zeros((3, 3), dtype="double", order="C")
        tag = self._header_tags
        lattice[0, 0] = tag["xlo_xhi"][1] - tag["xlo_xhi"][0]
        lattice[1, 1] = tag["ylo_yhi"][1] - tag["ylo_yhi"][0]
        lattice[2, 2] = tag["zlo_zhi"][1] - tag["zlo_zhi"][0]
        lattice[1, 0] = tag["xy_xz_yz"][0]  # xy
        lattice[2, 0] = tag["xy_xz_yz"][1]  # xz
        lattice[2, 1] = tag["xy_xz_yz"][2]  # yz

        if self._atom_type_labels:
            self._cell = PhonopyAtoms(
                cell=lattice, positions=self._atom_positions, symbols=self._atom_labels
            )
        else:
            self._cell = PhonopyAtoms(
                cell=lattice, positions=self._atom_positions, numbers=self._atom_labels
            )

    def _parse_AtomTypeLabels(self, lines):
        num_types = 0
        for i, line in enumerate(lines):
            _line = line.split("#")[0].strip()
            if _line == "":
                continue
            ary = _line.split()
            self._atom_type_labels[ary[1]] = int(ary[0])
            num_types += 1
            if num_types == self._header_tags["atom_types"]:
                break
        assert num_types == self._header_tags["atom_types"]
        return i + 1

    def _parse_Atoms(self, lines):
        positions = np.zeros((self._header_tags["atoms"], 3), dtype="double", order="C")
        if self._atom_type_labels:
            lables = []
        else:
            lables = np.zeros(self._header_tags["atoms"], dtype="int_")
        ids = np.zeros(self._header_tags["atoms"], dtype="int_")
        num_atoms = 0
        for line in lines:
            _line = line.split("#")[0].strip()
            if _line == "":
                continue
            ary = _line.split()
            ids[num_atoms] = int(ary[0])
            if self._atom_type_labels:
                assert ary[1] in self._atom_type_labels
                lables.append(ary[1])
            else:
                lables[num_atoms] = int(ary[1])
            positions[num_atoms] = [float(v) for v in ary[2:5]]
            num_atoms += 1
            if num_atoms == self._header_tags["atoms"]:
                break
        assert (
            num_atoms == self._header_tags["atoms"]
        ), f'{num_atoms} != {self._header_tags["atoms"]}'
        self._atom_ids = ids
        self._atom_labels = lables
        self._atom_positions = positions

    def _set_xlo_xhi(self, key, line):
        self._header_tags[key] = np.array(
            [float(v) for v in line.split("#")[0].split()]
        )

    def _set_number_of_atoms(self, key, line):
        self._header_tags[key] = int(line.split("#")[0])
