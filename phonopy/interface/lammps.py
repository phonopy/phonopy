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

import re

import numpy as np

from phonopy.structure.atoms import PhonopyAtoms


class LammpsStructureParser:
    """Class to create LAMMPS input structure file."""

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
        self._atom_ids = None
        self._atom_numbers = None
        self._atom_positions = None
        self._cell = None

    @property
    def cell(self):
        """Return parsed cell."""
        return self._cell

    def parse(self, lines):
        """Parse LAMMPS structure file."""
        for i, line in enumerate(lines):
            if "Atoms" in line:
                self._parse_Atoms(lines[(i + 1) :])
                break
            for key, (regex, method_name) in self._header_hooks.items():
                if regex.search(line):
                    getattr(self, method_name)(key, regex.sub("", line))
                    break
        lattice = np.zeros((3, 3), dtype="double", order="C")
        tag = self._header_tags
        lattice[0, 0] = tag["xlo_xhi"][1] - tag["xlo_xhi"][0]
        lattice[1, 1] = tag["ylo_yhi"][1] - tag["ylo_yhi"][0]
        lattice[2, 2] = tag["zlo_zhi"][1] - tag["zlo_zhi"][0]
        lattice[1, 0] = tag["xy_xz_yz"][0]  # xy
        lattice[2, 0] = tag["xy_xz_yz"][1]  # xz
        lattice[2, 1] = tag["xy_xz_yz"][2]  # yz
        self._cell = PhonopyAtoms(
            cell=lattice, positions=self._atom_positions, numbers=self._atom_numbers
        )

    def _parse_Atoms(self, lines):
        positions = np.zeros((self._header_tags["atoms"], 3), dtype="double", order="C")
        numbers = np.zeros(self._header_tags["atoms"], dtype="int_")
        ids = np.zeros(self._header_tags["atoms"], dtype="int_")
        count = 0
        for line in lines:
            _line = line.split("#")[0].strip()
            if _line == "":
                continue
            ary = _line.split()
            ids[count] = int(ary[0])
            numbers[count] = int(ary[1])
            positions[count] = [float(v) for v in ary[2:5]]
            count += 1
        self._atom_ids = ids
        self._atom_numbers = numbers
        self._atom_positions = positions

    def _set_xlo_xhi(self, key, line):
        self._header_tags[key] = np.array(
            [float(v) for v in line.split("#")[0].split()]
        )

    def _set_number_of_atoms(self, key, line):
        self._header_tags[key] = int(line.split("#")[0])
