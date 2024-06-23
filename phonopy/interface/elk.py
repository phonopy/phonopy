"""Elk calculator interface."""

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
from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    get_scaled_positions_lines,
    sort_positions_by_symbols,
)
from phonopy.structure.atoms import PhonopyAtoms, symbol_map


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "Forces :"
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        f = open(filename)
        elk_forces = collect_forces(f, num_atoms, hook, [3, 4, 5], word="total force")
        if check_forces(elk_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                elk_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(elk_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_elk(filename):
    """Read crystal structure."""
    elk_in = ElkIn(open(filename).readlines())
    tags = elk_in.get_variables()
    avec = [tags["scale"][i] * np.array(tags["avec"][i]) for i in range(3)]
    spfnames = tags["atoms"]["spfnames"]
    symbols = [x.split(".")[0] for x in spfnames]
    numbers = []
    for s in symbols:
        if s in symbols:
            numbers.append(symbol_map[s])
        else:
            numbers.append(0)

    for i, n in enumerate(numbers):
        if n == 0:
            for j in range(1, 119):
                if j not in numbers:
                    numbers[i] = j
                    break
    pos_all = []
    num_all = []
    for num, pos in zip(numbers, tags["atoms"]["positions"]):
        pos_all += pos
        num_all += [num] * len(pos)

    return PhonopyAtoms(numbers=num_all, cell=avec, scaled_positions=pos_all), spfnames


def write_elk(filename, cell, sp_filenames):
    """Write cell to file."""
    f = open(filename, "w")
    f.write(get_elk_structure(cell, sp_filenames))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    sp_filenames,
    pre_filename="supercell",
    width=3,
):
    """Write supercells with displacements to files."""
    write_elk("%s.in" % pre_filename, supercell, sp_filenames)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=pre_filename, width=width
        )
        write_elk(filename, cell, sp_filenames)


def get_elk_structure(cell, sp_filenames=None):
    """Return Elk structure in text."""
    lattice = cell.get_cell()
    (num_atoms, symbols, scaled_positions, sort_list) = sort_positions_by_symbols(
        cell.get_chemical_symbols(), cell.get_scaled_positions()
    )

    if sp_filenames is None:
        spfnames = [s + ".in" for s in symbols]
    else:
        spfnames = sp_filenames

    lines = ""
    lines += "avec\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "atoms\n"
    n_pos = 0
    lines += " %d\n" % len(num_atoms)
    for i, (n, s) in enumerate(zip(num_atoms, spfnames)):
        lines += " '%s'\n" % s
        lines += " %d\n" % n
        lines += get_scaled_positions_lines(scaled_positions[n_pos : (n_pos + n)])
        if i < len(num_atoms) - 1:
            lines += "\n"
        n_pos += n

    return lines


class ElkIn:
    """Class to create Elk input file."""

    def __init__(self, lines):
        """Init method."""
        self._set_methods = {
            "atoms": self._set_atoms,
            "avec": self._set_avec,
            "scale": self._set_scale,
            "scale1": self._set_scale1,
            "scale2": self._set_scale2,
            "scale3": self._set_scale3,
        }
        self._tags = {"atoms": None, "avec": None, "scale": [1.0, 1.0, 1.0]}
        self._lines = lines[:]
        self._collect()

    def get_variables(self):
        """Return tags."""
        return self._tags

    def _collect(self):
        while True:
            try:
                line_str = self._lines.pop(0).strip()
            except IndexError:
                break

            if len(line_str) == 0:
                continue
            if line_str[0] == "!":
                continue

            elems = line_str.split()
            if elems[0] in self._set_methods:
                self._set_methods[elems[0]]()

    def _set_atoms(self):
        nspecies = int(self._lines.pop(0).split()[0])
        spfnames = []
        positions = []
        for _ in range(nspecies):
            spfnames.append(self._lines.pop(0).split()[0].strip("'"))
            natoms = int(self._lines.pop(0).split()[0])
            pos_sp = []
            for _ in range(natoms):
                pos_sp.append([float(x) for x in self._lines.pop(0).split()[:3]])
            positions.append(pos_sp)

        self._tags["atoms"] = {"spfnames": spfnames, "positions": positions}

    def _set_avec(self):
        avec = []
        for _ in range(3):
            avec.append([float(x) for x in self._lines.pop(0).split()[:3]])
        self._tags["avec"] = avec

    def _set_scale(self):
        scale = float(self._lines.pop(0).split()[0])
        for i in range(3):
            self._tags["scale"][i] = scale

    def _set_scale1(self):
        self._tags["scale"][0] = float(self._lines.pop(0).split()[0])

    def _set_scale2(self):
        self._tags["scale"][1] = float(self._lines.pop(0).split()[0])

    def _set_scale3(self):
        self._tags["scale"][2] = float(self._lines.pop(0).split()[0])


if __name__ == "__main__":
    from phonopy.structure.symmetry import Symmetry

    cell, sp_filenames = read_elk(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
    print(get_elk_structure(cell, sp_filenames=sp_filenames))
