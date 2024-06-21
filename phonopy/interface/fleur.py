"""Fleur calculator interface."""

# Copyright (C) 2021 Alexander Neukirchen
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

import itertools
import math
import sys

import numpy as np

from phonopy.file_IO import collect_forces
from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    sort_positions_by_symbols,
)
from phonopy.structure.atoms import PhonopyAtoms, symbol_map


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "1 #"
    is_parsed = True
    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        f = open(filename)
        fleur_forces = collect_forces(f, num_atoms, hook, [0, 1, 2], word="force")
        if check_forces(fleur_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                fleur_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(fleur_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_fleur(filename):
    """Read crystal structure."""
    fleur_in = FleurIn(open(filename).readlines())
    tags = fleur_in.get_variables()
    avec = [tags["avec"][i] for i in range(3)]
    speci = tags["atoms"]["speci"]
    symbols = [
        list(symbol_map.keys())[list(symbol_map.values()).index(math.floor(float(x)))]
        for x in speci
    ]
    numbers = [math.floor(float(x)) for x in speci]

    for i, n in enumerate(numbers):
        if n == 0:
            for j in range(1, 119):
                if j not in numbers:
                    numbers[i] = j
                    break
    pos_all = []
    num_all = []
    for _, pos in zip(numbers, tags["atoms"]["positions"]):
        pos_all += pos
    num_all = [symbol_map[s] for s in symbols]

    return (
        PhonopyAtoms(numbers=num_all, cell=avec, scaled_positions=pos_all),
        speci,
        fleur_in._restlines,
    )


def write_fleur(filename, cell, speci, N, restlines):
    """Write crystal structure to file."""
    f = open(filename, "w")
    f.write(get_fleur_structure(cell, speci, N, restlines))


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    speci,
    N,
    restlines,
    pre_filename="supercell",
    width=3,
):
    """Write supercells with displacements to files."""
    write_fleur("%s.in" % pre_filename, supercell, speci, N, restlines)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=pre_filename, width=width
        )
        write_fleur(filename, cell, speci, N, restlines)


def get_fleur_structure(cell, speci, N, restlines):
    """Return Fleur structure in text."""
    lattice = cell.get_cell()
    (num_atoms, symbols, scaled_positions, sort_list) = sort_positions_by_symbols(
        cell.get_chemical_symbols(), cell.get_scaled_positions()
    )
    specilong = list(
        itertools.chain.from_iterable(itertools.repeat(x, N) for x in speci)
    )
    lines = restlines[0] + "\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "1.0 \n"
    lines += "1.0 1.0 1.0 \n \n"
    lines += str(sum(num_atoms)) + "\n"
    for i in range(sum(num_atoms)):
        lines += specilong[i].ljust(6)
        currentpos = str(scaled_positions[i]).replace("[", "")
        currentpos = currentpos.replace("]", "").split()

        for j in range(3):
            lines += " " + "{:.10f}".format(float(currentpos[j]))
        if i < sum(num_atoms) - 1:
            lines += "\n"
    for x in range(1, len(restlines)):
        if len(restlines[x]) == 0:
            lines += "\n"
            continue
        lines += restlines[x]
        if x != len(restlines) - 1:
            lines += "\n"
    return lines


class FleurIn:
    """Class to generate Fleur crystal structure."""

    def __init__(self, lines):
        """Init method."""
        self._set_methods = {"a1": self._set_avec, "atoms": self._set_atoms}
        self._tags = {"atoms": None, "a1": None}
        self._lines = lines[:]
        self._restlines = []
        self._collect()

    def get_variables(self):
        """Return variables."""
        return self._tags

    def _collect(self):
        firstline = True
        while True:
            try:
                line_str = self._lines.pop(0).strip()
            except IndexError:
                break

            if firstline is True:
                self._restlines.append(line_str)
                firstline = False
                continue
            if len(line_str) == 0:
                continue
            if line_str[0] == "!":
                continue

            elems = line_str.split()
            if elems[-1] in self._set_methods:
                self._lines.insert(0, line_str)
                self._set_methods[elems[-1]]()

    def _set_atoms(self):
        natoms = int(self._lines.pop(0).split()[0][0])
        speci = []
        positions = []
        positions1 = []
        for _ in range(natoms):
            currentline = self._lines.pop(0).split()
            speci.append(currentline[0])
            currentspeci = [float(x) for x in currentline[1:4]]
            positions1.append(currentspeci)
        positions.append(positions1)

        factors = [1.0, 1.0, 1.0]
        shifts = [0.0, 0.0, 0.0]

        for count, line in enumerate(self._lines):
            if "&factor" in line:
                currentline = self._lines.pop(count).split()
                factors = [
                    float(currentline[1]),
                    float(currentline[2]),
                    float(currentline[3]),
                ]

        for count, line in enumerate(self._lines):
            if "&shift" in line:
                currentline = self._lines.pop(count).split()
                shifts = [
                    float(currentline[1]),
                    float(currentline[2]),
                    float(currentline[3]),
                ]

        for x in positions[0]:
            x[0] = x[0] / factors[0] + shifts[0]
            x[1] = x[1] / factors[1] + shifts[1]
            x[2] = x[2] / factors[2] + shifts[2]

        self._tags["atoms"] = {"speci": speci, "positions": positions}
        for j in range(len(self._lines)):
            self._restlines.append(self._lines[j])

    def _set_avec(self):
        avec = []
        for _ in range(3):
            avec.append([float(x) for x in self._lines.pop(0).split()[:3]])
        lattcon = float(self._lines.pop(0).split()[0])
        scale = [float(x) for x in self._lines.pop(0).split()[:3]]
        for j in range(3):
            for k in range(3):
                if scale[k] < 0:
                    scale[k] = np.sqrt(np.abs(scale[k]))
                if scale[k] == 0.0:
                    scale[k] = 1.0
                avec[j][k] = lattcon * avec[j][k] * scale[k]
        self._tags["avec"] = avec


if __name__ == "__main__":
    from phonopy.structure.symmetry import Symmetry

    cell, speci, restlines = read_fleur(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
    print(get_fleur_structure(cell, speci, 1, restlines))
