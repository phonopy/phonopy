"""SIESTA calculator interface."""

# Copyright (C) 2015 Henrique Pereira Coutada Miranda
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

import numpy as np

from phonopy.file_IO import iter_collect_forces
from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import Bohr


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = ""  # Just for skipping the first line
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))
        siesta_forces = iter_collect_forces(
            filename, num_atoms, hook, [1, 2, 3], word=""
        )
        if check_forces(siesta_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                siesta_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(siesta_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_siesta(filename):
    """Read crystal structure."""
    siesta_in = SiestaIn(open(filename).read())
    numbers = siesta_in._tags["atomicnumbers"]
    alat = siesta_in._tags["latticeconstant"]
    lattice = siesta_in._tags["latticevectors"]
    positions = siesta_in._tags["atomiccoordinates"]
    atypes = siesta_in._tags["chemicalspecieslabel"]
    cell = PhonopyAtoms(numbers=numbers, cell=lattice, scaled_positions=positions)

    coordformat = siesta_in._tags["atomiccoordinatesformat"]
    if coordformat == "fractional" or coordformat == "scaledbylatticevectors":
        cell.set_scaled_positions(positions)
    elif coordformat == "scaledcartesian":
        cell.set_positions(np.array(positions) * alat)
    elif coordformat == "notscaledcartesianang" or coordformat == "ang":
        cell.set_positions(np.array(positions) / Bohr)
    elif coordformat == "notscaledcartesianbohr" or coordformat == "bohr":
        cell.set_positions(np.array(positions))
    else:
        print(
            "The format %s for the AtomicCoordinatesFormat is not "
            "implemented." % coordformat
        )
        sys.exit(1)

    return cell, atypes


def write_siesta(filename, cell, atypes):
    """Write cell to file."""
    with open(filename, "w") as w:
        w.write(get_siesta_structure(cell, atypes))


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, atypes, pre_filename="supercell", width=3
):
    """Write supercells with displacements to files."""
    write_siesta("%s.fdf" % pre_filename, supercell, atypes)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.fdf".format(
            i, pre_filename=pre_filename, width=width
        )
        write_siesta(filename, cell, atypes)


def get_siesta_structure(cell, atypes):
    """Return SIESTA structure in text."""
    lattice = cell.get_cell()
    positions = cell.get_scaled_positions()
    chemical_symbols = cell.get_chemical_symbols()

    lines = ""

    lines += "NumberOfAtoms %d\n\n" % len(positions)

    lines += "%block LatticeVectors\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "%endblock LatticeVectors\n\n"

    lines += "AtomicCoordinatesFormat  Fractional\n\n"

    lines += "LatticeConstant 1.0 Bohr\n\n"

    lines += "%block AtomicCoordinatesAndAtomicSpecies\n"
    for pos, i in zip(positions, chemical_symbols):
        lines += ("%21.16lf" * 3 + " %d\n") % tuple(pos.tolist() + [atypes[i]])
    lines += "%endblock AtomicCoordinatesAndAtomicSpecies\n"

    return lines


class SiestaIn:
    """Class to create SIESTA input file."""

    _num_regex = r"([+-]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    _tags = {
        "latticeconstant": 1.0,
        "latticeconstantunit": None,
        "chemicalspecieslabel": None,
        "atomiccoordinatesformat": None,
        "atomicnumbers": None,
        "atomicspecies": None,
        "atomiccoordinates": None,
    }

    def __init__(self, lines):
        """Init method."""
        self._collect(lines)

    def _collect(self, lines):
        """Collect values.

        This routine reads the following from the Siesta file:
        - atomic positions
        - cell_parameters
        - atomic_species

        """
        for tag, value, unit in re.findall(
            r"([\.A-Za-z]+)\s+%s\s+([A-Za-z]+)?" % self._num_regex, lines
        ):
            tag = tag.lower()
            unit = unit.lower()
            if tag == "latticeconstant":
                self._tags["latticeconstantunit"] = unit.capitalize()
                if unit == "ang":
                    self._tags[tag] = float(value) / Bohr
                elif unit == "bohr":
                    self._tags[tag] = float(value)
                else:
                    raise ValueError("Unknown LatticeConstant unit: {}".format(unit))

        for tag, value in re.findall(r"([\.A-Za-z]+)[ \t]+([a-zA-Z]+)", lines):
            tag = tag.replace("_", "").lower()
            if tag == "atomiccoordinatesformat":
                self._tags[tag] = value.strip().lower()

        # check if the necessary tags are present
        self._check_present("atomiccoordinatesformat")
        acell = self._tags["latticeconstant"]

        # capture the blocks
        blocks = re.findall(
            r"%block\s+([A-Za-z_]+)\s*\n((?:.+\n)+?(?=(?:\s+)?%endblock))",
            lines,
            re.MULTILINE,
        )
        for tag, block in blocks:
            tag = tag.replace("_", "").lower()
            if tag == "chemicalspecieslabel":
                block_array = block.split("\n")[:-1]
                self._tags["atomicnumbers"] = dict(
                    [map(int, species.split()[:2]) for species in block_array]
                )
                self._tags[tag] = dict(
                    [
                        (lambda x: (x[2], int(x[0])))(species.split())
                        for species in block_array
                    ]
                )
            elif tag == "latticevectors":
                self._tags[tag] = [
                    [float(v) * acell for v in vector.split()]
                    for vector in block.split("\n")[:3]
                ]
            elif tag == "atomiccoordinatesandatomicspecies":
                block_array = block.split("\n")[:-1]
                self._tags["atomiccoordinates"] = [
                    [float(x) for x in atom.split()[:3]] for atom in block_array
                ]
                self._tags["atomicspecies"] = [
                    int(atom.split()[3]) for atom in block_array
                ]

        # check if the block are present
        self._check_present("atomicspecies")
        self._check_present("atomiccoordinates")
        self._check_present("latticevectors")
        self._check_present("chemicalspecieslabel")

        # translate the atomicspecies to atomic numbers
        self._tags["atomicnumbers"] = [
            self._tags["atomicnumbers"][atype] for atype in self._tags["atomicspecies"]
        ]

    def _check_present(self, tag):
        if not self._tags[tag]:
            print("%s not present" % tag)
            sys.exit(1)

    def __str__(self):
        """Return tags."""
        return self._tags


if __name__ == "__main__":
    from phonopy.structure.symmetry import Symmetry

    cell, atypes = read_siesta(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
    print(get_siesta_structure(cell, atypes))
