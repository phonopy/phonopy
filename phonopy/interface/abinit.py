"""Abinit calculator interface."""

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

import io
import os
import sys
from typing import Union

import numpy as np

from phonopy.cui.settings import fracval
from phonopy.file_IO import collect_forces
from phonopy.interface.vasp import (
    check_forces,
    get_drift_forces,
    get_scaled_positions_lines,
)
from phonopy.structure.atoms import PhonopyAtoms, atom_data
from phonopy.units import Bohr


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "cartesian forces (eV/Angstrom)"
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        f = open(filename)
        abinit_forces = collect_forces(f, num_atoms, hook, [1, 2, 3])
        if check_forces(abinit_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                abinit_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(abinit_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_abinit(filename: Union[str, bytes, os.PathLike, io.IOBase]):
    """Read crystal structure."""
    if isinstance(filename, io.IOBase):
        abinit_in = AbinitIn(filename.readlines())
    else:
        with open(filename) as f:
            abinit_in = AbinitIn(f.readlines())
    tags = abinit_in.get_variables()
    acell = tags["acell"]
    rprim = tags["rprim"].T
    scalecart = tags["scalecart"]
    lattice = rprim * acell
    if scalecart is not None:
        for i in range(3):
            lattice[i] *= scalecart[i]

    if tags["xcart"] is not None:
        pos_bohr = np.transpose(tags["xcart"])
        positions = np.dot(np.linalg.inv(lattice), pos_bohr).T
    elif tags["xangst"] is not None:
        pos_bohr = np.transpose(tags["xangst"]) / Bohr
        positions = np.dot(np.linalg.inv(lattice), pos_bohr).T
    elif tags["xred"] is not None:
        positions = tags["xred"]

    numbers = [tags["znucl"][x - 1] for x in tags["typat"]]
    symbols = [atom_data[n][1] for n in numbers]

    return PhonopyAtoms(symbols=symbols, cell=lattice.T, scaled_positions=positions)


def write_abinit(filename, cell):
    """Write cell to file."""
    with open(filename, "w") as f:
        f.write(get_abinit_structure(cell))


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, pre_filename="supercell", width=3
):
    """Write supercells with displacements to files."""
    write_abinit("%s.in" % pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=pre_filename, width=width
        )
        write_abinit(filename, cell)


def get_abinit_structure(cell: PhonopyAtoms):
    """Return abinit structure in text."""
    znucl = []
    numbers = cell.numbers
    for n in numbers:
        if n not in znucl:
            znucl.append(n)
    typat = []
    for n in numbers:
        typat.append(znucl.index(n) + 1)

    lines = ""
    lines += "natom %d\n" % len(numbers)
    lines += "typat\n"
    lines += (" %d" * len(typat) + "\n") % tuple(typat)
    lines += "ntypat %d\n" % len(znucl)
    lines += ("znucl" + " %d" * len(znucl) + "\n") % tuple(znucl)
    lines += "acell 1 1 1\n"
    lines += "rprim\n"
    lines += ((" % 20.16f" * 3 + "\n") * 3) % tuple(cell.cell.ravel())
    lines += "xred\n"
    lines += get_scaled_positions_lines(cell.scaled_positions)

    return lines


class AbinitIn:
    """Class to create Abinit input file."""

    def __init__(self, lines):
        """Init method."""
        self._set_methods = {
            "acell": self._set_acell,
            "natom": self._set_natom,
            "ntypat": self._set_ntypat,
            "rprim": self._set_rprim,
            "typat": self._set_typat,
            "scalecart": self._set_scalecart,
            "xangst": self._set_xangst,
            "xcart": self._set_xcart,
            "xred": self._set_xred,
            "znucl": self._set_znucl,
        }
        self._tags = {
            "acell": None,
            "natom": None,
            "ntypat": None,
            "rprim": None,
            "typat": None,
            "scalecart": None,
            "xangst": None,
            "xcart": None,
            "xred": None,
            "znucl": None,
        }

        self._values = None
        self._collect(lines)

    def get_variables(self):
        """Return tags."""
        return self._tags

    def _collect(self, lines):
        elements = {}
        tag = None
        for line_tmp in lines:
            line = line_tmp.replace("!", "#").split("#")[0]
            for val in [x.lower() for x in line.split()]:
                if val in self._set_methods:
                    tag = val
                    elements[tag] = []
                elif tag is not None:
                    elements[tag].append(val)

        for tag in ["natom", "ntypat"]:
            if tag not in elements:
                print("%s is not found in the input file." % tag)
                sys.exit(1)

        for tag in elements:
            self._values = elements[tag]
            if tag == "natom" or tag == "ntypat":
                self._set_methods[tag]()

        for tag in elements:
            self._values = elements[tag]
            if tag != "natom" and tag != "ntypat":
                self._set_methods[tag]()

    def _get_numerical_values(self, char_string, num_type="float"):
        m = 1

        if "*" in char_string:
            m = int(char_string.split("*")[0])
            str_val = char_string.split("*")[1]
        else:
            m = 1
            str_val = char_string

        if num_type == "float":
            a = fracval(str_val)
        else:
            a = int(str_val)

        return [a] * m

    def _set_acell(self):
        acell = []
        for val in self._values:
            if len(acell) >= 3:
                if len(val) >= 6:
                    if val[:6] == "angstr":
                        for i in range(3):
                            acell[i] /= Bohr
                break

            acell += self._get_numerical_values(val)

        self._tags["acell"] = acell[:3]

    def _set_natom(self):
        self._tags["natom"] = int(self._values[0])

    def _set_ntypat(self):
        self._tags["ntypat"] = int(self._values[0])

    def _set_rprim(self):
        rprim = []
        for val in self._values:
            rprim += self._get_numerical_values(val)
            if len(rprim) >= 9:
                break

        self._tags["rprim"] = np.reshape(rprim[:9], (3, 3))

    def _set_scalecart(self):
        scalecart = []
        for val in self._values:
            scalecart += self._get_numerical_values(val)
            if len(scalecart) >= 3:
                break

        self._tags["scalecart"] = np.array(scalecart[:3])

    def _set_typat(self):
        typat = []
        natom = self._tags["natom"]
        for val in self._values:
            typat += self._get_numerical_values(val, num_type="int")
            if len(typat) >= natom:
                break

        self._tags["typat"] = typat[:natom]

    def _set_xangst(self):
        self._set_x_tags("xangst")

    def _set_xcart(self):
        self._set_x_tags("xcart")

    def _set_xred(self):
        self._set_x_tags("xred")

    def _set_x_tags(self, tagname):
        xtag = []
        natom = self._tags["natom"]
        for val in self._values:
            xtag += self._get_numerical_values(val)
            if len(xtag) >= natom * 3:
                break

        self._tags[tagname] = np.reshape(xtag[: natom * 3], (-1, 3))

    def _set_znucl(self):
        znucl = []
        ntypat = self._tags["ntypat"]
        for val in self._values:
            znucl += self._get_numerical_values(val, num_type="int")
            if len(znucl) >= ntypat:
                break

        self._tags["znucl"] = znucl[:ntypat]
