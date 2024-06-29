"""CASTEP calculator interface."""

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

import sys

import numpy as np

from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry


# Castep output contains blank lines surrounded by astericks which should be ignored.
# This is not implemented in collect_forces() function from file_IO module.
# Below is the reimplementation of the collect_forces() function for Castep with
# only one new variable skipafterhook. Setting this variable to zero (default value)
# is equivalent to original collect_forces() function.
def collect_forces_castep(f, num_atom, hook, force_pos, word=None, skipafterhook=0):
    """Collect forces from CASTEP output."""
    for line in f:
        if hook in line:
            break
    if skipafterhook > 0:
        for _ in range(skipafterhook):
            f.readline()

    forces = []
    for line in f:
        if line.strip() == "":
            continue
        if word is not None:
            if word not in line:
                continue

        elems = line.split()
        if len(elems) > force_pos[2]:
            try:
                forces.append([float(elems[i]) for i in force_pos])
            except ValueError:
                forces = []
                break
        else:
            return False

        if len(forces) == num_atom:
            break

    return forces


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    hook = "Cartesian components (eV/A)"
    # skipafterhook = 3
    is_parsed = True
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        f = open(filename)
        castep_forces = collect_forces_castep(
            f, num_atoms, hook, [3, 4, 5], skipafterhook=3
        )

        if check_forces(castep_forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(
                castep_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(castep_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_castep(filename):
    """Read crystal structure."""
    f_castep = open(filename)
    castep_in = CastepIn(f_castep.readlines())
    f_castep.close()
    tags = castep_in.get_tags()
    # 1st stage is to create Atoms object ignoring Spin polarization. General case.
    cell = PhonopyAtoms(
        cell=tags["lattice_vectors"],
        symbols=tags["atomic_species"],
        scaled_positions=tags["coordinates"],
    )
    # Analyse spin states and add data to Atoms instance "cell" if ones exist
    magmoms = tags["magnetic_moments"]
    if magmoms is not None:
        # Print out symmetry information for magnetic cases
        # Original code from structure/symmetry.py
        symmetry = Symmetry(cell, symprec=1e-5)
        print(
            "CASTEP-interface: Magnetic structure, "
            "number of operations without spin: %d"
            % len(symmetry.get_symmetry_operations()["rotations"])
        )
        print(
            "CASTEP-interface: Spacegroup without spin: %s"
            % symmetry.get_international_table()
        )

        cell.set_magnetic_moments(magmoms)
        symmetry = Symmetry(cell, symprec=1e-5)
        print(
            "CASTEP-interface: Magnetic structure, number of operations with spin: %d"
            % len(symmetry.get_symmetry_operations()["rotations"])
        )
        print("")

    return cell


def write_castep(filename, cell):
    """Write cell to file."""
    with open(filename, "w") as f:
        f.write(get_castep_structure(cell))


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, pre_filename="supercell", width=3
):
    """Write supercells with displacements to files."""
    write_castep("%s.cell" % pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.cell".format(
            i, pre_filename=pre_filename, width=width
        )
        write_castep(filename, cell)


def get_castep_structure(cell):
    """Return CASTEP structure in text."""
    lines = ""
    lines += "%BLOCK LATTICE_CART\n"
    lines += ((" % 20.16f" * 3 + "\n") * 3) % tuple(cell.get_cell().ravel())
    lines += "%ENDBLOCK LATTICE_CART\n\n"
    lines += "%BLOCK POSITIONS_FRAC\n"
    magmoms = cell.get_magnetic_moments()

    for i in range(len(cell.get_chemical_symbols())):
        atpos = "".join("% 12.10f " % ap for ap in cell.get_scaled_positions()[i])
        # Spin polarized case
        if (magmoms is not None) and (magmoms[i] != 0.0):
            lines += "".join(
                "%2s %s  spin=% 5.2f\n"
                % (cell.get_chemical_symbols()[i], atpos, magmoms[i])
            )
        # No spin ordering
        else:
            lines += "".join("%2s %s\n" % (cell.get_chemical_symbols()[i], atpos))
    lines += "%ENDBLOCK POSITIONS_FRAC\n\n"

    return lines


class CastepIn:
    """Class to create CASTEP input file."""

    def __init__(self, lines):
        """Init method."""
        self._tags = {
            "lattice_vectors": None,
            "atomic_species": None,
            "coordinates": None,
            "magnetic_moments": None,
        }

        self._values = None
        self._collect(lines)

    def get_tags(self):
        """Return tags."""
        return self._tags

    def _collect(self, lines):
        magmoms = []

        # numspins = 0
        units = 1.0  # Angstrom units
        lattvecs = []
        aspecies = []
        coords = []
        isSpinPol = 0
        # Read cell parameters
        for line in lines:
            if "%BLOCK LATTICE_CART" in line.upper():
                indx = lines.index(line)
                for i in range(indx + 1, len(lines)):
                    if "ENDBLOCK" in lines[i].upper():
                        break
                    if "BOHR" in lines[i].upper():
                        # The lattice vector units is Bohr. Convertion needed.
                        units = 0.529177211
                    elif len(lines[i].split()) >= 3:
                        lattvecs.append(
                            [(float(lines[i].split()[j]) * units) for j in range(3)]
                        )

        # Read atomic positions
        for line in lines:
            if "%BLOCK POSITIONS_FRAC" in line.upper():
                indx = lines.index(line)
                for i in range(indx + 1, len(lines)):
                    if "ENDBLOCK" in lines[i].upper():
                        break
                    if len(lines[i].split()) >= 3:
                        aspecies.append(lines[i].split()[0])
                        coords.append([float(lines[i].split()[j]) for j in (1, 2, 3)])
                        # If there is magmetic spin
                        if (len(lines[i].split()) > 4) and ("SPIN" in lines[i].upper()):
                            magmoms.append(
                                float(lines[i].upper().split("SPIN")[1].split("=")[1])
                            )
                            isSpinPol = 1
                        else:
                            magmoms.append(0.0)
        # Set magnetic tags if the structure with magnetic order
        if isSpinPol > 0:
            self._tags["magnetic_moments"] = magmoms

        if len(lattvecs) == 3 and len(aspecies) > 0 and len(aspecies) == len(coords):
            self._tags["lattice_vectors"] = lattvecs
            self._tags["atomic_species"] = aspecies
            self._tags["coordinates"] = coords
        else:
            print("CASTEP-interface: Error parsing CASTEP .cell file")


# ----------------------------------------------------------------------
