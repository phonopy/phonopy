"""CRYSTAL calculator interface."""

# Copyright (C) 2019 Antti J. Karttunen (antti.j.karttunen@iki.fi)
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

import os
import sys

import numpy as np

from phonopy.interface.vasp import check_forces, get_drift_forces
from phonopy.structure.atoms import PhonopyAtoms


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    """Parse forces from output files."""
    # Filenames = subdirectories supercell-001, supercell-002, ...
    force_sets = []
    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        f_gradient = open(os.path.join(filename, "gradient"))
        lines = f_gradient.readlines()
        f_gradient.close()
        # Structure of the gradient file:
        # $grad          cartesian gradients
        #  cycle =      1    SCF energy =     -578.5931883878   |dE/dxyz| =  0.000007
        # coordinates (num_atoms lines)
        # gradients (num_atoms lines)
        # $end
        turbomole_forces = []
        for line in lines[2 + num_atoms : 2 + 2 * num_atoms]:
            # Replace D with E in double precision floats
            turbomole_forces.append([float(x.replace("D", "E")) for x in line.split()])

        # Change from gradient to force by inverting the sign
        # Units: hartree / Bohr
        turbomole_forces = np.negative(turbomole_forces)

        if check_forces(turbomole_forces, num_atoms, filename, verbose=verbose):
            is_parsed = True
            drift_force = get_drift_forces(
                turbomole_forces, filename=filename, verbose=verbose
            )
            force_sets.append(np.array(turbomole_forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def read_turbomole(filename):
    """Read crystal structure."""
    # filename is typically "control"
    f_turbomole = open(filename)
    turbomole_in = TurbomoleIn(f_turbomole.readlines())
    f_turbomole.close()
    tags = turbomole_in.get_tags()
    cell = PhonopyAtoms(
        cell=tags["lattice_vectors"],
        symbols=tags["atomic_species"],
        positions=tags["coordinates"],
    )

    return cell


def write_turbomole(filename, cell):
    """Write cell to file."""
    # Write geometry in a new directory
    # Check if directory exists (directory supercell will already exist for phono3py)
    if not os.path.exists(filename):
        os.mkdir(filename)

    # Create control file
    lines = "$title " + filename + "\n"
    lines += "$symmetry c1\n"
    lines += "$coord    file=coord\n"
    lines += "$periodic 3\n"
    lines += "$kpoints\n"
    lines += "  nkpoints KPOINTS_HERE\n"
    lines += "$scfconv 10\n"
    lines += "$lattice\n"
    lattice = cell.get_cell()
    for lattvec in lattice:
        lines += ("%12.8f" * 3 + "\n") % tuple(lattvec)
    lines += "$end\n"
    f_control = open(os.path.join(filename, "control"), "w")
    f_control.write(lines)
    f_control.close()

    # Create coord file
    symbols = cell.get_chemical_symbols()
    positions = cell.get_positions()
    lines = "$coord\n"
    for atom, pos in zip(symbols, positions):
        lines += ("%16.12f" * 3 + "   %s\n") % (pos[0], pos[1], pos[2], atom.lower())
    lines += "$end\n"
    f_coord = open(os.path.join(filename, "coord"), "w")
    f_coord.write(lines)
    f_coord.close()


def write_supercells_with_displacements(
    supercell, cells_with_displacements, ids, pre_filename="supercell", width=3
):
    """Write supercells with displacements to files."""
    write_turbomole(pre_filename, supercell)
    for i, cell in zip(ids, cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}".format(
            i, pre_filename=pre_filename, width=width
        )
        write_turbomole(filename, cell)


class TurbomoleIn:
    """Class to create TURBOMOLE input file."""

    def __init__(self, lines):
        """Init method."""
        self._tags = {
            "lattice_vectors": None,
            "atomic_species": None,
            "coordinates": None,
        }

        self._values = None
        self._collect(lines)

    def get_tags(self):
        """Return tags."""
        return self._tags

    def _collect(self, lines):
        # Reads TURBOMOLE control and coord files
        #  (lattice vectors, cartesian atomic positions).
        ll = 0
        lattvecs = []
        aspecies = []
        coords = []
        while ll < len(lines):
            line = lines[ll]
            # Look for lattice vectors
            # Only supports atomic units, $lattice angs not supported
            if "$lattice" in line:
                # 0.00000000000   5.17898186576   5.17898186576
                for lattvec in lines[ll + 1 : ll + 4]:
                    lattvecs.append([float(x) for x in lattvec.split()])
                ll += 4
            # Look for Cartesian coordinates.
            # They can be in another file or embedded in control:
            # 1) $coord    file=coord
            # 2) $coord
            #      2.58949092075      2.58949092075      2.58949092075      si
            elif "$coord" in line:
                if line.strip() == "$coord":
                    # Embdedded coordinates.
                    ll += 1
                    while ll < len(lines):
                        atom = lines[ll].split()
                        if len(atom) == 4:
                            coords.append([float(x) for x in atom[0:3]])
                            aspecies.append(
                                atom[3].title()
                            )  # Convert si to Si, c to C, etc.
                            ll += 1
                        else:
                            # End of $coord, go back to the main while loop to interpret
                            # the current line
                            break
                elif line.find("file=") > 6:
                    # Cross-reference to another file
                    coordfile = line.split("=")[1].strip()
                    f_coord = open(coordfile)
                    for coordline in f_coord:
                        # 2.58949092075      2.58949092075      2.58949092075      si
                        atom = coordline.split()
                        if len(atom) == 4:
                            coords.append([float(x) for x in atom[0:3]])
                            aspecies.append(
                                atom[3].title()
                            )  # Convert si to Si, c to C, etc.
                    f_coord.close()
                    ll += 1
                else:
                    # $coordinateupdate or invalid $coord line
                    ll += 1
            else:
                ll += 1

        if len(lattvecs) == 3 and len(aspecies) > 0 and len(aspecies) == len(coords):
            self._tags["lattice_vectors"] = lattvecs
            self._tags["atomic_species"] = aspecies
            self._tags["coordinates"] = coords
        else:
            print("TURBOMOLE-interface: Error parsing TURBOMOLE output file")


if __name__ == "__main__":
    from phonopy.structure.symmetry import Symmetry

    cell = read_turbomole(sys.argv[1])
    symmetry = Symmetry(cell)
    print("# %s" % symmetry.get_international_table())
