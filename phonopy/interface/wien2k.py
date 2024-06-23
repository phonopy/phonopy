"""Wien2k calculator interface."""

# Copyright (C) 2011 Atsushi Togo
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
from phonopy.structure.cells import get_angles, get_cell_parameters
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation


def parse_set_of_forces(
    disps,
    forces_filenames,
    supercell,
    wien2k_P1_mode=False,  # Only for the test
    symmetry_tolerance=None,
    verbose=True,
):
    """Parse forces from output files."""
    if symmetry_tolerance is None:
        symprec = 1e-5
    else:
        symprec = symmetry_tolerance

    num_atoms = supercell.get_number_of_atoms()
    lattice = supercell.get_cell()
    is_parsed = True
    force_sets = []

    for i, (filename, disp) in enumerate(zip(forces_filenames, disps)):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        # Parse wien2k case.scf file
        wien2k_forces = _get_forces_wien2k(filename, lattice)
        if not wien2k_P1_mode:
            forces = _distribute_forces(
                supercell, disp, wien2k_forces, filename, symprec
            )
        else:
            if num_atoms != len(wien2k_forces):
                forces = []
            else:
                forces = wien2k_forces

        if check_forces(forces, num_atoms, filename, verbose=verbose):
            drift_force = get_drift_forces(forces, filename=filename, verbose=verbose)
            force_sets.append(np.array(forces) - drift_force)
        else:
            is_parsed = False

    if is_parsed:
        return force_sets
    else:
        return []


def parse_wien2k_struct(filename):
    """Read crystal structure."""
    with open(filename) as f:
        # 1
        _ = f.readline().rstrip()

        # 2
        num_site = int(f.readline()[27:30])

        # 3
        f.readline()

        # 4
        line = f.readline()
        a = float(line[0:10])
        b = float(line[10:20])
        c = float(line[20:30])
        alpha = float(line[30:40])
        beta = float(line[40:50])
        gamma = float(line[50:60])

        lattice = _transform_axis(alpha, beta, gamma, a, b, c)

        symbols = []
        positions = []
        npts = []
        r0s = []
        rmts = []

        for _ in range(num_site):
            # 5
            line = f.readline()
            x = float(line[12:22])
            y = float(line[25:35])
            z = float(line[38:48])
            positions.append([x, y, z])

            # 6
            line = f.readline()
            multi = int(line[15:17])

            for _ in range(multi - 1):
                line = f.readline()
                x = float(line[12:22])
                y = float(line[25:35])
                z = float(line[38:48])
                positions.append([x, y, z])

            # 7
            line = f.readline()
            chemical_symbol = line[0:2].strip()
            npt = int(line[15:20])
            r0 = float(line[25:35])
            rmt = float(line[40:50])

            for _ in range(multi):
                symbols.append(chemical_symbol)
                npts.append(npt)
                r0s.append(r0)
                rmts.append(rmt)

            # 8 - 10
            for _ in range(3):
                f.readline()

        cell = PhonopyAtoms(symbols=symbols, scaled_positions=positions, cell=lattice)

        return cell, npts, r0s, rmts


def write_supercells_with_displacements(
    supercell,
    cells_with_displacements,
    ids,
    npts,
    r0s,
    rmts,
    num_unitcells_in_supercell,
    pre_filename="wien2k",
    width=3,
):
    """Write supercells with displacements to files."""
    npts_super = []
    r0s_super = []
    rmts_super = []
    for i, j, k in zip(npts, r0s, rmts):
        for _ in range(num_unitcells_in_supercell):
            npts_super.append(i)
            r0s_super.append(j)
            rmts_super.append(k)

    _pre_filename = pre_filename.split("/")[-1] + "S"
    write_wein2k(_pre_filename, supercell, npts_super, r0s_super, rmts_super)

    for i, cell in zip(ids, cells_with_displacements):
        symmetry = Symmetry(cell)
        filename = "{pre_filename}-{0:0{width}}.in".format(
            i, pre_filename=_pre_filename, width=width
        )
        print(
            "Number of non-equivalent atoms in %s: %d"
            % (filename, len(symmetry.get_independent_atoms()))
        )
        write_wein2k(filename, cell, npts_super, r0s_super, rmts_super)


def write_wein2k(filename, cell, npts, r0s, rmts):
    """Write cell to file."""
    with open(filename, "w") as w:
        w.write(_get_wien2k_struct(cell, npts, r0s, rmts))


def _get_wien2k_struct(cell, npts, r0s, rmts):
    num_atom = cell.get_number_of_atoms()
    lattice = cell.get_cell()
    a, b, c = get_cell_parameters(lattice)
    alpha, beta, gamma = get_angles(lattice)
    positions = cell.get_scaled_positions()
    symbols = cell.get_chemical_symbols()
    numbers = cell.get_atomic_numbers()

    text = ""

    # 1
    text += "Title\n"

    # 2
    text += "%-4s%23s%3d\n" % ("P", "LATTICE,NONEQUIV.ATOMS:", num_atom)

    # 3
    text += "%13s%4s\n" % ("MODE OF CALC=", "RELA")

    # 4
    text += "%10.6f%10.6f%10.6f%10.6f%10.6f%10.6f\n" % (a, b, c, alpha, beta, gamma)

    for i, pos in enumerate(positions):
        for j in (0, 1, 2):
            if pos[j] < 0:
                pos[j] += 1
            if int(float("%10.8f" % pos[j])) == 1:
                pos[j] = 0.0

        # 5 format (4X,I4,4X,F10.8,3X,F10.8,3X,F10.8)
        text += "%4s%4d%4s%10.8f%3s%10.8f%3s%10.8f\n" % (
            "ATOM",
            -(i + 1),
            ": X=",
            pos[0],
            " Y=",
            pos[1],
            " Z=",
            pos[2],
        )

        # 6  format (15X,I2,17X,I2)
        text += "%15s%2d%17s%2d\n" % ("MULT=", 1, "ISPLIT=", 8)

        # 7 format (A10,5X,I5,5X,F10.8,5X,F10.5,5X,F5.2)
        npt = npts[i]
        r0 = r0s[i]
        rmt = rmts[i]
        text += "%-10s%5s%5d%5s%10.8f%5s%10.5f%5s%5.1f\n" % (
            symbols[i],
            "NPT=",
            npt,
            "R0=",
            r0,
            "RMT=",
            rmt,
            "Z:",
            numbers[i],
        )

        # 8 - 10 format (20X,3F10.7)
        text += "%-20s%10.7f%10.7f%10.7f\n" % ("LOCAL ROT MATRIX:", 1, 0, 0)
        text += "%-20s%10.7f%10.7f%10.7f\n" % ("", 0, 1, 0)
        text += "%-20s%10.7f%10.7f%10.7f\n" % ("", 0, 0, 1)

    text += "   0      NUMBER OF SYMMETRY OPERATIONS"

    return text


def _transform_axis(alpha, beta, gamma, a, b, c):
    alpha = alpha / 180 * np.pi
    beta = beta / 180 * np.pi
    gamma = gamma / 180 * np.pi

    cz = c

    bz = np.cos(alpha) * b
    by = np.sin(alpha) * b

    az = np.cos(beta) * a
    ay = (np.cos(gamma) - np.cos(beta) * np.cos(alpha)) / np.sin(alpha) * a
    ax = np.sqrt(a**2 - ay**2 - az**2)

    return [ax, ay, az], [0, by, bz], [0, 0, cz]


def _parse_core_param(file):
    npts = []
    r0s = []
    rmts = []

    for line in file:
        if line.strip()[0] == "#":
            continue

        vals = line.strip().split()
        npts.append(int(vals[1]))
        r0s.append(float(vals[2]))
        rmts.append(float(vals[3]))

    return npts, r0s, rmts


def _get_forces_wien2k(filename, lattice):
    forces = []
    red_lattice = []

    for v in lattice:
        red_lattice.append(v / np.sqrt(np.vdot(v, v)))

    num_atom = 0
    for line in open(filename):
        if line.count("total forces") > 0:
            if line[:4] == ":FGL":
                fx = float(line[29:45])
                fy = float(line[45:61])
                fz = float(line[61:77])
                forces.append(np.dot([fx, fy, fz], red_lattice))
                num_atom = int(line[4:7])

    return forces[-num_atom:]


def _distribute_forces(supercell, disp, forces, filename, symprec):
    natom = supercell.get_number_of_atoms()
    lattice = supercell.get_cell()
    symbols = supercell.get_chemical_symbols()
    positions = supercell.get_positions() + disp
    cell = PhonopyAtoms(cell=lattice, positions=positions, symbols=symbols)
    symmetry = Symmetry(cell, symprec)
    independent_atoms = symmetry.get_independent_atoms()

    # Rotation matrices in Cartesian
    rotations = []
    for r in symmetry.get_symmetry_operations()["rotations"]:
        rotations.append(similarity_transformation(lattice.T, r))

    map_operations = symmetry.get_map_operations()
    map_atoms = symmetry.get_map_atoms()

    atoms_in_dot_scf = _get_independent_atoms_in_dot_scf(filename)

    if len(forces) != len(atoms_in_dot_scf):
        print("%s does not contain necessary information." % filename)
        print('Plese check if there are "FGL" lines with')
        print('"total forces" are required.')
        return False

    if len(atoms_in_dot_scf) == natom:
        print("It is assumed that there is no symmetrically-equivalent " "atoms in ")
        print("'%s' at wien2k calculation." % filename)
        force_set = forces
    elif len(forces) != len(independent_atoms):
        print(
            "Non-equivalent atoms of %s could not be recognized by phonopy." % filename
        )
        return False
    else:
        # 1. Transform wien2k forces to those on independent atoms
        indep_atoms_to_wien2k = []
        forces_remap = []
        for i, pos_wien2k in enumerate(atoms_in_dot_scf):
            for j, pos in enumerate(cell.get_scaled_positions()):
                diff = pos_wien2k - pos
                diff -= np.rint(diff)
                if (abs(diff) < symprec).all():
                    forces_remap.append(np.dot(rotations[map_operations[j]], forces[i]))
                    indep_atoms_to_wien2k.append(map_atoms[j])
                    break

        if len(forces_remap) != len(forces):
            print("Atomic position mapping between Wien2k and phonopy failed.")
            print("If you think this is caused by a bug of phonopy")
            print("please report it in the phonopy mainling list.")
            return False

        # 2. Distribute forces from independent to dependent atoms.
        force_set = []
        for i in range(natom):
            j = indep_atoms_to_wien2k.index(map_atoms[i])
            force_set.append(np.dot(rotations[map_operations[i]].T, forces_remap[j]))

    return force_set


def _get_independent_atoms_in_dot_scf(filename):
    positions = []
    for line in open(filename):
        if line[:4] == ":POS":
            if "POSITION" in line:
                x = float(line[30:37])
                y = float(line[38:45])
                z = float(line[46:53])
            else:
                x = float(line[27:34])
                y = float(line[35:42])
                z = float(line[43:50])
            num_atom = int(line[4:7])
            positions.append([x, y, z])

    return np.array(positions)[-num_atom:]


if __name__ == "__main__":
    from optparse import OptionParser

    from phonopy.interface.vasp import read_vasp, write_vasp

    def _clean_scaled_positions(cell):
        positions = cell.get_scaled_positions()
        for pos in positions:
            for i in (0, 1, 2):
                # The following %19.16f follows write_vasp
                if float("%19.16f" % pos[i]) >= 1:
                    pos[i] -= 1.0
        cell.set_scaled_positions(positions)

    parser = OptionParser()
    parser.set_defaults(w2v=False, v2w=False)
    parser.add_option(
        "-w", dest="w2v", action="store_true", help="Convert WIEN2k to VASP"
    )
    parser.add_option(
        "-v", dest="v2w", action="store_true", help="Convert VASP to WIEN2k"
    )
    (options, args) = parser.parse_args()

    from phonopy.units import Bohr

    if options.v2w:
        cell = read_vasp(args[0])
        lattice = cell.get_cell() / Bohr
        cell.set_cell(lattice)
        npts, r0s, rmts = _parse_core_param(open(args[1]))
        text = _get_wien2k_struct(cell, npts, r0s, rmts)
        print(text)

    elif options.w2v:
        cell, npts, r0s, rmts = parse_wien2k_struct(args[0])
        positions = cell.get_scaled_positions()
        lattice = cell.get_cell() * Bohr
        cell.set_cell(lattice)
        cell.set_scaled_positions(positions)
        _clean_scaled_positions(cell)
        write_vasp("POSCAR.wien2k", cell, direct=True)
        w = open("wien2k_core.dat", "w")

        w.write("# symbol       npt       r0             rmt\n")
        for symbol, npt, r0, rmt in zip(cell.get_chemical_symbols(), npts, r0s, rmts):
            w.write("%-10s     %5d     %10.8f     %10.5f\n" % (symbol, npt, r0, rmt))
    else:
        print("You need to set -r or -w option.")
