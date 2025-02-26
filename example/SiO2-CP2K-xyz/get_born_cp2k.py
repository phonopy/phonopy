# Copyright (C) 2025 Eugene Roginskii
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


import argparse
from os import path

import numpy as np

from phonopy.cui import load

Debye2au = 2.54174


def get_cp2kver(fn):
    """Get CP2K version from the output file."""
    try:
        fh = open(fn, "r")
        for line in fh:
            if "CP2K| version string" in line:
                try:
                    ver = float(line.split("CP2K version")[1])
                    return int(ver)
                    break
                except ValueError:
                    print("Warning. Reading CP2K version failed. Suggest v8")
                    return 8
    except IOError:
        print("Warning. Reading CP2K version failed. Suggest v8 and above")
        return 8


def get_dipole(fn):
    """Get dipole moment from the output file."""
    dipole = []
    xyz = ["X=", "Y=", "Z="]
    try:
        fh = open(fn, "r")
        for line in fh:
            if "Dipole moment [Debye]" in line:
                break
        line = fh.readline()
        for i in range(3):
            dipole.append(float(line.split(xyz[i])[1].split()[0]) / Debye2au)
    except OSError as err:
        print("Error opening file %s: %s" % (fn, err))
    return np.array(dipole)


def get_epsilon_cp2k(fn, ucvol):
    """Get epsilon from the CP2K output file."""
    from numpy import pi

    epsilon = np.zeros(9)
    try:
        fh = open(fn, "r")
    except IOError:
        print("ERROR open output file %s for reading filed" % fn)
        return -1
    for line in fh:
        if "Polarizability tensor [a.u.]" in line:
            break
    # Components sequence in CP2K output
    # xx,yy,zz
    # xy,xz,yz
    # yx,zx,zy
    # Components sequence in epsilon array:
    #                     0    1    2    3    4    5    6    7    8
    #      9-components:  xx   xy   xz   yx   yy   yz   zx   zy   zz
    line = fh.readline()
    for i in range(3):
        epsilon[i * 4] = float(line.split()[i + 2]) / ucvol * 4 * pi + 1

    line = fh.readline()
    for i in range(2):
        epsilon[1 + i] = float(line.split()[i + 2]) / ucvol * 4 * pi
    epsilon[5] = float(line.split()[4]) / ucvol * 4 * pi

    line = fh.readline()
    epsilon[3] = float(line.split()[2]) / ucvol * 4 * pi
    for i in range(2):
        epsilon[6 + i] = float(line.split()[i + 3]) / ucvol * 4 * pi
    # Symmitrize
    epsilon[1] = (epsilon[1] + epsilon[3]) / 2
    epsilon[3] = epsilon[1]
    epsilon[2] = (epsilon[2] + epsilon[6]) / 2
    epsilon[6] = epsilon[2]
    epsilon[5] = (epsilon[5] + epsilon[7]) / 2
    epsilon[7] = epsilon[5]

    return epsilon


def get_epsilon_cp2kv6(fn, ucvol):
    """Get epsilon from the CP2K v6 output file."""
    from numpy import pi

    epsilon = np.zeros(9)
    print("cp2kv6")
    try:
        fh = open(fn, "r")
    except IOError:
        print("ERROR open output file %s for reading filed" % fn)
        return -1
    for line in fh:
        if "POLARIZABILITY TENSOR (atomic units)" in line:
            print(line)
            break

    line = fh.readline()
    for i in range(3):
        epsilon[i * 4] = float(line.split()[i + 1]) / ucvol * 4 * pi

    line = fh.readline()
    for i in range(2):
        epsilon[1 + i] = float(line.split()[i + 1]) / ucvol * 4 * pi
    epsilon[5] = float(line.split()[3]) / ucvol * 4 * pi

    line = fh.readline()
    epsilon[3] = float(line.split()[1]) / ucvol * 4 * pi
    for i in range(2):
        epsilon[6 + i] = float(line.split()[i + 2]) / ucvol * 4 * pi
    # Symmitrize
    epsilon[1] = (epsilon[1] + epsilon[3]) / 2
    epsilon[3] = epsilon[1]
    epsilon[2] = (epsilon[2] + epsilon[6]) / 2
    epsilon[6] = epsilon[2]
    epsilon[5] = (epsilon[5] + epsilon[7]) / 2
    epsilon[7] = epsilon[5]

    return epsilon


def is_minus_displacement(direction, site_symmetry):
    """Symmetrically check if minus displacement is necessary or not."""
    is_minus = True
    for r in site_symmetry:
        rot_direction = np.dot(direction, r.T)
        if (rot_direction + direction).any():
            continue
        else:
            is_minus = False
            break
    return is_minus


parser = argparse.ArgumentParser(description="Born charges file generator")

parser.add_argument(
    "-i",
    "--input",
    action="store",
    type=str,
    dest="eps_fn",
    help="CP2K output filename with LR calculations",
)
parser.add_argument(
    "-p",
    "--prefix",
    action="store",
    type=str,
    dest="pref",
    default="DISP",
    help="Prefix of directories with calculations",
)
parser.add_argument(
    "-m",
    "--dipolefn",
    action="store",
    type=str,
    dest="dipole_fn",
    default="force.out",
    help="CP2K output filename with dipole calculations",
)
parser.add_argument(
    "-o",
    "--out",
    action="store",
    type=str,
    dest="out_fn",
    default="BORN",
    help="Output filename (default: BORN)",
)

args = parser.parse_args()

b2a = 0.5291772109038
ph = load.load(phonopy_yaml="phonopy_disp.yaml")

basis = ph.supercell.cell
ucvol = ph.supercell.volume / (b2a**3)  # In a.u.

# Get epsilon
fn = path.join("%s_ideal" % args.pref, "polar.out")
if get_cp2kver(args.eps_fn) <= 6:
    epsilon = get_epsilon_cp2kv6(args.eps_fn, ucvol)
else:
    epsilon = get_epsilon_cp2k(args.eps_fn, ucvol)

dipoles = []
atlist = {}
for i, atom in enumerate(ph.dataset["first_atoms"]):
    wdir = "%s-%04d" % (args.pref, (i + 1))
    dipole = get_dipole(path.join(wdir, args.dipole_fn))
    dispmag = atom["displacement"].sum()
    dipoles.append([atom["number"], dispmag, dipole])
    if atom["number"] not in atlist.keys():
        # store index of displacement for each atom for positive and negative
        atlist.update({atom["number"]: [[-1, -1, -1], [-1, -1, -1]]})

for i, atom in enumerate(ph.dataset["first_atoms"]):
    ddir = np.where(atom["displacement"] != 0)[0][0]
    if atom["displacement"][ddir] > 0:
        atlist[atom["number"]][0][ddir] = i
    else:
        atlist[atom["number"]][1][ddir] = i

born = np.zeros(len(atlist) * 9).reshape(len(atlist), 3, 3)
for i, a in enumerate(atlist):
    for j in range(3):
        born[i][j] = (dipoles[atlist[a][0][j]][2] - dipoles[atlist[a][1][j]][2]) * b2a
        born[i][j] /= dipoles[atlist[a][0][j]][1] - dipoles[atlist[a][1][j]][1]
with open(args.out_fn, "w") as fh:
    fh.write("0.280028521\n")
    fh.write(" %12.9f " * 9 % tuple(epsilon))
    fh.write("\n ")
    for b in born:
        fh.write(" %12.9f " * 9 % tuple(b.flatten()))
        fh.write("\n ")
