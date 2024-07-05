# Copyright (C) 2018 Antti Karttunen (antti.j.karttunen@iki.fi)
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


def get_options():
    """Parse options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phonopy crystal-born command-line-tool"
    )
    parser.add_argument(
        "filename", nargs="*", help="Filename: CRYSTAL output file (default: crystal.o)"
    )
    args = parser.parse_args()
    return args


def read_crystal_epsilon(epscol, epslines):
    """Read epsilon in CRYSTAL14 format."""
    # XX, XY, XZ, YY, YZ, ZZ are given (6 rows). YX = XY, ZX = XZ, ZY = YZ
    # Format for CRYSTAL17 (epscol == 4):
    # 0.000   XX      57.1760       0.0000       3.2117       0.0000       2.2117
    # Format for CRYSTAL14 (epscol == 3):
    # XX      57.1760        0.0000        3.2117        2.2117
    eps = []  # Will have 9 elements 0..8
    eps.append(float(epslines[0].split()[epscol]))  # XX (0)
    eps.append(float(epslines[1].split()[epscol]))  # XY (1)
    eps.append(float(epslines[2].split()[epscol]))  # XZ (2)
    eps.append(eps[1])  # YX = XY (3)
    eps.append(float(epslines[3].split()[epscol]))  # YY (4)
    eps.append(float(epslines[4].split()[epscol]))  # YZ (5)
    eps.append(eps[2])  # ZX = XZ (6)
    eps.append(eps[5])  # ZY = YZ (7)
    eps.append(float(epslines[5].split()[epscol]))  # ZZ (8)
    return eps


def run():
    """Run phonopy-crystal-born."""
    args = get_options()

    if args.filename:
        crystal_filename = args.filename[0]
    else:
        crystal_filename = "crystal.o"

    # Read in the output file
    try:
        with open(crystal_filename, "r") as crystal_file:
            lines = crystal_file.readlines()
    except OSError:
        print(
            "CRYSTAL output file {} cannot be opened for reading".format(
                crystal_filename
            )
        )
        sys.exit(1)

    # Recommended CRYSTAL calculation type:
    # Gamma-point FREQCALC with INTCPHF and INTENS
    # The file will include both dielectric tensor and effective Born charges
    epsilon = []
    zeff = []
    ll = 0
    while ll < len(lines):
        line = lines[ll]
        # Parse the information about atoms
        if "PRIMITIVE CELL - CENTRING CODE" in line:
            ll += 4
            # ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    6
            N_asym_atoms = int(lines[ll].split()[5])
            N_atoms = int(lines[ll].split()[12])
            ll += 3
            # Check for each atom if it belongs to the asymmetric unit
            # 1 T  22 TI    4.721218104494E-21  3.307446203077E-21  1.413771901417E-21
            is_asym = []
            for _ in range(0, N_atoms):
                atomdata = lines[ll].split()
                is_asym.append(atomdata[1])
                ll += 1

        # Parse dielectric tensor from INTCPHF
        elif "FR.(eV) COMP.   ALPHA(Re,Im)           EPSILON(Re,Im)" in line:
            # CRYSTAL17
            epsilon = read_crystal_epsilon(4, lines[ll + 1 : ll + 7])
            ll += 7
        elif "COMPONENT    ALPHA(REAL, IMAGINARY)         EPSILON       CHI(1)" in line:
            epsilon = read_crystal_epsilon(3, lines[ll + 1 : ll + 7])
            ll += 7

        # Parse Born charges
        elif "ATOMIC BORN CHARGE TENSOR" in line:
            ll += 6
            for atom in range(0, N_atoms):
                if is_asym[atom] == "T":
                    zeffatom = []
                    for _ in range(0, 3):
                        # 1     8.3860E-01  3.4861E-01  3.4861E-01
                        zeffatom += [float(x) for x in lines[ll].split()[1:4]]
                        ll += 1
                    zeff.append(zeffatom)
                    ll += 4
                else:
                    ll += 7
        ll += 1  # while l < len(lines)

    # Output BORN if epsilon and zeff are available
    if len(epsilon) == 9 and len(zeff) == N_asym_atoms:
        # Conversion factor
        bornlines = "default\n"
        # Dielectric tensor
        bornlines += ("%6.4f " * 9 + "\n") % tuple(epsilon)
        # Effective charges
        for atom in range(0, N_asym_atoms):
            bornlines += ("%6.4f " * 9 + "\n") % tuple(zeff[atom])
        print(bornlines, end="")
