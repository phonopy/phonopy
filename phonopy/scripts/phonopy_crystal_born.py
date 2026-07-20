# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2018 Antti Karttunen (antti.j.karttunen@iki.fi)

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
