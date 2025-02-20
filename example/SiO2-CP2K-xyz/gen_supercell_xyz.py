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
#

import argparse
import sys

import numpy as np

from phonopy.api_phonopy import Phonopy
from phonopy.interface.cp2k import read_cp2k
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.structure.symmetry import Symmetry


def genxyz(fn, atnum, basis, species, cart, desc=""):
    """Generate an .xyz file with the given structure."""
    print("Generating file %s" % fn)
    try:
        fh = open(fn, "w")
    except IOError:
        print("ERROR Couldn't open output file %s for writing" % fn)
        return -1
    fh.write("%d\n" % len(species))
    fh.write("%s\n" % "Force calculations. %s" % desc)
    for i in range(len(species)):
        fh.write(
            "%s  %s\n" % (species[i], "".join("    % 15.10f" % c for c in cart[i]))
        )


parser = argparse.ArgumentParser(
    description="""Input files for phonon
                                 calculations in .xyz format generator"""
)

parser.add_argument(
    "-i",
    "--input",
    action="store",
    type=str,
    dest="str_fn",
    help="Input filename with structure",
)
parser.add_argument(
    "-o",
    "--output",
    action="store",
    type=str,
    dest="out_fn",
    default="supercell",
    help="Output filename prefix",
)
parser.add_argument(
    "-d",
    "--dim",
    action="store",
    type=str,
    dest="dim",
    default="1 1 1",
    help="Supercell dimentions",
)
parser.add_argument(
    "-a",
    "--amplitude",
    action="store",
    type=float,
    dest="ampl",
    default=0.01,
    help="Displacement amplitude",
)
parser.add_argument(
    "--born",
    dest="born",
    action="store_true",
    default=False,
    help="Generate supercells for BORN charges calculations",
)
parser.add_argument(
    "--nosym", dest="nosym", action="store_true", default=False, help="Ignore symmetry"
)

args = parser.parse_args()

if args.str_fn is None:
    print("Error. No input filename was given.")
    sys.exit(1)
if len(args.dim.split()) < 3:
    print("Size of dim should be 3 or 9")

if len(args.dim.split()) > 0:
    if len(args.dim.split()) > 3:
        dim = np.array([float(d) for d in args.dim.split()]).reshape(3, 3)
    else:
        dim = np.zeros(9).reshape(3, 3)
        for i in range(3):
            dim[i][i] = float(args.dim.split()[i])
else:
    dim = np.eye(3)

inph = read_cp2k(args.str_fn)
species = inph[0].symbols
numbers = inph[0].numbers
masses = inph[0].masses
natom = len(inph[0])
basis = inph[0].cell
cart = inph[0].positions
cvol = inph[0].volume
ph = Phonopy(inph[0], supercell_matrix=dim)
scideal = ph.supercell
displs = []

scMap = ph.symmetry.get_map_atoms()
s2uMap = ph._primitive.p2p_map

if args.nosym:
    print("Displacements with ignore symmetry option")
    # Iterate over all unique atoms in supercell mapped to unitcell ones
    for atom in s2uMap:
        for i in range(3):
            displ = [0.0, 0.0, 0.0]
            displ[i] = args.ampl
            displs.append({"number": atom, "displacement": displ})
            displ = [0.0, 0.0, 0.0]
            displ[i] = -args.ampl
            displs.append({"number": atom, "displacement": displ})
            ph.dataset = {"natom": len(scideal), "first_atoms": displs}
elif args.born:
    for atom in Symmetry(scideal).get_independent_atoms():
        for i in range(3):
            displ = [0.0, 0.0, 0.0]
            displ[i] = args.ampl
            displs.append({"number": atom, "displacement": displ})
            displ = [0.0, 0.0, 0.0]
            displ[i] = -args.ampl
            displs.append({"number": atom, "displacement": displ})
            ph.dataset = {"natom": len(scideal), "first_atoms": displs}
else:
    ph.generate_displacements(distance=args.ampl, is_plusminus="auto", is_diagonal=True)

scs = ph.supercells_with_displacements

scideal = ph.supercell

fn = "".join("%s-%04d.xyz" % (args.out_fn, 0))
genxyz(fn, len(scideal), scideal.cell, scideal.symbols, scideal.positions)

cellstr = "A " + "%14.8f " * 3 % tuple(scideal.cell[0])
cellstr += "B " + "%14.8f " * 3 % tuple(scideal.cell[1])
cellstr += "C " + "%14.8f " * 3 % tuple(scideal.cell[2])

for i in range(len(scs)):
    fn = "".join("%s-%04d.xyz" % (args.out_fn, i + 1))
    genxyz(
        fn,
        len(scs[i]),
        scs[i].cell,
        scs[i].symbols,
        scs[i].positions,
        desc="Lattice parameters: %s" % cellstr,
    )

print("Lattice parameters to put in pattern input file:")
for i, abc in enumerate(scideal.cell):
    print("%s " % "ABC"[i], "% 14.8f " * 3 % tuple(abc))

fn = "".join("%s-%04d.xyz" % (args.out_fn, 0))
settings = {
    "force_sets": False,
    "displacements": True,
    "force_constants": False,
    "born_effective_charge": True,
    "dielectric_constant": True,
}

conf = {"DIM": args.dim}
phyaml = PhonopyYaml(configuration=conf, calculator="cp2k", settings=settings)
phyaml.set_phonon_info(ph)


with open("phonopy_disp.yaml", "w") as fd:
    fd.write(phyaml.__str__())
