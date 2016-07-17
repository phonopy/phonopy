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
import phonopy.structure.spglib as spg
from phonopy.structure.symmetry import Symmetry, find_primitive, get_pointgroup
from phonopy.structure.cells import get_primitive, print_cell, get_supercell
from phonopy.interface.vasp import write_vasp
from phonopy.structure.atoms import PhonopyAtoms as Atoms

def check_symmetry(input_cell,
                   primitive_axis=None,
                   symprec=1e-5,
                   distance_to_A=1.0,
                   phonopy_version=None):
    if primitive_axis is None:
        cell = get_primitive(input_cell, np.eye(3), symprec=symprec)
    else:
        cell = get_primitive(input_cell, primitive_axis, symprec=symprec)
    lattice = cell.get_cell() * distance_to_A
    cell.set_cell(lattice)

    symmetry = Symmetry(cell, symprec)
    print(_get_symmetry_yaml(cell, symmetry, phonopy_version))

    if input_cell.get_magnetic_moments() is None:
        primitive = find_primitive(cell, symprec)
        if primitive is not None:
            print("# Primitive cell was found. It is written into PPOSCAR.")
            write_vasp('PPOSCAR', primitive)

            # Overwrite symmetry and cell
            symmetry = Symmetry(primitive, symprec)
            cell = primitive

        (bravais_lattice,
         bravais_pos,
         bravais_numbers) = spg.refine_cell(cell, symprec)
        bravais = Atoms(numbers=bravais_numbers,
                        scaled_positions=bravais_pos,
                        cell=bravais_lattice,
                        pbc=True)
        print("# Bravais lattice is written into BPOSCAR.")
        write_vasp('BPOSCAR', bravais)

def _get_symmetry_yaml(cell, symmetry, phonopy_version=None):
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']

    atom_sets = symmetry.get_map_atoms()
    independent_atoms = symmetry.get_independent_atoms()
    wyckoffs = symmetry.get_Wyckoff_letters()

    yaml = []

    if phonopy_version is not None:
        yaml.append("phonopy_version: %s" % phonopy_version)

    if cell.get_magnetic_moments() is None:
        yaml.append("space_group_type: " + symmetry.get_international_table())
    yaml.append("point_group_type: " + symmetry.get_pointgroup())
    yaml.append("space_group_operations:")
    for i, (r, t) in enumerate(zip(rotations, translations)):
        yaml.append("- rotation: # %d" % (i + 1))
        for vec in r:
            yaml.append("  - [%2d, %2d ,%2d]" % tuple(vec))
        line = "  translation: ["
        for j, x in enumerate(t):
            if abs(x - np.rint(x)) < 1e-5:
                line += " 0.00000"
            else:
                line += "%8.5f" % x
            if j < 2:
                line += ", "
            else:
                yaml.append(line)
    yaml.append("atom_mapping:")
    for i, atom_num in enumerate(atom_sets):
        yaml.append("  %d: %d" % (i + 1, atom_num + 1))
    yaml.append("site_symmetries:")
    for i in independent_atoms:
        sitesym = symmetry.get_site_symmetry(i)
        yaml.append("- atom: %d" % (i + 1))

        if cell.get_magnetic_moments() is None:
            yaml.append("  Wyckoff: %s" % wyckoffs[i])
        site_pointgroup = get_pointgroup(sitesym)
        yaml.append("  site_point_group: %s" % site_pointgroup[0])
        yaml.append("  orientation:")
        for v in site_pointgroup[1]:
            yaml.append("  - [%2d, %2d, %2d]" % tuple(v))

        yaml.append("  rotations:")
        for j, r in enumerate(sitesym):
            yaml.append("  - # %d" % (j + 1))
            for vec in r:
                yaml.append("    - [%2d, %2d, %2d]" % tuple(vec))

    return "\n".join(yaml)
