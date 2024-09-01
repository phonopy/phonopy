"""Show symmetry information invoked by --symmetry command option."""

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

import numpy as np
import spglib

from phonopy import Phonopy
from phonopy.interface.calculator import (
    get_default_cell_filename,
    write_crystal_structure,
)
from phonopy.structure.atoms import PhonopyAtoms, atom_data
from phonopy.structure.cells import get_primitive, guess_primitive_matrix
from phonopy.structure.symmetry import Symmetry


def check_symmetry(phonon: Phonopy, cell_info: dict):
    """Show symmetry information and write refined crystals to files."""
    # Assumed that primitive cell is the cell that user is interested in.
    print(
        _get_symmetry_yaml(phonon.primitive, phonon.primitive_symmetry, phonon.version)
    )

    if phonon.unitcell.magnetic_moments is None:
        base_fname = get_default_cell_filename(phonon.calculator)
        symprec = phonon.primitive_symmetry.tolerance
        (bravais_lattice, bravais_pos, bravais_numbers) = spglib.refine_cell(
            phonon.primitive.totuple(), symprec
        )
        bravais_symbols = [atom_data[n][1] for n in bravais_numbers]
        bravais = PhonopyAtoms(
            symbols=bravais_symbols, scaled_positions=bravais_pos, cell=bravais_lattice
        )
        trans_mat = guess_primitive_matrix(bravais, symprec=symprec)
        primitive = get_primitive(bravais, trans_mat, symprec=symprec)

        # Unless input cell is given as phonopy_yaml.
        if cell_info["phonopy_yaml"] is None:
            optional_structure_info = cell_info["optional_structure_info"]
            filename = "B" + base_fname
            print(
                f'# Symmetrized conventional unit cell is written into "{filename}" and'
            )
            write_crystal_structure(
                filename,
                bravais,
                interface_mode=phonon.calculator,
                optional_structure_info=optional_structure_info,
            )
            filename = "P" + base_fname
            print(f'# Symmetrized primitive is written into "{filename}" and ')
            write_crystal_structure(
                filename,
                primitive,
                interface_mode=phonon.calculator,
                optional_structure_info=optional_structure_info,
            )
        with open("phonopy_symcells.yaml", "w") as w:
            print("primitive_cell:", file=w)
            print(
                "\n".join(["  " + line for line in primitive.get_yaml_lines()]), file=w
            )
            print("unit_cell:", file=w)
            print("\n".join(["  " + line for line in bravais.get_yaml_lines()]), file=w)

        print('# Unit cell and primitive cell were written in "phonopy_symcells.yaml".')
        print("# These structures can be read in a python script as follows:")
        print("#")
        print("# from phonopy.interface.phonopy_yaml import read_cell_yaml")
        print('# unitcell = read_cell_yaml("phonopy_symcells.yaml", "unitcell")')
        print('# primitive_cell = read_cell_yaml("phonopy_symcells.yaml", "primitive")')


def _get_symmetry_yaml(cell: PhonopyAtoms, symmetry: Symmetry, phonopy_version=None):
    rotations = symmetry.symmetry_operations["rotations"]
    translations = symmetry.symmetry_operations["translations"]

    atom_sets = symmetry.get_map_atoms()
    independent_atoms = symmetry.get_independent_atoms()
    wyckoffs = symmetry.get_Wyckoff_letters()

    lines = []

    if phonopy_version is not None:
        lines.append("phonopy_version: '%s'" % phonopy_version)

    if cell.magnetic_moments is None:
        spg_symbol, spg_number = symmetry.get_international_table().split()
        spg_number = int(spg_number.replace("(", "").replace(")", ""))
        lines.append("space_group_type: '%s'" % spg_symbol)
        lines.append("space_group_number: %d" % spg_number)
        lines.append("point_group_type: '%s'" % symmetry.pointgroup_symbol)
    lines.append("space_group_operations:")
    for i, (r, t) in enumerate(zip(rotations, translations)):
        lines.append("- rotation: # %d" % (i + 1))
        for vec in r:
            lines.append("  - [%2d, %2d ,%2d]" % tuple(vec))
        line = "  translation: ["
        for j, x in enumerate(t):
            if abs(x - np.rint(x)) < 1e-5:
                line += " 0.00000"
            else:
                line += "%8.5f" % x
            if j < 2:
                line += ", "
            else:
                line += " ]"
                lines.append(line)
    lines.append("atom_mapping:")
    for i, atom_num in enumerate(atom_sets):
        lines.append("  %d: %d" % (i + 1, atom_num + 1))
    lines.append("site_symmetries:")
    for i in independent_atoms:
        sitesym = symmetry.get_site_symmetry(i)
        lines.append("- atom: %d" % (i + 1))

        if cell.magnetic_moments is None:
            lines.append("  Wyckoff: '%s'" % wyckoffs[i])
        site_pointgroup = spglib.get_pointgroup(sitesym)
        lines.append("  site_point_group: '%s'" % site_pointgroup[0].strip())
        lines.append("  orientation:")
        for v in site_pointgroup[2]:
            lines.append("  - [%2d, %2d, %2d]" % tuple(v))

        lines.append("  rotations:")
        for j, r in enumerate(sitesym):
            lines.append("  - # %d" % (j + 1))
            for vec in r:
                lines.append("    - [%2d, %2d, %2d]" % tuple(vec))

    return "\n".join(lines)
