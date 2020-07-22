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
from phonopy.structure.symmetry import get_pointgroup
from phonopy.interface.calculator import (
    write_crystal_structure, get_default_cell_filename)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import guess_primitive_matrix, get_primitive


def check_symmetry(phonon, optional_structure_info):
    # Assumed that primitive cell is the cell that user is interested in.
    print(_get_symmetry_yaml(phonon.primitive,
                             phonon.primitive_symmetry,
                             phonon.version))

    if phonon.unitcell.magnetic_moments is None:
        base_fname = get_default_cell_filename(phonon.calculator)
        symprec = phonon.primitive_symmetry.get_symmetry_tolerance()
        (bravais_lattice,
         bravais_pos,
         bravais_numbers) = spglib.refine_cell(phonon.primitive, symprec)
        bravais = PhonopyAtoms(numbers=bravais_numbers,
                               scaled_positions=bravais_pos,
                               cell=bravais_lattice)
        filename = 'B' + base_fname
        print("# Symmetrized conventional unit cell is written into %s."
              % filename)
        trans_mat = guess_primitive_matrix(bravais, symprec=symprec)
        primitive = get_primitive(bravais, trans_mat, symprec=symprec)
        write_crystal_structure(
            filename,
            bravais,
            interface_mode=phonon.calculator,
            optional_structure_info=optional_structure_info)

        filename = 'P' + base_fname
        print("# Symmetrized primitive is written into %s." % filename)
        write_crystal_structure(
            filename,
            primitive,
            interface_mode=phonon.calculator,
            optional_structure_info=optional_structure_info)


def _get_symmetry_yaml(cell, symmetry, phonopy_version=None):
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']

    atom_sets = symmetry.get_map_atoms()
    independent_atoms = symmetry.get_independent_atoms()
    wyckoffs = symmetry.get_Wyckoff_letters()

    lines = []

    if phonopy_version is not None:
        lines.append("phonopy_version: '%s'" % phonopy_version)

    if cell.get_magnetic_moments() is None:
        spg_symbol, spg_number = symmetry.get_international_table().split()
        spg_number = int(spg_number.replace('(', '').replace(')', ''))
        lines.append("space_group_type: '%s'" % spg_symbol)
        lines.append("space_group_number: %d" % spg_number)
        lines.append("point_group_type: '%s'" % symmetry.get_pointgroup())
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

        if cell.get_magnetic_moments() is None:
            lines.append("  Wyckoff: '%s'" % wyckoffs[i])
        site_pointgroup = get_pointgroup(sitesym)
        lines.append("  site_point_group: '%s'" % site_pointgroup[0])
        lines.append("  orientation:")
        for v in site_pointgroup[1]:
            lines.append("  - [%2d, %2d, %2d]" % tuple(v))

        lines.append("  rotations:")
        for j, r in enumerate(sitesym):
            lines.append("  - # %d" % (j + 1))
            for vec in r:
                lines.append("    - [%2d, %2d, %2d]" % tuple(vec))

    return "\n".join(lines)
