# Copyright (C) 2016 Atsushi Togo
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

from phonopy.structure.cells import get_angles, get_cell_parameters

def write_cif_P1(cell, U_cif=None, filename=None):
    if filename:
        with open(filename, 'w') as w:
            w.write(get_cif_P1(cell, U_cif=U_cif))

def get_cif_P1(cell, U_cif=None):
    a, b, c = get_cell_parameters(cell.get_cell())
    alpha, beta, gamma = get_angles(cell.get_cell())

    cif = """data_crystal_structure_P1

_symmetry_space_group_name_H-M     'P 1'
_symmetry_Int_Tables_number        1

_cell_length_a                     %.5f
_cell_length_b                     %.5f
_cell_length_c                     %.5f
_cell_angle_alpha                  %.5f
_cell_angle_beta                   %.5f
_cell_angle_gamma                  %.5f
_cell_volume                       %.5f
_cell_formula_units_Z              1

loop_
_space_group_symop_operation_xyz
x,y,z

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy\n""" % (a, b, c, alpha, beta, gamma, cell.get_volume())

    symbols = []
    for s, p in zip(cell.get_chemical_symbols(), cell.get_scaled_positions()):
        symbols.append(s)
        cif += ("%-7s%2s %10.5f%10.5f%10.5f   1.00000\n" %
                (s + "%d" % symbols.count(s), s, p[0], p[1], p[2]))

    if U_cif is not None:

        aniso_U = """loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12\n"""

        cif += aniso_U
    
        symbols = []
        for i, s in enumerate(cell.get_chemical_symbols()):
            symbols.append(s)
            m = U_cif[i]
            vals = (m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1])
            cif += ("%6s %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" %
                    ((s + "%d" % symbols.count(s),) + vals))

    return cif
