# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CIF tools."""

from __future__ import annotations

from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_angles, get_cell_parameters


def write_cif_P1(cell: PhonopyAtoms, U_cif=None, filename=None):
    """Write P1 symmetry CIF file."""
    if filename:
        with open(filename, "w") as w:
            w.write(get_cif_P1(cell, U_cif=U_cif))


def get_cif_P1(cell: PhonopyAtoms, U_cif=None):
    """Return P1 symmetry CIF text."""
    a, b, c = get_cell_parameters(cell.cell)
    alpha, beta, gamma = get_angles(cell.cell)

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
_atom_site_occupancy\n""" % (
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        cell.volume,
    )

    symbols = []
    for s, p in zip(cell.symbols, cell.scaled_positions, strict=True):
        symbols.append(s)
        cif += "%-7s%2s %10.5f%10.5f%10.5f   1.00000\n" % (
            s + "%d" % symbols.count(s),
            s,
            p[0],
            p[1],
            p[2],
        )

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
        for i, s in enumerate(cell.symbols):
            symbols.append(s)
            m = U_cif[i]
            vals = (m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1])
            cif += "%6s %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n" % (
                (s + "%d" % symbols.count(s),) + vals
            )

    return cif
