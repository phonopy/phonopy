# Copyright (C) 2018 Atsushi Togo
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
from phonopy.interface import read_crystal_structure
from phonopy.interface.vasp import read_vasp


def collect_cell_info(supercell_matrix=None,
                      primitive_matrix=None,
                      interface_mode=None,
                      cell_filename=None,
                      chemical_symbols=None,
                      enforce_primitive_matrix_auto=False,
                      command_name="phonopy",
                      symprec=1e-5):
    if supercell_matrix is None:
        _interface_mode = "phonopy_yaml"
    elif interface_mode is None:
        try:
            read_vasp(cell_filename)
        except (ValueError, TypeError):
            # TypeError occurs when cell_filename is None.
            # ValueError occurs in parsing POSCAR like file.
            _interface_mode = "phonopy_yaml"
    else:
        _interface_mode = interface_mode

    unitcell, optional_structure_info = read_crystal_structure(
        filename=cell_filename,
        interface_mode=_interface_mode,
        chemical_symbols=chemical_symbols,
        command_name=command_name)
    unitcell_filename = optional_structure_info[0]

    if _interface_mode == 'phonopy_yaml' and unitcell is not None:
        if optional_structure_info[1] is None:
            interface_mode_out = interface_mode
        else:
            interface_mode_out = optional_structure_info[1]
        if optional_structure_info[2] is None:
            _supercell_matrix = supercell_matrix
        else:
            _supercell_matrix = optional_structure_info[2]
        if primitive_matrix is not None:
            _primitive_matrix = primitive_matrix
        elif optional_structure_info[3] is not None:
            _primitive_matrix = optional_structure_info[3]
        else:
            _primitive_matrix = 'auto'
        has_read_phonopy_yaml = True
    else:
        interface_mode_out = _interface_mode
        _supercell_matrix = supercell_matrix
        _primitive_matrix = primitive_matrix
        has_read_phonopy_yaml = False

    if enforce_primitive_matrix_auto:
        _primitive_matrix = 'auto'

    if _supercell_matrix is None and _primitive_matrix == 'auto':
        supercell_matrix_out = np.eye(3, dtype='intc')
    else:
        supercell_matrix_out = _supercell_matrix

    primitive_matrix_out = _primitive_matrix

    if unitcell is None:
        fname_list = optional_structure_info
        if len(fname_list) == 1:
            msg = "Crystal structure file of \"%s\"" % fname_list[0]
            msg_list = ["%s was not found." % msg, ]
        elif len(fname_list) == 2:
            msg = "Crystal structure file of \"%s\" %s" % fname_list
            msg_list = ["%s was not found." % msg, ]
        elif len(fname_list) == 4:
            msg_list = []
            if cell_filename is None:
                msg = ("\"%s\" or \"%s\" should exist in the current directory"
                       % fname_list[:-1])
                msg_list.append(msg)
                msg = "to run without setting supercell matrix (DIM or --dim)."
                msg_list.append(msg)
            else:
                msg_list.append("Supercell matrix (DIM or --dim) may be "
                                "forgotten to be specified.")
        return "\n".join(msg_list)

    if supercell_matrix_out is None:
        return "Supercell matrix (DIM or --dim) is not specified."

    # Check unit cell
    if np.linalg.det(unitcell.get_cell()) < 0.0:
        return "Lattice vectors have to follow the right-hand rule."

    return (unitcell, supercell_matrix_out, primitive_matrix_out,
            unitcell_filename, optional_structure_info, interface_mode_out,
            has_read_phonopy_yaml)
