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
from phonopy.interface.calculator import (
    read_crystal_structure, get_default_cell_filename)
from phonopy.interface.vasp import read_vasp


def collect_cell_info(supercell_matrix=None,
                      primitive_matrix=None,
                      interface_mode=None,
                      cell_filename=None,
                      chemical_symbols=None,
                      enforce_primitive_matrix_auto=False,
                      command_name="phonopy",
                      symprec=1e-5):
    # Check fallback to phonopy_yaml interface.
    # If no fallback happens, fallback_reason is None.
    _interface_mode, fallback_reason = _fallback_to_phonopy_yaml(
        supercell_matrix,
        interface_mode,
        cell_filename)

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
            if fallback_reason == "no supercell matrix given":
                if cell_filename is None:
                    msg = ["Supercell matrix (DIM or --dim) is not specified. "
                           "To run phonopy without",
                           "explicitly setting supercell matrix, \"%s\" or \"%s\" "
                           % fname_list[:2],
                           "must exist in the current directory."]
                    msg_list += msg
                else:
                    msg_list.append("Supercell matrix (DIM or --dim) may be "
                                    "forgotten to be specified.")
            elif fallback_reason == "default file not found":
                msg_list = ["Any crystal structure file was not found.", ""]
            elif fallback_reason == "read_vasp failed":
                msg_list = []

                if cell_filename:
                    msg_list.append(
                        "Parsing crystal structure file of \"%s\" failed."
                        % cell_filename)
                else:
                    msg_list.append(
                        "Parsing crystal structure file failed.")
                msg_list += [
                    "Calculator interface may not be given correctly.", ""]

        return "\n".join(msg_list)

    if supercell_matrix_out is None:
        return "Supercell matrix (DIM or --dim) is not specified."

    # Check unit cell
    if np.linalg.det(unitcell.get_cell()) < 0.0:
        return "Lattice vectors have to follow the right-hand rule."

    return (unitcell, supercell_matrix_out, primitive_matrix_out,
            unitcell_filename, optional_structure_info, interface_mode_out,
            has_read_phonopy_yaml)


def _fallback_to_phonopy_yaml(supercell_matrix,
                              interface_mode,
                              cell_filename):
    fallback_reason = None
    if supercell_matrix is None:
        fallback_reason = "no supercell matrix given"
    elif interface_mode is None:
        try:
            if cell_filename is None:
                read_vasp(get_default_cell_filename('vasp'))
            else:
                read_vasp(cell_filename)
        except ValueError as e:
            # read_vasp parsing failed.
            fallback_reason = "read_vasp failed"
        except FileNotFoundError as e:
            if cell_filename is None:
                fallback_reason = "default file not found"
            else:
                # Do nothing. If this is a problem, this is handled in the
                # following part (read_crystal_structure).
                pass

    if fallback_reason:
        _interface_mode = 'phonopy_yaml'
    elif interface_mode is None:
        _interface_mode = None
    else:
        _interface_mode = interface_mode.lower()

    return _interface_mode, fallback_reason
