# Copyright (C) 2014 Atsushi Togo
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

import os
from phonopy.file_IO import parse_disp_yaml, write_FORCE_SETS

def read_crystal_structure(filename=None,
                           interface_mode='vasp',
                           chemical_symbols=None,
                           yaml_mode=False):
    if filename is None:
        unitcell_filename = get_default_cell_filename(interface_mode, yaml_mode)
    else:
        unitcell_filename = filename

    if not os.path.exists(unitcell_filename):
        if filename is None:
            return None, (unitcell_filename + " (default file name)",)
        else:
            return None, (unitcell_filename,)
    
    if yaml_mode:
        from phonopy.interface.phonopy_yaml import phonopyYaml
        unitcell = phonopyYaml(unitcell_filename).get_atoms()
        return unitcell, (unitcell_filename,)
        
    if interface_mode == 'vasp':
        from phonopy.interface.vasp import read_vasp
        if chemical_symbols is None:
            unitcell = read_vasp(unitcell_filename)
        else:
            unitcell = read_vasp(unitcell_filename, symbols=chemical_symbols)
        return unitcell, (unitcell_filename,)
        
    if interface_mode == 'abinit':
        from phonopy.interface.abinit import read_abinit
        unitcell = read_abinit(unitcell_filename)
        return unitcell, (unitcell_filename,)
        
    if interface_mode == 'pwscf':
        from phonopy.interface.pwscf import read_pwscf
        unitcell, pp_filenames = read_pwscf(unitcell_filename)
        return unitcell, (unitcell_filename, pp_filenames)
        
    if interface_mode == 'wien2k':
        from phonopy.interface.wien2k import parse_wien2k_struct
        unitcell, npts, r0s, rmts = parse_wien2k_struct(unitcell_filename)
        return unitcell, (unitcell_filename, npts, r0s, rmts)

    if interface_mode == 'elk':
        from phonopy.interface.elk import read_elk
        unitcell, sp_filenames = read_elk(unitcell_filename)
        return unitcell, (unitcell_filename, sp_filenames)

    if interface_mode == 'siesta':
        from phonopy.interface.siesta import read_siesta
        unitcell, atypes = read_siesta(unitcell_filename)
        return unitcell, (unitcell_filename, atypes)

def get_default_cell_filename(interface_mode, yaml_mode):
    if yaml_mode:
        return "POSCAR.yaml"
    if interface_mode == 'vasp':
        return "POSCAR"
    if interface_mode == 'abinit':
        return "unitcell.in"
    if interface_mode == 'pwscf':
        return "unitcell.in"
    if interface_mode == 'wien2k':
        return "case.struct"
    if interface_mode == 'elk':
        return "elk.in"
    if interface_mode == 'siesta':
        return "input.fdf" 

def create_FORCE_SETS(interface_mode,
                      force_filenames,
                      options,
                      log_level=0):
    if (interface_mode == 'vasp' or
        interface_mode == 'abinit' or
        interface_mode == 'elk' or
        interface_mode == 'pwscf' or
        interface_mode == 'siesta'):
        displacements = parse_disp_yaml(filename='disp.yaml')
        num_atoms = displacements['natom']
        if len(displacements['first_atoms']) != len(force_filenames):
            print('')
            print("Number of files to be read don't match "
                  "to number of displacements in disp.yaml.")
            return 1
        force_sets = get_force_sets(interface_mode, num_atoms, force_filenames)

    elif interface_mode == 'wien2k':
        displacements, supercell = parse_disp_yaml(filename='disp.yaml',
                                                   return_cell=True)
        if len(displacements['first_atoms']) != len(force_filenames):
            print('')
            print("Number of files to be read don't match "
                  "to number of displacements in disp.yaml.")
            return 1
        from phonopy.interface.wien2k import parse_set_of_forces
        force_sets = parse_set_of_forces(
            displacements,
            force_filenames,
            supercell,
            disp_keyword='first_atoms',
            is_distribute=(not options.is_wien2k_p1),
            symprec=options.symprec)
    else:
        force_sets = []

    if force_sets:
        for forces, disp in zip(force_sets, displacements['first_atoms']):
            disp['forces'] = forces
        write_FORCE_SETS(displacements, filename='FORCE_SETS')
        
    if log_level > 0:
        if force_sets:
            print("FORCE_SETS has been created.")
        else:
            print("FORCE_SETS could not be created.")

    return 0
            
def get_force_sets(interface_mode, num_atoms, force_filenames):
    if interface_mode == 'vasp':
        from phonopy.interface.vasp import parse_set_of_forces
        force_sets = parse_set_of_forces(num_atoms, force_filenames)
    elif interface_mode == 'abinit':
        from phonopy.interface.abinit import parse_set_of_forces
        force_sets = parse_set_of_forces(num_atoms, force_filenames)
    elif interface_mode == 'pwscf':
        from phonopy.interface.pwscf import parse_set_of_forces
        force_sets = parse_set_of_forces(num_atoms, force_filenames)
    elif interface_mode == 'elk':
        from phonopy.interface.elk import parse_set_of_forces
        force_sets = parse_set_of_forces(num_atoms, force_filenames)
    elif interface_mode == 'siesta':
        from phonopy.interface.siesta import parse_set_of_forces
        force_sets = parse_set_of_forces(num_atoms, force_filenames)

    return force_sets
