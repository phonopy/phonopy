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
                           interface_mode=None,
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
        from phonopy.interface.phonopy_yaml import PhonopyYaml
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(unitcell_filename)
        unitcell = phpy_yaml.get_unitcell()
        return unitcell, (unitcell_filename,)

    if interface_mode is None or interface_mode == 'vasp':
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

    if interface_mode == 'crystal':
        from phonopy.interface.crystal import read_crystal
        unitcell, conv_numbers = read_crystal(unitcell_filename)
        return unitcell, (unitcell_filename, conv_numbers)

def get_default_cell_filename(interface_mode, yaml_mode):
    if yaml_mode:
        return "POSCAR.yaml"
    if interface_mode is None or interface_mode == 'vasp':
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
    if interface_mode == 'crystal':
        return "crystal.o"

def create_FORCE_SETS(interface_mode,
                      force_filenames,
                      symprec=1e-5,
                      is_wien2k_p1=False,
                      force_sets_zero_mode=False,
                      disp_filename='disp.yaml',
                      force_sets_filename='FORCE_SETS',
                      log_level=0):
    if (interface_mode is None or
        interface_mode == 'vasp' or
        interface_mode == 'abinit' or
        interface_mode == 'elk' or
        interface_mode == 'pwscf' or
        interface_mode == 'siesta' or
        interface_mode == 'crystal'):
        disp_dataset = parse_disp_yaml(filename=disp_filename)
        num_atoms = disp_dataset['natom']
        num_displacements = len(disp_dataset['first_atoms'])
        if force_sets_zero_mode:
            num_displacements += 1
        force_sets = get_force_sets(interface_mode,
                                    num_atoms,
                                    num_displacements,
                                    force_filenames,
                                    disp_filename,
                                    verbose=(log_level > 0))

    elif interface_mode == 'wien2k':
        disp_dataset, supercell = parse_disp_yaml(filename=disp_filename,
                                                  return_cell=True)
        from phonopy.interface.wien2k import parse_set_of_forces
        num_displacements = len(disp_dataset['first_atoms'])
        if force_sets_zero_mode:
            num_displacements += 1
        if _check_number_of_files(num_displacements,
                                  force_filenames,
                                  disp_filename):
            force_sets = []
        else:
            disps = [[d['number'], d['displacement']]
                     for d in disp_dataset['first_atoms']]
            force_sets = parse_set_of_forces(
                disps,
                force_filenames,
                supercell,
                is_distribute=(not is_wien2k_p1),
                symprec=symprec,
                verbose=(log_level > 0))
    else:
        force_sets = []

    if force_sets:
        if force_sets_zero_mode:
            force_sets = _subtract_residual_forces(force_sets)
        for forces, disp in zip(force_sets, disp_dataset['first_atoms']):
            disp['forces'] = forces
        write_FORCE_SETS(disp_dataset, filename=force_sets_filename)

    if log_level > 0:
        if force_sets:
            print("%s has been created." % force_sets_filename)
        else:
            print("%s could not be created." % force_sets_filename)

    return 0

def get_force_sets(interface_mode,
                   num_atoms,
                   num_displacements,
                   force_filenames,
                   disp_filename,
                   verbose=True):
    if _check_number_of_files(num_displacements,
                              force_filenames,
                              disp_filename):
        return []

    if interface_mode is None or interface_mode == 'vasp':
        from phonopy.interface.vasp import parse_set_of_forces
    elif interface_mode == 'abinit':
        from phonopy.interface.abinit import parse_set_of_forces
    elif interface_mode == 'pwscf':
        from phonopy.interface.pwscf import parse_set_of_forces
    elif interface_mode == 'elk':
        from phonopy.interface.elk import parse_set_of_forces
    elif interface_mode == 'siesta':
        from phonopy.interface.siesta import parse_set_of_forces
    elif interface_mode == 'crystal':
        from phonopy.interface.crystal import parse_set_of_forces
    else:
        return []

    force_sets = parse_set_of_forces(num_atoms,
                                     force_filenames,
                                     verbose=verbose)

    return force_sets

def _check_number_of_files(num_displacements,
                           force_filenames,
                           disp_filename):
    if num_displacements != len(force_filenames):
        print('')
        print("Number of files to be read (%d) don't match to" %
              len(force_filenames))
        print("the number of displacements (%d) in %s." %
              (num_displacements, disp_filename))
        return 1
    else:
        return 0

def _subtract_residual_forces(force_sets):
    for i in range(1, len(force_sets)):
        force_sets[i] -= force_sets[0]
    return force_sets[1:]
