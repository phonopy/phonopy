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
import numpy as np
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.file_IO import parse_disp_yaml, write_FORCE_SETS


def get_interface_mode(args):
    """Return calculator name

    The calculator name is obtained from command option arguments where
    argparse is used. The argument attribute name has to be
    "{calculator}_mode". Then this method returns {calculator}.

    """

    calculator_list = ['wien2k', 'abinit', 'qe', 'elk', 'siesta', 'cp2k',
                       'crystal', 'vasp', 'dftbp']
    for calculator in calculator_list:
        mode = "%s_mode" % calculator
        if mode in args and args.__dict__[mode]:
            return calculator
    return None


def write_supercells_with_displacements(interface_mode,
                                        supercell,
                                        cells_with_disps,
                                        num_unitcells_in_supercell,
                                        optional_structure_info):
    if interface_mode is None or interface_mode == 'vasp':
        from phonopy.interface.vasp import write_supercells_with_displacements
        write_supercells_with_displacements(supercell, cells_with_disps)
    elif interface_mode is 'abinit':
        from phonopy.interface.abinit import write_supercells_with_displacements
        write_supercells_with_displacements(supercell, cells_with_disps)
    elif interface_mode is 'qe':
        from phonopy.interface.qe import write_supercells_with_displacements
        write_supercells_with_displacements(supercell,
                                            cells_with_disps,
                                            optional_structure_info[1])
    elif interface_mode == 'wien2k':
        from phonopy.interface.wien2k import write_supercells_with_displacements
        unitcell_filename, npts, r0s, rmts = optional_structure_info
        write_supercells_with_displacements(
            supercell,
            cells_with_disps,
            npts,
            r0s,
            rmts,
            num_unitcells_in_supercell,
            filename=unitcell_filename)
    elif interface_mode == 'elk':
        from phonopy.interface.elk import write_supercells_with_displacements
        write_supercells_with_displacements(supercell,
                                            cells_with_disps,
                                            optional_structure_info[1])
    elif interface_mode == 'siesta':
        from phonopy.interface.siesta import write_supercells_with_displacements
        write_supercells_with_displacements(supercell,
                                            cells_with_disps,
                                            optional_structure_info[1])
    elif interface_mode == 'cp2k':
        from phonopy.interface.cp2k import write_supercells_with_displacements
        write_supercells_with_displacements(supercell, cells_with_disps)
    elif interface_mode == 'crystal':
        from phonopy.interface.crystal import write_supercells_with_displacements
        write_supercells_with_displacements(supercell,
                                            cells_with_disps,
                                            optional_structure_info[1],
                                            num_unitcells_in_supercell,
                                            template_file="TEMPLATE")
    elif interface_mode == 'dftbp':
        from phonopy.interface.dftbp import write_supercells_with_displacements
        write_supercells_with_displacements(supercell, cells_with_disps)


def read_crystal_structure(filename=None,
                           interface_mode=None,
                           chemical_symbols=None,
                           command_name="phonopy"):
    if interface_mode == 'phonopy_yaml':
        return _read_phonopy_yaml(filename, command_name)

    if filename is None:
        cell_filename = get_default_cell_filename(interface_mode)
        if not os.path.isfile(cell_filename):
            return None, (cell_filename, "(default file name)")
    else:
        cell_filename = filename
        if not os.path.isfile(cell_filename):
            return None, (cell_filename,)

    if interface_mode is None or interface_mode == 'vasp':
        from phonopy.interface.vasp import read_vasp
        if chemical_symbols is None:
            unitcell = read_vasp(cell_filename)
        else:
            unitcell = read_vasp(cell_filename, symbols=chemical_symbols)
        return unitcell, (cell_filename,)
    elif interface_mode == 'abinit':
        from phonopy.interface.abinit import read_abinit
        unitcell = read_abinit(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == 'qe':
        from phonopy.interface.qe import read_pwscf
        unitcell, pp_filenames = read_pwscf(cell_filename)
        return unitcell, (cell_filename, pp_filenames)
    elif interface_mode == 'wien2k':
        from phonopy.interface.wien2k import parse_wien2k_struct
        unitcell, npts, r0s, rmts = parse_wien2k_struct(cell_filename)
        return unitcell, (cell_filename, npts, r0s, rmts)
    elif interface_mode == 'elk':
        from phonopy.interface.elk import read_elk
        unitcell, sp_filenames = read_elk(cell_filename)
        return unitcell, (cell_filename, sp_filenames)
    elif interface_mode == 'siesta':
        from phonopy.interface.siesta import read_siesta
        unitcell, atypes = read_siesta(cell_filename)
        return unitcell, (cell_filename, atypes)
    elif interface_mode == 'cp2k':
        from phonopy.interface.cp2k import read_cp2k
        unitcell = read_cp2k(cell_filename)
        return unitcell, (cell_filename,)
    elif interface_mode == 'crystal':
        from phonopy.interface.crystal import read_crystal
        unitcell, conv_numbers = read_crystal(cell_filename)
        return unitcell, (cell_filename, conv_numbers)
    elif interface_mode == 'dftbp':
        from phonopy.interface.dftbp import read_dftbp
        unitcell = read_dftbp(cell_filename)
        return unitcell, (cell_filename,)


def get_default_cell_filename(interface_mode):
    if interface_mode is None or interface_mode == 'vasp':
        return "POSCAR"
    elif interface_mode in ('abinit', 'qe'):
        return "unitcell.in"
    elif interface_mode == 'wien2k':
        return "case.struct"
    elif interface_mode == 'elk':
        return "elk.in"
    elif interface_mode == 'siesta':
        return "input.fdf"
    elif interface_mode == 'cp2k':
        return "unitcell.inp"
    elif interface_mode == 'crystal':
        return "crystal.o"
    elif interface_mode == 'dftbp':
        return "geo.gen"
    else:
        return None


def get_default_supercell_filename(interface_mode):
    if interface_mode == 'phonopy_yaml':
        return "phonopy_disp.yaml"
    elif interface_mode is None or interface_mode == 'vasp':
        return "SPOSCAR"
    elif interface_mode in ('abinit', 'elk', 'qe'):
        return "supercell.in"
    elif interface_mode == 'wien2k':
        return "case.structS"
    elif interface_mode == 'siesta':
        return "supercell.fdf"
    elif interface_mode == 'cp2k':
        return "supercell.inp"
    elif interface_mode == 'crystal':
        return None  # supercell.ext can not be parsed by crystal interface.
    elif interface_mode == 'dftbp':
        return "geo.genS"
    else:
        return None


def get_default_displacement_distance(interface_mode):
    if interface_mode in ('wien2k', 'abinit', 'elk', 'qe', 'siesta', 'cp2k'):
        displacement_distance = 0.02
    elif interface_mode == 'crystal':
        displacement_distance = 0.01
    else:  # default or vasp
        displacement_distance = 0.01
    return displacement_distance


def get_default_physical_units(interface_mode):
    """Return physical units used for calculators

    Physical units: energy,  distance,  atomic mass, force
    vasp          : eV,      Angstrom,  AMU,         eV/Angstrom
    wien2k        : Ry,      au(=borh), AMU,         mRy/au
    abinit        : hartree, au,        AMU,         eV/Angstrom
    elk           : hartree, au,        AMU,         hartree/au
    qe            : Ry,      au,        AMU,         Ry/au
    siesta        : eV,      au,        AMU,         eV/Angstroem
    CRYSTAL       : eV,      Angstrom,  AMU,         eV/Angstroem
    DFTB+         : hartree, au,        AMU          hartree/au

    """

    from phonopy.units import (Wien2kToTHz, AbinitToTHz, PwscfToTHz, ElkToTHz,
                               SiestaToTHz, VaspToTHz, CP2KToTHz, CrystalToTHz,
                               DftbpToTHz, Hartree, Bohr)

    units = {'factor': None,
             'nac_factor': None,
             'distance_to_A': None,
             'force_constants_unit': None,
             'length_unit': None}

    if interface_mode is None or interface_mode == 'vasp':
        units['factor'] = VaspToTHz
        units['nac_factor'] = Hartree * Bohr
        units['distance_to_A'] = 1.0
        units['force_constants_unit'] = 'eV/Angstrom^2'
        units['length_unit'] = 'Angstrom'
    elif interface_mode == 'abinit':
        units['factor'] = AbinitToTHz
        units['nac_factor'] = Hartree / Bohr
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'eV/Angstrom.au'
        units['length_unit'] = 'au'
    elif interface_mode == 'qe':
        units['factor'] = PwscfToTHz
        units['nac_factor'] = 2.0
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'Ry/au^2'
        units['length_unit'] = 'au'
    elif interface_mode == 'wien2k':
        units['factor'] = Wien2kToTHz
        units['nac_factor'] = 2000.0
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'mRy/au^2'
        units['length_unit'] = 'au'
    elif interface_mode == 'elk':
        units['factor'] = ElkToTHz
        units['nac_factor'] = 1.0
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'hartree/au^2'
        units['length_unit'] = 'au'
    elif interface_mode == 'siesta':
        units['factor'] = SiestaToTHz
        units['nac_factor'] = Hartree / Bohr
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'eV/Angstrom.au'
        units['length_unit'] = 'au'
    elif interface_mode == 'cp2k':
        units['factor'] = CP2KToTHz
        units['nac_factor'] = Hartree / Bohr  # in a.u.
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'hartree/au^2'
        units['length_unit'] = 'Angstrom'
    elif interface_mode == 'crystal':
        units['factor'] = CrystalToTHz
        units['nac_factor'] = Hartree * Bohr
        units['distance_to_A'] = 1.0
        units['force_constants_unit'] = 'eV/Angstrom^2'
        units['length_unit'] = 'Angstrom'
    elif interface_mode == 'dftbp':
        units['factor'] = DftbpToTHz
        units['nac_factor'] = Hartree * Bohr
        units['distance_to_A'] = Bohr
        units['force_constants_unit'] = 'hartree/au^2'
        units['length_unit'] = 'au'

    return units


def create_FORCE_SETS(interface_mode,
                      force_filenames,
                      symprec=1e-5,
                      is_wien2k_p1=False,
                      force_sets_zero_mode=False,
                      disp_filename='disp.yaml',
                      force_sets_filename='FORCE_SETS',
                      log_level=0):
    if log_level > 0:
        if interface_mode:
            print("Calculator interface: %s" % interface_mode)
        print("Displacements were read from \"%s\"." % disp_filename)
        if disp_filename == 'disp.yaml':
            print('')
            print("NOTE:")
            print("  From phonopy v2.0, displacements are written into "
                  "\"phonopy_disp.yaml\".")
            print("  \"disp.yaml\" is still supported for reading, but is "
                  "deprecated.")
            print('')
        if force_sets_zero_mode:
            print("Forces in %s are subtracted from forces in all "
                  "other files." % force_filenames[0])

    if interface_mode in (None, 'vasp', 'abinit', 'elk', 'qe', 'siesta',
                          'cp2k', 'crystal', 'dftbp'):
        disp_dataset = parse_disp_yaml(filename=disp_filename)
        num_atoms = disp_dataset['natom']
        num_displacements = len(disp_dataset['first_atoms'])
        if force_sets_zero_mode:
            num_displacements += 1
        force_sets = get_force_sets(interface_mode,
                                    num_atoms,
                                    num_displacements,
                                    force_filenames,
                                    disp_filename=disp_filename,
                                    check_number_of_files=True,
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
                   disp_filename=None,
                   check_number_of_files=False,
                   verbose=True):
    if check_number_of_files:
        if _check_number_of_files(num_displacements,
                                  force_filenames,
                                  disp_filename):
            return []

    if interface_mode is None or interface_mode == 'vasp':
        from phonopy.interface.vasp import parse_set_of_forces
    elif interface_mode == 'abinit':
        from phonopy.interface.abinit import parse_set_of_forces
    elif interface_mode == 'qe':
        from phonopy.interface.qe import parse_set_of_forces
    elif interface_mode == 'elk':
        from phonopy.interface.elk import parse_set_of_forces
    elif interface_mode == 'siesta':
        from phonopy.interface.siesta import parse_set_of_forces
    elif interface_mode == 'cp2k':
        from phonopy.interface.cp2k import parse_set_of_forces
    elif interface_mode == 'crystal':
        from phonopy.interface.crystal import parse_set_of_forces
    elif interface_mode == 'dftbp':
        from phonopy.interface.dftbp import parse_set_of_forces
    else:
        return []

    force_sets = parse_set_of_forces(num_atoms,
                                     force_filenames,
                                     verbose=verbose)

    return force_sets


def _read_phonopy_yaml(filename, command_name):
    cell_filename = None
    for fname in (filename,
                  "%s_disp.yaml" % command_name,
                  "%s.yaml" % command_name):
        if fname and os.path.isfile(fname):
            cell_filename = fname
            break
    if cell_filename is None:
        return None, ("%s_disp.yaml" % command_name,
                      "%s.yaml" % command_name, "")

    phpy = PhonopyYaml()
    try:
        phpy.read(cell_filename)
        cell = phpy.unitcell
        calculator = None
        if command_name in phpy.yaml:
            if 'calculator' in phpy.yaml[command_name]:
                calculator = phpy.yaml[command_name]['calculator']
        if ('supercell_matrix' in phpy.yaml and
            phpy.yaml['supercell_matrix'] is not None):
            smat = phpy.supercell_matrix
        else:
            smat = None
        if ('primitive_matrix' in phpy.yaml and
            phpy.yaml['primitive_matrix'] is not None):
            pmat = phpy.primitive_matrix
        else:
            pmat = None

        return cell, (cell_filename, calculator, smat, pmat)
    except TypeError:
        return None, (cell_filename, None, None, None)


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
