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
import sys
import numpy as np
from phonopy.file_IO import parse_disp_yaml, write_FORCE_SETS

try:
    import yaml
except ImportError:
    print("You need to install python-yaml.")
    sys.exit(1)

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from phonopy.structure.atoms import PhonopyAtoms as Atoms


def get_interface_mode(args):
    if args.wien2k_mode:
        return 'wien2k'
    elif args.abinit_mode:
        return 'abinit'
    elif args.qe_mode:
        return 'qe'
    elif args.elk_mode:
        return 'elk'
    elif args.siesta_mode:
        return 'siesta'
    elif args.cp2k_mode:
        return 'cp2k'
    elif args.crystal_mode:
        return 'crystal'
    elif args.vasp_mode:
        return 'vasp'
    else:
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


def read_crystal_structure(filename=None,
                           interface_mode=None,
                           chemical_symbols=None,
                           yaml_mode=False):
    if filename is None:
        unitcell_filename = get_default_cell_filename(interface_mode,
                                                      yaml_mode)
    else:
        unitcell_filename = filename

    if not os.path.isfile(unitcell_filename):
        if filename is None:
            return None, (unitcell_filename + " (default file name)",)
        else:
            return None, (unitcell_filename,)

    if yaml_mode:
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

    if interface_mode == 'qe':
        from phonopy.interface.qe import read_pwscf
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

    if interface_mode == 'cp2k':
        from phonopy.interface.cp2k import read_cp2k
        unitcell = read_cp2k(unitcell_filename)
        return unitcell, (unitcell_filename,)

    if interface_mode == 'crystal':
        from phonopy.interface.crystal import read_crystal
        unitcell, conv_numbers = read_crystal(unitcell_filename)
        return unitcell, (unitcell_filename, conv_numbers)


def get_default_cell_filename(interface_mode, yaml_mode):
    if yaml_mode:
        return "POSCAR.yaml"
    if interface_mode is None or interface_mode == 'vasp':
        return "POSCAR"
    if interface_mode in ('abinit', 'qe'):
        return "unitcell.in"
    if interface_mode == 'wien2k':
        return "case.struct"
    if interface_mode == 'elk':
        return "elk.in"
    if interface_mode == 'siesta':
        return "input.fdf"
    if interface_mode == 'cp2k':
        return "unitcell.inp"
    if interface_mode == 'crystal':
        return "crystal.o"


def get_default_supercell_filename(interface_mode, yaml_mode):
    if yaml_mode:
        return "SPOSCAR.yaml"
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

    """

    from phonopy.units import (Wien2kToTHz, AbinitToTHz, PwscfToTHz, ElkToTHz,
                               SiestaToTHz, VaspToTHz, CP2KToTHz, CrystalToTHz,
                               Hartree, Bohr)

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

    return units


def create_FORCE_SETS(interface_mode,
                      force_filenames,
                      symprec=1e-5,
                      is_wien2k_p1=False,
                      force_sets_zero_mode=False,
                      disp_filename='disp.yaml',
                      force_sets_filename='FORCE_SETS',
                      log_level=0):
    if interface_mode in (None, 'vasp', 'abinit', 'elk', 'qe', 'siesta',
                          'cp2k', 'crystal'):
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
    else:
        return []

    force_sets = parse_set_of_forces(num_atoms,
                                     force_filenames,
                                     verbose=verbose)

    return force_sets


def get_unitcell_from_phonopy_yaml(filename):
    ph_yaml = PhonopyYaml()
    ph_yaml.read(filename)
    return ph_yaml.get_unitcell()


class PhonopyYaml(object):
    def __init__(self,
                 configuration=None,
                 calculator=None,
                 show_force_constants=False):
        self._configuration = configuration
        self._calculator = calculator
        self._show_force_constants = show_force_constants

        self._unitcell = None
        self._primitive = None
        self._supercell = None
        self._supercell_matrix = None
        self._symmetry = None  # symmetry of supercell
        self._primitive_matrix = None
        self._force_constants = None
        self._s2p_map = None
        self._u2p_map = None
        self._nac_params = None
        self._version = None

    def get_unitcell(self):
        return self._unitcell

    def set_unitcell(self, cell):
        self._unitcell = cell

    def get_primitive(self):
        return self._primitive

    def get_supercell(self):
        return self._supercell

    def read(self, filename):
        with open(filename) as infile:
            self._load(infile)

    def set_phonon_info(self, phonopy):
        self._version = phonopy.get_version()
        self._unitcell = phonopy.get_unitcell()
        self._primitive = phonopy.get_primitive()
        self._supercell = phonopy.get_supercell()
        self._supercell_matrix = phonopy.get_supercell_matrix()
        self._symmetry = phonopy.get_symmetry()
        self._primitive_matrix = phonopy.get_primitive_matrix()
        self._force_constants = phonopy.get_force_constants()
        self._s2p_map = self._primitive.get_supercell_to_primitive_map()
        u2s_map = self._supercell.get_unitcell_to_supercell_map()
        u2u_map = self._supercell.get_unitcell_to_unitcell_map()
        s2u_map = self._supercell.get_supercell_to_unitcell_map()
        self._u2p_map = [u2u_map[i] for i in (s2u_map[self._s2p_map])[u2s_map]]
        self._nac_params = phonopy.get_nac_params()

    def get_yaml_lines(self):
        units = get_default_physical_units(self._calculator)
        lines = []
        nac_factor = None
        if self._primitive is None:
            symbols = None
        else:
            symbols = self._primitive.get_chemical_symbols()
        if self._nac_params is not None:
            born = self._nac_params['born']
            nac_factor = self._nac_params['factor']
            dielectric = self._nac_params['dielectric']

        if self._version:
            lines.append("phonopy:")
            lines.append("  version: %s" % self._version)
        if self._calculator:
            lines.append("  calculator: %s" % self._calculator)
        if self._nac_params:
            lines.append("  nac_unit_conversion_factor: %f" % nac_factor)
        if self._configuration is not None:
            lines.append("  configuration:")
            for key in self._configuration:
                lines.append("    %s: \"%s\"" %
                             (key, self._configuration[key]))
            lines.append("")

        lines.append("physical_unit:")
        lines.append("  atomic_mass: \"AMU\"")
        if units['length_unit'] is not None:
            lines.append("  length: \"%s\"" % units['length_unit'])
        if units['force_constants_unit'] is not None:
            lines.append("  force_constants: \"%s\"" %
                         units['force_constants_unit'])
        lines.append("")

        if self._supercell_matrix is not None:
            lines.append("supercell_matrix:")
            for v in self._supercell_matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")

        if self._symmetry.get_dataset() is not None:
            lines.append("space_group:")
            lines.append("  type: \"%s\"" %
                         self._symmetry.get_dataset()['international'])
            lines.append("  number: %d" %
                         self._symmetry.get_dataset()['number'])
            hall_symbol = self._symmetry.get_dataset()['hall']
            if "\"" in hall_symbol:
                hall_symbol = hall_symbol.replace("\"", "\\\"")
            lines.append("  Hall_symbol: \"%s\"" % hall_symbol)
            lines.append("")

        if self._primitive_matrix is not None:
            lines.append("primitive_matrix:")
            for v in self._primitive_matrix:
                lines.append("- [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            lines.append("")

        if self._primitive is not None:
            lines.append("primitive_cell:")
            for line in self._primitive.get_yaml_lines():
                lines.append("  " + line)
            lines.append("  reciprocal_lattice: # without 2pi")
            rec_lat = np.linalg.inv(self._primitive.get_cell())
            for v, a in zip(rec_lat.T, ('a*', 'b*', 'c*')):
                lines.append("  - [ %21.15f, %21.15f, %21.15f ] # %s" %
                             (v[0], v[1], v[2], a))
            lines.append("")

        if self._unitcell is not None:
            lines.append("unit_cell:")
            count = 0
            for line in self._unitcell.get_yaml_lines():
                lines.append("  " + line)
                if self._u2p_map is not None and "mass" in line:
                    lines.append("    reduced_to: %d" %
                                 (self._u2p_map[count] + 1))
                    count += 1
            lines.append("")

        if self._supercell is not None:
            lines.append("supercell:")
            count = 0
            for line in self._supercell.get_yaml_lines():
                lines.append("  " + line)
                if self._s2p_map is not None and "mass" in line:
                    lines.append("    reduced_to: %d" %
                                 (self._s2p_map[count] + 1))
                    count += 1
            lines.append("")

        if self._nac_params is not None:
            lines.append("born_effective_charge:")
            for i, z in enumerate(born):
                text = "- # %d" % (i + 1)
                if symbols:
                    text += " (%s)" % symbols[i]
                lines.append(text)
                for v in z:
                    lines.append("  - [ %18.15f, %18.15f, %18.15f ]" %
                                 tuple(v))
            lines.append("")

            lines.append("dielectric_constant:")
            for v in dielectric:
                lines.append("  - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            lines.append("")

        if self._show_force_constants and self._force_constants is not None:
            lines.append("force_constants:")
            natom = self._supercell.get_number_of_atoms()
            for (i, j) in list(np.ndindex((natom, natom))):
                lines.append("- # (%d, %d)" % (i + 1, j + 1))
                for v in self._force_constants[i, j]:
                    lines.append("  - [ %21.15f, %21.15f, %21.15f ]" %
                                 tuple(v))

        return lines

    def __str__(self):
        return "\n".join(self.get_yaml_lines())

    def _load(self, fp):
        self._data = yaml.load(fp, Loader=Loader)
        if 'unit_cell' in self._data:
            self._unitcell = self._parse_cell(self._data['unit_cell'])
        if 'primitive_cell' in self._data:
            self._primitive = self._parse_cell(self._data['primitive_cell'])
        if 'supercell' in self._data:
            self._supercell = self._parse_cell(self._data['supercell'])
        if self._unitcell is None:
            if 'lattice' in self._data and 'points' in self._data:
                self._unitcell = self._parse_cell(self._data)

    def _parse_cell(self, cell_yaml):
        lattice = None
        if 'lattice' in cell_yaml:
            lattice = cell_yaml['lattice']
        points = []
        symbols = []
        masses = []
        if 'points' in cell_yaml:
            for x in cell_yaml['points']:
                if 'coordinates' in x:
                    points.append(x['coordinates'])
                if 'symbol' in x:
                    symbols.append(x['symbol'])
                if 'mass' in x:
                    masses.append(x['mass'])
        # For version < 1.10.9
        elif 'atoms' in cell_yaml:
            for x in cell_yaml['atoms']:
                if 'coordinates' not in x and 'position' in x:
                    points.append(x['position'])
                if 'symbol' in x:
                    symbols.append(x['symbol'])
                if 'mass' in x:
                    masses.append(x['mass'])
        return self._get_cell(lattice, points, symbols, masses=masses)

    def _get_cell(self, lattice, points, symbols, masses=None):
        if lattice:
            _lattice = lattice
        else:
            _lattice = None
        if points:
            _points = points
        else:
            _points = None
        if symbols:
            _symbols = symbols
        else:
            _symbols = None
        if masses:
            _masses = masses
        else:
            _masses = None

        if _lattice and _points and _symbols:
            return Atoms(symbols=_symbols,
                         cell=_lattice,
                         masses=_masses,
                         scaled_positions=_points)
        else:
            return None


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
