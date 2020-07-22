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
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from phonopy.structure.atoms import PhonopyAtoms


def read_cell_yaml(filename, cell_type='unitcell'):
    ph_yaml = PhonopyYaml()
    ph_yaml.read(filename)
    if ph_yaml.unitcell and cell_type == 'unitcell':
        return ph_yaml.unitcell
    elif ph_yaml.primitive and cell_type == 'primitive':
        return ph_yaml.primitive
    elif ph_yaml.supercell and cell_type == 'supercell':
        return ph_yaml.supercell
    else:
        return None


class PhonopyYaml(object):
    """PhonopyYaml is a container of phonopy setting

    This contains the writer (__str__) and reader (read) of phonopy.yaml type
    file.

    Methods
    -------
    __str__
        Return string of phonopy.yaml.
    get_yaml_lines
        Return a list of string lines of phonopy.yaml.
    read
        Read specific properties written in phonopy.yaml.
    set_phonon_info
        Copy specific properties in Phonopy instance to self.

    Attributes
    ----------
    configuration : dict
        Phonopy setting tags or options (e.g., {"DIM": "2 2 2", ...})
    calculator : str
        Force calculator.
    physical_units : dict
        Set of physical units used in this phonon calculation.
    unitcell : PhonopyAtoms
        Unit cell.
    primitive : PhonopyAtoms
        Primitive cell. The instance of Primitive class is necessary has to
        be created from the instance of Supercell class with
        np.dot(np.linalg.inv(supercell_matrix), primitive_matrix).
    supercell : PhonopyAtoms
        Supercell. The instance of Supercell class is necessary has to be
        created from unitcell with supercel_matrix.
    dataset
    supercell_matrix
    primitive_matrix
    nac_params
    force_constants
    symmetry
    s2p_map
    u2p_map
    frequency_unit_conversion_factor
    version
    yaml_filename
    settings
    command_name
    default_filenames

    """

    command_name = "phonopy"
    default_filenames = ("phonopy_params.yaml",
                         "phonopy_disp.yaml",
                         "phonopy.yaml")
    default_settings = {'force_sets': True,
                        'displacements': True,
                        'force_constants': False,
                        'born_effective_charge': True,
                        'dielectric_constant': True}

    def __init__(self,
                 configuration=None,
                 calculator=None,
                 physical_units=None,
                 settings=None):
        self.configuration = configuration
        self.calculator = calculator
        self.physical_units = physical_units

        self.unitcell = None
        self.primitive = None
        self.supercell = None
        self.dataset = None
        self.supercell_matrix = None
        self.primitive_matrix = None
        self.nac_params = None
        self.force_constants = None

        self.symmetry = None  # symmetry of supercell
        self.s2p_map = None
        self.u2p_map = None
        self.frequency_unit_conversion_factor = None
        self.version = None
        self.yaml_filename = None

        self.settings = self.default_settings.copy()
        if type(settings) is dict:
            self.settings.update(settings)

        self._yaml = None

    def __str__(self):
        return "\n".join(self.get_yaml_lines())

    def read(self, filename):
        self.yaml_filename = filename
        with open(filename) as infile:
            self._load(infile)

    @property
    def yaml_data(self):
        return self._yaml

    @yaml_data.setter
    def yaml_data(self, yaml_data):
        self._yaml = yaml_data

    def parse(self):
        self._parse_transformation_matrices()
        self._parse_all_cells()
        self._parse_force_constants()
        self._parse_dataset()
        self._parse_nac_params()
        self._parse_calculator()

    def set_phonon_info(self, phonopy):
        self.unitcell = phonopy.unitcell
        self.primitive = phonopy.primitive
        self.supercell = phonopy.supercell
        self.version = phonopy.version
        self.supercell_matrix = phonopy.supercell_matrix
        self.symmetry = phonopy.symmetry
        self.primitive_matrix = phonopy.primitive_matrix
        s2p_map = self.primitive.s2p_map
        u2s_map = self.supercell.u2s_map
        u2u_map = self.supercell.u2u_map
        s2u_map = self.supercell.s2u_map
        self.u2p_map = [u2u_map[i] for i in (s2u_map[s2p_map])[u2s_map]]
        self.nac_params = phonopy.nac_params
        self.frequency_unit_conversion_factor = phonopy.unit_conversion_factor
        self.calculator = phonopy.calculator
        self.force_constants = phonopy.force_constants
        self.dataset = phonopy.dataset

    def get_yaml_lines(self):
        lines = self._header_yaml_lines()
        lines += self._physical_units_yaml_lines()
        lines += self._symmetry_yaml_lines()
        lines += self._cell_info_yaml_lines()
        lines += self._nac_yaml_lines()
        lines += self._dataset_yaml_lines()
        lines += self._force_constants_yaml_lines()
        return lines

    def _header_yaml_lines(self):
        lines = []
        lines.append("%s:" % self.command_name)
        lines.append("  version: %s" % self.version)
        if self.calculator:
            lines.append("  calculator: %s" % self.calculator)
        if self.frequency_unit_conversion_factor:
            lines.append("  frequency_unit_conversion_factor: %f" %
                         self.frequency_unit_conversion_factor)
        if self.symmetry:
            lines.append("  symmetry_tolerance: %.5e" %
                         self.symmetry.get_symmetry_tolerance())
        if self.nac_params:
            lines.append("  nac_unit_conversion_factor: %f"
                         % self.nac_params['factor'])
        if self.configuration is not None:
            lines.append("  configuration:")
            for key in self.configuration:
                val = self.configuration[key]
                if type(val) is str:
                    val = val.replace('\\', '\\\\')
                lines.append("    %s: \"%s\"" % (key, val))
        lines.append("")
        return lines

    def _physical_units_yaml_lines(self):
        lines = []
        lines.append("physical_unit:")
        lines.append("  atomic_mass: \"AMU\"")
        units = self.physical_units
        if units is not None:
            if units['length_unit'] is not None:
                lines.append("  length: \"%s\"" % units['length_unit'])
            if (self.command_name == "phonopy" and
                units['force_constants_unit'] is not None):
                lines.append("  force_constants: \"%s\"" %
                             units['force_constants_unit'])
        lines.append("")
        return lines

    def _symmetry_yaml_lines(self):
        lines = []
        if self.symmetry is not None and self.symmetry.dataset is not None:
            lines.append("space_group:")
            lines.append("  type: \"%s\"" %
                         self.symmetry.get_dataset()['international'])
            lines.append("  number: %d" %
                         self.symmetry.get_dataset()['number'])
            hall_symbol = self.symmetry.get_dataset()['hall']
            if "\"" in hall_symbol:
                hall_symbol = hall_symbol.replace("\"", "\\\"")
            lines.append("  Hall_symbol: \"%s\"" % hall_symbol)
            lines.append("")
        return lines

    def _cell_info_yaml_lines(self):
        lines = self._primitive_matrix_yaml_lines(
            self.primitive_matrix, "primitive_matrix")
        lines += self._supercell_matrix_yaml_lines(
            self.supercell_matrix, "supercell_matrix")
        lines += self._primitive_yaml_lines(self.primitive, "primitive_cell")
        lines += self._unitcell_yaml_lines()
        lines += self._supercell_yaml_lines()
        return lines

    def _primitive_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append("%s:" % name)
            for v in matrix:
                lines.append("- [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
            lines.append("")
        return lines

    def _supercell_matrix_yaml_lines(self, matrix, name):
        lines = []
        if matrix is not None:
            lines.append("%s:" % name)
            for v in matrix:
                lines.append("- [ %3d, %3d, %3d ]" % tuple(v))
            lines.append("")
        return lines

    def _primitive_yaml_lines(self, primitive, name):
        lines = []
        if primitive is not None:
            lines += self._cell_yaml_lines(
                self.primitive, name, None)
            lines.append("  reciprocal_lattice: # without 2pi")
            rec_lat = np.linalg.inv(primitive.cell)
            for v, a in zip(rec_lat.T, ('a*', 'b*', 'c*')):
                lines.append("  - [ %21.15f, %21.15f, %21.15f ] # %s" %
                             (v[0], v[1], v[2], a))
            lines.append("")
        return lines

    def _unitcell_yaml_lines(self):
        lines = []
        if self.unitcell is not None:
            lines += self._cell_yaml_lines(
                self.unitcell, "unit_cell", self.u2p_map)
            lines.append("")
        return lines

    def _supercell_yaml_lines(self):
        lines = []
        if self.supercell is not None:
            s2p_map = getattr(self.primitive, 's2p_map', None)
            lines += self._cell_yaml_lines(
                self.supercell, "supercell", s2p_map)
            lines.append("")
        return lines

    def _cell_yaml_lines(self, cell, name, map_to_primitive):
        lines = []
        lines.append("%s:" % name)
        count = 0
        for line in cell.get_yaml_lines():
            lines.append("  " + line)
            if map_to_primitive is not None and "mass" in line:
                lines.append("    reduced_to: %d" %
                             (map_to_primitive[count] + 1))
                count += 1
        return lines

    def _nac_yaml_lines(self):
        return self._nac_yaml_lines_given_symbols(self.primitive.symbols)

    def _nac_yaml_lines_given_symbols(self, symbols):
        lines = []
        if self.nac_params is not None:
            if self.settings['born_effective_charge']:
                lines.append("born_effective_charge:")
                for i, z in enumerate(self.nac_params['born']):
                    text = "- # %d" % (i + 1)
                    if symbols:
                        text += " (%s)" % symbols[i]
                    lines.append(text)
                    for v in z:
                        lines.append("  - [ %18.15f, %18.15f, %18.15f ]" %
                                     tuple(v))
                lines.append("")

            if self.settings['dielectric_constant']:
                lines.append("dielectric_constant:")
                for v in self.nac_params['dielectric']:
                    lines.append(
                        "  - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
                lines.append("")
        return lines

    def _dataset_yaml_lines(self):
        lines = []
        if self.settings['force_sets'] or self.settings['displacements']:
            disp_yaml_lines = self._displacements_yaml_lines(
                with_forces=self.settings['force_sets'])
            lines += disp_yaml_lines
        return lines

    def _displacements_yaml_lines(self, with_forces=False):
        return self._displacements_yaml_lines_2types(
            self.dataset, with_forces=with_forces)

    def _displacements_yaml_lines_2types(self, dataset, with_forces=False):
        if dataset is not None:
            if 'first_atoms' in dataset:
                return self._displacements_yaml_lines_type1(
                    dataset, with_forces=with_forces)
            elif 'displacements' in dataset:
                return self._displacements_yaml_lines_type2(
                    dataset, with_forces=with_forces)
        return []

    def _displacements_yaml_lines_type1(self, dataset, with_forces=False):
        lines = ["displacements:", ]
        for i, d in enumerate(dataset['first_atoms']):
            lines.append("- atom: %4d" % (d['number'] + 1))
            lines.append("  displacement:")
            lines.append("    [ %20.16f,%20.16f,%20.16f ]"
                         % tuple(d['displacement']))
            if with_forces and 'forces' in d:
                lines.append("  forces:")
                for f in d['forces']:
                    lines.append("  - [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")
        return lines

    def _displacements_yaml_lines_type2(self, dataset, with_forces=False):
        if 'random_seed' in dataset:
            lines = ["random_seed: %d" % dataset['random_seed'],
                     "displacements:"]
        else:
            lines = ["displacements:", ]
        for i, dset in enumerate(dataset['displacements']):
            lines.append("- # %4d" % (i + 1))
            for j, d in enumerate(dset):
                lines.append("  - displacement: # %d" % (j + 1))
                lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(d))
                if with_forces and 'forces' in dataset:
                    f = dataset['forces'][i][j]
                    lines.append("    force:")
                    lines.append("      [ %20.16f,%20.16f,%20.16f ]"
                                 % tuple(f))
        lines.append("")
        return lines

    def _force_constants_yaml_lines(self):
        lines = []
        if (self.settings['force_constants'] and
            self.force_constants is not None):
            shape = self.force_constants.shape[:2]
            lines = ["force_constants:", ]
            if shape[0] == shape[1]:
                lines.append("  format: \"full\"")
            else:
                lines.append("  format: \"compact\"")
            lines.append("  shape: [ %d, %d ]" % shape)
            lines.append("  elements:")
            for (i, j) in list(np.ndindex(shape)):
                lines.append("  - # (%d, %d)" % (i + 1, j + 1))
                for v in self.force_constants[i, j]:
                    lines.append("    - [ %21.15f, %21.15f, %21.15f ]"
                                 % tuple(v))
        return lines

    def _load(self, fp):
        self._yaml = yaml.load(fp, Loader=Loader)
        if type(self._yaml) is str:
            msg = "Could not open %s's yaml file." % self.command_name
            raise TypeError(msg)
        self.parse()

    def _parse_transformation_matrices(self):
        if 'supercell_matrix' in self._yaml:
            self.supercell_matrix = np.array(self._yaml['supercell_matrix'],
                                             dtype='intc', order='C')
        if 'primitive_matrix' in self._yaml:
            self.primitive_matrix = np.array(self._yaml['primitive_matrix'],
                                             dtype='double', order='C')

    def _parse_all_cells(self):
        if 'unit_cell' in self._yaml:
            self.unitcell = self._parse_cell(self._yaml['unit_cell'])
        if 'primitive_cell' in self._yaml:
            self.primitive = self._parse_cell(self._yaml['primitive_cell'])
        if 'supercell' in self._yaml:
            self.supercell = self._parse_cell(self._yaml['supercell'])
        if self.unitcell is None:
            if ('lattice' in self._yaml and
                ('points' in self._yaml or 'atoms' in self._yaml)):
                self.unitcell = self._parse_cell(self._yaml)

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
            return PhonopyAtoms(symbols=_symbols,
                                cell=_lattice,
                                masses=_masses,
                                scaled_positions=_points)
        else:
            return None

    def _parse_force_constants(self):
        if 'force_constants' in self._yaml:
            shape = tuple(self._yaml['force_constants']['shape']) + (3, 3)
            fc = np.reshape(self._yaml['force_constants']['elements'], shape)
            self.force_constants = np.array(fc, dtype='double', order='C')

    def _parse_dataset(self):
        self.dataset = self._get_dataset(self.supercell)

    def _get_dataset(self, supercell):
        dataset = None
        if 'displacements' in self._yaml:
            if supercell is not None:
                natom = len(supercell)
            else:
                natom = None
            disp = self._yaml['displacements'][0]
            if type(disp) is dict:  # type1
                dataset = self._parse_force_sets_type1(natom=natom)
            elif type(disp) is list:  # type2
                if 'displacement' in disp[0]:
                    dataset = self._parse_force_sets_type2()
        return dataset

    def _parse_force_sets_type1(self, natom=None):
        with_forces = False
        if 'forces' in self._yaml['displacements'][0]:
            with_forces = True
            dataset = {'natom': len(self._yaml['displacements'][0]['forces'])}
        elif natom is not None:
            dataset = {'natom': natom}
        elif 'natom' in self._yaml:
            dataset = {'natom': self._yaml['natom']}
        else:
            raise RuntimeError(
                "Number of atoms in supercell could not be found.")

        first_atoms = []
        for d in self._yaml['displacements']:
            data = {
                'number': d['atom'] - 1,
                'displacement': np.array(d['displacement'], dtype='double')}
            if with_forces:
                data['forces'] = np.array(d['forces'],
                                          dtype='double', order='C')
            first_atoms.append(data)
        dataset['first_atoms'] = first_atoms

        return dataset

    def _parse_force_sets_type2(self):
        nsets = len(self._yaml['displacements'])
        natom = len(self._yaml['displacements'][0])
        if 'force' in self._yaml['displacements'][0][0]:
            with_forces = True
            forces = np.zeros((nsets, natom, 3), dtype='double', order='C')
        else:
            with_forces = False
        displacements = np.zeros((nsets, natom, 3), dtype='double', order='C')
        for i, dfset in enumerate(self._yaml['displacements']):
            for j, df in enumerate(dfset):
                if with_forces:
                    forces[i, j] = df['force']
                displacements[i, j] = df['displacement']
        if with_forces:
            return {'forces': forces, 'displacements': displacements}
        else:
            return {'displacements': displacements}

    def _parse_nac_params(self):
        nac_params = {}
        if 'born_effective_charge' in self._yaml:
            nac_params['born'] = np.array(self._yaml['born_effective_charge'],
                                          dtype='double', order='C')
        if 'dielectric_constant' in self._yaml:
            nac_params['dielectric'] = np.array(
                self._yaml['dielectric_constant'], dtype='double', order='C')
        if (self.command_name in self._yaml and
            'nac_unit_conversion_factor' in self._yaml[self.command_name]):
            nac_params['factor'] = self._yaml[self.command_name][
                'nac_unit_conversion_factor']
        if 'born' in nac_params and 'dielectric' in nac_params:
            self.nac_params = nac_params

    def _parse_calculator(self):
        if (self.command_name in self._yaml and
            'calculator' in self._yaml[self.command_name]):
            self.calculator = self._yaml[self.command_name]['calculator']
