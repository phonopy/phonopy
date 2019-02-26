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
try:
    import yaml
except ImportError:
    raise ImportError("You need to install python-yaml.")

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from phonopy.structure.atoms import PhonopyAtoms as Atoms
from phonopy.file_IO import get_disp_yaml_lines


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
    def __init__(self,
                 configuration=None,
                 calculator=None,
                 physical_units=None,
                 settings=None):
        self._configuration = configuration
        self._calculator = calculator
        self._physical_units = physical_units

        self.unitcell = None
        self.primitive = None
        self.supercell = None
        self.yaml = None
        self.dataset = None
        self.supercell_matrix = None
        self.primitive_matrix = None
        self.nac_params = None
        self.force_constants = None

        self._supercell_matrix = None
        self._symmetry = None  # symmetry of supercell
        self._primitive_matrix = None
        self._force_constants = None
        self._s2p_map = None
        self._u2p_map = None
        self._nac_params = None
        self._frequency_unit_conversion_factor = None
        self._version = None

        self._command_name = "phonopy"
        self._settings = {'force_sets': True,
                          'displacements': True,
                          'force_constants': False,
                          'born_effective_charge': True,
                          'dielectric_constant': True}
        if type(settings) is dict:
            self._settings.update(settings)

    def read(self, filename):
        with open(filename) as infile:
            self._load(infile)

    def set_phonon_info(self, phonopy):
        self.unitcell = phonopy.unitcell
        self.primitive = phonopy.primitive
        self.supercell = phonopy.supercell
        self._version = phonopy.version
        self._supercell_matrix = phonopy.supercell_matrix
        self._symmetry = phonopy.symmetry
        self._primitive_matrix = phonopy.primitive_matrix
        self._s2p_map = self.primitive.s2p_map
        u2s_map = self.supercell.u2s_map
        u2u_map = self.supercell.u2u_map
        s2u_map = self.supercell.s2u_map
        self._u2p_map = [u2u_map[i] for i in (s2u_map[self._s2p_map])[u2s_map]]
        self._nac_params = phonopy.nac_params
        self._frequency_unit_conversion_factor = phonopy.unit_conversion_factor

        if self._command_name == "phonopy":
            self._force_constants = phonopy.force_constants
            self._displacements = phonopy.displacements
            self._dataset = phonopy.dataset

    def get_yaml_lines(self):
        lines = []
        nac_factor = None
        if self.primitive is None:
            symbols = None
        else:
            symbols = self.primitive.get_chemical_symbols()
        if self._nac_params is not None:
            born = self._nac_params['born']
            nac_factor = self._nac_params['factor']
            dielectric = self._nac_params['dielectric']

        lines.append("%s:" % self._command_name)
        lines.append("  version: %s" % self._version)
        if self._calculator:
            lines.append("  calculator: %s" % self._calculator)
        if self._frequency_unit_conversion_factor:
            lines.append("  frequency_unit_conversion_factor: %f" %
                         self._frequency_unit_conversion_factor)
        if self._nac_params:
            lines.append("  nac_unit_conversion_factor: %f" % nac_factor)
        if self._configuration is not None:
            lines.append("  configuration:")
            for key in self._configuration:
                val = self._configuration[key]
                if type(val) is str:
                    val = val.replace('\\', '\\\\')
                lines.append("    %s: \"%s\"" % (key, val))
        lines.append("")

        lines.append("physical_unit:")
        lines.append("  atomic_mass: \"AMU\"")
        units = self._physical_units
        if units is not None:
            if units['length_unit'] is not None:
                lines.append("  length: \"%s\"" % units['length_unit'])
            if (self._command_name == "phonopy" and
                units['force_constants_unit'] is not None):
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

        if self.primitive is not None:
            lines.append("primitive_cell:")
            for line in self.primitive.get_yaml_lines():
                lines.append("  " + line)
            lines.append("  reciprocal_lattice: # without 2pi")
            rec_lat = np.linalg.inv(self.primitive.get_cell())
            for v, a in zip(rec_lat.T, ('a*', 'b*', 'c*')):
                lines.append("  - [ %21.15f, %21.15f, %21.15f ] # %s" %
                             (v[0], v[1], v[2], a))
            lines.append("")

        if self.unitcell is not None:
            lines.append("unit_cell:")
            count = 0
            for line in self.unitcell.get_yaml_lines():
                lines.append("  " + line)
                if self._u2p_map is not None and "mass" in line:
                    lines.append("    reduced_to: %d" %
                                 (self._u2p_map[count] + 1))
                    count += 1
            lines.append("")

        if self.supercell is not None:
            lines.append("supercell:")
            count = 0
            for line in self.supercell.get_yaml_lines():
                lines.append("  " + line)
                if self._s2p_map is not None and "mass" in line:
                    lines.append("    reduced_to: %d" %
                                 (self._s2p_map[count] + 1))
                    count += 1
            lines.append("")

        if self._nac_params is not None:
            if self._settings['born_effective_charge']:
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

            if self._settings['dielectric_constant']:
                lines.append("dielectric_constant:")
                for v in dielectric:
                    lines.append("  - [ %18.15f, %18.15f, %18.15f ]" % tuple(v))
                lines.append("")

        if self._settings['force_sets']:
            lines += self._force_sets_yaml_lines()
        elif self._settings['displacements']:
            lines += self._displacements_yaml_lines()

        if self._settings['force_constants']:
            lines += self._force_constants_yaml_lines()

        return lines

    def _displacements_yaml_lines(self):
        lines = get_disp_yaml_lines(self._displacements, self.supercell)
        lines.append("")
        return lines

    def _force_sets_yaml_lines(self):
        if 'first_atoms' in self._dataset:
            return self._force_sets_yaml_lines_type1()
        elif 'displacements'  in self._dataset:
            return self._force_sets_yaml_lines_type2()
        else:
            return []

    def _force_sets_yaml_lines_type1(self):
        lines = ["displacements:", ]
        for i, d in enumerate(self._dataset['first_atoms']):
            lines.append("- atom: %4d" % (d['number'] + 1))
            lines.append("  displacement:")
            lines.append("    [ %20.16f,%20.16f,%20.16f ]"
                         % tuple(d['displacement']))
            if 'forces' in d:
                lines.append("  forces:")
                for f in d['forces']:
                    lines.append("  - [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")
        return lines

    def _force_sets_yaml_lines_type2(self):
        lines = ["displacements:", ]
        for i, (dset, fset) in enumerate(zip(self._dataset['displacements'],
                                             self._dataset['forces'])):
            lines.append("- # %4d" % (i + 1))
            for j, (d, f) in enumerate(zip(dset, fset)):
                lines.append("  - displacement: # %d" % (j + 1))
                lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(d))
                lines.append("    force:")
                lines.append("      [ %20.16f,%20.16f,%20.16f ]" % tuple(f))
        lines.append("")
        return lines

    def _force_constants_yaml_lines(self):
        if self._force_constants is None:
            return []

        shape = self._force_constants.shape[:2]
        lines = ["force_constants:", ]
        if shape[0] == shape[1]:
            lines.append("  format: \"full\"")
        else:
            lines.append("  format: \"compact\"")
        lines.append("  shape: [ %d, %d ]" % shape)
        lines.append("  elements:")
        for (i, j) in list(np.ndindex(shape)):
            lines.append("  - # (%d, %d)" % (i + 1, j + 1))
            for v in self._force_constants[i, j]:
                lines.append("    - [ %21.15f, %21.15f, %21.15f ]" % tuple(v))
        return lines

    def __str__(self):
        return "\n".join(self.get_yaml_lines())

    def _load(self, fp):
        self.yaml = yaml.load(fp, Loader=Loader)
        if type(self.yaml) is str:
            msg = "Could not open %s's yaml file." % self._command_name
            raise TypeError(msg)

        if 'unit_cell' in self.yaml:
            self.unitcell = self._parse_cell(self.yaml['unit_cell'])
        if 'primitive_cell' in self.yaml:
            self.primitive = self._parse_cell(self.yaml['primitive_cell'])
        if 'supercell' in self.yaml:
            self.supercell = self._parse_cell(self.yaml['supercell'])
        if self.unitcell is None:
            if ('lattice' in self.yaml and
                ('points' in self.yaml or 'atoms' in self.yaml)):
                self.unitcell = self._parse_cell(self.yaml)
        if 'force_constants' in self.yaml:
            shape = tuple(self.yaml['force_constants']['shape']) + (3, 3)
            fc = np.reshape(self.yaml['force_constants']['elements'], shape)
            self.force_constants = np.array(fc, dtype='double', order='C')
        elif 'displacements' in self.yaml:
            disp = self.yaml['displacements'][0]
            if type(disp) is dict and 'forces' in disp:
                    self.dataset = self._parse_force_sets_type1()
            elif type(disp) is list and 'forces' in disp[0]:
                self.dataset = self._parse_force_sets_type2()
        if 'supercell_matrix' in self.yaml:
            self.supercell_matrix = np.array(self.yaml['supercell_matrix'],
                                             dtype='intc', order='C')
        if 'primitive_matrix' in self.yaml:
            self.primitive_matrix = np.array(self.yaml['primitive_matrix'],
                                             dtype='double', order='C')
        _nac_params = {}
        if 'born_effective_charge' in self.yaml:
            _nac_params['born'] = np.array(self.yaml['born_effective_charge'],
                                           dtype='double', order='C')
        if 'dielectric_constant' in self.yaml:
            _nac_params['dielectric'] = np.array(
                self.yaml['dielectric_constant'], dtype='double', order='C')
        if (self._command_name in self.yaml and
            'nac_unit_conversion_factor' in self.yaml[self._command_name]):
            _nac_params['factor'] = self.yaml[self._command_name][
                'nac_unit_conversion_factor']
        if 'born' in _nac_params and 'dielectric' in _nac_params:
            self.nac_params = _nac_params

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

    def _parse_force_sets_type1(self):
        natom = len(self.yaml['displacements'][0]['forces'])
        dataset = {'natom': natom}
        first_atoms = []
        for d in self.yaml['displacements']:
            first_atoms.append({
                'number': d['atom'] - 1,
                'displacement': np.array(d['displacement'], dtype='double'),
                'forces': np.array(d['forces'], dtype='double', order='C')})
        dataset['first_atoms'] = first_atoms
        return dataset

    def _parse_force_sets_type2(self):
        nsets = len(self.yaml['displacements'])
        natom = len(self.yaml['displacements'][0])
        forces = np.zeros((nsets, natom, 3), dtype='double', order='C')
        displacements = np.zeros((nsets, natom, 3), dtype='double', order='C')
        for i, dfset in enumerate(self.yaml['displacements']):
            for j, df in enumerate(dfset):
                forces[i, j] = df['force']
                displacements[i, j] = df['displacement']
        return {'forces': forces, 'displacements': displacements}
