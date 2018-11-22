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
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from phonopy.structure.atoms import PhonopyAtoms as Atoms


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
                 show_force_constants=False):
        self._configuration = configuration
        self._calculator = calculator
        self._physical_units = physical_units
        self._show_force_constants = show_force_constants

        self.unitcell = None
        self.primitive = None
        self.supercell = None

        self._supercell_matrix = None
        self._symmetry = None  # symmetry of supercell
        self._primitive_matrix = None
        self._force_constants = None
        self._s2p_map = None
        self._u2p_map = None
        self._nac_params = None
        self._version = None

    def read(self, filename):
        with open(filename) as infile:
            self._load(infile)

    def set_phonon_info(self, phonopy):
        self.unitcell = phonopy.get_unitcell()
        self.primitive = phonopy.get_primitive()
        self.supercell = phonopy.get_supercell()
        self._version = phonopy.get_version()
        self._supercell_matrix = phonopy.get_supercell_matrix()
        self._symmetry = phonopy.get_symmetry()
        self._primitive_matrix = phonopy.get_primitive_matrix()
        self._force_constants = phonopy.get_force_constants()
        self._s2p_map = self.primitive.get_supercell_to_primitive_map()
        u2s_map = self.supercell.get_unitcell_to_supercell_map()
        u2u_map = self.supercell.get_unitcell_to_unitcell_map()
        s2u_map = self.supercell.get_supercell_to_unitcell_map()
        self._u2p_map = [u2u_map[i] for i in (s2u_map[self._s2p_map])[u2s_map]]
        self._nac_params = phonopy.get_nac_params()

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
        units = self._physical_units
        if units is not None:
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
            natom = self.supercell.get_number_of_atoms()
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
            self.unitcell = self._parse_cell(self._data['unit_cell'])
        if 'primitive_cell' in self._data:
            self.primitive = self._parse_cell(self._data['primitive_cell'])
        if 'supercell' in self._data:
            self.supercell = self._parse_cell(self._data['supercell'])
        if self.unitcell is None:
            if ('lattice' in self._data and
                ('points' in self._data or 'atoms' in self._data)):
                self.unitcell = self._parse_cell(self._data)

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
