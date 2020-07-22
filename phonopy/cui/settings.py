# Copyright (C) 2011 Atsushi Togo
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

import sys
import numpy as np


def fracval(frac):
    if frac.find('/') == -1:
        return float(frac)
    else:
        x = frac.split('/')
        return float(x[0]) / float(x[1])


class Settings(object):
    """Phonopy settings container

    This works almost like a dictionary.
    Method names without 'set_' and 'get_' and keys of self._v have to be same.

    """

    _default = {
        'band_indices': None,
        'band_paths': None,
        'band_points': None,
        'cell_filename': None,
        'chemical_symbols': None,
        'cutoff_frequency': None,
        'displacement_distance': None,
        'dm_decimals': None,
        'calculator': None,
        'create_displacements': False,
        'fc_calculator': None,
        'fc_calculator_options': None,
        'fc_decimals': None,
        'fc_symmetry': False,
        'frequency_pitch': None,
        'frequency_conversion_factor': None,
        'frequency_scale_factor': None,
        'group_velocity_delta_q': None,
        'hdf5_compression': 'gzip',
        'is_band_const_interval': False,
        'is_diagonal_displacement': True,
        'is_eigenvectors': False,
        'is_mesh_symmetry': True,
        'is_nac': False,
        'is_rotational_invariance': False,
        'is_plusminus_displacement': 'auto',
        'is_symmetry': True,
        'is_tetrahedron_method': True,
        'is_time_reversal_symmetry': True,
        'is_trigonal_displacement': False,
        'magnetic_moments': None,
        'masses': None,
        'mesh_numbers': None,
        'nac_method': None,
        'nac_q_direction': None,
        'num_frequency_points': None,
        'primitive_matrix': None,
        'qpoints': None,
        'read_qpoints': False,
        'sigma': None,
        'supercell_matrix': None,
        'symmetry_tolerance': None,
        'temperatures': None,
        'max_temperature': 1000,
        'min_temperature': 0,
        'temperature_step': 10
    }

    def __init__(self, default=None):
        self._v = Settings._default.copy()
        if default is not None:
            self._v.update(default)

    def __getattr__(self, attr):
        return self._v[attr]

    def set_band_paths(self, val):
        self._v['band_paths'] = val

    def set_band_points(self, val):
        self._v['band_points'] = val

    def set_band_indices(self, val):
        self._v['band_indices'] = val

    def set_cell_filename(self, val):
        self._v['cell_filename'] = val

    def set_chemical_symbols(self, val):
        self._v['chemical_symbols'] = val

    def set_create_displacements(self, val):
        self._v['create_displacements'] = val

    def set_cutoff_frequency(self, val):
        self._v['cutoff_frequency'] = val

    def set_dm_decimals(self, val):
        self._v['dm_decimals'] = val

    def set_displacement_distance(self, val):
        self._v['displacement_distance'] = val

    def set_calculator(self, val):
        self._v['calculator'] = val

    def set_fc_calculator(self, val):
        self._v['fc_calculator'] = val

    def set_fc_calculator_options(self, val):
        self._v['fc_calculator_options'] = val

    def set_fc_symmetry(self, val):
        self._v['fc_symmetry'] = val

    def set_fc_decimals(self, val):
        self._v['fc_decimals'] = val

    def set_frequency_conversion_factor(self, val):
        self._v['frequency_conversion_factor'] = val

    def set_frequency_pitch(self, val):
        self._v['frequency_pitch'] = val

    def set_frequency_scale_factor(self, val):
        self._v['frequency_scale_factor'] = val

    def set_group_velocity_delta_q(self, val):
        self._v['group_velocity_delta_q'] = val

    def set_hdf5_compression(self, val):
        self._v['hdf5_compression'] = val

    def set_is_band_const_interval(self, val):
        self._v['is_band_const_interval'] = val

    def set_is_diagonal_displacement(self, val):
        self._v['is_diagonal_displacement'] = val

    def set_is_eigenvectors(self, val):
        self._v['is_eigenvectors'] = val

    def set_is_mesh_symmetry(self, val):
        self._v['is_mesh_symmetry'] = val

    def set_is_nac(self, val):
        self._v['is_nac'] = val

    def set_is_plusminus_displacement(self, val):
        self._v['is_plusminus_displacement'] = val

    def set_is_rotational_invariance(self, val):
        self._v['is_rotational_invariance'] = val

    def set_is_tetrahedron_method(self, val):
        self._v['is_tetrahedron_method'] = val

    def set_is_trigonal_displacement(self, val):
        self._v['is_trigonal_displacement'] = val

    def set_is_symmetry(self, val):
        self._v['is_symmetry'] = val

    def set_magnetic_moments(self, val):
        self._v['magnetic_moments'] = val

    def set_masses(self, val):
        self._v['masses'] = val

    def set_max_temperature(self, val):
        self._v['max_temperature'] = val

    def set_mesh_numbers(self, val):
        self._v['mesh_numbers'] = val

    def set_min_temperature(self, val):
        self._v['min_temperature'] = val

    def set_nac_method(self, val):
        self._v['nac_method'] = val

    def set_nac_q_direction(self, val):
        self._v['nac_q_direction'] = val

    def set_num_frequency_points(self, val):
        self._v['num_frequency_points'] = val

    def set_primitive_matrix(self, val):
        self._v['primitive_matrix'] = val

    def set_qpoints(self, val):
        self._v['qpoints'] = val

    def set_read_qpoints(self, val):
        self._v['read_qpoints'] = val

    def set_sigma(self, val):
        self._v['sigma'] = val

    def set_supercell_matrix(self, val):
        self._v['supercell_matrix'] = val

    def set_symmetry_tolerance(self, val):
        self._v['symmetry_tolerance'] = val

    def set_temperatures(self, val):
        self._v['temperatures'] = val

    def set_temperature_step(self, val):
        self._v['temperature_step'] = val

    def set_is_time_reversal_symmetry(self, val):
        self._v['is_time_reversal_symmetry'] = val


# Parse phonopy setting filen
class ConfParser(object):
    def __init__(self, filename=None, args=None):
        self._confs = {}
        self._parameters = {}
        self._args = args
        self._filename = filename

    @property
    def confs(self):
        return self._confs

    def get_configures(self):
        return self.confs

    @property
    def settings(self):
        return self._settings

    def get_settings(self):
        return self.settings

    def setting_error(self, message):
        print(message)
        print("Please check the setting tags and options.")
        sys.exit(1)

    def read_file(self):
        file = open(self._filename, 'r')
        is_continue = False
        left = None

        for line in file:
            if line.strip() == '':
                is_continue = False
                continue

            if line.strip()[0] == '#':
                is_continue = False
                continue

            if is_continue and left is not None:
                self._confs[left] += line.strip()
                self._confs[left] = self._confs[left].replace('+++', ' ')
                is_continue = False

            if line.find('=') != -1:
                left, right = [x.strip() for x in line.split('=')]
                self._confs[left.lower()] = right

            if line.find('+++') != -1:
                is_continue = True

    def read_options(self):
        arg_list = vars(self._args)
        if 'band_indices' in arg_list:
            band_indices = self._args.band_indices
            if band_indices is not None:
                if type(band_indices) is list:
                    self._confs['band_indices'] = " ".join(band_indices)
                else:
                    self._confs['band_indices'] = band_indices

        if 'band_paths' in arg_list:
            if self._args.band_paths is not None:
                if type(self._args.band_paths) is list:
                    self._confs['band'] = " ".join(self._args.band_paths)
                else:
                    self._confs['band'] = self._args.band_paths

        if 'band_points' in arg_list:
            if self._args.band_points is not None:
                self._confs['band_points'] = self._args.band_points

        if 'cell_filename' in arg_list:
            if self._args.cell_filename is not None:
                self._confs['cell_filename'] = self._args.cell_filename

        if 'cutoff_frequency' in arg_list:
            if self._args.cutoff_frequency:
                self._confs['cutoff_frequency'] = self._args.cutoff_frequency

        if 'displacement_distance' in arg_list:
            if self._args.displacement_distance:
                self._confs['displacement_distance'] = \
                    self._args.displacement_distance

        if 'dynamical_matrix_decimals' in arg_list:
            if self._args.dynamical_matrix_decimals:
                self._confs['dm_decimals'] = \
                    self._args.dynamical_matrix_decimals

        if 'calculator' in arg_list:
            if self._args.calculator:
                self._confs['calculator'] = self._args.calculator

        if 'fc_calculator' in arg_list:
            if self._args.fc_calculator:
                self._confs['fc_calculator'] = self._args.fc_calculator

        if 'fc_calculator_options' in arg_list:
            fc_calc_opt = self._args.fc_calculator_options
            if fc_calc_opt:
                self._confs['fc_calculator_options'] = fc_calc_opt

        if 'fc_symmetry' in arg_list:
            if self._settings.fc_symmetry:
                if not self._args.fc_symmetry:
                    self._confs['fc_symmetry'] = '.false.'
            else:
                if self._args.fc_symmetry:
                    self._confs['fc_symmetry'] = '.true.'

        if 'force_constants_decimals' in arg_list:
            if self._args.force_constants_decimals:
                self._confs['fc_decimals'] = \
                    self._args.force_constants_decimals

        if 'fpitch' in arg_list:
            if self._args.fpitch:
                self._confs['fpitch'] = self._args.fpitch

        if 'frequency_conversion_factor' in arg_list:
            freq_factor = self._args.frequency_conversion_factor
            if freq_factor:
                self._confs['frequency_conversion_factor'] = freq_factor

        if 'frequency_scale_factor' in arg_list:
            freq_scale = self._args.frequency_scale_factor
            if freq_scale is not None:
                self._confs['frequency_scale_factor'] = freq_scale

        if 'gv_delta_q' in arg_list:
            if self._args.gv_delta_q:
                self._confs['gv_delta_q'] = self._args.gv_delta_q

        if 'hdf5_compression' in arg_list:
            if self._args.hdf5_compression:
                self._confs['hdf5_compression'] = self._args.hdf5_compression

        if 'is_band_const_interval' in arg_list:
            if self._args.is_band_const_interval:
                self._confs['band_const_interval'] = '.true.'

        if 'is_displacement' in arg_list:
            if self._args.is_displacement:
                self._confs['create_displacements'] = '.true.'

        if 'is_eigenvectors' in arg_list:
            if self._args.is_eigenvectors:
                self._confs['eigenvectors'] = '.true.'

        if 'is_nac' in arg_list:
            if self._settings.is_nac:  # Check default settings
                if not self._args.is_nac:
                    self._confs['nac'] = '.false.'
            else:
                if self._args.is_nac:
                    self._confs['nac'] = '.true.'

        if 'is_nodiag' in arg_list:
            if self._args.is_nodiag:
                self._confs['diag'] = '.false.'

        if 'is_nomeshsym' in arg_list:
            if self._args.is_nomeshsym:
                self._confs['mesh_symmetry'] = '.false.'

        if 'is_nosym' in arg_list:
            if self._args.is_nosym:
                self._confs['symmetry'] = '.false.'

        if 'is_plusminus_displacements' in arg_list:
            if self._args.is_plusminus_displacements:
                self._confs['pm'] = '.true.'

        if 'is_trigonal_displacements' in arg_list:
            if self._args.is_trigonal_displacements:
                self._confs['trigonal'] = '.true.'

        if 'masses' in arg_list:
            if self._args.masses is not None:
                if type(self._args.masses) is list:
                    self._confs['mass'] = " ".join(self._args.masses)
                else:
                    self._confs['mass'] = self._args.masses

        if 'magmoms' in arg_list:
            if self._args.magmoms is not None:
                if type(self._args.magmoms) is list:
                    self._confs['magmom'] = " ".join(self._args.magmoms)
                else:
                    self._confs['magmom'] = self._args.magmoms

        if 'mesh_numbers' in arg_list:
            mesh = self._args.mesh_numbers
            if mesh is not None:
                if type(mesh) is list:
                    self._confs['mesh_numbers'] = " ".join(mesh)
                else:
                    self._confs['mesh_numbers'] = mesh

        if 'num_frequency_points' in arg_list:
            opt_num_freqs = self._args.num_frequency_points
            if opt_num_freqs:
                self._confs['num_frequency_points'] = opt_num_freqs

        # For backword compatibility
        if 'primitive_axis' in arg_list:
            if self._args.primitive_axis is not None:
                if type(self._args.primitive_axis) is list:
                    primitive_axes = " ".join(self._args.primitive_axis)
                    self._confs['primitive_axes'] = primitive_axes
                else:
                    self._confs['primitive_axes'] = self._args.primitive_axis

        if 'primitive_axes' in arg_list:
            if self._args.primitive_axes:
                if type(self._args.primitive_axes) is list:
                    primitive_axes = " ".join(self._args.primitive_axes)
                    self._confs['primitive_axes'] = primitive_axes
                else:
                    self._confs['primitive_axes'] = self._args.primitive_axes

        if 'supercell_dimension' in arg_list:
            dim = self._args.supercell_dimension
            if dim is not None:
                if type(dim) is list:
                    self._confs['dim'] = " ".join(dim)
                else:
                    self._confs['dim'] = dim

        if 'qpoints' in arg_list:
            if self._args.qpoints is not None:
                if type(self._args.qpoints) is list:
                    self._confs['qpoints'] = " ".join(self._args.qpoints)
                else:
                    self._confs['qpoints'] = self._args.qpoints

        if 'nac_q_direction' in arg_list:
            q_dir = self._args.nac_q_direction
            if q_dir is not None:
                if type(q_dir) is list:
                    self._confs['q_direction'] = " ".join(q_dir)
                else:
                    self._confs['q_direction'] = q_dir

        if 'nac_method' in arg_list:
            if self._args.nac_method is not None:
                self._confs['nac_method'] = self._args.nac_method

        if 'read_qpoints' in arg_list:
            if self._args.read_qpoints:
                self._confs['read_qpoints'] = '.true.'

        if 'sigma' in arg_list:
            if self._args.sigma is not None:
                if type(self._args.sigma) is list:
                    self._confs['sigma'] = " ".join(self._args.sigma)
                else:
                    self._confs['sigma'] = self._args.sigma

        if 'symmetry_tolerance' in arg_list:
            if self._args.symmetry_tolerance:
                symtol = self._args.symmetry_tolerance
                self._confs['symmetry_tolerance'] = symtol

        if 'temperature' in arg_list:
            if self._args.temperature is not None:
                self._confs['temperature'] = self._args.temperature

        if 'temperatures' in arg_list:
            if self._args.temperatures is not None:
                self._confs['temperatures'] = " ".join(self._args.temperatures)

        if 'tmax' in arg_list:
            if self._args.tmax:
                self._confs['tmax'] = self._args.tmax

        if 'tmin' in arg_list:
            if self._args.tmin:
                self._confs['tmin'] = self._args.tmin

        if 'tstep' in arg_list:
            if self._args.tstep:
                self._confs['tstep'] = self._args.tstep

        from phonopy.interface.calculator import get_interface_mode
        calculator = get_interface_mode(arg_list)
        if calculator:
            self._confs['calculator'] = calculator

        if 'use_alm' in arg_list:
            if self._args.use_alm:
                self._confs['fc_calculator'] = 'alm'

        if 'use_hiphive' in arg_list:
            if self._args.use_hiphive:
                self._confs['fc_calculator'] = 'hiphive'

    def parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'band_indices':
                vals = []
                for sum_set in confs['band_indices'].split(','):
                    vals.append([int(x) - 1 for x in sum_set.split()])
                self.set_parameter('band_indices', vals)

            if conf_key == 'cell_filename':
                self.set_parameter('cell_filename', confs['cell_filename'])

            if conf_key == 'create_displacements':
                if confs['create_displacements'].lower() == '.true.':
                    self.set_parameter('create_displacements', True)
                elif confs['create_displacements'].lower() == '.false.':
                    self.set_parameter('create_displacements', False)

            if conf_key == 'dim':
                matrix = [int(x) for x in confs['dim'].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of DIM tag has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            'Determinant of supercell matrix has to be '
                            'positive.')
                    else:
                        self.set_parameter('supercell_matrix', matrix)

            if conf_key in ('primitive_axis', 'primitive_axes'):
                if confs[conf_key].strip().lower() == 'auto':
                    self.set_parameter('primitive_axes', 'auto')
                elif not len(confs[conf_key].split()) == 9:
                    self.setting_error(
                        "Number of elements in %s has to be 9." %
                        conf_key.upper())
                else:
                    p_axis = []
                    for x in confs[conf_key].split():
                        p_axis.append(fracval(x))
                    p_axis = np.array(p_axis).reshape(3, 3)
                    if np.linalg.det(p_axis) < 1e-8:
                        self.setting_error(
                            "%s has to have positive determinant." %
                            conf_key.upper())
                    self.set_parameter('primitive_axes', p_axis)

            if conf_key == 'mass':
                self.set_parameter(
                    'mass',
                    [float(x) for x in confs['mass'].split()])

            if conf_key == 'magmom':
                self.set_parameter(
                    'magmom',
                    [float(x) for x in confs['magmom'].split()])

            if conf_key == 'atom_name':
                self.set_parameter(
                    'atom_name',
                    [x.capitalize() for x in confs['atom_name'].split()])

            if conf_key == 'displacement_distance':
                self.set_parameter('displacement_distance',
                                   float(confs['displacement_distance']))

            if conf_key == 'diag':
                if confs['diag'].lower() == '.false.':
                    self.set_parameter('diag', False)
                elif confs['diag'].lower() == '.true.':
                    self.set_parameter('diag', True)

            if conf_key == 'pm':
                if confs['pm'].lower() == '.false.':
                    self.set_parameter('pm_displacement', False)
                elif confs['pm'].lower() == '.true.':
                    self.set_parameter('pm_displacement', True)

            if conf_key == 'trigonal':
                if confs['trigonal'].lower() == '.false.':
                    self.set_parameter('is_trigonal_displacement', False)
                elif confs['trigonal'].lower() == '.true.':
                    self.set_parameter('is_trigonal_displacement', True)

            if conf_key == 'eigenvectors':
                if confs['eigenvectors'].lower() == '.false.':
                    self.set_parameter('is_eigenvectors', False)
                elif confs['eigenvectors'].lower() == '.true.':
                    self.set_parameter('is_eigenvectors', True)

            if conf_key == 'nac':
                if confs['nac'].lower() == '.false.':
                    self.set_parameter('is_nac', False)
                elif confs['nac'].lower() == '.true.':
                    self.set_parameter('is_nac', True)

            if conf_key == 'symmetry':
                if confs['symmetry'].lower() == '.false.':
                    self.set_parameter('is_symmetry', False)
                    self.set_parameter('is_mesh_symmetry', False)
                elif confs['symmetry'].lower() == '.true.':
                    self.set_parameter('is_symmetry', True)

            if conf_key == 'mesh_symmetry':
                if confs['mesh_symmetry'].lower() == '.false.':
                    self.set_parameter('is_mesh_symmetry', False)
                elif confs['mesh_symmetry'].lower() == '.true.':
                    self.set_parameter('is_mesh_symmetry', True)

            if conf_key == 'rotational':
                if confs['rotational'].lower() == '.false.':
                    self.set_parameter('is_rotational', False)
                elif confs['rotational'].lower() == '.true.':
                    self.set_parameter('is_rotational', True)

            if conf_key == 'calculator':
                self.set_parameter('calculator', confs['calculator'])

            if conf_key == 'fc_calculator':
                self.set_parameter('fc_calculator', confs['fc_calculator'])

            if conf_key == 'fc_calculator_options':
                self.set_parameter('fc_calculator_options',
                                   confs['fc_calculator_options'])

            if conf_key == 'fc_symmetry':
                if confs['fc_symmetry'].lower() == '.false.':
                    self.set_parameter('fc_symmetry', False)
                elif confs['fc_symmetry'].lower() == '.true.':
                    self.set_parameter('fc_symmetry', True)
                else:
                    self.setting_error(
                        "FC_SYMMETRY has to be specified by .TRUE. or .FALSE.")

            if conf_key == 'fc_decimals':
                self.set_parameter('fc_decimals', confs['fc_decimals'])

            if conf_key == 'dm_decimals':
                self.set_parameter('dm_decimals', confs['dm_decimals'])

            if conf_key in ['mesh_numbers', 'mp', 'mesh']:
                vals = [x for x in confs[conf_key].split()]
                if len(vals) == 1:
                    self.set_parameter('mesh_numbers', float(vals[0]))
                elif len(vals) < 3:
                    self.setting_error("Mesh numbers are incorrectly set.")
                else:
                    self.set_parameter('mesh_numbers',
                                       [int(x) for x in vals[:3]])

            if conf_key == 'band_points':
                self.set_parameter('band_points', int(confs['band_points']))

            if conf_key == 'band_const_interval':
                if confs['band_const_interval'].lower() == '.false.':
                    self.set_parameter('is_band_const_interval', False)
                elif confs['band_const_interval'].lower() == '.true.':
                    self.set_parameter('is_band_const_interval', True)

            if conf_key == 'band':
                bands = []
                if confs['band'].strip().lower() == 'auto':
                    self.set_parameter('band_paths', 'auto')
                else:
                    for section in confs['band'].split(','):
                        points = [fracval(x) for x in section.split()]
                        if len(points) % 3 != 0 or len(points) < 6:
                            self.setting_error("BAND is incorrectly set.")
                            break
                        bands.append(np.array(points).reshape(-1, 3))
                    self.set_parameter('band_paths', bands)

            if conf_key == 'qpoints':
                if confs['qpoints'].lower() == '.true.':
                    self.set_parameter('read_qpoints', True)
                elif confs['qpoints'].lower() == '.false.':
                    self.set_parameter('read_qpoints', False)
                else:
                    vals = [fracval(x) for x in confs['qpoints'].split()]
                    if len(vals) == 0 or len(vals) % 3 != 0:
                        self.setting_error("Q-points are incorrectly set.")
                    else:
                        self.set_parameter('qpoints',
                                           list(np.reshape(vals, (-1, 3))))

            if conf_key == 'read_qpoints':
                if confs['read_qpoints'].lower() == '.false.':
                    self.set_parameter('read_qpoints', False)
                elif confs['read_qpoints'].lower() == '.true.':
                    self.set_parameter('read_qpoints', True)

            if conf_key == 'nac_method':
                self.set_parameter('nac_method', confs['nac_method'].lower())

            if conf_key == 'q_direction':
                q_direction = [fracval(x)
                               for x in confs['q_direction'].split()]
                if len(q_direction) < 3:
                    self.setting_error("Number of elements of q_direction "
                                       "is less than 3")
                else:
                    self.set_parameter('nac_q_direction', q_direction)

            if conf_key == 'frequency_conversion_factor':
                val = float(confs['frequency_conversion_factor'])
                self.set_parameter('frequency_conversion_factor', val)

            if conf_key == 'frequency_scale_factor':
                self.set_parameter('frequency_scale_factor',
                                   float(confs['frequency_scale_factor']))

            if conf_key == 'fpitch':
                val = float(confs['fpitch'])
                self.set_parameter('fpitch', val)

            if conf_key == 'num_frequency_points':
                val = int(confs['num_frequency_points'])
                self.set_parameter('num_frequency_points', val)

            if conf_key == 'cutoff_frequency':
                val = float(confs['cutoff_frequency'])
                self.set_parameter('cutoff_frequency', val)

            if conf_key == 'sigma':
                vals = [float(x) for x in str(confs['sigma']).split()]
                if len(vals) == 1:
                    self.set_parameter('sigma', vals[0])
                else:
                    self.set_parameter('sigma', vals)

            if conf_key == 'tetrahedron':
                if confs['tetrahedron'].lower() == '.false.':
                    self.set_parameter('is_tetrahedron_method', False)
                if confs['tetrahedron'].lower() == '.true.':
                    self.set_parameter('is_tetrahedron_method', True)

            if conf_key == 'symmetry_tolerance':
                val = float(confs['symmetry_tolerance'])
                self.set_parameter('symmetry_tolerance', val)

            # For multiple T values.
            if conf_key == 'temperatures':
                vals = [fracval(x) for x in confs['temperatures'].split()]
                if len(vals) < 1:
                    self.setting_error("Temperatures are incorrectly set.")
                else:
                    self.set_parameter('temperatures', vals)

            # For single T value.
            if conf_key == 'temperature':
                self.set_parameter('temperatures', [confs['temperature'], ])

            if conf_key == 'tmin':
                val = float(confs['tmin'])
                self.set_parameter('tmin', val)

            if conf_key == 'tmax':
                val = float(confs['tmax'])
                self.set_parameter('tmax', val)

            if conf_key == 'tstep':
                val = float(confs['tstep'])
                self.set_parameter('tstep', val)

            # Group velocity finite difference
            if conf_key == 'gv_delta_q':
                self.set_parameter('gv_delta_q', float(confs['gv_delta_q']))

            # Use ALM for generating force constants
            if conf_key == 'alm':
                if confs['alm'].lower() == '.true.':
                    self.set_parameter('alm', True)

            # Compression option for writing int hdf5
            if conf_key == 'hdf5_compression':
                hdf5_compression = confs['hdf5_compression']
                try:
                    compression = int(hdf5_compression)
                except ValueError:  # str
                    compression = hdf5_compression
                    if compression.lower() == "none":
                        compression = None
                except TypeError:  # None (this will not happen)
                    compression = hdf5_compression
                self.set_parameter('hdf5_compression', compression)

    def set_parameter(self, key, val):
        self._parameters[key] = val

    def set_settings(self):
        params = self._parameters

        # Chemical symbols
        if 'atom_name' in params:
            self._settings.set_chemical_symbols(params['atom_name'])

        # Sets of band indices that are summed
        if 'band_indices' in params:
            self._settings.set_band_indices(params['band_indices'])

        # Filename of input unit cell
        if 'cell_filename' in params:
            self._settings.set_cell_filename(params['cell_filename'])

        # Is getting least displacements?
        if 'create_displacements' in params:
            self._settings.set_create_displacements(
                params['create_displacements'])

        # Cutoff frequency
        if 'cutoff_frequency' in params:
            self._settings.set_cutoff_frequency(params['cutoff_frequency'])

        # Diagonal displacement
        if 'diag' in params:
            self._settings.set_is_diagonal_displacement(params['diag'])

        # Distance of finite displacements introduced
        if 'displacement_distance' in params:
            self._settings.set_displacement_distance(
                params['displacement_distance'])

        # Decimals of values of dynamical matrxi
        if 'dm_decimals' in params:
            self._settings.set_dm_decimals(int(params['dm_decimals']))

        # Force calculator
        if 'calculator' in params:
            self._settings.set_calculator(params['calculator'])

        # Force constants calculator
        if 'fc_calculator' in params:
            self._settings.set_fc_calculator(params['fc_calculator'])

        # Force constants calculator options as str
        if 'fc_calculator_options' in params:
            self._settings.set_fc_calculator_options(
                params['fc_calculator_options'])

        # Decimals of values of force constants
        if 'fc_decimals' in params:
            self._settings.set_fc_decimals(int(params['fc_decimals']))

        # Enforce translational invariance and index permutation symmetry
        # to force constants?
        if 'fc_symmetry' in params:
            self._settings.set_fc_symmetry(params['fc_symmetry'])

        # Frequency unit conversion factor
        if 'frequency_conversion_factor' in params:
            self._settings.set_frequency_conversion_factor(
                params['frequency_conversion_factor'])

        # This scale factor is multiplied to force constants by
        # fc * scale_factor ** 2, therefore only changes
        # frequencies but does not change NAC part.
        if 'frequency_scale_factor' in params:
            self._settings.set_frequency_scale_factor(
                params['frequency_scale_factor'])

        # Spectram drawing step
        if 'fpitch' in params:
            self._settings.set_frequency_pitch(params['fpitch'])

        # Number of sampling points for spectram drawing
        if 'num_frequency_points' in params:
            self._settings.set_num_frequency_points(params['num_frequency_points'])

        # Group velocity finite difference
        if 'gv_delta_q' in params:
            self._settings.set_group_velocity_delta_q(params['gv_delta_q'])

        # Mesh sampling numbers
        if 'mesh_numbers' in params:
            self._settings.set_mesh_numbers(params['mesh_numbers'])

        # Is getting eigenvectors?
        if 'is_eigenvectors' in params:
            self._settings.set_is_eigenvectors(params['is_eigenvectors'])

        # Is reciprocal mesh symmetry searched?
        if 'is_mesh_symmetry' in params:
            self._settings.set_is_mesh_symmetry(params['is_mesh_symmetry'])

        # Non analytical term correction?
        if 'is_nac' in params:
            self._settings.set_is_nac(params['is_nac'])

        # Is rotational invariance ?
        if 'is_rotational' in params:
            self._settings.set_is_rotational_invariance(params['is_rotational'])

        # Is crystal symmetry searched?
        if 'is_symmetry' in params:
            self._settings.set_is_symmetry(params['is_symmetry'])

        # Tetrahedron method
        if 'is_tetrahedron_method' in params:
            self._settings.set_is_tetrahedron_method(
                params['is_tetrahedron_method'])

        # Trigonal displacement
        if 'is_trigonal_displacement' in params:
            self._settings.set_is_trigonal_displacement(
                params['is_trigonal_displacement'])

        # Magnetic moments
        if 'magmom' in params:
            self._settings.set_magnetic_moments(params['magmom'])

        # Atomic mass
        if 'mass' in params:
            self._settings.set_masses(params['mass'])

        # Plus minus displacement
        if 'pm_displacement' in params:
            self._settings.set_is_plusminus_displacement(
                params['pm_displacement'])

        # Primitive cell shape
        if 'primitive_axes' in params:
            self._settings.set_primitive_matrix(params['primitive_axes'])

        # Q-points mode
        if 'qpoints' in params:
            self._settings.set_qpoints(params['qpoints'])

        if 'read_qpoints' in params:
            if params['read_qpoints']:
                self._settings.set_read_qpoints(params['read_qpoints'])

        # non analytical term correction method
        if 'nac_method' in params:
            self._settings.set_nac_method(params['nac_method'])

        # q-direction for non analytical term correction
        if 'nac_q_direction' in params:
            self._settings.set_nac_q_direction(params['nac_q_direction'])

        # Smearing width
        if 'sigma' in params:
            self._settings.set_sigma(params['sigma'])

        # Symmetry tolerance
        if 'symmetry_tolerance' in params:
            self._settings.set_symmetry_tolerance(params['symmetry_tolerance'])

        # Supercell size
        if 'supercell_matrix' in params:
            self._settings.set_supercell_matrix(params['supercell_matrix'])

        # Temperatures or temerature range
        if 'temperatures' in params:
            self._settings.set_temperatures(params['temperatures'])
        if 'tmax' in params:
            self._settings.set_max_temperature(params['tmax'])
        if 'tmin' in params:
            self._settings.set_min_temperature(params['tmin'])
        if 'tstep' in params:
            self._settings.set_temperature_step(params['tstep'])

        # Band paths
        # BAND = 0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5
        # [array([[ 0. ,  0. ,  0. ],
        #         [ 0.5,  0. ,  0. ],
        #         [ 0.5,  0.5,  0. ],
        #         [ 0. ,  0. ,  0. ],
        #         [ 0.5,  0.5,  0.5]])]
        #
        # BAND = 0.0 0.0 0.0  0.5 0.0 0.0, 0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5
        # [array([[ 0. ,  0. ,  0. ],
        #         [ 0.5,  0. ,  0. ]]),
        #  array([[ 0.5,  0.5,  0. ],
        #         [ 0. ,  0. ,  0. ],
        #         [ 0.5,  0.5,  0.5]])]
        # or
        # BAND = AUTO
        if 'band_paths' in params:
            self._settings.set_band_paths(params['band_paths'])

        # This number includes end points
        if 'band_points' in params:
            self._settings.set_band_points(params['band_points'])

        if 'is_band_const_interval' in params:
            self._settings.set_is_band_const_interval(
                params['is_band_const_interval'])

        # Use ALM to generating force constants
        if 'alm' in params:
            self._settings.set_use_alm(params['alm'])

        # Compression option for writing int hdf5
        if 'hdf5_compression' in params:
            self._settings.set_hdf5_compression(params['hdf5_compression'])


#
# For phonopy
#
class PhonopySettings(Settings):
    """Phonopy settings container

    Basic part is stored in Settings and extended part is stored in this class.

    This works almost like a dictionary.
    Method names without 'set_' and 'get_' and keys of self._v have to be same.

    """

    _default = {
        'anime_band_index': None,
        'anime_amplitude': None,
        'anime_division': None,
        'anime_qpoint': None,
        'anime_shift': None,
        'anime_type': 'v_sim',
        'band_format': 'yaml',
        'band_labels': None,
        'create_force_sets': None,
        'create_force_sets_zero': None,
        'create_force_constants': None,
        'cutoff_radius': None,
        'dos': None,
        'fc_spg_symmetry': False,
        'fits_Debye_model': False,
        'max_frequency': None,
        'min_frequency': None,
        'irreps_q_point': None,
        'irreps_tolerance': 1e-5,
        'is_band_connection': False,
        'is_dos_mode': False,
        'is_full_fc': False,
        'is_group_velocity': False,
        'is_gamma_center': False,
        'is_hdf5': False,
        'is_legacy_plot': False,
        'is_little_cogroup': False,
        'is_moment': False,
        'is_plusminus_displacement': 'auto',
        'is_thermal_displacements': False,
        'is_thermal_displacement_matrices': False,
        'is_thermal_distances': False,
        'is_thermal_properties': False,
        'is_projected_thermal_properties': False,
        'include_force_constants': False,
        'include_force_sets': False,
        'include_nac_params': False,
        'include_displacements': False,
        'lapack_solver': False,
        'mesh_shift': None,
        'mesh_format': 'yaml',
        'modulation': None,
        'moment_order': None,
        'random_displacements': None,
        'pdos_indices': None,
        'pretend_real': False,
        'projection_direction': None,
        'random_seed': None,
        'qpoints_format': 'yaml',
        'read_force_constants': False,
        'readfc_format': 'text',
        'run_mode': None,
        'save_params': False,
        'show_irreps': False,
        'thermal_atom_pairs': None,
        'thermal_displacement_matrix_temperatue': None,
        'write_dynamical_matrices': False,
        'write_mesh': True,
        'write_force_constants': False,
        'writefc_format': 'text',
        'xyz_projection': False
    }

    def __init__(self, default=None):
        Settings.__init__(self)
        self._v.update(PhonopySettings._default.copy())
        if default is not None:
            self._v.update(default)

    def set_anime_band_index(self, val):
        self._v['anime_band_index'] = val

    def set_anime_amplitude(self, val):
        self._v['anime_amplitude'] = val

    def set_anime_division(self, val):
        self._v['anime_division'] = val

    def set_anime_qpoint(self, val):
        self._v['anime_qpoint'] = val

    def set_anime_shift(self, val):
        self._v['anime_shift'] = val

    def set_anime_type(self, val):
        self._v['anime_type'] = val

    def set_band_format(self, val):
        self._v['band_format'] = val

    def set_band_labels(self, val):
        self._v['band_labels'] = val

    def set_create_force_sets(self, val):
        self._v['create_force_sets'] = val

    def set_create_force_sets_zero(self, val):
        self._v['create_force_sets_zero'] = val

    def set_create_force_constants(self, val):
        self._v['create_force_constants'] = val

    def set_cutoff_radius(self, val):
        self._v['cutoff_radius'] = val

    def set_fc_spg_symmetry(self, val):
        self._v['fc_spg_symmetry'] = val

    def set_fits_Debye_model(self, val):
        self._v['fits_Debye_model'] = val

    def set_max_frequency(self, val):
        self._v['max_frequency'] = val

    def set_mesh_shift(self, val):
        self._v['mesh_shift'] = val

    def set_min_frequency(self, val):
        self._v['min_frequency'] = val

    def set_irreps_q_point(self, val):
        self._v['irreps_q_point'] = val

    def set_irreps_tolerance(self, val):
        self._v['irreps_tolerance'] = val

    def set_is_band_connection(self, val):
        self._v['is_band_connection'] = val

    def set_is_dos_mode(self, val):
        self._v['is_dos_mode'] = val

    def set_is_full_fc(self, val):
        self._v['is_full_fc'] = val

    def set_is_gamma_center(self, val):
        self._v['is_gamma_center'] = val

    def set_is_group_velocity(self, val):
        self._v['is_group_velocity'] = val

    def set_is_hdf5(self, val):
        self._v['is_hdf5'] = val

    def set_is_legacy_plot(self, val):
        self._v['is_legacy_plot'] = val

    def set_is_little_cogroup(self, val):
        self._v['is_little_cogroup'] = val

    def set_is_moment(self, val):
        self._v['is_moment'] = val

    def set_is_projected_thermal_properties(self, val):
        self._v['is_projected_thermal_properties'] = val

    def set_is_thermal_displacements(self, val):
        self._v['is_thermal_displacements'] = val

    def set_is_thermal_displacement_matrices(self, val):
        self._v['is_thermal_displacement_matrices'] = val

    def set_is_thermal_distances(self, val):
        self._v['is_thermal_distances'] = val

    def set_is_thermal_properties(self, val):
        self._v['is_thermal_properties'] = val

    def set_include_force_constants(self, val):
        self._v['include_force_constants'] = val

    def set_include_force_sets(self, val):
        self._v['include_force_sets'] = val

    def set_include_nac_params(self, val):
        self._v['include_nac_params'] = val

    def set_include_displacements(self, val):
        self._v['include_displacements'] = val

    def set_lapack_solver(self, val):
        self._v['lapack_solver'] = val

    def set_mesh_format(self, val):
        self._v['mesh_format'] = val

    def set_modulation(self, val):
        self._v['modulation'] = val

    def set_moment_order(self, val):
        self._v['moment_order'] = val

    def set_random_displacements(self, val):
        self._v['random_displacements'] = val

    def set_pdos_indices(self, val):
        self._v['pdos_indices'] = val

    def set_pretend_real(self, val):
        self._v['pretend_real'] = val

    def set_projection_direction(self, val):
        self._v['projection_direction'] = val

    def set_qpoints_format(self, val):
        self._v['qpoints_format'] = val

    def set_random_seed(self, val):
        self._v['random_seed'] = val

    def set_read_force_constants(self, val):
        self._v['read_force_constants'] = val

    def set_readfc_format(self, val):
        self._v['readfc_format'] = val

    def set_run_mode(self, val):
        self._v['run_mode'] = val

    def set_thermal_atom_pairs(self, val):
        self._v['thermal_atom_pairs'] = val

    def set_thermal_displacement_matrix_temperature(self, val):
        self._v['thermal_displacement_matrix_temperatue'] = val

    def set_save_params(self, val):
        self._v['save_params'] = val

    def set_show_irreps(self, val):
        self._v['show_irreps'] = val

    def set_write_dynamical_matrices(self, val):
        self._v['write_dynamical_matrices'] = val

    def set_write_force_constants(self, val):
        self._v['write_force_constants'] = val

    def set_write_mesh(self, val):
        self._v['write_mesh'] = val

    def set_writefc_format(self, val):
        self._v['writefc_format'] = val

    def set_xyz_projection(self, val):
        self._v['xyz_projection'] = val


class PhonopyConfParser(ConfParser):
    def __init__(self, filename=None, args=None, default_settings=None):
        # This is fragile implementation.
        # Remember that options have to be activated only
        # when it changes the default value, i.e.,
        # _read_options has to be written in this way.

        self._settings = PhonopySettings(default=default_settings)
        confs = {}
        if filename is not None:
            ConfParser.__init__(self, filename=filename)
            self.read_file()  # store .conf file setting in self._confs
            self._parse_conf()  # self.parameters[key] = val
            self._set_settings()  # self.parameters -> PhonopySettings
            confs.update(self._confs)
        if args is not None:
            # To invoke ConfParser.__init__() to flush variables.
            ConfParser.__init__(self, args=args)
            self._read_options()  # store options in self._confs
            self._parse_conf()  # self.parameters[key] = val
            self._set_settings()  # self.parameters -> PhonopySettings
            confs.update(self._confs)
        self._confs = confs

    def _read_options(self):
        ConfParser.read_options(self)  # store data in self._confs
        arg_list = vars(self._args)
        if 'band_format' in arg_list:
            if self._args.band_format:
                self._confs['band_format'] = self._args.band_format

        if 'band_labels' in arg_list:
            if self._args.band_labels is not None:
                self._confs['band_labels'] = " ".join(self._args.band_labels)

        if 'is_gamma_center' in arg_list:
            if self._args.is_gamma_center:
                self._confs['gamma_center'] = '.true.'

        if 'create_force_sets' in arg_list:
            if self._args.create_force_sets:
                self._confs['create_force_sets'] = " ".join(
                    self._args.create_force_sets)

        if 'create_force_sets_zero' in arg_list:
            if self._args.create_force_sets_zero:
                self._confs['create_force_sets_zero'] = " ".join(
                    self._args.create_force_sets_zero)

        if 'create_force_constants' in arg_list:
            if self._args.create_force_constants:
                self._confs['create_force_constants'] = " ".join(
                    self._args.create_force_constants)

        if 'is_dos_mode' in arg_list:
            if self._args.is_dos_mode:
                self._confs['dos'] = '.true.'

        if 'pdos' in arg_list:
            if self._args.pdos is not None:
                self._confs['pdos'] = " ".join(self._args.pdos)

        if 'xyz_projection' in arg_list:
            if self._args.xyz_projection:
                self._confs['xyz_projection'] = '.true.'

        if 'fc_spg_symmetry' in arg_list:
            if self._args.fc_spg_symmetry:
                self._confs['fc_spg_symmetry'] = '.true.'

        if 'is_full_fc' in arg_list:
            if self._args.is_full_fc:
                self._confs['full_force_constants'] = '.true.'

        if 'fits_debye_model' in arg_list:
            if self._args.fits_debye_model:
                self._confs['debye_model'] = '.true.'

        if 'fmax' in arg_list:
            if self._args.fmax:
                self._confs['fmax'] = self._args.fmax

        if 'fmin' in arg_list:
            if self._args.fmin:
                self._confs['fmin'] = self._args.fmin

        if 'is_thermal_properties' in arg_list:
            if self._args.is_thermal_properties:
                self._confs['tprop'] = '.true.'

        if 'pretend_real' in arg_list:
            if self._args.pretend_real:
                self._confs['pretend_real'] = '.true.'

        if 'is_projected_thermal_properties' in arg_list:
            if self._args.is_projected_thermal_properties:
                self._confs['ptprop'] = '.true.'

        if 'is_thermal_displacements' in arg_list:
            if self._args.is_thermal_displacements:
                self._confs['tdisp'] = '.true.'

        if 'is_thermal_displacement_matrices' in arg_list:
            if self._args.is_thermal_displacement_matrices:
                self._confs['tdispmat'] = '.true.'

        if 'thermal_displacement_matrices_cif' in arg_list:
            opt_tdm_cif = self._args.thermal_displacement_matrices_cif
            if opt_tdm_cif:
                self._confs['tdispmat_cif'] = opt_tdm_cif

        if 'projection_direction' in arg_list:
            opt_proj_dir = self._args.projection_direction
            if opt_proj_dir is not None:
                self._confs['projection_direction'] = " ".join(opt_proj_dir)

        if 'read_force_constants' in arg_list:
            if self._args.read_force_constants:
                self._confs['read_force_constants'] = '.true.'

        if 'write_force_constants' in arg_list:
            if self._args.write_force_constants:
                self._confs['write_force_constants'] = '.true.'

        if 'readfc_format' in arg_list:
            if self._args.readfc_format:
                self._confs['readfc_format'] = self._args.readfc_format

        if 'writefc_format' in arg_list:
            if self._args.writefc_format:
                self._confs['writefc_format'] = self._args.writefc_format

        if 'fc_format' in arg_list:
            if self._args.fc_format:
                self._confs['fc_format'] = self._args.fc_format

        if 'is_hdf5' in arg_list:
            if self._args.is_hdf5:
                self._confs['hdf5'] = '.true.'

        if 'write_dynamical_matrices' in arg_list:
            if self._args.write_dynamical_matrices:
                self._confs['writedm'] = '.true.'

        if 'write_mesh' in arg_list:
            if not self._args.write_mesh:
                self._confs['write_mesh'] = '.false.'

        if 'mesh_format' in arg_list:
            if self._args.mesh_format:
                self._confs['mesh_format'] = self._args.mesh_format

        if 'qpoints_format' in arg_list:
            if self._args.qpoints_format:
                self._confs['qpoints_format'] = self._args.qpoints_format

        if 'irreps_qpoint' in arg_list:
            if self._args.irreps_qpoint is not None:
                self._confs['irreps'] = " ".join(self._args.irreps_qpoint)

        if 'save_params' in arg_list:
            if self._args.save_params:
                self._confs['save_params'] = '.true.'

        if 'show_irreps' in arg_list:
            if self._args.show_irreps:
                self._confs['show_irreps'] = '.true.'

        if 'is_little_cogroup' in arg_list:
            if self._args.is_little_cogroup:
                self._confs['little_cogroup'] = '.true.'

        if 'is_legacy_plot' in arg_list:
            if self._args.is_legacy_plot:
                self._confs['legacy_plot'] = '.true.'

        if 'is_band_connection' in arg_list:
            if self._args.is_band_connection:
                self._confs['band_connection'] = '.true.'

        if 'cutoff_radius' in arg_list:
            if self._args.cutoff_radius:
                self._confs['cutoff_radius'] = self._args.cutoff_radius

        if 'modulation' in arg_list:
            if self._args.modulation:
                self._confs['modulation'] = " ".join(self._args.modulation)

        if 'anime' in arg_list:
            if self._args.anime:
                self._confs['anime'] = " ".join(self._args.anime)

        if 'is_group_velocity' in arg_list:
            if self._args.is_group_velocity:
                self._confs['group_velocity'] = '.true.'

        if 'is_moment' in arg_list:
            if self._args.is_moment:
                self._confs['moment'] = '.true.'

        if 'moment_order' in arg_list:
            if self._args.moment_order:
                self._confs['moment_order'] = self._args.moment_order

        if 'random_displacements' in arg_list:
            nrand = self._args.random_displacements
            if nrand:
                self._confs['random_displacements'] = nrand

        if 'random_seed' in arg_list:
            if self._args.random_seed:
                seed = self._args.random_seed
                if (np.issubdtype(type(seed), np.integer) and
                    seed >= 0 and seed < 2 ** 32):
                    self._confs['random_seed'] = seed

        # Overwrite
        if 'is_check_symmetry' in arg_list:
            if self._args.is_check_symmetry:
                # Dummy 'dim' setting for sym-check
                self._confs['dim'] = '1 1 1'

        if 'lapack_solver' in arg_list:
            if self._args.lapack_solver:
                self._confs['lapack_solver'] = '.true.'

        # Select yaml summary contents
        if 'include_fc' in arg_list:
            if self._args.include_fc:
                self._confs['include_fc'] = '.true.'

        if 'include_fs' in arg_list:
            if self._args.include_fs:
                self._confs['include_fs'] = '.true.'

        if 'include_nac_params' in arg_list:
            if self._settings.include_nac_params:
                if not self._args.include_nac_params:
                    self._confs['include_nac_params'] = '.false.'
            else:
                if self._args.include_nac_params:
                    self._confs['include_nac_params'] = '.true.'

        if 'include_disp' in arg_list:
            if self._args.include_disp:
                self._confs['include_disp'] = '.true.'

        if 'include_all' in arg_list:
            if self._args.include_all:
                self._confs['include_all'] = '.true.'

    def _parse_conf(self):
        ConfParser.parse_conf(self)
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'band_format':
                self.set_parameter('band_format', confs['band_format'].lower())

            if conf_key == 'band_labels':
                labels = [x for x in confs['band_labels'].split()]
                self.set_parameter('band_labels', labels)

            if conf_key == 'band_connection':
                if confs['band_connection'].lower() == '.true.':
                    self.set_parameter('band_connection', True)
                elif confs['band_connection'].lower() == '.false.':
                    self.set_parameter('band_connection', False)

            if conf_key == 'legacy_plot':
                if confs['legacy_plot'].lower() == '.true.':
                    self.set_parameter('legacy_plot', True)
                elif confs['legacy_plot'].lower() == '.false.':
                    self.set_parameter('legacy_plot', False)

            if conf_key == 'create_force_sets':
                fnames = [v for v in confs['create_force_sets'].split()]
                self.set_parameter('create_force_sets', fnames)

            if conf_key == 'create_force_sets_zero':
                fnames = [v for v in confs['create_force_sets_zero'].split()]
                self.set_parameter('create_force_sets_zero', fnames)

            if conf_key == 'create_force_constants':
                fnames = [v for v in confs['create_force_constants'].split()]
                self.set_parameter('create_force_constants', fnames[0])

            if conf_key == 'force_constants':
                self.set_parameter('force_constants',
                                   confs['force_constants'].lower())

            if conf_key == 'read_force_constants':
                if confs['read_force_constants'].lower() == '.true.':
                    self.set_parameter('read_force_constants', True)
                elif confs['read_force_constants'].lower() == '.false.':
                    self.set_parameter('read_force_constants', False)

            if conf_key == 'write_force_constants':
                if confs['write_force_constants'].lower() == '.true.':
                    self.set_parameter('write_force_constants', True)
                elif confs['write_force_constants'].lower() == '.false.':
                    self.set_parameter('write_force_constants', False)

            if conf_key == 'full_force_constants':
                if confs['full_force_constants'].lower() == '.true.':
                    self.set_parameter('is_full_fc', True)
                elif confs['full_force_constants'].lower() == '.false.':
                    self.set_parameter('is_full_fc', False)

            if conf_key == 'cutoff_radius':
                val = float(confs['cutoff_radius'])
                self.set_parameter('cutoff_radius', val)

            if conf_key == 'writedm':
                if confs['writedm'].lower() == '.true.':
                    self.set_parameter('write_dynamical_matrices', True)
                elif confs['writedm'].lower() == '.false.':
                    self.set_parameter('write_dynamical_matrices', False)

            if conf_key == 'write_mesh':
                if confs['write_mesh'].lower() == '.true.':
                    self.set_parameter('write_mesh', True)
                elif confs['write_mesh'].lower() == '.false.':
                    self.set_parameter('write_mesh', False)

            if conf_key == 'hdf5':
                if confs['hdf5'].lower() == '.true.':
                    self.set_parameter('hdf5', True)
                elif confs['hdf5'].lower() == '.false.':
                    self.set_parameter('hdf5', False)

            if conf_key == 'mp_shift':
                vals = [fracval(x) for x in confs['mp_shift'].split()]
                if len(vals) < 3:
                    self.setting_error("MP_SHIFT is incorrectly set.")
                self.set_parameter('mp_shift', vals[:3])

            if conf_key == 'mesh_format':
                self.set_parameter('mesh_format', confs['mesh_format'].lower())

            if conf_key == 'qpoints_format':
                self.set_parameter('qpoints_format',
                                   confs['qpoints_format'].lower())

            if conf_key == 'time_reversal_symmetry':
                if confs['time_reversal_symmetry'].lower() == '.true.':
                    self.set_parameter('is_time_reversal_symmetry', True)
                elif confs['time_reversal_symmetry'].lower() == '.false.':
                    self.set_parameter('is_time_reversal_symmetry', False)

            if conf_key == 'gamma_center':
                if confs['gamma_center'].lower() == '.true.':
                    self.set_parameter('is_gamma_center', True)
                elif confs['gamma_center'].lower() == '.false.':
                    self.set_parameter('is_gamma_center', False)

            if conf_key == 'fc_spg_symmetry':
                if confs['fc_spg_symmetry'].lower() == '.true.':
                    self.set_parameter('fc_spg_symmetry', True)
                elif confs['fc_spg_symmetry'].lower() == '.false.':
                    self.set_parameter('fc_spg_symmetry', False)

            if conf_key == 'readfc_format':
                self.set_parameter('readfc_format',
                                   confs['readfc_format'].lower())

            if conf_key == 'writefc_format':
                self.set_parameter('writefc_format',
                                   confs['writefc_format'].lower())

            if conf_key == 'fc_format':
                self.set_parameter('readfc_format', confs['fc_format'].lower())
                self.set_parameter('writefc_format',
                                   confs['fc_format'].lower())

            # Animation
            if conf_key == 'anime':
                vals = []
                data = confs['anime'].split()
                if len(data) < 3:
                    self.setting_error("ANIME is incorrectly set.")
                else:
                    self.set_parameter('anime', data)

            if conf_key == 'anime_type':
                anime_type = confs['anime_type'].lower()
                if anime_type in ('arc', 'v_sim', 'poscar', 'xyz', 'jmol'):
                    self.set_parameter('anime_type', anime_type)
                else:
                    self.setting_error("%s is not available for ANIME_TYPE tag."
                                       % confs['anime_type'])

            # Modulation
            if conf_key == 'modulation':
                self._parse_conf_modulation(confs['modulation'])

            # Character table
            if conf_key == 'irreps':
                vals = [fracval(x) for x in confs['irreps'].split()]
                if len(vals) == 3 or len(vals) == 4:
                    self.set_parameter('irreps_qpoint', vals)
                else:
                    self.setting_error("IRREPS is incorrectly set.")

            if conf_key == 'show_irreps':
                if confs['show_irreps'].lower() == '.true.':
                    self.set_parameter('show_irreps', True)
                elif confs['show_irreps'].lower() == '.false.':
                    self.set_parameter('show_irreps', False)

            if conf_key == 'little_cogroup':
                if confs['little_cogroup'].lower() == '.true.':
                    self.set_parameter('little_cogroup', True)
                elif confs['little_cogroup'].lower() == '.false.':
                    self.set_parameter('little_cogroup', False)

            # DOS
            if conf_key == 'pdos':
                if confs['pdos'].strip().lower() == 'auto':
                    self.set_parameter('pdos', 'auto')
                else:
                    vals = []
                    for index_set in confs['pdos'].split(','):
                        vals.append([int(x) - 1 for x in index_set.split()])
                    self.set_parameter('pdos', vals)

            if conf_key == 'xyz_projection':
                if confs['xyz_projection'].lower() == '.true.':
                    self.set_parameter('xyz_projection', True)
                elif confs['xyz_projection'].lower() == '.false.':
                    self.set_parameter('xyz_projection', False)

            if conf_key == 'dos':
                if confs['dos'].lower() == '.true.':
                    self.set_parameter('dos', True)
                elif confs['dos'].lower() == '.false.':
                    self.set_parameter('dos', False)

            if conf_key == 'debye_model':
                if confs['debye_model'].lower() == '.true.':
                    self.set_parameter('fits_debye_model', True)
                elif confs['debye_model'].lower() == '.false.':
                    self.set_parameter('fits_debye_model', False)

            if conf_key == 'dos_range':
                vals = [float(x) for x in confs['dos_range'].split()]
                self.set_parameter('dos_range', vals)

            if conf_key == 'fmax':
                self.set_parameter('fmax', float(confs['fmax']))

            if conf_key == 'fmin':
                self.set_parameter('fmin', float(confs['fmin']))

            # Thermal properties
            if conf_key == 'tprop':
                if confs['tprop'].lower() == '.true.':
                    self.set_parameter('tprop', True)
                if confs['tprop'].lower() == '.false.':
                    self.set_parameter('tprop', False)

            # Projected thermal properties
            if conf_key == 'ptprop':
                if confs['ptprop'].lower() == '.true.':
                    self.set_parameter('ptprop', True)
                elif confs['ptprop'].lower() == '.false.':
                    self.set_parameter('ptprop', False)

            # Use imaginary frequency as real for thermal property calculation
            if conf_key == 'pretend_real':
                if confs['pretend_real'].lower() == '.true.':
                    self.set_parameter('pretend_real', True)
                elif confs['pretend_real'].lower() == '.false.':
                    self.set_parameter('pretend_real', False)

            # Thermal displacement
            if conf_key == 'tdisp':
                if confs['tdisp'].lower() == '.true.':
                    self.set_parameter('tdisp', True)
                elif confs['tdisp'].lower() == '.false.':
                    self.set_parameter('tdisp', False)

            # Thermal displacement matrices
            if conf_key == 'tdispmat':
                if confs['tdispmat'].lower() == '.true.':
                    self.set_parameter('tdispmat', True)
                elif confs['tdispmat'].lower() == '.false.':
                    self.set_parameter('tdispmat', False)

            # Write thermal displacement matrices to cif file,
            # for which the temperature to execute is stored.
            if conf_key == 'tdispmat_cif':
                self.set_parameter('tdispmat_cif', float(confs['tdispmat_cif']))

            # Thermal distance
            if conf_key == 'tdistance':
                atom_pairs = []
                for atoms in confs['tdistance'].split(','):
                    pair = [int(x) - 1 for x in atoms.split()]
                    if len(pair) == 2:
                        atom_pairs.append(pair)
                    else:
                        self.setting_error(
                            "TDISTANCE is incorrectly specified.")
                if len(atom_pairs) > 0:
                    self.set_parameter('tdistance', atom_pairs)

            # Projection direction used for thermal displacements and PDOS
            if conf_key == 'projection_direction':
                vals = [float(x) for x in confs['projection_direction'].split()]
                if len(vals) < 3:
                    self.setting_error(
                        "PROJECTION_DIRECTION (--pd) is incorrectly specified.")
                else:
                    self.set_parameter('projection_direction', vals)

            # Group velocity
            if conf_key == 'group_velocity':
                if confs['group_velocity'].lower() == '.true.':
                    self.set_parameter('is_group_velocity', True)
                elif confs['group_velocity'].lower() == '.false.':
                    self.set_parameter('is_group_velocity', False)

            # Moment of phonon states distribution
            if conf_key == 'moment':
                if confs['moment'].lower() == '.true.':
                    self.set_parameter('moment', True)
                elif confs['moment'].lower() == '.false.':
                    self.set_parameter('moment', False)

            if conf_key == 'moment_order':
                self.set_parameter('moment_order', int(confs['moment_order']))

            # Number of supercells with random displacements
            if conf_key == 'random_displacements':
                self.set_parameter('random_displacements',
                                   int(confs['random_displacements']))

            if conf_key == 'random_seed':
                self.set_parameter('random_seed', int(confs['random_seed']))

            # Use Lapack solver via Lapacke
            if conf_key == 'lapack_solver':
                if confs['lapack_solver'].lower() == '.true.':
                    self.set_parameter('lapack_solver', True)
                elif confs['lapack_solver'].lower() == '.false.':
                    self.set_parameter('lapack_solver', False)

            # Select yaml summary contents
            if conf_key == 'save_params':
                if confs['save_params'].lower() == '.true.':
                    self.set_parameter('save_params', True)
                elif confs['save_params'].lower() == '.false.':
                    self.set_parameter('save_params', False)

            if conf_key == 'include_fc':
                if confs['include_fc'].lower() == '.true.':
                    self.set_parameter('include_fc', True)
                elif confs['include_fc'].lower() == '.false.':
                    self.set_parameter('include_fc', False)

            if conf_key == 'include_fs':
                if confs['include_fs'].lower() == '.true.':
                    self.set_parameter('include_fs', True)
                elif confs['include_fs'].lower() == '.false.':
                    self.set_parameter('include_fs', False)

            if conf_key in ('include_born', 'include_nac_params'):
                if confs[conf_key].lower() == '.true.':
                    self.set_parameter('include_nac_params', True)
                elif confs[conf_key].lower() == '.false.':
                    self.set_parameter('include_nac_params', False)

            if conf_key == 'include_disp':
                if confs['include_disp'].lower() == '.true.':
                    self.set_parameter('include_disp', True)
                elif confs['include_disp'].lower() == '.false.':
                    self.set_parameter('include_disp', False)

            if conf_key == 'include_all':
                if confs['include_all'].lower() == '.true.':
                    self.set_parameter('include_all', True)
                elif confs['include_all'].lower() == '.false.':
                    self.set_parameter('include_all', False)

    def _parse_conf_modulation(self, conf_modulation):
        modulation = {}
        modulation['dimension'] = [1, 1, 1]
        modulation['order'] = None
        mod_list = conf_modulation.split(',')
        header = mod_list[0].split()
        if len(header) > 2 and len(mod_list) > 1:
            if len(header) > 8:
                dimension = [int(x) for x in header[:9]]
                modulation['dimension'] = dimension
                if len(header) > 11:
                    delta_q = [float(x) for x in header[9:12]]
                    modulation['delta_q'] = delta_q
                if len(header) == 13:
                    modulation['order'] = int(header[12])
            else:
                dimension = [int(x) for x in header[:3]]
                modulation['dimension'] = dimension
                if len(header) > 3:
                    delta_q = [float(x) for x in header[3:6]]
                    modulation['delta_q'] = delta_q
                if len(header) == 7:
                    modulation['order'] = int(header[6])

            vals = []
            for phonon_mode in mod_list[1:]:
                mode_conf = [x for x in phonon_mode.split()]
                if len(mode_conf) < 4 or len(mode_conf) > 6:
                    self.setting_error("MODULATION tag is wrongly set.")
                    break
                else:
                    q = [fracval(x) for x in mode_conf[:3]]

                if len(mode_conf) == 4:
                    vals.append([q, int(mode_conf[3]) - 1, 1.0, 0])
                elif len(mode_conf) == 5:
                    vals.append([q,
                                 int(mode_conf[3]) - 1,
                                 float(mode_conf[4]),
                                 0])
                else:
                    vals.append([q,
                                 int(mode_conf[3]) - 1,
                                 float(mode_conf[4]),
                                 float(mode_conf[5])])

            modulation['modulations'] = vals
            self.set_parameter('modulation', modulation)
        else:
            self.setting_error("MODULATION tag is wrongly set.")

    def _set_settings(self):
        self.set_settings()
        params = self._parameters

        # Create FORCE_SETS
        if 'create_force_sets' in params:
            self._settings.set_create_force_sets(params['create_force_sets'])

        if 'create_force_sets_zero' in params:
            self._settings.set_create_force_sets_zero(
                params['create_force_sets_zero'])

        if 'create_force_constants' in params:
            self._settings.set_create_force_constants(
                params['create_force_constants'])

        # Is force constants written or read?
        if 'force_constants' in params:
            if params['force_constants'] == 'write':
                self._settings.set_write_force_constants(True)
            elif params['force_constants'] == 'read':
                self._settings.set_read_force_constants(True)

        if 'read_force_constants' in params:
            self._settings.set_read_force_constants(
                params['read_force_constants'])

        if 'write_force_constants' in params:
            self._settings.set_write_force_constants(
                params['write_force_constants'])

        if 'is_full_fc' in params:
            self._settings.set_is_full_fc(params['is_full_fc'])

        # Enforce space group symmetyr to force constants?
        if 'fc_spg_symmetry' in params:
            self._settings.set_fc_spg_symmetry(params['fc_spg_symmetry'])

        if 'readfc_format' in params:
            self._settings.set_readfc_format(params['readfc_format'])

        if 'writefc_format' in params:
            self._settings.set_writefc_format(params['writefc_format'])

        # Use hdf5?
        if 'hdf5' in params:
            self._settings.set_is_hdf5(params['hdf5'])

        # Cutoff radius of force constants
        if 'cutoff_radius' in params:
            self._settings.set_cutoff_radius(params['cutoff_radius'])

        # Mesh
        if 'mesh_numbers' in params:
            self._settings.set_run_mode('mesh')
            self._settings.set_mesh_numbers(params['mesh_numbers'])
            if 'mp_shift' in params:
                self._settings.set_mesh_shift(params['mp_shift'])
            if 'is_time_reversal_symmetry' in params:
                self._settings.set_is_time_reversal_symmetry(
                    params['is_time_reversal_symmetry'])
            if 'is_mesh_symmetry' in params:
                self._settings.set_is_mesh_symmetry(params['is_mesh_symmetry'])
            if 'is_gamma_center' in params:
                self._settings.set_is_gamma_center(params['is_gamma_center'])
            if 'mesh_format' in params:
                self._settings.set_mesh_format(params['mesh_format'])

        # band mode
        if 'band_paths' in params:
            self._settings.set_run_mode('band')
            if 'band_format' in params:
                self._settings.set_band_format(params['band_format'])
            if 'band_labels' in params:
                self._settings.set_band_labels(params['band_labels'])
            if 'band_connection' in params:
                self._settings.set_is_band_connection(params['band_connection'])
            if 'legacy_plot' in params:
                self._settings.set_is_legacy_plot(params['legacy_plot'])

        # Q-points mode
        if 'qpoints' in params or 'read_qpoints' in params:
            self._settings.set_run_mode('qpoints')
        if self._settings.run_mode == 'qpoints':
            if 'qpoints_format' in params:
                self._settings.set_qpoints_format(params['qpoints_format'])

        # Whether write out dynamical matrices or not
        if 'write_dynamical_matrices' in params:
            self._settings.set_write_dynamical_matrices(
                params['write_dynamical_matrices'])

        # Whether write out mesh.yaml or mesh.hdf5
        if 'write_mesh' in params:
            self._settings.set_write_mesh(params['write_mesh'])

        # Anime mode
        if 'anime_type' in params:
            self._settings.set_anime_type(params['anime_type'])

        if 'anime' in params:
            self._settings.set_run_mode('anime')
            anime_type = self._settings.anime_type
            if anime_type == 'v_sim':
                qpoints = [fracval(x) for x in params['anime'][0:3]]
                self._settings.set_anime_qpoint(qpoints)
                if len(params['anime']) > 3:
                    self._settings.set_anime_amplitude(float(params['anime'][3]))
            else:
                self._settings.set_anime_band_index(int(params['anime'][0]))
                self._settings.set_anime_amplitude(float(params['anime'][1]))
                self._settings.set_anime_division(int(params['anime'][2]))
            if len(params['anime']) == 6:
                self._settings.set_anime_shift(
                    [fracval(x) for x in params['anime'][3:6]])

        # Modulation mode
        if 'modulation' in params:
            self._settings.set_run_mode('modulation')
            self._settings.set_modulation(params['modulation'])

        # Character table mode
        if 'irreps_qpoint' in params:
            self._settings.set_run_mode('irreps')
            self._settings.set_irreps_q_point(
                params['irreps_qpoint'][:3])
            if len(params['irreps_qpoint']) == 4:
                self._settings.set_irreps_tolerance(params['irreps_qpoint'][3])
            if self._settings.run_mode == 'irreps':
                if 'show_irreps' in params:
                    self._settings.set_show_irreps(params['show_irreps'])
                if 'little_cogroup' in params:
                    self._settings.set_is_little_cogroup(params['little_cogroup'])

        # DOS
        if 'dos_range' in params:
            fmin =  params['dos_range'][0]
            fmax =  params['dos_range'][1]
            fpitch = params['dos_range'][2]
            self._settings.set_min_frequency(fmin)
            self._settings.set_max_frequency(fmax)
            self._settings.set_frequency_pitch(fpitch)
        if 'dos' in params:
            self._settings.set_is_dos_mode(params['dos'])

        if 'fits_debye_model' in params:
            self._settings.set_fits_Debye_model(params['fits_debye_model'])

        if 'fmax' in params:
            self._settings.set_max_frequency(params['fmax'])

        if 'fmin' in params:
            self._settings.set_min_frequency(params['fmin'])

        # Project PDOS x, y, z directions in Cartesian coordinates
        if 'xyz_projection' in params:
            self._settings.set_xyz_projection(params['xyz_projection'])
            if ('pdos' not in params and
                self._settings.pdos_indices is None):
                self.set_parameter('pdos', [])

        if 'pdos' in params:
            self._settings.set_pdos_indices(params['pdos'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)

        if ('projection_direction' in params and
            not self._settings.xyz_projection):
            self._settings.set_projection_direction(
                params['projection_direction'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)

        # Thermal properties
        if 'tprop' in params:
            self._settings.set_is_thermal_properties(params['tprop'])
            # Exclusive conditions
            self._settings.set_is_thermal_displacements(False)
            self._settings.set_is_thermal_displacement_matrices(False)
            self._settings.set_is_thermal_distances(False)

        # Projected thermal properties
        if 'ptprop' in params and params['ptprop']:
            self._settings.set_is_thermal_properties(True)
            self._settings.set_is_projected_thermal_properties(True)
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)
            # Exclusive conditions
            self._settings.set_is_thermal_displacements(False)
            self._settings.set_is_thermal_displacement_matrices(False)
            self._settings.set_is_thermal_distances(False)

        # Use imaginary frequency as real for thermal property calculation
        if 'pretend_real' in params:
            self._settings.set_pretend_real(params['pretend_real'])

        # Thermal displacements
        if 'tdisp' in params and params['tdisp']:
            self._settings.set_is_thermal_displacements(True)
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)
            # Exclusive conditions
            self._settings.set_is_thermal_properties(False)
            self._settings.set_is_thermal_displacement_matrices(False)
            self._settings.set_is_thermal_distances(True)

        # Thermal displacement matrices
        if ('tdispmat' in params and params['tdispmat'] or
            'tdispmat_cif' in params):
            self._settings.set_is_thermal_displacement_matrices(True)
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)
            # Exclusive conditions
            self._settings.set_is_thermal_properties(False)
            self._settings.set_is_thermal_displacements(False)
            self._settings.set_is_thermal_distances(False)

            # Temperature used to calculate thermal displacement matrix
            # to write aniso_U to cif
            if 'tdispmat_cif' in params:
                self._settings.set_thermal_displacement_matrix_temperature(
                    params['tdispmat_cif'])

        # Thermal distances
        if 'tdistance' in params:
            self._settings.set_is_thermal_distances(True)
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)
            self._settings.set_thermal_atom_pairs(params['tdistance'])
            # Exclusive conditions
            self._settings.set_is_thermal_properties(False)
            self._settings.set_is_thermal_displacements(False)
            self._settings.set_is_thermal_displacement_matrices(False)

        # Group velocity
        if 'is_group_velocity' in params:
            self._settings.set_is_group_velocity(params['is_group_velocity'])

        # Moment mode
        if 'moment' in params:
            self._settings.set_is_moment(params['moment'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_mesh_symmetry(False)

        if self._settings.is_moment:
            if 'moment_order' in params:
                self._settings.set_moment_order(params['moment_order'])

        # Number of supercells with random displacements
        if 'random_displacements' in params:
            self._settings.set_random_displacements(
                params['random_displacements'])

        if 'random_seed' in params:
            self._settings.set_random_seed(params['random_seed'])

        # Use Lapack solver via Lapacke
        if 'lapack_solver' in params:
            self._settings.set_lapack_solver(params['lapack_solver'])

        # Select yaml summary contents
        if 'save_params' in params:
            self._settings.set_save_params(params['save_params'])

        if 'include_fc' in params:
            self._settings.set_include_force_constants(params['include_fc'])

        if 'include_fs' in params:
            self._settings.set_include_force_sets(params['include_fs'])

        if 'include_nac_params' in params:
            self._settings.set_include_nac_params(params['include_nac_params'])

        if 'include_disp' in params:
            self._settings.set_include_displacements(params['include_disp'])

        if 'include_all' in params:
            self._settings.set_include_force_constants(True)
            self._settings.set_include_force_sets(True)
            self._settings.set_include_nac_params(True)
            self._settings.set_include_displacements(True)

        # ***********************************************************
        # This has to come last in this method to overwrite run_mode.
        # ***********************************************************
        if 'pdos' in params and params['pdos'] == 'auto':
            if 'band_paths' in params:
                self._settings.set_run_mode('band_mesh')
            else:
                self._settings.set_run_mode('mesh')

        if 'mesh_numbers' in params and 'band_paths' in params:
            self._settings.set_run_mode('band_mesh')
