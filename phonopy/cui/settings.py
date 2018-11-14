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
    def __init__(self):
        self._band_indices = None
        self._band_paths = None
        self._band_points = None
        self._cell_filename = None
        self._chemical_symbols = None
        self._cutoff_frequency = None
        self._displacement_distance = None
        self._dm_decimals = None
        self._fc_decimals = None
        self._fc_symmetry = False
        self._fpitch = None
        self._frequency_conversion_factor = None
        self._frequency_scale_factor = None
        self._gv_delta_q = None
        self._is_diagonal_displacement = True
        self._is_eigenvectors = False
        self._is_mesh_symmetry = True
        self._is_nac = False
        self._is_rotational_invariance = False
        self._is_plusminus_displacement = 'auto'
        self._is_symmetry = True
        self._is_tetrahedron_method = False
        self._is_time_reversal_symmetry = True
        self._is_trigonal_displacement = False
        self._magmoms = None
        self._masses = None
        self._mesh = None
        self._mesh_shift = None
        self._nac_method = None
        self._nac_q_direction = None
        self._num_frequency_points = None
        self._primitive_matrix = None
        self._qpoints = None
        self._read_qpoints = False
        self._sigma = None
        self._supercell_matrix = None
        self._tmax = 1000
        self._tmin = 0
        self._tstep = 10
        self._use_alm = False
        self._yaml_mode = False

    def set_band_paths(self, band_paths):
        self._band_paths = band_paths

    def get_band_paths(self):
        return self._band_paths

    def set_band_points(self, band_points):
        self._band_points = band_points

    def get_band_points(self):
        return self._band_points

    def set_band_indices(self, band_indices):
        self._band_indices = band_indices

    def get_band_indices(self):
        return self._band_indices

    def set_cell_filename(self, cell_filename):
        self._cell_filename = cell_filename

    def get_cell_filename(self):
        return self._cell_filename

    def set_chemical_symbols(self, symbols):
        self._chemical_symbols = symbols

    def get_chemical_symbols(self):
        return self._chemical_symbols

    def set_cutoff_frequency(self, cutoff_frequency):
        self._cutoff_frequency = cutoff_frequency

    def get_cutoff_frequency(self):
        return self._cutoff_frequency

    def set_dm_decimals(self, decimals):
        self._dm_decimals = decimals

    def get_dm_decimals(self):
        return self._dm_decimals

    def set_displacement_distance(self, distance):
        self._displacement_distance = distance

    def get_displacement_distance(self):
        return self._displacement_distance

    def set_fc_symmetry(self, fc_symmetry):
        self._fc_symmetry = fc_symmetry

    def get_fc_symmetry(self):
        return self._fc_symmetry

    def set_fc_decimals(self, decimals):
        self._fc_decimals = decimals

    def get_fc_decimals(self):
        return self._fc_decimals

    def set_frequency_conversion_factor(self, frequency_conversion_factor):
        self._frequency_conversion_factor = frequency_conversion_factor

    def get_frequency_conversion_factor(self):
        return self._frequency_conversion_factor

    def set_frequency_pitch(self, fpitch):
        self._fpitch = fpitch

    def get_frequency_pitch(self):
        return self._fpitch

    def set_frequency_scale_factor(self, frequency_scale_factor):
        self._frequency_scale_factor = frequency_scale_factor

    def get_frequency_scale_factor(self):
        return self._frequency_scale_factor

    def set_num_frequency_points(self, num_frequency_points):
        self._num_frequency_points = num_frequency_points

    def get_num_frequency_points(self):
        return self._num_frequency_points

    def set_group_velocity_delta_q(self, gv_delta_q):
        self._gv_delta_q = gv_delta_q

    def get_group_velocity_delta_q(self):
        return self._gv_delta_q

    def set_is_diagonal_displacement(self, is_diag):
        self._is_diagonal_displacement = is_diag

    def get_is_diagonal_displacement(self):
        return self._is_diagonal_displacement

    def set_is_eigenvectors(self, is_eigenvectors):
        self._is_eigenvectors = is_eigenvectors

    def get_is_eigenvectors(self):
        return self._is_eigenvectors

    def set_is_mesh_symmetry(self, is_mesh_symmetry):
        self._is_mesh_symmetry = is_mesh_symmetry

    def get_is_mesh_symmetry(self):
        return self._is_mesh_symmetry

    def set_is_nac(self, is_nac):
        self._is_nac = is_nac

    def get_is_nac(self):
        return self._is_nac

    def set_is_plusminus_displacement(self, is_pm):
        self._is_plusminus_displacement = is_pm

    def get_is_plusminus_displacement(self):
        return self._is_plusminus_displacement

    def set_is_rotational_invariance(self, is_rotational_invariance):
        self._is_rotational_invariance = is_rotational_invariance

    def get_is_rotational_invariance(self):
        return self._is_rotational_invariance

    def set_is_tetrahedron_method(self, is_thm):
        self._is_tetrahedron_method = is_thm

    def get_is_tetrahedron_method(self):
        return self._is_tetrahedron_method

    def set_is_trigonal_displacement(self, is_trigonal):
        self._is_trigonal_displacement = is_trigonal

    def get_is_trigonal_displacement(self):
        return self._is_trigonal_displacement

    def set_is_symmetry(self, is_symmetry):
        self._is_symmetry = is_symmetry

    def get_is_symmetry(self):
        return self._is_symmetry

    def set_magnetic_moments(self, magmoms):
        self._magmoms = magmoms

    def get_magnetic_moments(self):
        return self._magmoms

    def set_masses(self, masses):
        self._masses = masses

    def get_masses(self):
        return self._masses

    def set_max_temperature(self, tmax):
        self._tmax = tmax

    def get_max_temperature(self):
        return self._tmax

    def set_mesh_numbers(self, mesh):
        self._mesh = mesh

    def get_mesh_numbers(self):
        return self._mesh

    def set_mesh_shift(self, mesh_shift):
        self._mesh_shift = mesh_shift

    def get_mesh_shift(self):
        return self._mesh_shift

    def set_min_temperature(self, tmin):
        self._tmin = tmin

    def get_min_temperature(self):
        return self._tmin

    def set_nac_method(self, nac_method):
        self._nac_method = nac_method

    def get_nac_method(self):
        return self._nac_method

    def set_nac_q_direction(self, nac_q_direction):
        self._nac_q_direction = nac_q_direction

    def get_nac_q_direction(self):
        return self._nac_q_direction

    def set_primitive_matrix(self, primitive_matrix):
        self._primitive_matrix = primitive_matrix

    def get_primitive_matrix(self):
        return self._primitive_matrix

    def set_qpoints(self, qpoints):
        self._qpoints = qpoints

    def get_qpoints(self):
        return self._qpoints

    def set_read_qpoints(self, read_qpoints):
        self._read_qpoints = read_qpoints

    def get_read_qpoints(self):
        return self._read_qpoints

    def set_sigma(self, sigma):
        self._sigma = sigma

    def get_sigma(self):
        return self._sigma

    def set_supercell_matrix(self, matrix):
        self._supercell_matrix = matrix

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def set_temperature_step(self, tstep):
        self._tstep = tstep

    def get_temperature_step(self):
        return self._tstep

    def set_time_reversal_symmetry(self, time_reversal_symmetry=True):
        self._is_time_reversal_symmetry = time_reversal_symmetry

    def get_time_reversal_symmetry(self):
        return self._is_time_reversal_symmetry

    def set_use_alm(self, use_alm):
        self._use_alm = use_alm

    def get_use_alm(self):
        return self._use_alm

    def set_yaml_mode(self, yaml_mode):
        self._yaml_mode = yaml_mode

    def get_yaml_mode(self):
        return self._yaml_mode


# Parse phonopy setting filen
class ConfParser(object):
    def __init__(self, filename=None, args=None):
        self._confs = {}
        self._parameters = {}
        self._args = args
        self._filename = filename

    def get_configures(self):
        return self._confs

    def get_settings(self):
        return self._settings

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

        if 'fc_symmetry' in arg_list:
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

        if 'frequency_scale_factor' in self._args:
            freq_scale = self._args.frequency_scale_factor
            if freq_scale is not None:
                self._confs['frequency_scale_factor'] = freq_scale

        if 'gv_delta_q' in arg_list:
            if self._args.gv_delta_q:
                self._confs['gv_delta_q'] = self._args.gv_delta_q

        if 'is_eigenvectors' in arg_list:
            if self._args.is_eigenvectors:
                self._confs['eigenvectors'] = '.true.'

        if 'is_nac' in arg_list:
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

        if 'is_tetrahedron_method' in arg_list:
            if self._args.is_tetrahedron_method:
                self._confs['tetrahedron'] = '.true.'

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

        if 'tmax' in arg_list:
            if self._args.tmax:
                self._confs['tmax'] = self._args.tmax

        if 'tmin' in arg_list:
            if self._args.tmin:
                self._confs['tmin'] = self._args.tmin

        if 'tstep' in arg_list:
            if self._args.tstep:
                self._confs['tstep'] = self._args.tstep

        if 'use_alm' in arg_list:
            if self._args.use_alm:
                self._confs['alm'] = '.true.'

        if 'yaml_mode' in arg_list:
            if self._args.yaml_mode:
                self._confs['yaml_mode'] = '.true.'

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
                vals = [int(x) for x in confs[conf_key].split()]
                if len(vals) < 3:
                    self.setting_error("Mesh numbers are incorrectly set.")
                self.set_parameter('mesh_numbers', vals[:3])

            if conf_key == 'band_points':
                self.set_parameter('band_points', int(confs['band_points']))

            if conf_key == 'band':
                bands = []
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
                q_direction = [fracval(x) for x in confs['q_direction'].split()]
                if len(q_direction) < 3:
                    self.setting_error("Number of elements of q_direction is less than 3")
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

            # Phonopy YAML mode
            if conf_key == 'yaml_mode':
                if confs['yaml_mode'].lower() == '.true.':
                    self.set_parameter('yaml_mode', True)

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

        # Supercell size
        if 'supercell_matrix' in params:
            self._settings.set_supercell_matrix(params['supercell_matrix'])

        # Temerature range
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
        if 'band_paths' in params:
            self._settings.set_band_paths(params['band_paths'])

        # This number includes end points
        if 'band_points' in params:
            self._settings.set_band_points(params['band_points'])

        # Use ALM to generating force constants
        if 'alm' in params:
            self._settings.set_use_alm(params['alm'])

        # Activate phonopy YAML mode
        if 'yaml_mode' in params:
            self._settings.set_yaml_mode(params['yaml_mode'])

#
# For phonopy
#
class PhonopySettings(Settings):
    def __init__(self):
        Settings.__init__(self)

        self._anime_band_index = None
        self._anime_amplitude = None
        self._anime_division = None
        self._anime_qpoint = None
        self._anime_shift = None
        self._anime_type = 'v_sim'
        self._band_format = 'yaml'
        self._band_labels = None
        self._band_connection = False
        self._cutoff_radius = None
        self._dos = None
        self._fc_spg_symmetry = False
        self._fits_Debye_model = False
        self._fmax = None
        self._fmin = None
        self._irreps_q_point = None
        self._irreps_tolerance = 1e-5
        self._is_dos_mode = False
        self._is_full_fc = False
        self._is_group_velocity = False
        self._is_gamma_center = False
        self._is_hdf5 = False
        self._is_little_cogroup = False
        self._is_moment = False
        self._is_plusminus_displacement = 'auto'
        self._is_thermal_displacements = False
        self._is_thermal_displacement_matrices = False
        self._is_thermal_distances = False
        self._is_thermal_properties = False
        self._is_projected_thermal_properties = False
        self._lapack_solver = False
        self._mesh_format = 'yaml'
        self._modulation = None
        self._moment_order = None
        self._pdos_indices = None
        self._pretend_real = False
        self._projection_direction = None
        self._qpoints_format = 'yaml'
        self._read_force_constants = False
        self._readfc_format = 'text'
        self._run_mode = None
        self._show_irreps = False
        self._thermal_atom_pairs = None
        self._thermal_displacement_matrix_temperatue = None
        self._write_dynamical_matrices = False
        self._write_mesh = True
        self._write_force_constants = False
        self._writefc_format = 'text'
        self._xyz_projection = False

    def set_anime_band_index(self, band_index):
        self._anime_band_index = band_index

    def get_anime_band_index(self):
        return self._anime_band_index

    def set_anime_amplitude(self, amplitude):
        self._anime_amplitude = amplitude

    def get_anime_amplitude(self):
        return self._anime_amplitude

    def set_anime_division(self, division):
        self._anime_division = division

    def get_anime_division(self):
        return self._anime_division

    def set_anime_qpoint(self, qpoint):
        self._anime_qpoint = qpoint

    def get_anime_qpoint(self):
        return self._anime_qpoint

    def set_anime_shift(self, shift):
        self._anime_shift = shift

    def get_anime_shift(self):
        return self._anime_shift

    def set_anime_type(self, anime_type):
        self._anime_type = anime_type

    def get_anime_type(self):
        return self._anime_type

    def set_band_format(self, band_format):
        self._band_format = band_format

    def get_band_format(self):
        return self._band_format

    def set_band_labels(self, labels):
        self._band_labels = labels

    def get_band_labels(self):
        return self._band_labels

    def set_cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = cutoff_radius

    def get_cutoff_radius(self):
        return self._cutoff_radius

    def set_fc_spg_symmetry(self, fc_spg_symmetry):
        self._fc_spg_symmetry = fc_spg_symmetry

    def get_fc_spg_symmetry(self):
        return self._fc_spg_symmetry

    def set_fits_Debye_model(self, fits_Debye_model):
        self._fits_Debye_model = fits_Debye_model

    def get_fits_Debye_model(self):
        return self._fits_Debye_model

    def set_max_frequency(self, fmax):
        self._fmax = fmax

    def get_max_frequency(self):
        return self._fmax

    def set_min_frequency(self, fmin):
        self._fmin = fmin

    def get_min_frequency(self):
        return self._fmin

    def set_irreps_q_point(self, q_point):
        self._irreps_q_point = q_point

    def get_irreps_q_point(self):
        return self._irreps_q_point

    def set_irreps_tolerance(self, tolerance):
        self._irreps_tolerance = tolerance

    def get_irreps_tolerance(self):
        return self._irreps_tolerance

    def set_is_band_connection(self, band_connection):
        self._band_connection = band_connection

    def get_is_band_connection(self):
        return self._band_connection

    def set_is_dos_mode(self, is_dos_mode):
        self._is_dos_mode = is_dos_mode

    def get_is_dos_mode(self):
        return self._is_dos_mode

    def set_is_full_fc(self, is_full_fc):
        self._is_full_fc = is_full_fc

    def get_is_full_fc(self):
        return self._is_full_fc

    def set_is_gamma_center(self, is_gamma_center):
        self._is_gamma_center = is_gamma_center

    def get_is_gamma_center(self):
        return self._is_gamma_center

    def set_is_group_velocity(self, is_group_velocity):
        self._is_group_velocity = is_group_velocity

    def get_is_group_velocity(self):
        return self._is_group_velocity

    def set_is_hdf5(self, is_hdf5):
        self._is_hdf5 = is_hdf5

    def get_is_hdf5(self):
        return self._is_hdf5

    def set_is_little_cogroup(self, is_little_cogroup):
        self._is_little_cogroup = is_little_cogroup

    def get_is_little_cogroup(self):
        return self._is_little_cogroup

    def set_is_moment(self, is_moment):
        self._is_moment = is_moment

    def get_is_moment(self):
        return self._is_moment

    def set_is_projected_thermal_properties(self, is_ptp):
        self._is_projected_thermal_properties = is_ptp

    def get_is_projected_thermal_properties(self):
        return self._is_projected_thermal_properties

    def set_is_thermal_displacements(self, is_thermal_displacements):
        self._is_thermal_displacements = is_thermal_displacements

    def get_is_thermal_displacements(self):
        return self._is_thermal_displacements

    def set_is_thermal_displacement_matrices(self, is_displacement_matrices):
        self._is_thermal_displacement_matrices = is_displacement_matrices

    def get_is_thermal_displacement_matrices(self):
        return self._is_thermal_displacement_matrices

    def set_is_thermal_distances(self, is_thermal_distances):
        self._is_thermal_distances = is_thermal_distances

    def get_is_thermal_distances(self):
        return self._is_thermal_distances

    def set_is_thermal_properties(self, is_thermal_properties):
        self._is_thermal_properties = is_thermal_properties

    def get_is_thermal_properties(self):
        return self._is_thermal_properties

    def set_lapack_solver(self, lapack_solver):
        self._lapack_solver = lapack_solver

    def get_lapack_solver(self):
        return self._lapack_solver

    def set_mesh(self,
                 mesh,
                 mesh_shift=None,
                 is_time_reversal_symmetry=True,
                 is_mesh_symmetry=True,
                 is_gamma_center=False):
        if mesh_shift is None:
            mesh_shift = [0.,0.,0.]
        self._mesh = mesh
        self._mesh_shift = mesh_shift
        self._is_time_reversal_symmetry = is_time_reversal_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._is_gamma_center = is_gamma_center

    def get_mesh(self):
        return (self._mesh,
                self._mesh_shift,
                self._is_time_reversal_symmetry,
                self._is_mesh_symmetry,
                self._is_gamma_center)

    def set_mesh_format(self, mesh_format):
        self._mesh_format = mesh_format

    def get_mesh_format(self):
        return self._mesh_format

    def set_modulation(self, modulation):
        self._modulation = modulation

    def get_modulation(self):
        return self._modulation

    def set_moment_order(self, moment_order):
        self._moment_order = moment_order

    def get_moment_order(self):
        return self._moment_order

    def set_pdos_indices(self, indices):
        self._pdos_indices = indices

    def get_pdos_indices(self):
        return self._pdos_indices

    def set_pretend_real(self, pretend_real):
        self._pretend_real = pretend_real

    def get_pretend_real(self):
        return self._pretend_real

    def set_projection_direction(self, direction):
        self._projection_direction = direction

    def get_projection_direction(self):
        return self._projection_direction

    def set_qpoints_format(self, qpoints_format):
        self._qpoints_format = qpoints_format

    def get_qpoints_format(self):
        return self._qpoints_format

    def set_read_force_constants(self, read_force_constants):
        self._read_force_constants = read_force_constants

    def get_read_force_constants(self):
        return self._read_force_constants

    def set_readfc_format(self, readfc_format):
        self._readfc_format = readfc_format

    def get_readfc_format(self):
        return self._readfc_format

    def set_run_mode(self, run_mode):
        modes = ['qpoints',
                 'mesh',
                 'band',
                 'band_mesh',
                 'anime',
                 'modulation',
                 'displacements',
                 'irreps']
        for mode in modes:
            if run_mode.lower() == mode:
                self._run_mode = run_mode

    def get_run_mode(self):
        return self._run_mode

    def set_thermal_property_range(self, tmin, tmax, tstep):
        self._tmax = tmax
        self._tmin = tmin
        self._tstep = tstep

    def get_thermal_property_range(self):
        return {'min':  self._tmin,
                'max':  self._tmax,
                'step': self._tstep}

    def set_thermal_atom_pairs(self, atom_pairs):
        self._thermal_atom_pairs = atom_pairs

    def get_thermal_atom_pairs(self):
        return self._thermal_atom_pairs

    def set_thermal_displacement_matrix_temperature(self, t):
        self._thermal_displacement_matrix_temperatue = t

    def get_thermal_displacement_matrix_temperature(self):
        return self._thermal_displacement_matrix_temperatue

    def set_show_irreps(self, show_irreps):
        self._show_irreps = show_irreps

    def get_show_irreps(self):
        return self._show_irreps

    def set_write_dynamical_matrices(self, write_dynamical_matrices):
        self._write_dynamical_matrices = write_dynamical_matrices

    def get_write_dynamical_matrices(self):
        return self._write_dynamical_matrices

    def set_write_force_constants(self, write_force_constants):
        self._write_force_constants = write_force_constants

    def get_write_force_constants(self):
        return self._write_force_constants

    def set_write_mesh(self, write_mesh):
        self._write_mesh = write_mesh

    def get_write_mesh(self):
        return self._write_mesh

    def set_writefc_format(self, writefc_format):
        self._writefc_format = writefc_format

    def get_writefc_format(self):
        return self._writefc_format

    def set_xyz_projection(self, xyz_projection):
        self._xyz_projection = xyz_projection

    def get_xyz_projection(self):
        return self._xyz_projection

class PhonopyConfParser(ConfParser):
    def __init__(self, filename=None, args=None):
        self._settings = PhonopySettings()
        if filename is not None:
            ConfParser.__init__(self, filename=filename)
            self.read_file()  # store .conf file setting in self._confs
            self._parse_conf()  # self.parameters[key] = val
            self._set_settings()  # self.parameters -> PhonopySettings
        if args is not None:
            # To invoke ConfParser.__init__() to flush variables.
            ConfParser.__init__(self, args=args)
            self._read_options()  # store options in self._confs
            self._parse_conf()  # self.parameters[key] = val
            self._set_settings()  # self.parameters -> PhonopySettings

    def _read_options(self):
        self.read_options() # store data in self._confs
        arg_list = vars(self._args)
        if 'band_format' in arg_list:
            if self._args.band_format:
                self._confs['band_format'] = self._args.band_format

        if 'band_labels' in arg_list:
            if self._args.band_labels is not None:
                self._confs['band_labels'] = " ".join(self._args.band_labels)

        if 'is_displacement' in arg_list:
            if self._args.is_displacement:
                self._confs['create_displacements'] = '.true.'

        if 'is_gamma_center' in arg_list:
            if self._args.is_gamma_center:
                self._confs['gamma_center'] = '.true.'

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

        if 'show_irreps' in arg_list:
            if self._args.show_irreps:
                self._confs['show_irreps'] = '.true.'

        if 'is_little_cogroup' in arg_list:
            if self._args.is_little_cogroup:
                self._confs['little_cogroup'] = '.true.'

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

        # Overwrite
        if 'is_check_symmetry' in arg_list:
            if self._args.is_check_symmetry:
                # Dummy 'dim' setting for sym-check
                self._confs['dim'] = '1 1 1'

        if 'lapack_solver' in arg_list:
            if self._args.lapack_solver:
                self._confs['lapack_solver'] = '.true.'

    def _parse_conf(self):
        self.parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'create_displacements':
                if confs['create_displacements'].lower() == '.true.':
                    self.set_parameter('create_displacements', True)

            if conf_key == 'band_format':
                self.set_parameter('band_format', confs['band_format'].lower())

            if conf_key == 'band_labels':
                labels = [x for x in confs['band_labels'].split()]
                self.set_parameter('band_labels', labels)

            if conf_key == 'band_connection':
                if confs['band_connection'].lower() == '.true.':
                    self.set_parameter('band_connection', True)

            if conf_key == 'force_constants':
                self.set_parameter('force_constants',
                                   confs['force_constants'].lower())

            if conf_key == 'read_force_constants':
                if confs['read_force_constants'].lower() == '.true.':
                    self.set_parameter('read_force_constants', True)

            if conf_key == 'write_force_constants':
                if confs['write_force_constants'].lower() == '.true.':
                    self.set_parameter('write_force_constants', True)

            if conf_key == 'full_force_constants':
                if confs['full_force_constants'].lower() == '.true.':
                    self.set_parameter('is_full_fc', True)

            if conf_key == 'cutoff_radius':
                val = float(confs['cutoff_radius'])
                self.set_parameter('cutoff_radius', val)

            if conf_key == 'writedm':
                if confs['writedm'].lower() == '.true.':
                    self.set_parameter('write_dynamical_matrices', True)

            if conf_key == 'write_mesh':
                if confs['write_mesh'].lower() == '.false.':
                    self.set_parameter('write_mesh', False)

            if conf_key == 'hdf5':
                if confs['hdf5'].lower() == '.true.':
                    self.set_parameter('hdf5', True)

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
                if confs['time_reversal_symmetry'].lower() == '.false.':
                    self.set_parameter('is_time_reversal_symmetry', False)

            if conf_key == 'gamma_center':
                if confs['gamma_center'].lower() == '.true.':
                    self.set_parameter('is_gamma_center', True)

            if conf_key == 'fc_spg_symmetry':
                if confs['fc_spg_symmetry'].lower() == '.true.':
                    self.set_parameter('fc_spg_symmetry', True)

            if conf_key == 'readfc_format':
                self.set_parameter('readfc_format', confs['readfc_format'].lower())

            if conf_key == 'writefc_format':
                self.set_parameter('writefc_format', confs['writefc_format'].lower())

            if conf_key == 'fc_format':
                self.set_parameter('readfc_format', confs['fc_format'].lower())
                self.set_parameter('writefc_format', confs['fc_format'].lower())

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

            if conf_key == 'little_cogroup':
                if confs['little_cogroup'].lower() == '.true.':
                    self.set_parameter('little_cogroup', True)

            # DOS
            if conf_key == 'pdos':
                vals = []
                for index_set in confs['pdos'].split(','):
                    vals.append([int(x) - 1 for x in index_set.split()])
                self.set_parameter('pdos', vals)

            if conf_key == 'xyz_projection':
                if confs['xyz_projection'].lower() == '.true.':
                    self.set_parameter('xyz_projection', True)

            if conf_key == 'dos':
                if confs['dos'].lower() == '.true.':
                    self.set_parameter('dos', True)

            if conf_key == 'debye_model':
                if confs['debye_model'].lower() == '.true.':
                    self.set_parameter('fits_debye_model', True)

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

            # Projected thermal properties
            if conf_key == 'ptprop':
                if confs['ptprop'].lower() == '.true.':
                    self.set_parameter('ptprop', True)

            # Use imaginary frequency as real for thermal property calculation
            if conf_key == 'pretend_real':
                if confs['pretend_real'].lower() == '.true.':
                    self.set_parameter('pretend_real', True)

            # Thermal displacement
            if conf_key == 'tdisp':
                if confs['tdisp'].lower() == '.true.':
                    self.set_parameter('tdisp', True)

            # Thermal displacement matrices
            if conf_key == 'tdispmat':
                if confs['tdispmat'].lower() == '.true.':
                    self.set_parameter('tdispmat', True)

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

            # Moment of phonon states distribution
            if conf_key == 'moment':
                if confs['moment'].lower() == '.true.':
                    self.set_parameter('moment', True)

            if conf_key == 'moment_order':
                self.set_parameter('moment_order', int(confs['moment_order']))

            # Use Lapack solver via Lapacke
            if conf_key == 'lapack_solver':
                if confs['lapack_solver'].lower() == '.true.':
                    self.set_parameter('lapack_solver', True)

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

        # Is getting least displacements?
        if 'create_displacements' in params:
            if params['create_displacements']:
                self._settings.set_run_mode('displacements')

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

        # band & mesh mode
        # This has to come before 'mesh_numbers' and 'band_paths'
        if 'mesh_numbers' in params and 'band_paths' in params:
            self._settings.set_run_mode('band_mesh')

        # Mesh
        if 'mesh_numbers' in params:
            if self._settings.get_run_mode() != 'band_mesh':
                self._settings.set_run_mode('mesh')
            self._settings.set_mesh_numbers(params['mesh_numbers'])
        if (self._settings.get_run_mode() == 'mesh' or
            self._settings.get_run_mode() == 'band_mesh'):
            if 'mp_shift' in params:
                shift = params['mp_shift']
            else:
                shift = [0.,0.,0.]
            self._settings.set_mesh_shift(shift)
            if 'is_time_reversal_symmetry' in params:
                if not params['is_time_reversal_symmetry']:
                    self._settings.set_time_reversal_symmetry(False)
            if 'is_mesh_symmetry' in params:
                if not params['is_mesh_symmetry']:
                    self._settings.set_is_mesh_symmetry(False)
            if 'is_gamma_center' in params:
                if params['is_gamma_center']:
                    self._settings.set_is_gamma_center(True)
            if 'mesh_format' in params:
                self._settings.set_mesh_format(params['mesh_format'])

        # band mode
        if 'band_paths' in params:
            if self._settings.get_run_mode() != 'band_mesh':
                self._settings.set_run_mode('band')
        if (self._settings.get_run_mode() == 'band' or
            self._settings.get_run_mode() == 'band_mesh'):
            if 'band_format' in params:
                self._settings.set_band_format(params['band_format'])
            if 'band_labels' in params:
                self._settings.set_band_labels(params['band_labels'])
            if 'band_connection' in params:
                self._settings.set_is_band_connection(params['band_connection'])

        # Q-points mode
        if 'qpoints' in params or 'read_qpoints' in params:
            self._settings.set_run_mode('qpoints')
        if self._settings.get_run_mode() == 'qpoints':
            if 'qpoints_format' in params:
                self._settings.set_qpoints_format(params['qpoints_format'])

        # Whether write out dynamical matrices or not
        if 'write_dynamical_matrices' in params:
            if params['write_dynamical_matrices']:
                self._settings.set_write_dynamical_matrices(True)

        # Whether write out mesh.yaml or mesh.hdf5
        if 'write_mesh' in params:
            self._settings.set_write_mesh(params['write_mesh'])

        # Anime mode
        if 'anime_type' in params:
            self._settings.set_anime_type(params['anime_type'])

        if 'anime' in params:
            self._settings.set_run_mode('anime')
            anime_type = self._settings.get_anime_type()
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
            if self._settings.get_run_mode() == 'irreps':
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
                self._settings.get_pdos_indices() is None):
                self.set_parameter('pdos', [])

        if 'pdos' in params:
            self._settings.set_pdos_indices(params['pdos'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_dos_mode(True)
            self._settings.set_is_mesh_symmetry(False)

        if ('projection_direction' in params and
            not self._settings.get_xyz_projection()):
            self._settings.set_projection_direction(
                params['projection_direction'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_dos_mode(True)
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

        if self._settings.get_is_moment():
            if 'moment_order' in params:
                self._settings.set_moment_order(params['moment_order'])

        # Use Lapack solver via Lapacke
        if 'lapack_solver' in params:
            self._settings.set_lapack_solver(params['lapack_solver'])
