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

class Settings:
    def __init__(self):
        self._chemical_symbols = None
        self._is_eigenvectors = False
        self._is_diagonal_displacement = True
        self._is_plusminus_displacement = 'auto'
        self._is_trigonal_displacement = False
        self._is_tensor_symmetry = False
        self._is_translational_invariance = False
        self._is_rotational_invariance = False
        self._is_nac = False
        self._is_symmetry = True
        self._is_mesh_symmetry = True
        self._fc_decimals = None
        self._fc_symmetry_iteration = 0
        self._masses = None
        self._magmoms = None
        self._mesh = None
        self._omega_step = None
        self._primitive_matrix = np.eye(3, dtype=float)
        self._run_mode = None
        self._sigma = None
        self._supercell_matrix = None
        self._is_time_symmetry = True
        self._tmax = 1000
        self._tmin = 0
        self._tstep = 10

    def set_run_mode(self, run_mode):
        self._run_mode = run_mode

    def get_run_mode(self):
        return self._run_mode

    def set_supercell_matrix(self, matrix):
        self._supercell_matrix = matrix

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def set_is_diagonal_displacement(self, is_diag):
        self._is_diagonal_displacement = is_diag

    def get_is_diagonal_displacement(self):
        return self._is_diagonal_displacement

    def set_is_plusminus_displacement(self, is_pm):
        self._is_plusminus_displacement = is_pm

    def get_is_plusminus_displacement(self):
        return self._is_plusminus_displacement

    def set_is_trigonal_displacement(self, is_trigonal):
        self._is_trigonal_displacement = is_trigonal

    def get_is_trigonal_displacement(self):
        return self._is_trigonal_displacement

    def set_is_nac(self, is_nac):
        self._is_nac = is_nac

    def get_is_nac(self):
        return self._is_nac

    def set_masses(self, masses):
        self._masses = masses

    def get_masses(self):
        return self._masses

    def set_magnetic_moments(self, magmoms):
        self._magmoms = magmoms

    def get_magnetic_moments(self):
        return self._magmoms

    def set_chemical_symbols(self, symbols):
        self._chemical_symbols = symbols

    def get_chemical_symbols(self):
        return self._chemical_symbols

    def set_mesh_numbers(self, mesh):
        self._mesh = mesh

    def get_mesh_numbers(self):
        return self._mesh

    def set_mesh_shift(self, mesh_shift):
        self._mesh_shift = mesh_shift

    def get_mesh_shift(self):
        return self._mesh_shift

    def set_mesh_symmetry(self, mesh_symmetry=True):
        self._is_mesh_symmetry = mesh_symmetry

    def get_mesh_symmetry(self):
        return self._is_mesh_symmetry

    def set_time_symmetry(self, time_symmetry=True):
        self._is_time_symmetry = time_symmetry

    def get_time_symmetry(self):
        return self._is_time_symmetry

    def set_primitive_matrix(self, primitive_matrix):
        self._primitive_matrix = primitive_matrix

    def get_primitive_matrix(self):
        return self._primitive_matrix
        
    def set_is_eigenvectors(self, is_eigenvectors):
        self._is_eigenvectors = is_eigenvectors

    def get_is_eigenvectors(self):
        return self._is_eigenvectors

    def set_is_tensor_symmetry(self, is_tensor_symmetry):
        self._is_tensor_symmetry = is_tensor_symmetry

    def get_is_tensor_symmetry(self):
        return self._is_tensor_symmetry

    def set_is_translational_invariance(self, is_translational_invariance):
        self._is_translational_invariance = is_translational_invariance

    def get_is_translational_invariance(self):
        return self._is_translational_invariance

    def set_is_rotational_invariance(self, is_rotational_invariance):
        self._is_rotational_invariance = is_rotational_invariance

    def get_is_rotational_invariance(self):
        return self._is_rotational_invariance

    def set_fc_symmetry_iteration(self, iteration):
        self._fc_symmetry_iteration = iteration

    def get_fc_symmetry_iteration(self):
        return self._fc_symmetry_iteration

    def set_fc_decimals(self, decimals):
        self._fc_decimals = decimals

    def get_fc_decimals(self):
        return self._fc_decimals

    def set_is_symmetry(self, is_symmetry):
        self._is_symmetry = is_symmetry

    def get_is_symmetry(self):
        return self._is_symmetry

    def set_is_mesh_symmetry(self, is_mesh_symmetry):
        self._is_mesh_symmetry = is_mesh_symmetry

    def get_is_mesh_symmetry(self):
        return self._is_mesh_symmetry

    def set_sigma(self, sigma):
        self._sigma = sigma

    def get_sigma(self):
        return self._sigma

    def set_omega_step(self, omega_step):
        self._omega_step = omega_step

    def get_omega_step(self):
        return self._omega_step

    def set_max_temperature(self, tmax):
        self._tmax = tmax

    def get_max_temperature(self):
        return self._tmax

    def set_min_temperature(self, tmin):
        self._tmin = tmin

    def get_min_temperature(self):
        return self._tmin

    def set_temperature_step(self, tstep):
        self._tstep = tstep

    def get_temperature_step(self):
        return self._tstep



# Parse phonopy setting filen
class ConfParser:
    def __init__(self, filename=None, options=None, option_list=None):
        self._confs = {}
        self._parameters = {}
        self._options = options
        self._option_list = option_list

        if not filename==None:
            self.read_file(filename) # store data in self._confs
        if (not options==None) and (not option_list==None):
            self.read_options() # store data in self._confs
        self.parse_conf() # self.parameters[key] = val

    def get_settings(self):
        return self._settings

    def setting_error(self, message):
        print message
        print "Please check the setting tags and options."
        sys.exit(1)

    def set_settings(self):
        params = self._parameters

        # Supercell size
        if params.has_key('supercell_matrix'):
            self._settings.set_supercell_matrix(params['supercell_matrix'])

        # Atomic mass
        if params.has_key('mass'):
            self._settings.set_masses(params['mass'])
    
        # Magnetic moments
        if params.has_key('magmom'):
            self._settings.set_magnetic_moments(params['magmom'])
    
        # Chemical symbols
        if params.has_key('atom_name'):
            self._settings.set_chemical_symbols(params['atom_name'])
            
        # Diagonal displacement
        if params.has_key('diag'):
            self._settings.set_is_diagonal_displacement(params['diag'])
    
        # Plus minus displacement
        if params.has_key('pm_displacement'):
            self._settings.set_is_plusminus_displacement(params['pm_displacement'])
    
        # Trigonal displacement
        if params.has_key('is_trigonal_displacement'):
            self._settings.set_is_trigonal_displacement(params['is_trigonal_displacement'])
    
        # Primitive cell shape
        if params.has_key('primitive_axis'):
            self._settings.set_primitive_matrix(params['primitive_axis'])
    
        # Is getting eigenvectors?
        if params.has_key('is_eigenvectors'):
            self._settings.set_is_eigenvectors(params['is_eigenvectors'])
    
        # Non analytical term correction?
        if params.has_key('is_nac'):
            self._settings.set_is_nac(params['is_nac'])
    
        # Is crystal symmetry searched?
        if params.has_key('is_symmetry'):
            self._settings.set_is_symmetry(params['is_symmetry'])
    
        # Is reciprocal mesh symmetry searched?
        if params.has_key('is_mesh_symmetry'):
            self._settings.set_is_mesh_symmetry(params['is_mesh_symmetry'])
    
        # Is translational invariance ?
        if params.has_key('is_translation'):
            self._settings.set_is_translational_invariance(params['is_translation'])
    
        # Is rotational invariance ?
        if params.has_key('is_rotational'):
            self._settings.set_is_rotational_invariance(params['is_rotational'])
    
        # Enforce force constant symmetry?
        if params.has_key('fc_symmetry'):
            self._settings.set_fc_symmetry_iteration(int(params['fc_symmetry']))
    
        # Decimals of values of force constants
        if params.has_key('fc_decimals'):
            self._settings.set_fc_decimals(int(params['fc_decimals']))
    
        # Is force constants symmetry forced?
        if params.has_key('is_tensor_symmetry'):
            self._settings.set_is_tensor_symmetry(params['is_tensor_symmetry'])

        # Mesh sampling numbers
        if params.has_key('mesh_numbers'):
            self._settings.set_mesh_numbers(params['mesh_numbers'])

        # Spectram drawing step
        if params.has_key('omega_step'):
            self._settings.set_omega_step(params['omega_step'])

        # Smearing width
        if params.has_key('sigma'):
            self._settings.set_sigma(params['sigma'])

        # Temerature range
        if params.has_key('tmax'):
            self._settings.set_max_temperature(params['tmax'])
        if params.has_key('tmin'):
            self._settings.set_min_temperature(params['tmin'])
        if params.has_key('tstep'):
            self._settings.set_temperature_step(params['tstep'])

    def read_file(self, filename):
        file = open(filename, 'r')
        confs = self._confs
        is_continue = False
        for line in file:
            if line.strip() == '':
                is_continue = False
                continue
            
            if line.strip()[0] == '#':
                is_continue = False
                continue

            if is_continue:
                confs[left] += line.strip()
                confs[left] = confs[left].replace('+++', ' ')
                is_continue = False
                
            if line.find('=') != -1:
                left, right = [x.strip().lower() for x in line.split('=')]
                if left == 'band_labels':
                    right = [x.strip() for x in line.split('=')][1]
                confs[left] = right

            if line.find('+++') != -1:
                is_continue = True

    def read_options(self):
        for opt in self._option_list:

            if opt.dest=='is_plusminus_displacements':
                if self._options.is_plusminus_displacements:
                    self._confs['pm'] = '.true.'

            if opt.dest=='is_trigonal_displacements':
                if self._options.is_trigonal_displacements:
                    self._confs['trigonal'] = '.true.'

            if opt.dest=='mesh_numbers':
                if self._options.mesh_numbers:
                    self._confs['mesh_numbers'] = self._options.mesh_numbers

            if opt.dest=='primitive_axis':
                if self._options.primitive_axis:
                    self._confs['primitive_axis'] = self._options.primitive_axis
                    
            if opt.dest=='supercell_dimension':
                if self._options.supercell_dimension:
                    self._confs['dim'] = self._options.supercell_dimension

            if opt.dest=='is_nodiag':
                if self._options.is_nodiag:
                    self._confs['diag'] = '.false.'

            if opt.dest=='is_eigenvectors':
                if self._options.is_eigenvectors:
                    self._confs['eigenvectors'] = '.true.'
                    
            if opt.dest=='is_nac':
                if self._options.is_nac:
                    self._confs['nac'] = '.true.'

            if opt.dest=='is_nosym':
                if self._options.is_nosym:
                    self._confs['symmetry'] = '.false.'
                    self._confs['mesh_symmetry'] = '.false.'

            if opt.dest=='is_nomeshsym':
                if self._options.is_nomeshsym:
                    self._confs['mesh_symmetry'] = '.false.'

            if opt.dest=='omega_step':
                if self._options.omega_step:
                    self._confs['omega_step'] = self._options.omega_step

            if opt.dest=='sigma':
                if self._options.sigma:
                    self._confs['sigma'] = self._options.sigma

            if opt.dest=='tmin':
                if self._options.tmin:
                    self._confs['tmin'] = self._options.tmin

            if opt.dest=='tmax':
                if self._options.tmax:
                    self._confs['tmax'] = self._options.tmax
                    
            if opt.dest=='tstep':
                if self._options.tstep:
                    self._confs['tstep'] = self._options.tstep

            if opt.dest=='force_constants_decimals':
                if self._options.force_constants_decimals:
                    self._confs['fc_decimals'] = \
                        self._options.force_constants_decimals

    def parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim':
                matrix = [ int(x) for x in confs['dim'].split() ]
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
                            'Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('supercell_matrix', matrix)

            if conf_key == 'primitive_axis':
                if not len(confs['primitive_axis'].split()) == 9:
                    self.setting_error(
                        "Number of elements in PRIMITIVE_AXIS has to be 9.")
                p_axis = []
                for x in confs['primitive_axis'].split():
                    p_axis.append(fracval(x))
                p_axis = np.array(p_axis).reshape(3,3)
                if np.linalg.det(p_axis) < 1e-8:
                    self.setting_error(
                        "PRIMITIVE_AXIS has to have positive determinant.")
                self.set_parameter('primitive_axis', p_axis)

            if conf_key == 'mass':
                self.set_parameter(
                    'mass',
                    [ float(x) for x in confs['mass'].split()])

            if conf_key == 'magmom':
                self.set_parameter(
                    'magmom',
                    [ float(x) for x in confs['magmom'].split()])

            if conf_key == 'atom_name':
                self.set_parameter(
                    'atom_name',
                    [ x.capitalize() for x in confs['atom_name'].split() ])
                        
            if conf_key == 'diag':
                if confs['diag'] == '.false.':
                    self.set_parameter('diag', False)
                if confs['diag'] == '.true.':
                    self.set_parameter('diag', True)

            if conf_key == 'pm':
                if confs['pm'] == '.false.':
                    self.set_parameter('pm_displacement', False)
                if confs['pm'] == '.true.':
                    self.set_parameter('pm_displacement', True)

            if conf_key == 'trigonal':
                if confs['trigonal'] == '.false.':
                    self.set_parameter('is_trigonal_displacement', False)
                if confs['trigonal'] == '.true.':
                    self.set_parameter('is_trigonal_displacement', True)

            if conf_key == 'eigenvectors':
                if confs['eigenvectors'] == '.true.':
                    self.set_parameter('is_eigenvectors', True)

            if conf_key == 'nac':
                if confs['nac'] == '.true.':
                    self.set_parameter('is_nac', True)

            if conf_key == 'symmetry':
                if confs['symmetry'] == '.false.':
                    self.set_parameter('is_symmetry', False)

            if conf_key == 'mesh_symmetry':
                if confs['mesh_symmetry'] == '.false.':
                    self.set_parameter('is_mesh_symmetry', False)
                
            if conf_key == 'translation':
                if confs['translation'] == '.true.':
                    self.set_parameter('is_translation', True)

            if conf_key == 'rotational':
                if confs['rotational'] == '.true.':
                    self.set_parameter('is_rotational', True)

            if conf_key == 'fc_symmetry':
                self.set_parameter('fc_symmetry', confs['fc_symmetry'])

            if conf_key == 'fc_decimals':
                self.set_parameter('fc_decimals', confs['fc_decimals'])

            if conf_key == 'tensor_symmetry':
                if confs['tensor_symmetry'] == '.true.':
                    self.set_parameter('is_tensor_symmetry', True)

            if conf_key == 'mesh_numbers':
                vals = [ int(x) for x in confs['mesh_numbers'].split() ]
                if len(vals) < 3:
                    self.setting_error("Mesh numbers are incorrectly set.")
                self.set_parameter('mesh_numbers', vals[:3])

            if conf_key == 'omega_step':
                if isinstance(confs['omega_step'], str):
                    val = float(confs['omega_step'].split()[0])
                else:
                    val = confs['omega_step']
                self.set_parameter('omega_step', val)

            if conf_key == 'sigma':
                if isinstance(confs['sigma'], str):
                    val = float(confs['sigma'].split()[0])
                else:
                    val = confs['sigma']
                self.set_parameter('sigma', val)

            if conf_key == 'tmin':
                val = float(confs['tmin'].split()[0])
                self.set_parameter('tmin', val)

            if conf_key == 'tmax':
                val = float(confs['tmax'].split()[0])
                self.set_parameter('tmax', val)

            if conf_key == 'tstep':
                val = float(confs['tstep'].split()[0])
                self.set_parameter('tstep', val)

    def set_parameter(self, key, val):
        self._parameters[key] = val



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
        self._bands = None
        self._band_labels = None
        self._band_connection = False
        self._cutoff_radius = None
        self._dos = None
        self._dos_range = { 'min':  None,
                            'max':  None }
        self._fits_Debye_model = False
        self._thermal_atom_pairs = None
        self._is_dos_mode = False
        self._is_force_constants = False
        self._is_gamma_center = False
        self._is_plusminus_displacement = 'auto'
        self._is_thermal_displacements = False
        self._is_thermal_distances = False
        self._is_thermal_properties = False
        self._write_dynamical_matrices = False
        self._qpoints = None
        self._modulation = None
        self._pdos_indices = None
        self._projection_direction = None
        self._character_table_q_point = None
        self._character_table_tolerance = 1e-5
        self._character_table_show_irreps = False

    def set_run_mode(self, run_mode):
        modes = ['qpoints',
                 'mesh',
                 'band',
                 'anime',
                 'modulation',
                 'displacements',
                 'character_table']
        for mode in modes:
            if run_mode.lower() == mode:
                self._run_mode = run_mode

    def get_run_mode(self):
        return self._run_mode

    def set_is_force_constants(self, is_force_constants):
        self._is_force_constants = is_force_constants

    def get_is_force_constants(self):
        return self._is_force_constants

    def set_bands(self, bands):
        self._bands = bands

    def get_bands(self):
        return self._bands

    def set_band_labels(self, labels):
        self._band_labels = labels

    def get_band_labels(self):
        return self._band_labels

    def set_is_band_connection(self, band_connection):
        self._band_connection = band_connection

    def get_is_band_connection(self):
        return self._band_connection

    def set_cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = cutoff_radius

    def get_cutoff_radius(self):
        return self._cutoff_radius

    def set_mesh(self,
                 mesh,
                 mesh_shift=[0.,0.,0.],
                 is_time_symmetry=True,
                 is_mesh_symmetry=True,
                 is_gamma_center=False):
        self._mesh = mesh
        self._mesh_shift = mesh_shift
        self._is_time_symmetry = is_time_symmetry
        self._is_mesh_symmetry = is_mesh_symmetry
        self._is_gamma_center = is_gamma_center

    def get_mesh(self):
        return (self._mesh,
                self._mesh_shift,
                self._is_time_symmetry,
                self._is_mesh_symmetry,
                self._is_gamma_center)

    def set_is_gamma_center(self, is_gamma_center):
        self._is_gamma_center = is_gamma_center

    def get_is_gamma_center(self):
        return self._is_gamma_center

    def set_is_dos_mode(self, is_dos_mode):
        self._is_dos_mode = is_dos_mode

    def get_is_dos_mode(self):
        return self._is_dos_mode

    def set_dos_range(self, dos_min, dos_max, dos_step):
        self._dos_range = {'min':  dos_min,
                           'max':  dos_max}
        self._omega_step = dos_step

    def get_dos_range(self):
        dos_range = {'min': self._dos_range['min'],
                     'max': self._dos_range['max'],
                     'step': self._omega_step}
        return dos_range

    def set_fits_Debye_model(self, fits_Debye_model):
        self._fits_Debye_model = fits_Debye_model

    def get_fits_Debye_model(self):
        return self._fits_Debye_model

    def set_pdos_indices(self, indices):
        self._pdos_indices = indices

    def get_pdos_indices(self):
        return self._pdos_indices

    def set_is_thermal_properties(self, is_thermal_properties):
        self._is_thermal_properties = is_thermal_properties

    def get_is_thermal_properties(self):
        return self._is_thermal_properties

    def set_thermal_property_range(self, tmin, tmax, tstep):
        self._tmax = tmax
        self._tmin = tmin
        self._tstep = tstep

    def get_thermal_property_range(self):
        return {'min':  self._tmin,
                'max':  self._tmax,
                'step': self._tstep}

    def set_is_thermal_displacements(self, is_thermal_displacements):
        self._is_thermal_displacements = is_thermal_displacements

    def get_is_thermal_displacements(self):
        return self._is_thermal_displacements

    def set_is_thermal_distances(self, is_thermal_distances):
        self._is_thermal_distances = is_thermal_distances

    def get_is_thermal_distances(self):
        return self._is_thermal_distances

    def set_write_dynamical_matrices(self, write_dynamical_matrices):
        self._write_dynamical_matrices = write_dynamical_matrices

    def get_write_dynamical_matrices(self):
        return self._write_dynamical_matrices

    def set_projection_direction(self, direction):
        self._projection_direction = direction

    def get_projection_direction(self):
        return self._projection_direction

    def set_thermal_atom_pairs(self, atom_pairs):
        self._thermal_atom_pairs = atom_pairs

    def get_thermal_atom_pairs(self):
        return self._thermal_atom_pairs

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

    def set_modulation(self, modulation):
        self._modulation = modulation

    def get_modulation(self):
        return self._modulation

    def set_qpoints(self, qpoints):
        self._qpoints = qpoints

    def get_qpoints(self):
        return self._qpoints

    def set_character_table_q_point(self, q_point):
        self._character_table_q_point = q_point
        
    def get_character_table_q_point(self):
        return self._character_table_q_point

    def set_character_table_tolerance(self, tolerance):
        self._character_table_tolerance = tolerance
        
    def get_character_table_tolerance(self):
        return self._character_table_tolerance

    def set_character_table_show_irreps(self, show_irreps):
        self._character_table_show_irreps = show_irreps
        
    def get_character_table_show_irreps(self):
        return self._character_table_show_irreps
        
class PhonopyConfParser(ConfParser):
    def __init__(self, filename=None, options=None, option_list=None):
        ConfParser.__init__(self, filename, options, option_list)
        self._read_options()
        self._parse_conf()
        self._settings = PhonopySettings()
        self._set_settings()

    def _read_options(self):
        for opt in self._option_list:
            if opt.dest == 'is_displacement':
                if self._options.is_displacement:
                    self._confs['create_displacements'] = '.true.'

            if opt.dest == 'is_gamma_center':
                if self._options.is_gamma_center:
                    self._confs['gamma_center'] = '.true.'
    
            if opt.dest == 'is_dos_mode':
                if self._options.is_dos_mode:
                    self._confs['dos'] = '.true.'

            if opt.dest == 'fits_debye_model':
                if self._options.fits_debye_model:
                    self._confs['debye_model'] = '.true.'

            if opt.dest == 'is_thermal_properties':
                if self._options.is_thermal_properties:
                    self._confs['tprop'] = '.true.'

            if opt.dest == 'is_thermal_displacements':
                if self._options.is_thermal_displacements:
                    self._confs['tdisp'] = '.true.'
                    
            if opt.dest == 'projection_direction':
                if self._options.projection_direction is not None:
                    self._confs['projection_direction'] = self._options.projection_direction

            if opt.dest == 'is_read_force_constants':
                if self._options.is_read_force_constants:
                    self._confs['force_constants'] = 'read'
    
            if opt.dest == 'write_force_constants':
                if self._options.write_force_constants:
                    self._confs['force_constants'] = 'write'
    
            if opt.dest == 'write_dynamical_matrices':
                if self._options.write_dynamical_matrices:
                    self._confs['writedm'] = '.true.'
    
            if opt.dest == 'q_character_table':
                if not self._options.q_character_table == None:
                    self._confs['character_table'] = self._options.q_character_table

            if opt.dest == 'show_irreps':
                if self._options.show_irreps:
                    self._confs['irreps'] = '.true.'

            if opt.dest == 'qpoints':
                if not self._options.qpoints == None:
                    self._confs['qpoints'] = self._options.qpoints

            if opt.dest == 'band_paths':
                if not self._options.band_paths == None:
                    self._confs['band'] = self._options.band_paths

            if opt.dest == 'band_points':
                if not self._options.band_points == None:
                    self._confs['band_points'] = self._options.band_points

            if opt.dest == 'is_band_connection':
                if self._options.is_band_connection:
                    self._confs['band_connection'] = '.true.'

            if opt.dest == 'cutoff_radius':
                if self._options.cutoff_radius:
                    self._confs['cutoff_radius'] = self._options.cutoff_radius

            if opt.dest == 'modulation':
                if self._options.modulation:
                    self._confs['modulation'] = self._options.modulation

            if opt.dest == 'anime':
                if self._options.anime:
                    self._confs['anime'] = self._options.anime
    
            # Overwrite
            if opt.dest == 'is_check_symmetry':
                if self._options.is_check_symmetry: # Dummy 'dim' setting for sym-check
                    self._confs['dim'] = '1 1 1'

    def _parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'create_displacements':
                if confs['create_displacements'] == '.true.':
                    self.set_parameter('create_displacements', True)

            if conf_key == 'band_points':
                self.set_parameter('band_points',
                                   int(confs['band_points']))

            if conf_key == 'band_labels':
                labels = [x for x in confs['band_labels'].split()]
                self.set_parameter('band_labels', labels)

            if conf_key == 'band':
                bands = []
                for section in confs['band'].split(','):
                    points = [fracval(x) for x in section.split()]
                    if len(points) % 3 != 0:
                        self.setting_error("BAND is incorrectly set.")
                        break
                    bands.append(np.array(points).reshape(-1, 3))
                self.set_parameter('band', bands)

            if conf_key == 'band_connection':
                if confs['band_connection'] == '.true.':
                    self.set_parameter('band_connection', True)

            if conf_key == 'force_constants':
                self.set_parameter('force_constants',
                                   confs['force_constants'])

            if conf_key == 'cutoff_radius':
                if isinstance(confs['cutoff_radius'], str):
                    val = float(confs['cutoff_radius'].split()[0])
                else:
                    val = confs['cutoff_radius']
                self.set_parameter('cutoff_radius', val)

            if conf_key == 'qpoints':
                if confs['qpoints'] == '.true.':
                    self.set_parameter('qpoints', True)
                else:
                    vals = [fracval(x) for x in confs['qpoints'].split()]
                    if len(vals) == 0 or len(vals) % 3 != 0:
                        self.setting_error("Q-points are incorrectly set.")
                    else:
                        self.set_parameter('qpoints',
                                           list(np.reshape(vals, (-1, 3))))

            if conf_key == 'writedm':
                if confs['writedm'] == '.true.':
                    self.set_parameter('write_dynamical_matrices', True)

            if conf_key == 'mp':
                vals = [ int(x) for x in confs['mp'].split() ]
                if len(vals) < 3:
                    self.setting_error("Mesh numbers are incorrectly set.")
                self.set_parameter('mesh_numbers', vals[:3])

            if conf_key == 'mp_shift':
                vals = [ fracval(x) for x in confs['mp_shift'].split()]
                if len(vals) < 3:
                    self.setting_error("MP_SHIFT is incorrectly set.")
                self.set_parameter('mp_shift', vals[:3])
                
            if conf_key == 'time_symmetry':
                if confs['time_symmetry'] == '.false.':
                    self.set_parameter('is_time_symmetry', False)

            if conf_key == 'gamma_center':
                if confs['gamma_center'] == '.true.':
                    self.set_parameter('is_gamma_center', True)

            # Animation
            if conf_key == 'anime':
                vals = []
                data = confs['anime'].split()
                if len(data) < 3:
                    self.setting_error("ANIME is incorrectly set.")
                else:
                    self.set_parameter('anime', data)

            if conf_key == 'anime_type':
                if (confs['anime_type'] == 'arc' or
                     confs['anime_type'] == 'v_sim' or
                     confs['anime_type'] == 'poscar' or
                     confs['anime_type'] == 'xyz' or
                     confs['anime_type'] == 'jmol'):
                    self.set_parameter('anime_type', confs['anime_type'])
                else:
                    self.setting_error("%s is not available for ANIME_TYPE tag." % confs['anime_type'])

            # Modulation
            if conf_key == 'modulation':
                self._parse_conf_modulation(confs)

            # Character table
            if conf_key == 'character_table':
                vals = [ fracval(x) for x in confs['character_table'].split()]
                if len(vals) == 3 or len(vals) == 4:
                    self.set_parameter('character_table', vals)
                else:
                    self.setting_error("CHARACTER_TABLE is incorrectly set.")

            if conf_key == 'irreps':
                if confs['irreps'] == '.true.':
                    self.set_parameter('irreps', True)

            # DOS
            if conf_key == 'pdos':
                vals = []
                for sum_set in confs['pdos'].split(','):
                    indices = [ int(x) - 1 for x in sum_set.split() ]
                    vals.append(indices)
                self.set_parameter('pdos', vals)

            if conf_key == 'dos':
                self.set_parameter('dos', confs['dos'])

            if conf_key == 'debye_model':
                self.set_parameter('fits_debye_model', confs['debye_model'])

            if conf_key == 'dos_range':
                vals = [ float(x) for x in confs['dos_range'].split() ]
                self.set_parameter('dos_range', vals)

            # Thermal properties
            if conf_key == 'tprop':
                self.set_parameter('tprop', confs['tprop'])

            # Thermal displacement
            if conf_key == 'tdisp':
                self.set_parameter('tdisp', confs['tdisp'])

            # Thermal distance
            if conf_key == 'tdistance':
                atom_pairs = []
                for atoms in confs['tdistance'].split(','):
                    pair = [ int(x)-1 for x in atoms.split() ]
                    if len(pair) == 2:
                        atom_pairs.append(pair)
                    else:
                        self.setting_error("TDISTANCE is incorrectly specified.")
                if len(atom_pairs) > 0:
                    self.set_parameter('tdistance', atom_pairs)
            
            # Projection direction used for thermal displacements
            if conf_key == 'projection_direction':
                vals = [float(x) for x in confs['projection_direction'].split()]
                if len(vals) < 3:
                    self.setting_error("PROJECTION_DIRECTION (--pd) is incorrectly specified.")
                else:
                    self.set_parameter('projection_direction', vals)


    def _parse_conf_modulation(self, confs):
        modulation = {}
        modulation['dimension'] = [1, 1, 1]
        mod_list = confs['modulation'].split(',')
        header = mod_list[0].split()
        if len(header) == 3 and len(mod_list) > 1:
            dimension = [int(x) for x in header]
            modulation['dimension'] = dimension
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
        ConfParser.set_settings(self)
        params = self._parameters
    
        # Is getting least displacements?
        if params.has_key('create_displacements'):
            if params['create_displacements']:
                self._settings.set_run_mode('displacements')
    
        # Is force constants written or read ?
        if params.has_key('force_constants'):
            if params['force_constants'] == 'write':
                self._settings.set_is_force_constants("write")
            elif params['force_constants'] == 'read':
                self._settings.set_is_force_constants("read")

        # Cutoff radius of force constants
        if params.has_key('cutoff_radius'):
            self._settings.set_cutoff_radius(params['cutoff_radius'])
    
        # Mesh
        if params.has_key('mesh_numbers'):
            self._settings.set_run_mode('mesh')
            if params.has_key('mp_shift'):
                shift = params['mp_shift']
            else:
                shift = [0.,0.,0.]
    
            time_symmetry = True
            if params.has_key('is_time_symmetry'):
                if not params['is_time_symmetry']:
                    time_symmetry = False
    
            mesh_symmetry = True
            if params.has_key('is_mesh_symmetry'):
                if not params['is_mesh_symmetry']:
                    mesh_symmetry = False

            gamma_center = False
            if params.has_key('is_gamma_center'):
                if params['is_gamma_center']:
                    gamma_center = True
    
            self._settings.set_mesh(params['mesh_numbers'],
                                    mesh_shift=shift,
                                    is_time_symmetry=time_symmetry,
                                    is_mesh_symmetry=mesh_symmetry,
                                    is_gamma_center=gamma_center)
    
        # band mode
        if params.has_key('band'):
            if params.has_key('band_points'):
                npoints = params['band_points'] - 1
            else:
                npoints = 50
                
            self._settings.set_run_mode('band')
            bands = []
            
            for band_path in params['band']:
                nd = len(band_path)
                for i in range(nd - 1):
                    diff = (band_path[i + 1] - band_path[i]) / npoints
                    band = [band_path[i].copy()]
                    q = np.zeros(3)
                    for j in range(npoints):
                        q += diff
                        band.append(band_path[i] + q)
                    bands.append(band)
            self._settings.set_bands(bands)

        if params.has_key('band_labels'):
            self._settings.set_band_labels(params['band_labels'])

        if params.has_key('band_connection'):
            self._settings.set_is_band_connection(params['band_connection'])
    
        # Q-points mode
        if params.has_key('qpoints'):
            self._settings.set_run_mode('qpoints')
            if params['qpoints'] != True:
                self._settings.set_qpoints(params['qpoints'])

        if params.has_key('write_dynamical_matrices'):
            if params['write_dynamical_matrices']:
                self._settings.set_write_dynamical_matrices(True)
                
        # Anime mode
        if params.has_key('anime_type'):
            self._settings.set_anime_type(params['anime_type'])
    
        if params.has_key('anime'):
            self._settings.set_run_mode('anime')
            anime_type = self._settings.get_anime_type()
            if anime_type=='v_sim':
                qpoints = [ fracval(x) for x in params['anime'][0:3] ]
                self._settings.set_anime_qpoint(qpoints)
                if len(params['anime']) > 3:
                    self._settings.set_anime_amplitude(float(params['anime'][3]))
            else:
                self._settings.set_anime_band_index(int(params['anime'][0]))
                self._settings.set_anime_amplitude(float(params['anime'][1]))
                self._settings.set_anime_division(int(params['anime'][2]))
            if len(params['anime']) == 6:
                self._settings.set_anime_shift(
                    [ fracval(x) for x in params['anime'][3:6] ])
    
        # Modulation mode
        if params.has_key('modulation'):
            self._settings.set_run_mode('modulation')
            self._settings.set_modulation(params['modulation'])
    
        # Character table mode
        if params.has_key('character_table'):
            self._settings.set_run_mode('character_table')
            self._settings.set_character_table_q_point(
                params['character_table'][:3])
            if len(params['character_table']) == 4:
                self._settings.set_character_table_tolerance(
                    params['character_table'][3])

            if params.has_key('irreps'):
                self._settings.set_character_table_show_irreps(params['irreps'])
                
        # DOS
        if params.has_key('dos_range'):
            dos_min =  params['dos_range'][0]
            dos_max =  params['dos_range'][1]
            dos_step = params['dos_range'][2]
            self._settings.set_dos_range(dos_min, dos_max, dos_step)
            self._settings.set_is_dos_mode(True)
    
        if params.has_key('dos'):
            if params['dos'] == '.true.':
                self._settings.set_is_dos_mode(True)

        if params.has_key('fits_debye_model'):
            if params['fits_debye_model'] == '.true.':
                self._settings.set_fits_Debye_model(True)
    
        if params.has_key('pdos'):
            self._settings.set_pdos_indices(params['pdos'])
            self._settings.set_is_eigenvectors(True)
            self._settings.set_is_dos_mode(True)
    
        # Thermal properties
        if params.has_key('tprop'):
            if params['tprop'] == '.true.':
                self._settings.set_is_thermal_properties(True)
    
        # Thermal displacement
        if params.has_key('tdisp'):
            if params['tdisp'] == '.true.':
                self._settings.set_is_thermal_displacements(True)
                self._settings.set_is_eigenvectors(True)
                self._settings.set_mesh_symmetry(False)
    
        # Thermal distance
        if params.has_key('tdistance'): 
            self._settings.set_is_thermal_distances(True)
            self._settings.set_is_eigenvectors(True)
            self._settings.set_mesh_symmetry(False)
            self._settings.set_thermal_atom_pairs(params['tdistance'])
    
        # Projection direction (currently only used for thermal displacements
        if params.has_key('projection_direction'): 
            self._settings.set_projection_direction(
                params['projection_direction'])
            
