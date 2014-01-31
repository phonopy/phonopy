import numpy as np
from phonopy.cui.settings import Settings, ConfParser, fracval

class Phono3pySettings(Settings):
    def __init__(self):
        Settings.__init__(self)

        self._coarse_mesh_shifts = None
        self._cutoff_fc3_distance = None
        self._cutoff_pair_distance = None
        self._cutoff_frequency = 1e-4
        self._cutoff_lifetime = 1e-4 # in second
        self._grid_points = None
        self._ion_clamped = False
        self._is_bterta = False
        self._is_linewidth = False
        self._is_frequency_shift = False
        self._mass_variances = None
        self._max_freepath = None
        self._mesh_divisors = None
        self._no_kappa_stars = False
        self._read_amplitude = False
        self._read_gamma = False
        self._phonon_supercell_matrix = None
        self._temperatures = None
        self._write_amplitude = False
        self._write_gamma = False
        
    def set_coarse_mesh_shifts(self, coarse_mesh_shifts):
        self._coarse_mesh_shifts = coarse_mesh_shifts

    def get_coarse_mesh_shifts(self):
        return self._coarse_mesh_shifts

    def set_cutoff_fc3_distance(self, cutoff_fc3_distance):
        self._cutoff_fc3_distance = cutoff_fc3_distance

    def get_cutoff_fc3_distance(self):
        return self._cutoff_fc3_distance

    def set_cutoff_pair_distance(self, cutoff_pair_distance):
        self._cutoff_pair_distance = cutoff_pair_distance

    def get_cutoff_pair_distance(self):
        return self._cutoff_pair_distance

    def set_cutoff_frequency(self, cutoff_frequency):
        self._cutoff_frequency = cutoff_frequency

    def get_cutoff_frequency(self):
        return self._cutoff_frequency

    def set_cutoff_lifetime(self, cutoff_lifetime):
        self._cutoff_lifetime = cutoff_lifetime

    def get_cutoff_lifetime(self):
        return self._cutoff_lifetime

    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def get_grid_points(self):
        return self._grid_points

    def set_ion_clamped(self, ion_clamped):
        self._ion_clamped = ion_clamped

    def get_ion_clamped(self):
        return self._ion_clamped

    def set_is_bterta(self, is_bterta):
        self._is_bterta = is_bterta

    def get_is_bterta(self):
        return self._is_bterta

    def set_is_linewidth(self, is_linewidth):
        self._is_linewidth = is_linewidth

    def get_is_linewidth(self):
        return self._is_linewidth

    def set_is_frequency_shift(self, is_frequency_shift):
        self._is_frequency_shift = is_frequency_shift

    def get_is_frequency_shift(self):
        return self._is_frequency_shift

    def set_mass_variances(self, mass_variances):
        self._mass_variances = mass_variances

    def get_mass_variances(self):
        return self._mass_variances

    def set_max_freepath(self, max_freepath):
        self._max_freepath = max_freepath

    def get_max_freepath(self):
        return self._max_freepath

    def set_mesh_divisors(self, mesh_divisors):
        self._mesh_divisors = mesh_divisors

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def set_no_kappa_stars(self, no_kappa_stars):
        self._no_kappa_stars = no_kappa_stars

    def get_no_kappa_stars(self):
        return self._no_kappa_stars

    def set_read_gamma(self, read_gamma):
        self._read_gamma = read_gamma

    def get_read_gamma(self):
        return self._read_gamma

    def set_read_amplitude(self, read_amplitude):
        self._read_amplitude = read_amplitude

    def get_read_amplitude(self):
        return self._read_amplitude

    def set_phonon_supercell_matrix(self, matrix):
        self._phonon_supercell_matrix = matrix

    def get_phonon_supercell_matrix(self):
        return self._phonon_supercell_matrix

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures

    def set_write_amplitude(self, write_amplitude):
        self._write_amplitude = write_amplitude

    def get_write_amplitude(self):
        return self._write_amplitude

    def set_write_gamma(self, write_gamma):
        self._write_gamma = write_gamma

    def get_write_gamma(self):
        return self._write_gamma



class Phono3pyConfParser(ConfParser):
    def __init__(self, filename=None, options=None, option_list=None):
        ConfParser.__init__(self, filename, options, option_list)
        self._read_options()
        self._parse_conf()
        self._settings = Phono3pySettings()
        self._set_settings()

    def _read_options(self):
        for opt in self._option_list:
            if opt.dest == 'supercell_dimension_extra':
                if self._options.supercell_dimension_extra is not None:
                    self._confs['dim_extra'] = \
                        self._options.supercell_dimension_extra

            if opt.dest == 'cutoff_fc3_distance':
                if self._options.cutoff_fc3_distance is not None:
                    self._confs['cutoff_fc3_distance'] = \
                        self._options.cutoff_fc3_distance

            if opt.dest == 'cutoff_pair_distance':
                if self._options.cutoff_pair_distance is not None:
                    self._confs['cutoff_pair_distance'] = \
                        self._options.cutoff_pair_distance

            if opt.dest == 'cutoff_frequency':
                if self._options.cutoff_frequency is not None:
                    self._confs['cutoff_frequency'] = \
                        self._options.cutoff_frequency

            if opt.dest == 'cutoff_lifetime':
                if self._options.cutoff_lifetime is not None:
                    self._confs['cutoff_lifetime'] = \
                        self._options.cutoff_lifetime

            if opt.dest == 'grid_points':
                if self._options.grid_points is not None:
                    self._confs['grid_points'] = self._options.grid_points

            if opt.dest == 'ion_clamped':
                if self._options.ion_clamped:
                    self._confs['ion_clamped'] = '.true.'

            if opt.dest == 'is_bterta':
                if self._options.is_bterta:
                    self._confs['bterta'] = '.true.'

            if opt.dest == 'is_linewidth':
                if self._options.is_linewidth:
                    self._confs['linewidth'] = '.true.'

            if opt.dest == 'is_frequency_shift':
                if self._options.is_frequency_shift:
                    self._confs['frequency_shift'] = '.true.'

            if opt.dest == 'mass_variances':
                if self._options.mass_variances is not None:
                    self._confs['mass_variances'] = self._options.mass_variances

            if opt.dest == 'max_freepath':
                if self._options.max_freepath is not None:
                    self._confs['max_freepath'] = self._options.max_freepath

            if opt.dest == 'mesh_divisors':
                if self._options.mesh_divisors is not None:
                    self._confs['mesh_divisors'] = self._options.mesh_divisors

            if opt.dest == 'no_kappa_stars':
                if self._options.no_kappa_stars:
                    self._confs['no_kappa_stars'] = '.true.'

            if opt.dest == 'read_amplitude':
                if self._options.read_amplitude:
                    self._confs['read_amplitude'] = '.true.'

            if opt.dest == 'read_gamma':
                if self._options.read_gamma:
                    self._confs['read_gamma'] = '.true.'

            if opt.dest == 'temperatures':
                if self._options.temperatures is not None:
                    self._confs['temperatures'] = self._options.temperatures

            if opt.dest == 'write_amplitude':
                if self._options.write_amplitude:
                    self._confs['write_amplitude'] = '.true.'

            if opt.dest == 'write_gamma':
                if self._options.write_gamma:
                    self._confs['write_gamma'] = '.true.'

    def _parse_conf(self):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim_extra':
                matrix = [ int(x) for x in confs['dim_extra'].split() ]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            "Determinant of supercell matrix has " +
                            "to be positive.")
                    else:
                        self.set_parameter('dim_extra', matrix)

            if conf_key == 'cutoff_fc3_distance':
                self.set_parameter('cutoff_fc3_distance',
                                   float(confs['cutoff_fc3_distance']))

            if conf_key == 'cutoff_pair_distance':
                self.set_parameter('cutoff_pair_distance',
                                   float(confs['cutoff_pair_distance']))

            if conf_key == 'cutoff_frequency':
                self.set_parameter('cutoff_frequency',
                                   float(confs['cutoff_frequency']))

            if conf_key == 'cutoff_lifetime':
                self.set_parameter('cutoff_lifetime',
                                   float(confs['cutoff_lifetime']))

            if conf_key == 'grid_points':
                vals = [int(x) for x in
                        confs['grid_points'].replace(',', ' ').split()]
                self.set_parameter('grid_points', vals)

            if conf_key == 'ion_clamped':
                if confs['ion_clamped'] == '.true.':
                    self.set_parameter('ion_clamped', True)

            if conf_key == 'bterta':
                if confs['bterta'] == '.true.':
                    self.set_parameter('is_bterta', True)

            if conf_key == 'linewidth':
                if confs['linewidth'] == '.true.':
                    self.set_parameter('is_linewidth', True)

            if conf_key == 'frequency_shift':
                if confs['frequency_shift'] == '.true.':
                    self.set_parameter('is_frequency_shift', True)

            if conf_key == 'mass_variances':
                vals = [fracval(x) for x in confs['mass_variances'].split()]
                if len(vals) < 1:
                    self.setting_error("Mass variance parameters are incorrectly set.")
                else:
                    self.set_parameter('mass_variances', vals)

            if conf_key == 'max_freepath':
                self.set_parameter('max_freepath', float(confs['max_freepath']))

            if conf_key == 'mesh_divisors':
                vals = [x for x in confs['mesh_divisors'].split()]
                if len(vals) == 3:
                    self.set_parameter('mesh_divisors', [int(x) for x in vals])
                elif len(vals) == 6:
                    divs = [int(x) for x in vals[:3]]
                    is_shift = [x.lower() == 't' for x in vals[3:]]
                    for i in range(3):
                        if is_shift[i] and (divs[i] % 2 != 0):
                            is_shift[i] = False
                            self.setting_error("Coarse grid shift along the " +
                                               ["first", "second", "third"][i] +
                                               " axis is not allowed.")
                    self.set_parameter('mesh_divisors', divs + is_shift)
                else:
                    self.setting_error("Mesh divisors are incorrectly set.")

            if conf_key == 'no_kappa_stars':
                if confs['no_kappa_stars'] == '.true.':
                    self.set_parameter('no_kappa_stars', True)

            if conf_key == 'read_amplitude':
                if confs['read_amplitude'] == '.true.':
                    self.set_parameter('read_amplitude', True)

            if conf_key == 'read_gamma':
                if confs['read_gamma'] == '.true.':
                    self.set_parameter('read_gamma', True)

            if conf_key == 'temperatures':
                vals = [fracval(x) for x in confs['temperatures'].split()]
                if len(vals) < 1:
                    self.setting_error("Temperatures are incorrectly set.")
                else:
                    self.set_parameter('temperatures', vals)

            if conf_key == 'write_amplitude':
                if confs['write_amplitude'] == '.true.':
                    self.set_parameter('write_amplitude', True)

            if conf_key == 'write_gamma':
                if confs['write_gamma'] == '.true.':
                    self.set_parameter('write_gamma', True)


    def _set_settings(self):
        ConfParser.set_settings(self)
        params = self._parameters

        # Supercell size for fc2
        if params.has_key('dim_extra'):
            self._settings.set_phonon_supercell_matrix(params['dim_extra'])

        # Cutoff distance of third-order force constants. Elements where any 
        # pair of atoms has larger distance than cut-off distance are set zero.
        if params.has_key('cutoff_fc3_distance'):
            self._settings.set_cutoff_fc3_distance(params['cutoff_fc3_distance'])

        # Cutoff distance between pairs of displaced atoms used for supercell
        # creation with displacements and making third-order force constants
        if params.has_key('cutoff_pair_distance'):
            self._settings.set_cutoff_pair_distance(
                params['cutoff_pair_distance'])

        # Phonon modes below this frequency are ignored.
        if params.has_key('cutoff_frequency'):
            self._settings.set_cutoff_frequency(params['cutoff_frequency'])

        # Cutoff lifetime used for thermal conductivity calculation
        if params.has_key('cutoff_lifetime'):
            self._settings.set_cutoff_lifetime(params['cutoff_lifetime'])

        # Grid points
        if params.has_key('grid_points'):
            self._settings.set_grid_points(params['grid_points'])

        # Atoms are clamped under applied strain in Gruneisen parameter calculation
        if params.has_key('ion_clamped'):
            self._settings.set_ion_clamped(params['ion_clamped'])

        # Calculate thermal conductivity in BTE-RTA
        if params.has_key('is_bterta'):
            self._settings.set_is_bterta(params['is_bterta'])

        # Calculate linewidths
        if params.has_key('is_linewidth'):
            self._settings.set_is_linewidth(params['is_linewidth'])

        # Calculate frequency_shifts
        if params.has_key('is_frequency_shift'):
            self._settings.set_is_frequency_shift(params['is_frequency_shift'])

        # Mass variance parameters
        if params.has_key('mass_variances'):
            self._settings.set_mass_variances(params['mass_variances'])

        # Maximum mean free path
        if params.has_key('max_freepath'):
            self._settings.set_max_freepath(params['max_freepath'])

        # Divisors for mesh numbers
        if params.has_key('mesh_divisors'):
            self._settings.set_mesh_divisors(params['mesh_divisors'][:3])
            if len(params['mesh_divisors']) > 3:
                self._settings.set_coarse_mesh_shifts(
                    params['mesh_divisors'][3:])

        # Read phonon-phonon interaction amplitudes from hdf5
        if params.has_key('read_amplitude'):
            self._settings.set_read_amplitude(params['read_amplitude'])

        # Read gammas from hdf5
        if params.has_key('read_gamma'):
            self._settings.set_read_gamma(params['read_gamma'])
            
        # Sum partial kappa at q-stars
        if params.has_key('no_kappa_stars'):
            self._settings.set_no_kappa_stars(params['no_kappa_stars'])

        # Temperatures
        if params.has_key('temperatures'):
            self._settings.set_temperatures(params['temperatures'])

        # Write phonon-phonon interaction amplitudes to hdf5
        if params.has_key('write_amplitude'):
            self._settings.set_write_amplitude(params['write_amplitude'])

        # Write gamma to hdf5
        if params.has_key('write_gamma'):
            self._settings.set_write_gamma(params['write_gamma'])


        

