import numpy as np
from phonopy.cui.settings import Settings, ConfParser, fracval

class Phono3pySettings(Settings):
    def __init__(self):
        Settings.__init__(self)

        self._supercell_matrix_extra = None
        self._band_indices = None
        self._q_direction = None
        self._is_bterta = False
        self._is_linewidth = False
        self._qpoints = [[0, 0, 0]]
        self._no_kappa_stars = False
        
    def set_supercell_matrix_extra(self, matrix):
        self._supercell_matrix_extra = matrix

    def get_supercell_matrix_extra(self):
        return self._supercell_matrix_extra

    def set_band_indices(self, band_indices):
        self._band_indices = band_indices

    def get_band_indices(self):
        return self._band_indices

    def set_is_bterta(self, is_bterta):
        self._is_bterta = is_bterta

    def get_is_bterta(self):
        return self._is_bterta

    def set_is_linewidth(self, is_linewidth):
        self._is_linewidth = is_linewidth

    def get_is_linewidth(self):
        return self._is_linewidth

    def set_multiple_sigmas(self, multiple_sigmas):
        self._multiple_sigmas = multiple_sigmas

    def get_multiple_sigmas(self):
        return self._multiple_sigmas

    def set_no_kappa_stars(self, no_kappa_stars):
        self._no_kappa_stars = no_kappa_stars

    def get_no_kappa_stars(self):
        return self._no_kappa_stars

    def set_q_direction(self, q_direction):
        self._q_direction = q_direction

    def get_q_direction(self):
        return self._q_direction

    def set_qpoints(self, qpoints):
        self._qpoints = qpoints

    def get_qpoints(self):
        return self._qpoints


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
                if not self._options.supercell_dimension_extra==None:
                    self._confs['dim_extra'] = self._options.supercell_dimension_extra

            if opt.dest == 'band_indices':
                if self._options.band_indices is not None:
                    self._confs['band_indices'] = self._options.band_indices

            if opt.dest == 'is_bterta':
                if self._options.is_bterta:
                    self._confs['bterta'] = '.true.'

            if opt.dest == 'is_linewidth':
                if self._options.is_linewidth:
                    self._confs['linewidth'] = '.true.'

            if opt.dest == 'multiple_sigmas':
                if self._options.multiple_sigmas is not None:
                    self._confs['multiple_sigmas'] = self._options.multiple_sigmas

            if opt.dest == 'no_kappa_stars':
                if self._options.no_kappa_stars:
                    self._confs['no_kappa_stars'] = '.true.'

            if opt.dest == 'q_direction':
                if self._options.q_direction is not None:
                    self._confs['q_direction'] = self._options.q_direction

            if opt.dest == 'qpoints':
                if self._options.qpoints is not None:
                    self._confs['qpoints'] = self._options.qpoints

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
                    self.setting_error("Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error('Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter('dim_extra', matrix)

            if conf_key == 'band_indices':
                vals = []
                for sum_set in confs['band_indices'].split(','):
                    vals.append([int(x) - 1 for x in sum_set.split()])
                self.set_parameter('band_indices', vals)

            if conf_key == 'bterta':
                if confs['bterta'] == '.true.':
                    self.set_parameter('is_bterta', True)

            if conf_key == 'linewidth':
                if confs['linewidth'] == '.true.':
                    self.set_parameter('is_linewidth', True)

            if conf_key == 'multiple_sigmas':
                vals = [fracval(x) for x in confs['multiple_sigmas'].split()]
                if len(vals) < 1:
                    self.setting_error("Mutiple sigmas are incorrectly set.")
                else:
                    self.set_parameter('multiple_sigmas', vals)

            if conf_key == 'no_kappa_stars':
                if confs['no_kappa_stars'] == '.true.':
                    self.set_parameter('no_kappa_stars', True)

            if conf_key == 'q_direction':
                q_direction = [ float(x) for x in confs['q_direction'].split() ]
                if len(q_direction) < 3:
                    self.setting_error("Number of elements of q_direction is less than 3")
                else:
                    self.set_parameter('q_direction', q_direction)

            if conf_key == 'qpoints':
                vals = [fracval(x) for x in confs['qpoints'].split()]
                if len(vals) == 0 or len(vals) % 3 != 0:
                    self.setting_error("Q-points are incorrectly set.")
                else:
                    self.set_parameter('qpoints',
                                       list(np.reshape(vals, (-1, 3))))



    def _set_settings(self):
        ConfParser.set_settings(self)
        params = self._parameters

        # Supercell size for fc2
        if params.has_key('dim_extra'):
            self._settings.set_supercell_matrix_extra(params['dim_extra'])

        # Sets of band indices that are summed
        if params.has_key('band_indices'):
            self._settings.set_band_indices(params['band_indices'])

        # Calculate thermal conductivity in BTE-RTA
        if params.has_key('is_bterta'):
            self._settings.set_is_bterta(params['is_bterta'])

        # Calculate linewidths
        if params.has_key('is_linewidth'):
            self._settings.set_is_linewidth(params['is_linewidth'])

        # Multiple sigmas
        if params.has_key('multiple_sigmas'):
            self._settings.set_multiple_sigmas(params['multiple_sigmas'])

        # q-vector direction at q->0 for non-analytical term correction
        if params.has_key('q_direction'):
            self._settings.set_q_direction(params['q_direction'])
            
        # Q-points mode
        if params.has_key('qpoints'):
            self._settings.set_qpoints(params['qpoints'])
        
        # Sum partial kappa at q-stars
        if params.has_key('no_kappa_stars'):
            self._settings.set_no_kappa_stars(params['no_kappa_stars'])

        

