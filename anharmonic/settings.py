import numpy as np
from phonopy.cui.settings import Settings, ConfParser, fracval

class Phono3pySettings( Settings ):
    def __init__( self ):
        Settings.__init__( self )

        self._supercell_matrix_extra = None
        self._band_indices = None
        self._q_direction = None
        self._is_lifetime = False
        self._qpoints = [[0, 0, 0]]
        
    def set_supercell_matrix_extra(self, matrix):
        self._supercell_matrix_extra = matrix

    def get_supercell_matrix_extra(self):
        return self._supercell_matrix_extra

    def set_band_indices(self, band_indices):
        self._band_indices = band_indices

    def get_band_indices(self):
        return self._band_indices

    def set_is_lifetime(self, is_lifetime):
        self._is_lifetime = is_lifetime

    def get_is_lifetime(self):
        return self._is_lifetime

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
                if not self._options.band_indices==None:
                    self._confs['band_indices'] = self._options.band_indices

            if opt.dest == 'is_lifetime':
                if self._options.is_lifetime:
                    self._confs['lifetime'] = '.true.'

            if opt.dest == 'q_direction':
                if not self._options.q_direction==None:
                    self._confs['q_direction'] = self._options.q_direction

            if opt.dest == 'qpoints':
                if not self._options.qpoints == None:
                    self._confs['qpoints'] = self._options.qpoints


    def _parse_conf( self ):
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == 'dim_extra':
                matrix = [ int(x) for x in confs['dim_extra'].split() ]
                if len( matrix ) == 9:
                    matrix = np.array( matrix ).reshape( 3, 3 )
                elif len( matrix ) == 3:
                    matrix = np.diag( matrix )
                else:
                    self.setting_error("Number of elements of dim2 has to be 3 or 9.")

                if matrix.shape == ( 3, 3 ):
                    if np.linalg.det( matrix ) < 1:
                        self.setting_error('Determinant of supercell matrix has to be positive.')
                    else:
                        self.set_parameter( 'dim_extra', matrix )

            if conf_key == 'band_indices':
                vals = []
                for sum_set in confs['band_indices'].split(','):
                    vals.append( [ int(x) for x in sum_set.split() ] )
                self.set_parameter('band_indices', vals)

            if conf_key == 'lifetime':
                if confs['lifetime'] == '.true.':
                    self.set_parameter('is_lifetime', True)

            if conf_key == 'q_direction':
                q_direction = [ float(x) for x in confs['q_direction'].split() ]
                if len( q_direction ) < 3:
                    self.setting_error("Number of elements of q_direction is less than 3")
                else:
                    self.set_parameter( 'q_direction', q_direction )

            if conf_key == 'qpoints':
                vals = [fracval(x) for x in confs['qpoints'].split()]
                if len(vals) == 0 or len(vals) % 3 != 0:
                    self.setting_error("Q-points are incorrectly set.")
                else:
                    self.set_parameter('qpoints',
                                       list(np.reshape(vals, (-1, 3))))

    def _set_settings( self ):
        ConfParser.set_settings( self )
        params = self._parameters

        # Supercell size for fc2
        if params.has_key('dim_extra'):
            self._settings.set_supercell_matrix_extra(params['dim_extra'])

        # Sets of band indices that are summed
        if params.has_key('band_indices'):
            self._settings.set_band_indices( params['band_indices'] )

        # Calculate lifetimes
        if params.has_key('is_lifetime'):
            self._settings.set_is_lifetime( params['is_lifetime'] )

        # q-vector direction at q->0 for non-analytical term correction
        if params.has_key('q_direction'):
            self._settings.set_q_direction( params['q_direction'] )
            
        # Q-points mode
        if params.has_key('qpoints'):
            self._settings.set_qpoints(params['qpoints'])
        
        

