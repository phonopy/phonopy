import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from anharmonic.phonon3.conductivity import Conductivity
from anharmonic.phonon3.collision_matrix import CollisionMatrix

def get_thermal_conductivity_LBTE(
        interaction,
        symmetry,
        temperatures=np.arange(0, 1001, 10, dtype='double'),
        sigmas=[],
        mass_variances=None,
        grid_points=None,
        mesh_divisors=None,
        coarse_mesh_shifts=None,
        cutoff_lifetime=1e-4, # in second
        no_kappa_stars=False,
        gv_delta_q=1e-4, # for group velocity
        write_gamma=False,
        read_gamma=False,
        input_filename=None,
        output_filename=None,
        log_level=0):

    if log_level:
        print "-------------------- Lattice thermal conducitivity (LBTE) --------------------"
    br = Conductivity_LBTE(interaction,
                           symmetry,
                           temperatures=temperatures,
                           sigmas=sigmas,
                           mass_variances=mass_variances,
                           mesh_divisors=mesh_divisors,
                           coarse_mesh_shifts=coarse_mesh_shifts,
                           cutoff_lifetime=cutoff_lifetime,
                           no_kappa_stars=no_kappa_stars,
                           gv_delta_q=gv_delta_q,
                           log_level=log_level)
    br.initialize(grid_points)

    for i in br:
        pass

    return br

class Conductivity_LBTE(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 temperatures=np.arange(0, 1001, 10, dtype='double'),
                 sigmas=[],
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 cutoff_lifetime=1e-4, # in second
                 no_kappa_stars=False,
                 gv_delta_q=None, # finite difference for group veolocity
                 log_level=0):

        self._pp = None
        self._temperatures = None
        self._sigmas = None
        self._no_kappa_stars = None
        self._gv_delta_q = None
        self._log_level = None
        self._primitive = None
        self._dm = None
        self._frequency_factor_to_THz = None
        self._cutoff_frequency = None
        self._cutoff_lifetime = None

        self._symmetry = None
        self._point_operations = None
        self._rotations_cartesian = None
        
        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._gamma = None
        self._collision_matrix = None
        self._read_gamma = False
        self._frequencies = None
        self._gv = None
        self._gamma_iso = None

        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._conversion_factor = None
        self._sum_num_kstar = None

        self._isotope = None
        self._mass_variances = None
        self._grid_point_count = None

        Conductivity.__init__(self,
                 interaction,
                 symmetry,
                 temperatures=temperatures,
                 sigmas=sigmas,
                 mass_variances=mass_variances,
                 mesh_divisors=mesh_divisors,
                 coarse_mesh_shifts=coarse_mesh_shifts,
                 cutoff_lifetime=cutoff_lifetime,
                 no_kappa_stars=no_kappa_stars,
                 gv_delta_q=gv_delta_q,
                 log_level=log_level)

        self._collision = CollisionMatrix(self._pp, self._symmetry)
        
    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]
        if not self._read_gamma:
            self._collision.set_grid_point(grid_point)
            
            if self._log_level:
                print "Number of triplets:",
                print len(self._pp.get_triplets_at_q()[0])
                print "Calculating interaction..."
                
            self._collision.run_interaction()
            self._set_gamma_at_sigmas()
            self._set_collision_matrix_at_sigmas()

        if self._isotope is not None:
            self._set_gamma_isotope_at_sigmas()

    def _allocate_values(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        num_ir_grid_points = len(self._ir_grid_points)
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    num_grid_points,
                                    len(self._temperatures),
                                    num_band), dtype='double')
            self._collision_matrix = np.zeros((len(self._sigmas),
                                               num_grid_points,
                                               num_ir_grid_points,
                                               len(self._temperatures),
                                               num_band,
                                               num_band,
                                               ), dtype='double')
            
        self._gv = np.zeros((num_grid_points,
                             num_band,
                             3), dtype='double')
        self._gamma_iso = np.zeros((len(self._sigmas),
                                    num_grid_points,
                                    num_band), dtype='double')
        


    def _set_collision_matrix_at_sigmas(self):
        i = self._grid_point_count
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating collision matrix with",
                if sigma is None:
                    print "tetrahedron method"
                else:
                    print "sigma=%s" % sigma
            self._collision.set_sigma(sigma)
            if not sigma:
                self._collision.set_integration_weights()
            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                self._collision.run_collision_matrix()
                self._gamma[j, i, k] = self._collision.get_imag_self_energy()
