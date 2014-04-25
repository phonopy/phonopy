import numpy as np
from anharmonic.phonon3.conductivity import Conductivity
from anharmonic.phonon3.collision_matrix import CollisionMatrix
from anharmonic.phonon3.triplets import get_grid_point_from_address
from phonopy.units import THzToEv, Kb

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

    if read_gamma:
        print "The option --read_gamma is not yet implemented."
        import sys
        sys.exit(1)

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

    br.run_pinv()
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

    def initialize(self, grid_points=None):
        Conductivity.initialize(self, grid_points=grid_points)
        self._collision = CollisionMatrix(self._pp,
                                          self._symmetry,
                                          self._ir_grid_points)
        
    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]

        if self._isotope is not None:
            self._set_gamma_isotope_at_sigmas()

        if not self._read_gamma:
            self._collision.set_grid_point(grid_point)
            
            if self._log_level:
                print "Number of triplets:",
                print len(self._pp.get_triplets_at_q()[0])
                print "Calculating interaction..."
                
            self._collision.run_interaction()
            self._set_collision_matrix_at_sigmas()
            self._set_gv()
            self._show_log()

    def _allocate_values(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        num_ir_grid_points = len(self._ir_grid_points)
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    len(self._temperatures),
                                    num_grid_points,
                                    num_band), dtype='double')
            
        self._collision_matrix = np.zeros(
            (len(self._sigmas),
             len(self._temperatures),
             num_grid_points, num_band, 3,
             num_ir_grid_points, num_band, 3),
            dtype='double')
        self._gv = np.zeros((num_grid_points,
                             num_band,
                             3), dtype='double')
        self._gamma_iso = np.zeros((len(self._sigmas),
                                    num_grid_points,
                                    num_band), dtype='double')

    def _set_collision_matrix_at_sigmas(self):
        i = self._grid_point_count
        ir_gp = self._grid_points[i]
        ir_address = self._grid_address[ir_gp]
        r_address = np.dot(self._point_operations.reshape(-1, 3),
                           ir_address).reshape(-1, 3)
        r_gps = get_grid_point_from_address(r_address.T, self._mesh)
        order_r_gp = np.sqrt(len(r_gps) / len(np.unique(r_gps)))
        
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating collision matrix with",
                if sigma is None:
                    print "tetrahedron method"
                else:
                    print "sigma=%s" % sigma
            self._collision.set_sigma(sigma)
            self._collision.set_integration_weights()
            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                if self._isotope is not None:
                    self._collision.run(self._gamma_iso[j, i])
                else:
                    self._collision.run()
                self._gamma[j, k, i] = self._collision.get_imag_self_energy()
                self._collision_matrix[j, k, i] = (
                    self._collision.get_collision_matrix() / order_r_gp)

    def _get_X(self, t):
        X = self._gv.copy()
        freqs = self._frequencies[self._ir_grid_points] * THzToEv
        sinh = np.where(
            freqs > 1e-5, np.sinh(freqs / (2 * Kb * t)), -1)
        inv_sinh = np.where(sinh > 0, 1 / sinh, 0)
        freqs_sinh = freqs * inv_sinh
        num_band = self._primitive.get_number_of_atoms() * 3
                
        for i, (ir_gp, f) in enumerate(zip(self._grid_points, freqs_sinh)):
            ir_address = self._grid_address[ir_gp]
            r_address = np.dot(self._point_operations.reshape(-1, 3),
                               ir_address).reshape(-1, 3)
            r_gps = get_grid_point_from_address(r_address.T, self._mesh)
            order_r_gp = np.sqrt(len(r_gps) / len(np.unique(r_gps)))
            X[i] *= 1.0 / (4 * Kb * t ** 2) / order_r_gp
            for j in range(num_band):
                X[i, j] *= f[j]
        
        if t > 0:
            # return (freqs / (4 * Kb * t ** 2) / np.sinh(freqs / (2 * Kb * t))
            #         * self._gv.reshape(-1, 3).T).T
            return X.reshape(-1, 3)
        else:
            return np.zeros_like(self._gv.reshape(-1, 3))

    def run_pinv(self):
        if len(self._grid_points) != len(self._ir_grid_points):
            print "Collision matrix is not well created."
            import sys
            sys.exit(1)
            
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        orders = self._get_order_of_star()
        rot_order = len(self._rotations_cartesian)

        for j, sigma in enumerate(self._sigmas):
            for k, t in enumerate(self._temperatures):
                X = self._get_X(t)
                col_mat = self._collision_matrix[j, k].reshape(
                    num_ir_grid_points * num_band * 3,
                    num_ir_grid_points * num_band * 3)

                for cutoff in (1e-3, 1e-5, 1e-7, 1e-9, 1e-11):
                    print cutoff
                    # inv_col = np.linalg.pinv((col_mat + col_mat.T) / 2)
                    inv_col = np.zeros_like(col_mat)
                    import anharmonic._forcefit as forcefit
                    forcefit.pinv(((col_mat + col_mat.T) / 2).copy(), inv_col, cutoff)
    
                    Y = np.dot(inv_col, X.ravel()).reshape(-1, 3)
                    RX = np.dot(self._rotations_cartesian.reshape(-1, 3), X.T).T
                    RY = np.dot(self._rotations_cartesian.reshape(-1, 3), Y.T).T
                    
                    sum_outer = np.zeros((3, 3), dtype='double')
                    for order, irX, irY in zip(
                            orders,
                            RX.reshape(num_ir_grid_points, num_band, rot_order, 3),
                            RY.reshape(num_ir_grid_points, num_band, rot_order, 3)):
                        sum_outer_kp = np.zeros((3, 3), dtype='double')
                        for X_band, Y_band in zip(irX, irY):
                            for RX_band, RY_band in zip(X_band, Y_band):
                                sum_outer_kp += np.outer(RX_band, RY_band)
                        # sum_outer += sum_outer_kp * order / rot_order
                        sum_outer += sum_outer_kp
                        
                    sum_outer *= self._conversion_factor * 2 * Kb * t ** 2 / np.prod(self._mesh)
    
                    print sum_outer

    def _get_order_of_star(self):
        orders = []
        num_band = self._primitive.get_number_of_atoms() * 3
        for i, ir_gp in enumerate(self._ir_grid_points):
            ir_address = self._grid_address[ir_gp]
            r_address = np.dot(self._point_operations.reshape(-1, 3),
                               ir_address).reshape(-1, 3)
            r_gps = get_grid_point_from_address(r_address.T, self._mesh)
            orders.append(len(np.unique(r_gps)))
            
        return np.array(orders, dtype='intc')
        
    def _show_log(self):
        i = self._grid_point_count
        q = self._qpoints[i]
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        gv = self._gv[i]

        print "Frequency, projected group velocity (x, y, z), group velocity norm",
        if self._gv_delta_q is None:
            print
        else:
            print " (dq=%3.1e)" % self._gv_delta_q
        for f, v in zip(frequencies, gv):
            print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                f, v[0], v[1], v[2], np.linalg.norm(v))

