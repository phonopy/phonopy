import sys
import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.units import THz, Angstrom
from anharmonic.phonon3.conductivity import Conductivity
from anharmonic.phonon3.collision_matrix import CollisionMatrix
from anharmonic.phonon3.triplets import (get_grid_points_by_rotations,
                                         get_BZ_grid_points_by_rotations)
from anharmonic.file_IO import (write_kappa_to_hdf5,
                                write_collision_to_hdf5,
                                read_collision_from_hdf5,
                                write_collision_eigenvalues_to_hdf5)
from phonopy.units import THzToEv, Kb

def get_thermal_conductivity_LBTE(
        interaction,
        symmetry,
        temperatures=np.arange(0, 1001, 10, dtype='double'),
        sigmas=None,
        is_isotope=False,
        mass_variances=None,
        grid_points=None,
        boundary_mfp=None, # in micrometre
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=1e-4, # for group velocity
        is_full_pp=False,
        pinv_cutoff=1.0e-8,
        write_collision=False,
        read_collision=False,
        write_kappa=False,
        input_filename=None,
        output_filename=None,
        log_level=0):

    if sigmas is None:
        sigmas = []
    if log_level:
        print("-------------------- Lattice thermal conducitivity (LBTE) "
              "--------------------")
        print("Cutoff frequency of pseudo inversion of collision matrix: %s" %
              pinv_cutoff)

    if read_collision:
        temps = None
    else:
        temps = temperatures
        
    lbte = Conductivity_LBTE(
        interaction,
        symmetry,
        grid_points=grid_points,
        temperatures=temps,
        sigmas=sigmas,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        pinv_cutoff=pinv_cutoff,
        log_level=log_level)
    
    if read_collision:
        read_from = _set_collision_from_file(
            lbte,
            indices=read_collision,
            filename=input_filename)
        if not read_from:
            print("Reading collisions failed.")
            return False

    for i in lbte:
        if write_collision:
            _write_collision(lbte, i=i, filename=output_filename)

    if not read_collision or read_from == "grid_points":
        _write_collision(lbte, filename=output_filename)

    if write_kappa and grid_points is None:
        lbte.set_kappa_at_sigmas()
        _write_kappa(lbte, filename=output_filename, log_level=log_level)
    
    return lbte

def _write_collision(lbte, i=None, filename=None):
    temperatures = lbte.get_temperatures()
    sigmas = lbte.get_sigmas()
    gamma = lbte.get_gamma()
    gamma_isotope = lbte.get_gamma_isotope()
    collision_matrix = lbte.get_collision_matrix()
    mesh = lbte.get_mesh_numbers()
    
    if i is not None:
        gp = lbte.get_grid_points()[i]
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j, i]
            else:
                gamma_isotope_at_sigma = None
            write_collision_to_hdf5(temperatures,
                                    mesh,
                                    gamma=gamma[j, :, i],
                                    gamma_isotope=gamma_isotope_at_sigma,
                                    collision_matrix=collision_matrix[j, :, i],
                                    grid_point=gp,
                                    sigma=sigma,
                                    filename=filename)
    else:    
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j]
            else:
                gamma_isotope_at_sigma = None
            write_collision_to_hdf5(temperatures,
                                    mesh,
                                    gamma=gamma[j],
                                    gamma_isotope=gamma_isotope_at_sigma,
                                    collision_matrix=collision_matrix[j],
                                    sigma=sigma,
                                    filename=filename)
    
def _write_kappa(lbte, filename=None, log_level=0):
    temperatures = lbte.get_temperatures()
    sigmas = lbte.get_sigmas()
    gamma = lbte.get_gamma()
    mesh = lbte.get_mesh_numbers()
    frequencies = lbte.get_frequencies()
    gv = lbte.get_group_velocities()
    ave_pp = lbte.get_averaged_pp_interaction()
    qpoints = lbte.get_qpoints()
    kappa = lbte.get_kappa()
    mode_kappa = lbte.get_mode_kappa()
    
    coleigs = lbte.get_collision_eigenvalues()

    for i, sigma in enumerate(sigmas):
        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            kappa=kappa[i],
                            mode_kappa=mode_kappa[i],
                            gamma=gamma[i],
                            averaged_pp_interaction=ave_pp,
                            qpoint=qpoints,
                            sigma=sigma,
                            filename=filename,
                            verbose=log_level)
        write_collision_eigenvalues_to_hdf5(temperatures,
                                            mesh,
                                            coleigs[i],
                                            sigma=sigma,
                                            filename=filename,
                                            verbose=log_level)
                                            

def _set_collision_from_file(lbte,
                             indices='all',
                             filename=None):
    sigmas = lbte.get_sigmas()
    mesh = lbte.get_mesh_numbers()
    grid_points = lbte.get_grid_points()

    gamma = []
    collision_matrix = []

    read_from = None

    for j, sigma in enumerate(sigmas):
        collisions = read_collision_from_hdf5(mesh,
                                              sigma=sigma,
                                              filename=filename)
        if collisions is False:
            gamma_of_gps = []
            collision_matrix_of_gps = []
            for i, gp in enumerate(grid_points):
                collision_gp = read_collision_from_hdf5(
                    mesh,
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename)
                if collision_gp is False:
                    print("Gamma at grid point %d doesn't exist." % gp)
                    return False
                else:
                    (collision_matrix_at_gp,
                     gamma_at_gp,
                     temperatures_at_gp) = collision_gp
                    gamma_at_t = []
                    collision_matrix_at_t = []
                    if indices == 'all':
                        gamma_of_gps.append(gamma_at_gp)
                        collision_matrix_of_gps.append(collision_matrix_at_gp)
                        temperatures = temperatures_at_gp
                    else:                        
                        gamma_of_gps.append(gamma_at_gp[indices])
                        collision_matrix_of_gps.append(
                            collision_matrix_at_gp[indices])
                        temperatures = temperatures_at_gp[indices]
                        
            gamma_at_sigma = np.zeros((len(temperatures),
                                       len(grid_points),
                                       len(gamma_of_gps[0][0])),
                                       dtype='double')
            collision_matrix_at_sigma = np.zeros((len(temperatures),
                                                  len(grid_points),
                                                  len(gamma_of_gps[0][0]),
                                                  3,
                                                  len(grid_points),
                                                  len(gamma_of_gps[0][0]),
                                                  3),
                                                  dtype='double')
            for i in range(len(temperatures)):
                for j in range(len(grid_points)):
                    gamma_at_sigma[i, j] = gamma_of_gps[j][i]
                    collision_matrix_at_sigma[
                        i, j] = collision_matrix_of_gps[j][i]
                
            gamma.append(gamma_at_sigma)
            collision_matrix.append(collision_matrix_at_sigma)

            read_from = "grid_points"
        else:            
            (collision_matrix_at_sigma,
             gamma_at_sigma,
             temperatures_at_sigma) = collisions

            if indices == 'all':
                collision_matrix.append(collision_matrix_at_sigma)
                gamma.append(gamma_at_sigma)
                temperatures = temperatures_at_sigma
            else:
                collision_matrix.append(collision_matrix_at_sigma[indices])
                gamma.append(gamma_at_sigma[indices])
                temperatures = temperatures_at_sigma[indices]

            read_from = "full_matrix"
        
    temperatures = np.array(temperatures, dtype='double', order='C')
    gamma = np.array(gamma, dtype='double', order='C')
    collision_matrix = np.array(collision_matrix, dtype='double', order='C')

    lbte.set_temperatures(temperatures)
    lbte.set_gamma(gamma)
    lbte.set_collision_matrix(collision_matrix)
    
    return read_from
        
class Conductivity_LBTE(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 is_isotope=False,
                 mass_variances=None,
                 boundary_mfp=None, # in micrometre
                 is_reducible_collision_matrix=False,
                 is_kappa_star=True,
                 gv_delta_q=None, # finite difference for group veolocity
                 is_full_pp=False,
                 pinv_cutoff=1.0e-8,
                 log_level=0):
        if sigmas is None:
            sigmas = []
        self._pp = None
        self._temperatures = None
        self._sigmas = None
        self._is_kappa_star = None
        self._gv_delta_q = None
        self._is_full_pp = is_full_pp
        self._log_level = None
        self._primitive = None
        self._dm = None
        self._frequency_factor_to_THz = None
        self._cutoff_frequency = None
        self._boundary_mfp = None

        self._symmetry = None
        self._point_operations = None
        self._rotations_cartesian = None
        
        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._ir_grid_points = None
        self._ir_grid_weights = None

        self._gamma = None
        self._read_gamma = False
        self._read_gamma_iso = False
        self._frequencies = None
        self._gv = None
        self._gamma_iso = None
        self._averaged_pp_interaction = None
        
        self._mesh = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._conversion_factor = None
        
        self._is_isotope = None
        self._isotope = None
        self._mass_variances = None
        self._grid_point_count = None

        self._collision_eigenvalues = None

        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              is_isotope=is_isotope,
                              mass_variances=mass_variances,
                              boundary_mfp=boundary_mfp,
                              is_kappa_star=is_kappa_star,
                              gv_delta_q=gv_delta_q,
                              log_level=log_level)

        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        if not self._is_kappa_star:
            self._is_reducible_collision_matrix = True
        self._collision_matrix = None
        self._pinv_cutoff = pinv_cutoff
        
        if self._temperatures is not None:
            self._allocate_values()

    def set_kappa_at_sigmas(self):
        if len(self._grid_points) != len(self._ir_grid_points):
            print("Collision matrix is not well created.")
            import sys
            sys.exit(1)
        else:
            self._set_kappa_at_sigmas()

    def set_collision_matrix(self, collision_matrix):
        self._collision_matrix = collision_matrix
        
    def get_collision_matrix(self):
        return self._collision_matrix

    def get_collision_eigenvalues(self):
        return self._collision_eigenvalues
        
    def get_averaged_pp_interaction(self):
        return self._averaged_pp_interaction

    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]

        if not self._read_gamma:
            self._collision.set_grid_point(grid_point)
            
            if self._log_level:
                print("Number of triplets: %s" %
                      len(self._pp.get_triplets_at_q()[0]))
                print("Calculating interaction...")
                
            self._set_collision_matrix_at_sigmas(i)
            
        if self._isotope is not None:
            self._set_gamma_isotope_at_sigmas(i)

        self._set_gv(i)
        if self._log_level:
            self._show_log(i)

    def _allocate_values(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        num_ir_grid_points = len(self._ir_grid_points)

        self._kappa = np.zeros((len(self._sigmas),
                                len(self._temperatures),
                                6), dtype='double')
        self._gv = np.zeros((num_grid_points,
                             num_band,
                             3), dtype='double')
        if self._is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band), dtype='double')
        self._gamma = np.zeros((len(self._sigmas),
                                len(self._temperatures),
                                num_grid_points,
                                num_band), dtype='double')
        if self._isotope is not None:
            self._gamma_iso = np.zeros((len(self._sigmas),
                                        num_grid_points,
                                        num_band), dtype='double')

        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._mesh)
            self._mode_kappa = np.zeros((len(self._sigmas),
                                         len(self._temperatures),
                                         num_mesh_points,
                                         num_band,
                                         6), dtype='double')
            self._collision = CollisionMatrix(
                self._pp,
                is_reducible_collision_matrix=True)
            self._collision_matrix = np.zeros(
                (len(self._sigmas),
                 len(self._temperatures),
                 num_grid_points, num_band, num_mesh_points, num_band),
                dtype='double')
        else:
            self._mode_kappa = np.zeros((len(self._sigmas),
                                         len(self._temperatures),
                                         num_grid_points,
                                         num_band,
                                         6), dtype='double')
            self._rot_grid_points = np.zeros(
                (len(self._ir_grid_points), len(self._point_operations)),
                dtype='intc')
            self._rot_BZ_grid_points = np.zeros(
                (len(self._ir_grid_points), len(self._point_operations)),
                dtype='intc')
            for i, ir_gp in enumerate(self._ir_grid_points):
                self._rot_grid_points[i] = get_grid_points_by_rotations(
                    self._grid_address[ir_gp],
                    self._point_operations,
                    self._mesh)
                self._rot_BZ_grid_points[i] = get_BZ_grid_points_by_rotations(
                    self._grid_address[ir_gp],
                    self._point_operations,
                    self._mesh,
                    self._pp.get_bz_map())
            self._collision = CollisionMatrix(
                self._pp,
                point_operations=self._point_operations,
                ir_grid_points=self._ir_grid_points,
                rotated_grid_points=self._rot_BZ_grid_points)
            self._collision_matrix = np.zeros(
                (len(self._sigmas),
                 len(self._temperatures),
                 num_grid_points, num_band, 3,
                 num_ir_grid_points, num_band, 3),
                dtype='double')

            self._collision_eigenvalues = np.zeros(
                (len(self._sigmas),
                 len(self._temperatures),
                 num_ir_grid_points * num_band * 3),
                dtype='double')

    def _set_collision_matrix_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating collision matrix with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)
            self._collision.set_sigma(sigma)
            self._collision.set_integration_weights()

            if self._is_full_pp and j != 0:
                pass
            else:
                self._collision.run_interaction(is_full_pp=self._is_full_pp)
            if self._is_full_pp and j == 0:
                self._averaged_pp_interaction[i] = (
                    self._pp.get_averaged_interaction())

            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                self._collision.run()
                self._gamma[j, k, i] = self._collision.get_imag_self_energy()
                self._collision_matrix[j, k, i] = (
                    self._collision.get_collision_matrix())

    def _set_kappa_at_sigmas(self):
        if self._log_level:
            print("Symmetrizing collision matrix...")
            sys.stdout.flush()
            
        if self._is_reducible_collision_matrix:
            if self._is_kappa_star:
                self._expand_collisions()
            self._combine_reducible_collisions()
            weights = np.ones(np.prod(self._mesh), dtype='intc')
            self._symmetrize_reducible_collision_matrix()
        else:
            self._combine_collisions()
            weights = self._get_weights()
            for i, w_i in enumerate(weights):
                for j, w_j in enumerate(weights):
                    self._collision_matrix[:, :, i, :, :, j, :, :] *= w_i * w_j
            self._symmetrize_collision_matrix()
            
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "----------- Thermal conductivity (W/m-k) "
                if sigma:
                    text += "for sigma=%s -----------" % sigma
                else:
                    text += "with tetrahedron method -----------"
                print(text)
                print(("#%6s       " + " %-10s" * 6) %
                      ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
            for k, t in enumerate(self._temperatures):
                if t > 0:
                    if self._is_reducible_collision_matrix:
                        self._set_inv_reducible_collision_matrix(j, k)
                    else:
                        self._set_inv_collision_matrix(j, k)
                    X = self._get_X(t, weights)
                    self._set_kappa(j, k, X)

                if self._log_level:
                    print(("%7.1f " + " %10.3f" * 6) % 
                          ((t,) + tuple(self._kappa[j, k])))

        if self._log_level:
            print('')

    def _combine_collisions(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        for j, k in list(np.ndindex((len(self._sigmas),
                                     len(self._temperatures)))):
            for i, ir_gp in enumerate(self._ir_grid_points):
                multi = ((self._rot_grid_points == ir_gp).sum() //
                         (self._rot_BZ_grid_points == ir_gp).sum())
                for r, r_BZ_gp in zip(self._rotations_cartesian,
                                      self._rot_BZ_grid_points[i]):
                    if ir_gp != r_BZ_gp:
                        continue

                    main_diagonal = self._get_main_diagonal(i, j, k)
                    main_diagonal *= multi
                    for l in range(num_band):
                        self._collision_matrix[
                            j, k, i, l, :, i, l, :] += main_diagonal[l] * r

    def _combine_reducible_collisions(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_mesh_points = np.prod(self._mesh)

        for j, k in list(
                np.ndindex((len(self._sigmas), len(self._temperatures)))):
            for i in range(num_mesh_points):
                main_diagonal = self._get_main_diagonal(i, j, k)
                for l in range(num_band):
                    self._collision_matrix[
                        j, k, i, l, i, l] += main_diagonal[l]

    def _expand_collisions(self):
        num_mesh_points = np.prod(self._mesh)
        num_rot = len(self._point_operations)
        num_band = self._primitive.get_number_of_atoms() * 3
        rot_grid_points = np.zeros(
            (num_rot, num_mesh_points), dtype='intc')
        collision_matrix = np.zeros(
            (len(self._sigmas),
             len(self._temperatures),
             num_mesh_points, num_band, num_mesh_points, num_band),
            dtype='double')
        gamma = np.zeros((len(self._sigmas),
                          len(self._temperatures),
                          num_mesh_points,
                          num_band), dtype='double')
        gv = np.zeros((num_mesh_points,
                       num_band,
                       3), dtype='double')
        
        if self._gamma_iso is not None:
            gamma_iso = np.zeros((len(self._sigmas),
                                  num_mesh_points,
                                  num_band), dtype='double')
        
        for i in range(num_mesh_points):
            rot_grid_points[:, i] = get_grid_points_by_rotations(
                self._grid_address[i],
                self._point_operations,
                self._mesh)
            
        for i, ir_gp in enumerate(self._ir_grid_points):
            multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
            g_elem = self._gamma[:, :, i, :] / multi
            if self._gamma_iso is not None:
                giso_elem = self._gamma_iso[:, i, :] / multi
            for j, r in enumerate(self._rotations_cartesian):
                gp_r = rot_grid_points[j, ir_gp]
                gamma[:, :, gp_r, :] += g_elem
                if self._gamma_iso is not None:
                    gamma_iso[:, gp_r, :] += giso_elem
                for k in range(num_mesh_points):
                    colmat_elem = self._collision_matrix[:, :, i, :, k, :]
                    colmat_elem = colmat_elem.copy() / multi
                    gp_c = rot_grid_points[j, k]
                    collision_matrix[:, :, gp_r, :, gp_c, :] += colmat_elem
                    
                gv[gp_r] += np.dot(self._gv[i], r.T) / multi

        self._gamma = gamma
        self._collision_matrix = collision_matrix
        if self._gamma_iso is not None:
            self._gamma_iso = gamma_iso
        self._gv = gv
                            
    def _get_weights(self):
        weights = []
        for r_gps in self._rot_grid_points:
            weights.append(np.sqrt(len(np.unique(r_gps))) / np.sqrt(len(r_gps)))
        return weights

    def _symmetrize_collision_matrix(self):
        import anharmonic._phono3py as phono3c
        phono3c.symmetrize_collision_matrix(self._collision_matrix)

        # Average matrix elements belonging to degenerate bands
        col_mat = self._collision_matrix
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                sum_col = (col_mat[:, :, i, bi_set, :, :, :, :].sum(axis=2) /
                           len(bi_set))
                for j in bi_set:
                    col_mat[:, :, i, j, :, :, :, :] = sum_col

        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                sum_col = (col_mat[:, :, :, :, :, i, bi_set, :].sum(axis=5) /
                           len(bi_set))
                for j in bi_set:
                    col_mat[:, :, :, :, :, i, j, :] = sum_col
        
    def _symmetrize_reducible_collision_matrix(self):
        import anharmonic._phono3py as phono3c
        phono3c.symmetrize_collision_matrix(self._collision_matrix)
        
        # Average matrix elements belonging to degenerate bands
        col_mat = self._collision_matrix
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                sum_col = (col_mat[:, :, i, bi_set, :, :].sum(axis=2) /
                           len(bi_set))
                for j in bi_set:
                    col_mat[:, :, i, j, :, :] = sum_col

        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                sum_col = (col_mat[:, :, :, :, i, bi_set].sum(axis=4) /
                           len(bi_set))
                for j in bi_set:
                    col_mat[:, :, :, :, i, j] = sum_col
        
    def _get_X(self, t, weights):
        num_band = self._primitive.get_number_of_atoms() * 3
        X = self._gv.copy()
        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._mesh)
            freqs = self._frequencies[:num_mesh_points]
        else:
            freqs = self._frequencies[self._ir_grid_points]
            
        sinh = np.where(freqs > self._cutoff_frequency,
                        np.sinh(freqs * THzToEv / (2 * Kb * t)),
                        -1.0)
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)
        freqs_sinh = freqs * THzToEv * inv_sinh / (4 * Kb * t ** 2)
                
        for i, f in enumerate(freqs_sinh):
            X[i] *= weights[i]
            for j in range(num_band):
                X[i, j] *= f[j]
        
        if t > 0:
            return X.reshape(-1, 3)
        else:
            return np.zeros_like(X.reshape(-1, 3))

    def _set_inv_collision_matrix(self,
                                  i_sigma,
                                  i_temp,
                                  method=0):
        num_ir_grid_points = len(self._ir_grid_points)
        num_band = self._primitive.get_number_of_atoms() * 3
        
        if method == 0:
            col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                num_ir_grid_points * num_band * 3,
                num_ir_grid_points * num_band * 3)
            w, col_mat[:] = np.linalg.eigh(col_mat)
            e = np.zeros_like(w)
            v = col_mat
            for l, val in enumerate(w):
                if val > self._pinv_cutoff:
                    e[l] = 1 / np.sqrt(val)
            v[:] = e * v
            v[:] = np.dot(v, v.T) # inv_col
        elif method == 1:
            import anharmonic._phono3py as phono3c
            w = np.zeros(num_ir_grid_points * num_band * 3, dtype='double')
            phono3c.inverse_collision_matrix(
                self._collision_matrix, w, i_sigma, i_temp, self._pinv_cutoff)
        elif method == 2:
            import anharmonic._phono3py as phono3c
            w = np.zeros(num_ir_grid_points * num_band * 3, dtype='double')
            phono3c.inverse_collision_matrix_libflame(
                self._collision_matrix, w, i_sigma, i_temp, self._pinv_cutoff)

        self._collision_eigenvalues[i_sigma, i_temp] = w

    def _set_inv_reducible_collision_matrix(self, i_sigma, i_temp):
        t = self._temperatures[i_temp]
        num_mesh_points = np.prod(self._mesh)
        num_band = self._primitive.get_number_of_atoms() * 3
        col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
            num_mesh_points * num_band, num_mesh_points * num_band)
        w, col_mat[:] = np.linalg.eigh(col_mat)
        v = col_mat
        e = np.zeros(len(w), dtype='double')
        for l, val in enumerate(w):
            if val > self._pinv_cutoff:
                e[l] = 1 / np.sqrt(val)
        v[:] = e * v
        v[:] = np.dot(v, v.T) # inv_col
        
    def _set_kappa(self, i_sigma, i_temp, X):
        num_band = self._primitive.get_number_of_atoms() * 3

        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._mesh)
            num_grid_points = num_mesh_points
            from phonopy.harmonic.force_constants import similarity_transformation
            point_operations = self._symmetry.get_reciprocal_operations()
            rec_lat = np.linalg.inv(self._primitive.get_cell())
            rotations_cartesian = np.array(
                [similarity_transformation(rec_lat, r)
                 for r in point_operations], dtype='double')
            inv_col_mat = np.kron(
                self._collision_matrix[i_sigma, i_temp].reshape(
                    num_mesh_points * num_band,
                    num_mesh_points * num_band), np.eye(3))
        else:
            num_ir_grid_points = len(self._ir_grid_points)
            num_grid_points = num_ir_grid_points
            rotations_cartesian = self._rotations_cartesian
            inv_col_mat = self._collision_matrix[i_sigma, i_temp].reshape(
                num_ir_grid_points * num_band * 3,
                num_ir_grid_points * num_band * 3)

        Y = np.dot(inv_col_mat, X.ravel()).reshape(-1, 3)

        for i, (v_gp, f_gp) in enumerate(zip(X.reshape(num_grid_points,
                                                       num_band, 3),
                                             Y.reshape(num_grid_points,
                                                       num_band, 3))):
            for j, (v, f) in enumerate(zip(v_gp, f_gp)):
                sum_k = np.zeros((3, 3), dtype='double')
                for r in rotations_cartesian:
                    sum_k += np.outer(np.dot(r, v), np.dot(r, f))
                sum_k = sum_k + sum_k.T
                for k, vxf in enumerate(
                        ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
                    self._mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]

        t = self._temperatures[i_temp]
        self._mode_kappa *= (self._conversion_factor * Kb * t ** 2
                             / np.prod(self._mesh))
        
        if self._is_reducible_collision_matrix:
            self._mode_kappa /= len(point_operations)
        
        self._kappa[i_sigma, i_temp] = (
            self._mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0))
                    
    def _show_log(self, i):
        q = self._qpoints[i]
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        gv = self._gv[i]
        if self._is_full_pp:
            ave_pp = self._averaged_pp_interaction[i]
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"

        if self._gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._gv_delta_q
        print(text)
        if self._is_full_pp:
            for f, v, pp in zip(frequencies, gv, ave_pp):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
        else:
            for f, v in zip(frequencies, gv):
                print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                      (f, v[0], v[1], v[2], np.linalg.norm(v)))

        sys.stdout.flush()

    def _py_symmetrize_collision_matrix(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(3):
                    for l in range(num_ir_grid_points):
                        for m in range(num_band):
                            for n in range(3):
                                self._py_set_symmetrized_element(
                                    i, j, k, l, m, n)

    def _py_set_symmetrized_element(self, i, j, k, l, m, n):
        sym_val = (self._collision_matrix[:, :, i, j, k, l, m, n] +
                   self._collision_matrix[:, :, l, m, n, i, j, k]) / 2
        self._collision_matrix[:, :, i, j, k, l, m, n] = sym_val
        self._collision_matrix[:, :, l, m, n, i, j, k] = sym_val

    def _py_symmetrize_collision_matrix_no_kappa_stars(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(num_ir_grid_points):
                    for l in range(num_band):
                        self._py_set_symmetrized_element_no_kappa_stars(
                            i, j, k, l)

    def _py_set_symmetrized_element_no_kappa_stars(self, i, j, k, l):
        sym_val = (self._collision_matrix[:, :, i, j, k, l] +
                   self._collision_matrix[:, :, k, l, i, j]) / 2
        self._collision_matrix[:, :, i, j, k, l] = sym_val
        self._collision_matrix[:, :, k, l, i, j] = sym_val
        
