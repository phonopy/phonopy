import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.units import EV, THz, Angstrom
from anharmonic.phonon3.triplets import get_grid_address, reduce_grid_points, get_ir_grid_points, from_coarse_to_dense_grid_points
from anharmonic.phonon3.interaction import set_phonon_c
from anharmonic.other.isotope import Isotope

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class Conductivity:
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=np.arange(0, 1001, 10, dtype='double'),
                 sigmas=[],
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 cutoff_lifetime=1e-4, # in second
                 no_kappa_stars=False,
                 gv_delta_q=None, # finite difference for group veolocity
                 log_level=0):
        self._pp = interaction
        self._collision = None # has to be set derived class
        
        self._temperatures = temperatures
        self._sigmas = sigmas
        self._no_kappa_stars = no_kappa_stars
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._primitive = self._pp.get_primitive()
        self._dm = self._pp.get_dynamical_matrix()
        self._frequency_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._cutoff_lifetime = cutoff_lifetime

        self._symmetry = symmetry
        self._point_operations = symmetry.get_reciprocal_operations()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._point_operations], dtype='double')

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

        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._set_mesh_numbers(mesh_divisors=mesh_divisors,
                               coarse_mesh_shifts=coarse_mesh_shifts)
        volume = self._primitive.get_volume()
        self._conversion_factor = unit_to_WmK / volume

        self._isotope = None
        self._mass_variances = None
        if mass_variances is not None:
            self._set_isotope(mass_variances)

        self._grid_point_count = None
        self._set_grid_properties(grid_points)

    def __iter__(self):
        return self
            
    def next(self):
        if self._grid_point_count == len(self._grid_points):
            if self._log_level:
                print ("=================== End of collection of collisions "
                       "===================")
            raise StopIteration
        else:
            self._run_at_grid_point()
            self._grid_point_count += 1
            return self._grid_point_count - 1

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_group_velocities(self):
        return self._gv

    def get_frequencies(self):
        return self._frequencies[self._grid_points]
        
    def get_qpoints(self):
        return self._qpoints
            
    def get_grid_points(self):
        return self._grid_points

    def get_grid_weights(self):
        return self._grid_weights
            
    def get_temperatures(self):
        return self._temperatures

    def set_gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True
        
    def set_gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    def get_gamma(self):
        return self._gamma
        
    def get_gamma_isotope(self):
        return self._gamma_iso
        
    def get_collision_matrix(self):
        return self._collision_matrix
        
    def get_kappa(self):
        return self._kappa

    def get_sigmas(self):
        return self._sigmas

    def get_grid_point_count(self):
        return self._grid_point_count

    def _run_at_grid_point(self):
        """This has to be implementated in the derived class"""
        pass

    def _allocate_values(self):
        """This has to be implementated in the derived class"""
        pass

    def _set_grid_properties(self, grid_points):
        self._grid_address = self._pp.get_grid_address()

        if grid_points is not None: # Specify grid points
            self._grid_points = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                grid_points,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            (self._ir_grid_points,
             self._ir_grid_weights) = self._get_ir_grid_points()
        elif self._no_kappa_stars: # All grid points
            coarse_grid_address = get_grid_address(self._coarse_mesh)
            coarse_grid_points = np.arange(np.prod(self._coarse_mesh),
                                           dtype='intc')
            self._grid_points = from_coarse_to_dense_grid_points(
                self._mesh,
                self._mesh_divisors,
                coarse_grid_points,
                coarse_grid_address,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            self._grid_weights = np.ones(len(self._grid_points), dtype='intc')
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights
        else: # Automatic sampling
            self._grid_points, self._grid_weights = self._get_ir_grid_points()
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights

        self._qpoints = np.array(self._grid_address[self._grid_points] /
                                 self._mesh.astype('double'),
                                 dtype='double', order='C')

        self._grid_point_count = 0
        self._pp.set_phonon(self._grid_points)
        self._frequencies = self._pp.get_phonons()[0]

    def _set_gamma_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating Gamma of ph-ph with",
                if sigma is None:
                    print "tetrahedron method"
                else:
                    print "sigma=%s" % sigma
            self._collision.set_sigma(sigma)
            if not sigma:
                self._collision.set_integration_weights()
            for k, t in enumerate(self._temperatures):
                self._collision.set_temperature(t)
                self._collision.run()
                self._gamma[j, k, i] = self._collision.get_imag_self_energy()
                
    def _set_gamma_isotope_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating Gamma of ph-isotope with",
                if sigma is None:
                    print "tetrahedron method"
                else:
                    print "sigma=%s" % sigma
            pp_freqs, pp_eigvecs, pp_phonon_done = self._pp.get_phonons()
            self._isotope.set_sigma(sigma)
            self._isotope.set_phonons(pp_freqs,
                                      pp_eigvecs,
                                      pp_phonon_done,
                                      dm=self._dm)
            gp = self._grid_points[i]
            self._isotope.set_grid_point(gp)
            self._isotope.run()
            self._gamma_iso[j, i] = self._isotope.get_gamma()

    def _set_mesh_numbers(self, mesh_divisors=None, coarse_mesh_shifts=None):
        self._mesh = self._pp.get_mesh_numbers()

        if mesh_divisors is None:
            self._mesh_divisors = np.array([1, 1, 1], dtype='intc')
        else:
            self._mesh_divisors = []
            for i, (m, n) in enumerate(zip(self._mesh, mesh_divisors)):
                if m % n == 0:
                    self._mesh_divisors.append(n)
                else:
                    self._mesh_divisors.append(1)
                    print ("Mesh number %d for the " +
                           ["first", "second", "third"][i] + 
                           " axis is not dividable by divisor %d.") % (m, n)
            self._mesh_divisors = np.array(self._mesh_divisors, dtype='intc')
            if coarse_mesh_shifts is None:
                self._coarse_mesh_shifts = [False, False, False]
            else:
                self._coarse_mesh_shifts = coarse_mesh_shifts
            for i in range(3):
                if (self._coarse_mesh_shifts[i] and
                    (self._mesh_divisors[i] % 2 != 0)):
                    print ("Coarse grid along " +
                           ["first", "second", "third"][i] + 
                           " axis can not be shifted. Set False.")
                    self._coarse_mesh_shifts[i] = False

        self._coarse_mesh = self._mesh / self._mesh_divisors

        if self._log_level:
            print ("Lifetime sampling mesh: [ %d %d %d ]" %
                   tuple(self._mesh / self._mesh_divisors))

    def _get_ir_grid_points(self):
        if self._coarse_mesh_shifts is None:
            mesh_shifts = [False, False, False]
        else:
            mesh_shifts = self._coarse_mesh_shifts
        (coarse_grid_points,
         coarse_grid_weights,
         coarse_grid_address) = get_ir_grid_points(
            self._coarse_mesh,
            self._symmetry.get_pointgroup_operations(),
            mesh_shifts=mesh_shifts)
        grid_points = from_coarse_to_dense_grid_points(
            self._mesh,
            self._mesh_divisors,
            coarse_grid_points,
            coarse_grid_address,
            coarse_mesh_shifts=self._coarse_mesh_shifts)
        grid_weights = coarse_grid_weights

        assert grid_weights.sum() == np.prod(self._mesh /
                                             self._mesh_divisors)

        return grid_points, grid_weights
            
    def _set_isotope(self, mass_variances):
        self._mass_variances = np.array(mass_variances, dtype='double')
        self._isotope = Isotope(
            self._mesh,
            mass_variances,
            primitive=self._primitive,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symmetry.get_symmetry_tolerance(),
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.get_lapack_zheev_uplo())
        
    def _set_gv(self, i):
        # Group velocity [num_freqs, 3]
        self._gv[i] = get_group_velocity(
            self._qpoints[i],
            self._dm,
            q_length=self._gv_delta_q,
            symmetry=self._symmetry,
            frequency_factor_to_THz=self._frequency_factor_to_THz)
                
    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print ("======================= Grid point %d (%d/%d) "
                   "=======================" %
                   (gp, i + 1, len(self._grid_points)))
            print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i])
            print "Lifetime cutoff (sec): %-10.3e" % self._cutoff_lifetime
            if self._isotope is not None:
                print "Mass variance parameters:",
                print ("%5.2e " * len(self._mass_variances)) % tuple(
                    self._mass_variances)
                        
