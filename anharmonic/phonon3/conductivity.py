import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.units import EV, THz, Angstrom
from anharmonic.phonon3.triplets import (get_grid_address, reduce_grid_points,
                                         get_ir_grid_points,
                                         from_coarse_to_dense_grid_points)
from anharmonic.other.isotope import Isotope

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class Conductivity(object):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 is_isotope=False,
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 boundary_mfp=None, # in micrometre
                 is_kappa_star=True,
                 gv_delta_q=None, # finite difference for group veolocity
                 log_level=0):
        if sigmas is None:
            self._sigmas = []
        else:
            self._sigmas = sigmas
        self._pp = interaction
        self._collision = None # has to be set derived class

        self._temperatures = temperatures
        self._is_kappa_star = is_kappa_star
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._primitive = self._pp.get_primitive()
        self._dm = self._pp.get_dynamical_matrix()
        self._frequency_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._boundary_mfp = boundary_mfp

        self._symmetry = symmetry

        if not self._is_kappa_star:
            self._point_operations = np.array([np.eye(3, dtype='intc')],
                                              dtype='intc')
        else:
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

        self._kappa = None
        self._mode_kappa = None
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
        self._is_isotope = is_isotope
        if mass_variances is not None:
            self._is_isotope = True
        if self._is_isotope:
            self._set_isotope(mass_variances)

        self._grid_point_count = None
        self._set_grid_properties(grid_points)

    def __iter__(self):
        return self

    def __next__(self):
        if self._grid_point_count == len(self._grid_points):
            if self._log_level:
                print("=================== End of collection of collisions "
                      "===================")
            raise StopIteration
        else:
            self._run_at_grid_point()
            self._grid_point_count += 1
            return self._grid_point_count - 1

    def next(self):
        return self.__next__()

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

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures
        self._allocate_values()

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

    def get_kappa(self):
        return self._kappa

    def get_mode_kappa(self):
        return self._mode_kappa

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
        elif not self._is_kappa_star: # All grid points
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
        self._pp.set_phonons(self._grid_points)
        self._frequencies = self._pp.get_phonons()[0]

    def _set_gamma_isotope_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating Gamma of ph-isotope with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)
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
                    print(("Mesh number %d for the " +
                           ["first", "second", "third"][i] +
                           " axis is not dividable by divisor %d.") % (m, n))
            self._mesh_divisors = np.array(self._mesh_divisors, dtype='intc')
            if coarse_mesh_shifts is None:
                self._coarse_mesh_shifts = [False, False, False]
            else:
                self._coarse_mesh_shifts = coarse_mesh_shifts
            for i in range(3):
                if (self._coarse_mesh_shifts[i] and
                    (self._mesh_divisors[i] % 2 != 0)):
                    print("Coarse grid along " +
                          ["first", "second", "third"][i] +
                          " axis can not be shifted. Set False.")
                    self._coarse_mesh_shifts[i] = False

        self._coarse_mesh = self._mesh // self._mesh_divisors

        if self._log_level:
            print("Lifetime sampling mesh: [ %d %d %d ]" %
                  tuple(self._mesh // self._mesh_divisors))

    def _get_ir_grid_points(self):
        if self._coarse_mesh_shifts is None:
            mesh_shifts = [False, False, False]
        else:
            mesh_shifts = self._coarse_mesh_shifts
        (coarse_grid_points,
         coarse_grid_weights,
         coarse_grid_address, _) = get_ir_grid_points(
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

        assert grid_weights.sum() == np.prod(self._mesh // self._mesh_divisors)

        return grid_points, grid_weights

    def _set_isotope(self, mass_variances):
        if mass_variances is True:
            mv = None
        else:
            mv = mass_variances
        self._isotope = Isotope(
            self._mesh,
            self._primitive,
            mass_variances=mv,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symmetry.get_symmetry_tolerance(),
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.get_lapack_zheev_uplo())
        self._mass_variances = self._isotope.get_mass_variances()

    def _set_gv(self, i):
        # Group velocity [num_freqs, 3]
        gv = self._get_gv(self._qpoints[i])
        self._gv[i] = gv[self._pp.get_band_indices(), :]

    def _get_gv(self, q):
        return get_group_velocity(
            q,
            self._dm,
            q_length=self._gv_delta_q,
            symmetry=self._symmetry,
            frequency_factor_to_THz=self._frequency_factor_to_THz)

    def _get_main_diagonal(self, i, j, k):
        num_band = self._primitive.get_number_of_atoms() * 3
        main_diagonal = self._gamma[j, k, i].copy()
        if self._gamma_iso is not None:
            main_diagonal += self._gamma_iso[j, i]
        if self._boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)

        # if self._boundary_mfp is not None:
        #     for l in range(num_band):
        #         # Acoustic modes at Gamma are avoided.
        #         if i == 0 and l < 3:
        #             continue
        #         gv_norm = np.linalg.norm(self._gv[i, l])
        #         mean_free_path = (gv_norm * Angstrom * 1e6 /
        #                           (4 * np.pi * main_diagonal[l]))
        #         if mean_free_path > self._boundary_mfp:
        #             main_diagonal[l] = (
        #                 gv_norm / (4 * np.pi * self._boundary_mfp))

        return main_diagonal

    def _get_boundary_scattering(self, i):
        num_band = self._primitive.get_number_of_atoms() * 3
        g_boundary = np.zeros(num_band, dtype='double')
        for l in range(num_band):
            g_boundary[l] = (np.linalg.norm(self._gv[i, l]) * Angstrom * 1e6 /
                             (4 * np.pi * self._boundary_mfp))
        return g_boundary

    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print("======================= Grid point %d (%d/%d) "
                  "=======================" %
                  (gp, i + 1, len(self._grid_points)))
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i]))
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print("Boundary mean free path (millimetre): %.3f" %
                          (self._boundary_mfp / 1000.0))
                else:
                    print("Boundary mean free path (micrometre): %.5f" %
                          self._boundary_mfp)
            if self._is_isotope:
                print(("Mass variance parameters: " +
                       "%5.2e " * len(self._mass_variances)) %
                      tuple(self._mass_variances))
