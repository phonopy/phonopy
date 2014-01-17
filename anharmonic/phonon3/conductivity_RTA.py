import numpy as np
import phonopy.structure.spglib as spg
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from phonopy.phonon.thermal_properties import mode_cv
from anharmonic.file_IO import write_kappa_to_hdf5, write_triplets
from anharmonic.phonon3.triplets import get_grid_address, reduce_grid_points, get_ir_grid_points, from_coarse_to_dense_grid_points
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.interaction import set_phonon_c
from anharmonic.other.isotope import Isotope

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class conductivity_RTA:
    def __init__(self,
                 interaction,
                 symmetry,
                 temperatures=[300.0],
                 sigmas=[0.1],
                 tetrahedron_method=False,
                 mass_variances=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 cutoff_lifetime=1e-4, # in second
                 no_kappa_stars=False,
                 gv_delta_q=1e-4, # finite difference for group veolocity
                 log_level=0,
                 filename=None):
        self._pp = interaction
        self._ise = ImagSelfEnergy(self._pp)
        self._temperatures = temperatures
        self._sigmas = sigmas
        self._no_kappa_stars = no_kappa_stars
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._filename = filename
        self._primitive = self._pp.get_primitive()
        self._dm = self._pp.get_dynamical_matrix()
        self._frequency_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._cutoff_lifetime = cutoff_lifetime

        self._symmetry = symmetry
        if self._no_kappa_stars:
            self._point_operations = [np.identity(3, dtype='intc')]
        else:
            self._point_operations = symmetry.get_reciprocal_operations()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = [similarity_transformation(rec_lat, r)
                                     for r in self._point_operations]
        
        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None

        self._gamma = None
        self._read_gamma = False
        self._frequencies = None
        self._cv = None
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
        self._sum_num_kstar = 0

        self._isotope = None
        self._mass_variances = None
        if mass_variances is not None:
            self._set_isotope(mass_variances)

    def run(self,
            write_amplitude=False,
            read_amplitude=False,
            write_gamma=False):
        self._allocate_values()
        self._pp.set_phonon(self._grid_points)
        self._frequencies = self._pp.get_phonons()[0]

        for i, grid_point in enumerate(self._grid_points):
            self._show_log_header(i)            
            if not self._read_gamma:
                self._ise.set_grid_point(grid_point)
                
                if self._log_level:
                    print "Number of triplets:",
                    print len(self._pp.get_triplets_at_q()[0])
                    print "Calculating interaction..."
                    
                self._ise.run_interaction()
                self._set_gamma_at_sigmas(i)

            if self._isotope is not None:
                self._set_gamma_isotope_at_sigmas(i)

            self._set_kappa_at_sigmas(i)

            if write_gamma:
                self._write_gamma(i)
                if self._log_level > 1:
                    self._write_triplets(grid_point)

    def set_grid_points(self, grid_points=None):
        self._grid_address = self._pp.get_grid_address()

        if grid_points is not None: # Specify grid points
            self._grid_points = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                grid_points,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
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
        else: # Automatic sampling
            if self._coarse_mesh_shifts is None:
                mesh_shifts = [False, False, False]
            else:
                mesh_shifts = self._coarse_mesh_shifts
            (coarse_grid_points,
             coarse_grid_weights,
             coarse_grid_address) = get_ir_grid_points(
                self._coarse_mesh,
                self._primitive,
                mesh_shifts=mesh_shifts)
            self._grid_points = from_coarse_to_dense_grid_points(
                self._mesh,
                self._mesh_divisors,
                coarse_grid_points,
                coarse_grid_address,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
            self._grid_weights = coarse_grid_weights

            assert self._grid_weights.sum() == np.prod(self._mesh /
                                                       self._mesh_divisors)

        self._qpoints = np.array(self._grid_address[self._grid_points] /
                                 self._mesh.astype('double'), dtype='double')

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_group_velocities(self):
        return self._gv

    def get_mode_heat_capacities(self):
        return self._cv

    def get_frequencies(self):
        return self._frequencies[self._grid_points]
        
    def get_qpoints(self):
        return self._qpoints
            
    def get_grid_points(self):
        return self._grid_points

    def get_grid_weights(self):
        return self._grid_weights
            
    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures

    def set_gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True

    def get_gamma(self):
        return self._gamma
        
    def get_kappa(self):
        return self._kappa / self._sum_num_kstar

    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print ("===================== Grid point %d (%d/%d) "
                   "=====================" %
                   (gp, i + 1, len(self._grid_points)))
            print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i])
            print "Lifetime cutoff (sec): %-10.3e" % self._cutoff_lifetime
            if self._isotope is not None:
                print "Mass variance parameters:",
                print ("%5.2e " * len(self._mass_variances)) % tuple(
                    self._mass_variances)
                        
    def _allocate_values(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        self._kappa = np.zeros((len(self._sigmas),
                                num_grid_points,
                                len(self._temperatures),
                                num_band,
                                6), dtype='double')
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    num_grid_points,
                                    len(self._temperatures),
                                    num_band), dtype='double')
        self._gv = np.zeros((num_grid_points,
                             num_band,
                             3), dtype='double')
        self._cv = np.zeros((num_grid_points,
                             len(self._temperatures),
                             num_band), dtype='double')
        self._gamma_iso = np.zeros((len(self._sigmas),
                                    num_grid_points,
                                    num_band), dtype='double')
        
    def _set_gamma_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating Gamma of ph-ph with sigma=%s" % sigma
            self._ise.set_sigma(sigma)
            for k, t in enumerate(self._temperatures):
                self._ise.set_temperature(t)
                self._ise.run()
                self._gamma[j, i, k] = self._ise.get_imag_self_energy()
                
    def _set_gamma_isotope_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                print "Calculating Gamma of ph-isotope with sigma=%s" % sigma
            pp_freqs, pp_eigvecs, pp_phonon_done = self._pp.get_phonons()
            self._isotope.set_phonons(pp_freqs,
                                      pp_eigvecs,
                                      pp_phonon_done,
                                      dm=self._dm)
            self._isotope.run(i)
            self._gamma_iso[j, i] = self._isotope.get_gamma()
    
    def _set_kappa_at_sigmas(self, i):
        freqs = self._frequencies[self._grid_points[i]]
        
        # Heat capacity [num_temps, num_freqs]
        cv = self._get_cv(freqs)
        self._cv[i] = cv

        # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
        gv_by_gv_tensor = self._get_gv_by_gv(i)

        # Sum all vxv at k*
        gv_sum2 = np.zeros((6, len(freqs)), dtype='double')
        for j, vxv in enumerate(
            ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            gv_sum2[j] = gv_by_gv_tensor[:, :, vxv[0], vxv[1]].sum(axis=0)

        # Kappa
        for j, sigma in enumerate(self._sigmas):
            for k, l in list(np.ndindex(len(self._temperatures), len(freqs))):
                g_phph = self._gamma[j, i, k, l]
                if g_phph < 0.5 / self._cutoff_lifetime / THz:
                    continue
                if self._isotope is None:
                    g_sum = g_phph
                else:
                    g_iso = self._gamma_iso[j, i, l]
                    g_sum = g_phph + g_iso
                self._kappa[j, i, k, l, :] = (
                    gv_sum2[:, l] * cv[k, l] / (g_sum * 2) *
                    self._conversion_factor)

    def _get_gv_by_gv(self, i):
        gp = self._grid_points[i]
        grid_address = self._grid_address[gp]
        if self._no_kappa_stars:
            gv_by_gv_tmp = []
            rotation_map = [0]
            for address in self._grid_address:
                if ((grid_address - address) % self._mesh == 0).all():
                    qpoint = address.astype('double') / self._mesh
                    gv = get_group_velocity(
                        qpoint,
                        self._dm,
                        q_length=self._gv_delta_q,
                        symmetry=self._symmetry,
                        frequency_factor_to_THz=self._frequency_factor_to_THz)
                    self._gv[i] = gv
                    gv_by_gv_tmp.append(
                        self._get_gv_by_gv_on_star(gv, rotation_map)[0])

                    if self._log_level:
                        self._show_log(qpoint,
                                       self._frequencies[gp],
                                       gv,
                                       rotation_map)

            gv_by_gv = [np.sum(gv_by_gv_tmp, axis=0) / len(gv_by_gv_tmp)]
        else:
            # Group velocity [num_freqs, 3]
            gv = get_group_velocity(
                self._qpoints[i],
                self._dm,
                q_length=self._gv_delta_q,
                symmetry=self._symmetry,
                frequency_factor_to_THz=self._frequency_factor_to_THz)
            self._gv[i] = gv
            rotation_map = self._get_rotation_map_for_star(grid_address)
            gv_by_gv = self._get_gv_by_gv_on_star(gv, rotation_map)

            if self._log_level:
                self._show_log(self._qpoints[i],
                               self._frequencies[gp],
                               gv,
                               rotation_map)

        # check if the number of rotations is correct.
        if self._grid_weights is not None:
            if len(set(rotation_map)) != self._grid_weights[i]:
                if self._log_level:
                    print "*" * 33  + "Warning" + "*" * 33
                    print (" Number of elements in k* is unequal "
                           "to number of equivalent grid-points.")
                    print "*" * 73
            # assert len(rotations) == self._grid_weights[i], \
            #     "Num rotations %d, weight %d" % (
            #     len(rotations), self._grid_weights[i])

        self._sum_num_kstar += len(gv_by_gv)

        return np.array(gv_by_gv, dtype='double')

    def _get_gv_by_gv_on_star(self, group_velocity, rotation_map):
        gv2_tensor = []
        for j in np.unique(rotation_map):
            gv_by_gv = np.zeros((len(group_velocity), 3, 3), dtype='double')
            multiplicity = 0
            for k, rot_c in enumerate(self._rotations_cartesian):
                if rotation_map[k] == j:
                    gvs_rot = np.dot(rot_c, group_velocity.T).T
                    gv_by_gv += [np.outer(gv, gv) for gv in gvs_rot]
                    multiplicity += 1
            gv_by_gv /= multiplicity
            gv2_tensor.append(gv_by_gv)

        return gv2_tensor
    
    def _get_rotation_map_for_star(self, orig_address):
        rot_addresses = [np.dot(rot, orig_address)
                         for rot in self._point_operations]
        rotation_map = []
        for rot_adrs in rot_addresses:
            for i, rot_adrs_comp in enumerate(rot_addresses):
                if ((rot_adrs - rot_adrs_comp) % self._mesh == 0).all():
                    rotation_map.append(i)
                    break

        return rotation_map

    def _get_cv(self, freqs):
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype='double')
        # T/freq has to be large enough to avoid divergence.
        # Otherwise just set 0.
        for i, f in enumerate(freqs):
            finite_t = (self._temperatures > f / 100)
            if f > self._cutoff_frequency:
                cv[:, i] = np.where(
                    finite_t, mode_cv(
                        np.where(finite_t, self._temperatures, 10000),
                        f * THzToEv), 0)
        return cv

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

    def _set_isotope(self, mass_variances):
        self._mass_variances = np.array(mass_variances, dtype='double')
        self._isotope = Isotope(
            self._mesh,
            mass_variances,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symmetry.get_symmetry_tolerance(),
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.get_lapack_zheev_uplo())

    def _show_log(self,
                  q,
                  frequencies,
                  group_velocity,
                  rotation_map):
        print "Frequency, projected group velocity (x, y, z), group velocity norm",
        if self._gv_delta_q is None:
            print
        else:
            print " (dq=%3.1e)" % self._gv_delta_q

        if self._log_level > 1:
            for i, j in enumerate(np.unique(rotation_map)):
                for k, (rot, rot_c) in enumerate(zip(self._point_operations,
                                                     self._rotations_cartesian)):
                    if rotation_map[k] != j:
                        continue
    
                    print " k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) +
                                                           tuple(np.dot(rot, q)))
                    for f, v in zip(frequencies,
                                    np.dot(rot_c, group_velocity.T).T):
                        print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                            f, v[0], v[1], v[2], np.linalg.norm(v))
            print
        else:
            num_ks = len(np.unique(rotation_map))
            if num_ks == 1:
                print " 1 orbit",
            else:
                print " %d orbits" % num_ks,
            print "of k* at (%5.2f %5.2f %5.2f)" % tuple(q)
            for f, v in zip(frequencies, group_velocity):
                print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                    f, v[0], v[1], v[2], np.linalg.norm(v))
    
    def _write_gamma(self, i):
        gp = self._grid_points[i]
        for j, sigma in enumerate(self._sigmas):
            write_kappa_to_hdf5(
                self._gamma[j, i],
                self._temperatures,
                self._mesh,
                frequency=self._frequencies[gp],
                group_velocity=self._gv[i],
                heat_capacity=self._cv[i],
                kappa=self._kappa[j, i],
                mesh_divisors=self._mesh_divisors,
                grid_point=grid_point,
                sigma=sigma,
                filename=self._filename)

    def _write_triplets(self, grid_point):
        triplets, weights = self._pp.get_triplets_at_q()
        grid_address = self._pp.get_grid_address()
        write_triplets(triplets,
                       weights,
                       self._mesh,
                       grid_address,
                       grid_point=grid_point,
                       filename=self._filename)
