import numpy as np
import phonopy.structure.spglib as spg
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from phonopy.phonon.thermal_properties import mode_cv
from anharmonic.file_IO import write_kappa_to_hdf5, write_triplets
from anharmonic.phonon3.triplets import get_grid_address, reduce_grid_points, get_ir_grid_points, from_coarse_to_dense_grid_points, get_grid_points_in_Brillouin_zone
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class conductivity_RTA:
    def __init__(self,
                 interaction,
                 symmetry,
                 sigmas=[0.1],
                 t_max=1500,
                 t_min=0,
                 t_step=10,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 cutoff_lifetime=1e-4, # in second
                 no_kappa_stars=False,
                 gv_delta_q=1e-4, # finite difference for group veolocity
                 log_level=0,
                 filename=None):
        self._pp = interaction
        self._ise = ImagSelfEnergy(self._pp)

        self._sigmas = sigmas
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._no_kappa_stars = no_kappa_stars
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._filename = filename

        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)
        self._primitive = self._pp.get_primitive()
        self._dynamical_matrix = self._pp.get_dynamical_matrix()
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

        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._set_mesh_numbers(mesh_divisors=mesh_divisors,
                               coarse_mesh_shifts=coarse_mesh_shifts)
        volume = self._primitive.get_volume()
        self._conversion_factor = unit_to_WmK / volume
        self._sum_num_kstar = 0

    def get_mesh_divisors(self):
        return self._mesh_divisors

    def get_mesh_numbers(self):
        return self._mesh

    def get_group_velocities(self):
        return self._gv

    def get_mode_heat_capacities(self):
        return self._cv

    def get_frequencies(self):
        return self._frequencies
        
    def set_grid_points(self, grid_points=None):
        if grid_points is not None: # Specify grid points
            self._grid_address = get_grid_address(self._mesh)
            self._grid_points = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                grid_points,
                coarse_mesh_shifts=self._coarse_mesh_shifts)
        elif self._no_kappa_stars: # All grid points
            self._grid_address = get_grid_address(self._mesh)
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
            self._grid_address = get_grid_address(self._mesh)
            self._grid_weights = coarse_grid_weights

            assert self._grid_weights.sum() == np.prod(self._mesh /
                                                       self._mesh_divisors)

        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        multiplicity = spg.relocate_BZ_grid_address(self._grid_address,
                                                    self._mesh,
                                                    primitive_lattice)

        # rec_lat = np.linalg.inv(self._primitive.get_cell())
        # look_at_grid_address_in_BZ(rec_lat,
        #                            self._mesh,
        #                            self._grid_address,
        #                            self._grid_points)
            
    def get_qpoints(self):
        qpoints = np.array(self._grid_address[self._grid_points] /
                           self._mesh.astype('double'), dtype='double')
        return qpoints
            
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

    def calculate_kappa(self,
                        write_amplitude=False,
                        read_amplitude=False,
                        write_gamma=False):
        self._allocate_values()
        num_band = self._primitive.get_number_of_atoms()
        for i, grid_point in enumerate(self._grid_points):
            self._qpoint = (self._grid_address[grid_point].astype('double') /
                            self._mesh)
            
            if self._log_level:
                print ("===================== Grid point %d (%d/%d) "
                       "=====================" %
                       (grid_point, i + 1, len(self._grid_points)))
                print "q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoint)
                print "Lifetime cutoff (sec): %-10.3e" % self._cutoff_lifetime

            if self._read_gamma:
                self._frequencies[i] = self._get_phonon_c()
            else:
                if self._log_level > 0:
                    print "Number of triplets:",

                self._ise.set_grid_point(grid_point)
                
                if self._log_level > 0:
                    print len(self._pp.get_triplets_at_q()[0])
                    print "Calculating interaction..."
                    
                self._ise.run_interaction()
                self._frequencies[i] = self._ise.get_phonon_at_grid_point()[0]
                self._set_gamma_at_sigmas(i)

            self._set_kappa_at_sigmas(i)

            if write_gamma:
                self._write_gamma(i, grid_point)
                self._write_triplets(grid_point)

    def _allocate_values(self):
        num_freqs = self._primitive.get_number_of_atoms() * 3
        self._kappa = np.zeros((len(self._sigmas),
                                len(self._grid_points),
                                len(self._temperatures),
                                num_freqs,
                                6), dtype='double')
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    len(self._grid_points),
                                    len(self._temperatures),
                                    num_freqs), dtype='double')
        self._gv = np.zeros((len(self._grid_points),
                             num_freqs,
                             3), dtype='double')
        self._cv = np.zeros((len(self._grid_points),
                             len(self._temperatures),
                             num_freqs), dtype='double')

        self._frequencies = np.zeros((len(self._grid_points),
                                      num_freqs), dtype='double')
        
    def _set_gamma_at_sigmas(self, i):
        freqs = self._frequencies[i]
        for j, sigma in enumerate(self._sigmas):
            if self._log_level > 0:
                print "Calculating Gamma with sigma=%s" % sigma
            self._ise.set_sigma(sigma)
            for k, t in enumerate(self._temperatures):
                self._ise.set_temperature(t)
                self._ise.run()
                self._gamma[j, i, k] = self._ise.get_imag_self_energy()
    
    def _set_kappa_at_sigmas(self, i):
        freqs = self._frequencies[i]
        
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
                if self._gamma[j, i, k, l] < 0.5 / self._cutoff_lifetime / THz:
                    continue
                self._kappa[j, i, k, l, :] = (
                    gv_sum2[:, l] * cv[k, l] / (self._gamma[j, i, k, l] * 2) *
                    self._conversion_factor)

    def _get_gv_by_gv(self, i):
        # If q-point is on the boundary,
        # group velocities are also calculated on the other boundaries
        # and the average is taken.
        gv_by_gv = []
        grid_addresses = self._get_grid_address(i)
        for j, grid_address in enumerate(grid_addresses):
            qpoint = grid_address.astype('double') / self._mesh
            # Group velocity [num_freqs, 3]
            gv = get_group_velocity(
                qpoint,
                self._dynamical_matrix,
                q_length=self._gv_delta_q,
                symmetry=self._symmetry,
                frequency_factor_to_THz=self._frequency_factor_to_THz)
            if j == 0:
                self._gv[i] = gv
            rotation_map = self._get_rotation_map_for_star(grid_address)
            gv_by_gv += self._get_gv_by_gv_on_star(gv, rotation_map)

            if self._log_level:
                self._show_log(qpoint,
                               self._frequencies[i],
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

        self._sum_num_kstar += len(gv_by_gv) / len(grid_addresses)

        return np.array(gv_by_gv, dtype='double') / len(grid_addresses)

    def _get_grid_address(self, i):
        orig_address = self._grid_address[self._grid_points[i]]
        numbers = []
        for j in range(3):
            mesh_j = self._mesh[j]
            adrs = orig_address[j]
            if (mesh_j % 2 == 0 and adrs == mesh_j / 2):
                numbers.append([adrs, -adrs])
            else:
                numbers.append([adrs])

        grid_address = []
        for a in numbers[0]:
            for b in numbers[1]:
                for c in numbers[2]:
                    grid_address.append([a, b, c])

        return np.array(grid_address, dtype='intc')

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

    def _get_phonon_c(self):
        import anharmonic._phono3py as phono3c

        dm = self._dynamical_matrix
        svecs, multiplicity = dm.get_shortest_vectors()
        masses = np.array(dm.get_primitive().get_masses(), dtype='double')
        rec_lattice = np.array(np.linalg.inv(dm.get_primitive().get_cell()),
                               dtype='double').copy()
        if dm.is_nac():
            born = dm.get_born_effective_charges()
            nac_factor = dm.get_nac_factor()
            dielectric = dm.get_dielectric_constant()
        else:
            born = None
            nac_factor = 0
            dielectric = None
        uplo = self._pp.get_lapack_zheev_uplo()
        num_freqs = len(masses) * 3
        frequencies = np.zeros(num_freqs, dtype='double')
        eigenvectors = np.zeros((num_freqs, num_freqs), dtype='complex128')

        phono3c.phonon(frequencies,
                       eigenvectors,
                       np.array(self._qpoint, dtype='double'),
                       dm.get_force_constants(),
                       svecs,
                       multiplicity,
                       masses,
                       dm.get_primitive_to_supercell_map(),
                       dm.get_supercell_to_primitive_map(),
                       self._frequency_factor_to_THz,
                       born,
                       dielectric,
                       rec_lattice,
                       None,
                       nac_factor,
                       uplo)
        # dm.set_dynamical_matrix(self._qpoint)
        # dynmat = dm.get_dynamical_matrix()
        # eigvals = np.linalg.eigvalsh(dynmat).real
        # frequencies = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals) *
        #                self._frequency_factor_to_THz)

        return frequencies

    def _show_log(self,
                  q,
                  frequencies,
                  group_velocity,
                  rotation_map):
        print "Frequency, projected group velocity (x, y, z), norm at k-stars",
        if self._gv_delta_q is None:
            print
        else:
            print " (dq=%3.1e)" % self._gv_delta_q
        
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

    def _write_gamma(self, i, grid_point):
        for j, sigma in enumerate(self._sigmas):
            write_kappa_to_hdf5(
                self._gamma[j, i],
                self._temperatures,
                self._mesh,
                frequency=self._frequencies[i],
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
        

def look_at_grid_address_in_BZ(rec_lat, mesh, grid_address, grid_points):
    bz_adrs = get_grid_points_in_Brillouin_zone(rec_lat,
                                                mesh,
                                                grid_address,
                                                grid_points,
                                                with_boundary=False)

    for gp, bza in zip(grid_points, bz_adrs):
        q = np.dot(rec_lat, grid_address[gp]) / mesh
        print gp, grid_address[gp], q, np.linalg.norm(q)
        for a in bza:
            q = np.dot(rec_lat, a) / mesh
            print "-->", a, q, np.linalg.norm(q)
        print
