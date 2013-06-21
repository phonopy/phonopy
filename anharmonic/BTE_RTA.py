import numpy as np
import phonopy.structure.spglib as spg
from anharmonic.im_self_energy import get_gamma
from phonopy.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from phonopy.phonon.thermal_properties import mode_cv
from anharmonic.file_IO import write_kappa_to_hdf5
from anharmonic.triplets import get_grid_address, reduce_grid_points

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi)) # 2pi comes from definition of lifetime.

class BTE_RTA:
    def __init__(self,
                 interaction_strength,
                 sigmas=[0.1],
                 t_max=1500,
                 t_min=0,
                 t_step=10,
                 max_freepath=0.01, # in meter
                 mesh_divisors=None,
                 coase_mesh_shift=None,
                 is_mesh_shift=None,
                 no_kappa_stars=False,
                 gamma_option=0,
                 log_level=0,
                 filename=None):
        self._pp = interaction_strength
        self._sigmas = sigmas
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._max_freepath = max_freepath
        self._no_kappa_stars = no_kappa_stars
        self._gamma_option = gamma_option
        self._log_level = log_level
        self._filename = filename

        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)
        self._primitive = self._pp.get_primitive()
        self._cutoff_frequency = self._pp.get_cutoff_frequency()
        self._freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        self._reciprocal_lattice = np.linalg.inv(
            self._primitive.get_cell()) # a*, b*, c* are column vectors.
        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None

        self._point_operations = get_pointgroup_operations(
            self._pp.get_symmetry().get_pointgroup_operations())
        self._gamma = None
        self._read_gamma = False
        self._frequencies = None
        self._cv = None
        self._gv = None

        self._mesh = None
        self._mesh_divisors = None
        self._coase_mesh_shift = None
        self._set_mesh_numbers(mesh_divisors=mesh_divisors,
                               coase_mesh_shift=coase_mesh_shift)
        volume = self._primitive.get_volume()
        num_grid = np.prod(self._mesh / self._mesh_divisors)
        self._conversion_factor = unit_to_WmK / volume / num_grid

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
            dense_grid_points = grid_points
            self._grid_points = reduce_grid_points(self._mesh_divisors,
                                                   self._grid_address,
                                                   dense_grid_points)
        elif self._no_kappa_stars: # All grid points
            self._grid_address = get_grid_address(self._mesh)
            dense_grid_points = range(np.prod(self._mesh))
            self._grid_points = reduce_grid_points(self._mesh_divisors,
                                                   self._grid_address,
                                                   dense_grid_points)
        else: # Automatic sampling
            (grid_mapping_table,
             self._grid_address) = spg.get_ir_reciprocal_mesh(self._mesh,
                                                              self._primitive)
            dense_grid_points = np.unique(grid_mapping_table)
            weights = np.zeros_like(grid_mapping_table)
            for g in grid_mapping_table:
                weights[g] += 1
            dense_grid_weights = weights[dense_grid_points]

            self._grid_points, self._grid_weights = reduce_grid_points(
                self._mesh_divisors,
                self._grid_address,
                dense_grid_points,
                dense_grid_weights)

            assert self._grid_weights.sum() == np.prod(self._mesh /
                                                       self._mesh_divisors)

    def get_qpoints(self):
        qpoints = np.double([self._grid_address[gp].astype(float) / self._mesh
                             for gp in self._grid_points])
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
        return self._kappa
        
    def calculate_kappa(self,
                        write_amplitude=False,
                        read_amplitude=False,
                        write_gamma=False):
        self._allocate_values()
        for i in range(len(self._grid_points)):
            grid_point = self._grid_points[i]
            if self._log_level:
                print ("================= %d/%d =================" %
                       (i + 1, len(self._grid_points)))

            if self._read_gamma:
                self._pp.set_qpoint(
                    self._grid_address[grid_point].astype(float) / self._mesh)
                self._pp.set_harmonic_phonons()
            else:
                self._pp.set_triplets_at_q(grid_point)
                self._pp.set_interaction_strength(
                    write_amplitude=write_amplitude,
                    read_amplitude=read_amplitude)
                self._pp.set_harmonic_phonons()
                self._set_gamma_at_sigmas(i)

            self._set_kappa_at_sigmas(i)
            
            if write_gamma:
                for j, sigma in enumerate(self._sigmas):
                    write_kappa_to_hdf5(self._gamma[j, i],
                                        self._temperatures,
                                        self._mesh,
                                        frequency=self._pp.get_frequencies(),
                                        group_velocity=self._gv[i],
                                        heat_capacity=self._cv[i],
                                        mesh_divisors=self._mesh_divisors,
                                        grid_point=grid_point,
                                        sigma=sigma,
                                        filename=self._filename)

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
        for j, sigma in enumerate(self._sigmas):
            if self._log_level > 0:
                print "Calculating Gamma using sigma=%s" % sigma
            self._gamma[j, i] = self._get_gamma(sigma)
    
    def _set_kappa_at_sigmas(self, i):
        freqs = self._pp.get_frequencies()
        self._frequencies[i] = freqs
        
        # Group velocity [num_freqs, 3]
        gv = get_group_velocity(
            self._pp.get_qpoint(),
            self._pp.get_dynamical_matrix(),
            self._reciprocal_lattice,
            eigenvectors=self._pp.get_eigenvectors(),
            frequencies=freqs)
        self._gv[i] = gv
        
        # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
        gv_by_gv_tensor = self._get_gv_by_gv(gv, i)

        # Sum all vxv at k*
        gv_sum2 = np.zeros((6, len(freqs)), dtype='double')
        for j, vxv in enumerate(([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            gv_sum2[j] = gv_by_gv_tensor[:, :, vxv[0], vxv[1]].sum(axis=0)

        # Heat capacity [num_temps, num_freqs]
        cv = self._get_cv()
        self._cv[i] = cv

        # Kappa
        for j, sigma in enumerate(self._sigmas):
            for k in range(len(self._temperatures)):
                for l in range(len(freqs)):
                    if self._gamma[j, i, k, l] > 0:
                        self._kappa[j, i, k, l, :] = (
                            gv_sum2[:, l] * cv[k, l] /
                            (self._gamma[j, i, k, l] * 2) *
                            self._conversion_factor)

    def _get_gv_by_gv(self, gv, index):
        grid_point = self._grid_points[index]

        # Sum group velocities at symmetrically equivalent q-points
        if self._no_kappa_stars: # [1, 3, 3] 
            rotations = [np.eye(3, dtype=int)]
        else: # [num_k*, 3, 3]
            rotations = self._get_rotations_for_star(grid_point)

            # check if the number of rotations is correct.
            if self._grid_weights is not None:
                assert len(rotations) == self._grid_weights[index], \
                    "Num rotations %d, weight %d" % (
                    len(rotations), self._grid_weights[index])
            
        gv2_tensor = []
        inv_rec_lat = self._primitive.get_cell()
        rec_lat = np.linalg.inv(inv_rec_lat)
        rotations_cartesian = [np.dot(rec_lat, np.dot(r, inv_rec_lat))
                               for r in rotations]
        for rot in rotations_cartesian:
            gv2_tensor.append([np.outer(gv_rot_band_index, gv_rot_band_index)
                               for gv_rot_band_index in np.dot(rot, gv.T).T])

        if self._log_level:
            self._show_log(grid_point, gv, rotations_cartesian, rotations)
            print

        return np.array(gv2_tensor)
    
    def _get_cv(self):
        freqs = self._pp.get_frequencies()
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype='double')
        for i, t in enumerate(self._temperatures):
            if t > 0:
                for j, f in enumerate(freqs):
                    if f > self._cutoff_frequency:
                        cv[i, j] = mode_cv(
                            t, f * THzToEv / self._freq_conv_factor) # eV/K

        return cv

    def _get_gamma(self, sigma):
        unit_conversion = self._pp.get_unit_conversion_factor()
        freqs = self._pp.get_frequencies()

        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()

        gamma = -1 * np.ones((len(self._temperatures), len(freqs)), dtype='double')

        for i, t in enumerate(self._temperatures):
            for j, f in enumerate(freqs):
                if f > self._cutoff_frequency:
                    g = get_gamma(
                        amplitude_at_q,
                        np.array([f], dtype='double'),
                        weights_at_q,
                        frequencies_at_q,
                        j,
                        t,
                        sigma,
                        self._freq_conv_factor,
                        cutoff_frequency=self._cutoff_frequency,
                        gamma_option=self._gamma_option
                        )[0] * unit_conversion # THz in default
                    gamma[i, j] = g
        return gamma

    def _get_rotations_for_star(self, grid_point):
        orig_address = self._grid_address[grid_point]
        orbits = []
        rotations = []
        for rot in self._point_operations:
            rot_address = np.dot(rot, orig_address) % self._mesh
            in_orbits = False
            for orbit in orbits:
                if (rot_address == orbit).all():
                    in_orbits = True
                    break
            if not in_orbits:
                orbits.append(rot_address)
                rotations.append(rot)

        return rotations

    def _set_mesh_numbers(self, mesh_divisors=None, coase_mesh_shift=None):
        self._mesh = self._pp.get_mesh_numbers()

        if mesh_divisors is None:
            self._mesh_divisors = np.intc([1, 1, 1])
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
            self._mesh_divisors = np.intc(self._mesh_divisors)
            if coase_mesh_shift is None:
                self._coase_mesh_shift = [False, False, False]
            else:
                self._coase_mesh_shift = coase_mesh_shift
            for i in range(3):
                if (self._coase_mesh_shift[i] and
                    (self._mesh_divisors[i] % 2 != 0)):
                    print ("Coase grid along " +
                           ["first", "second", "third"][i] + 
                           " axis can not be shifted. Set False.")
                    self._coase_mesh_shift[i] = False
        if self._log_level:
            print ("Lifetime sampling mesh: [ %d %d %d ]" %
                   tuple(self._mesh / self._mesh_divisors))

    def _sort_normal_umklapp(self):
        normal_a = []
        normal_w = []
        normal_f = []
        umklapp_a = []
        umklapp_w = []
        umklapp_f = []
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        triplets_at_q = self._pp.get_triplets_at_q()
        for a, w, f, g3 in zip(amplitude_at_q,
                               weights_at_q,
                               frequencies_at_q,
                               triplets_at_q):
            sum_q = (self._grid_address[g3[0]] +
                     self._grid_address[g3[1]] +
                     self._grid_address[g3[2]])
            if (sum_q == 0).all():
                normal_a.append(a)
                normal_w.append(w)
                normal_f.append(f)
                print "N",
            else:
                umklapp_a.append(a)
                umklapp_w.append(w)
                umklapp_f.append(f)
                print "U",

        print
        return ((np.double(normal_a),
                 np.intc(normal_w),
                 np.double(normal_f)),
                (np.double(umklapp_a),
                 np.intc(umklapp_w),
                 np.double(umklapp_f)))
    
    def _show_log(self,
                  grid_point,
                  group_velocity,
                  rotations_cartesian,
                  rotations):
        print "----- Partial kappa at grid address %d -----" % grid_point
        print "Frequency, projected group velocity (x, y, z) at k* (k-star)"
        q = self._grid_address[grid_point].astype(float) / self._mesh
        for i, (rotc, rot) in enumerate(zip(rotations_cartesian, rotations)):
            q_rot = np.dot(rot, q)
            q_rot -= np.rint(q_rot)
            print " k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) + tuple(q_rot))
            for f, v in zip(self._pp.get_frequencies(),
                            np.dot(rot, group_velocity.T).T):
                print "%8.3f   (%8.3f %8.3f %8.3f)" % ((f,) + tuple(v))

        
def get_pointgroup_operations(point_operations_real):
    exist_r_inv = False
    for rot in point_operations_real:
        if (rot + np.eye(3, dtype='intc') == 0).all():
            exist_r_inv = True
            break

    point_operations = [rot.T for rot in point_operations_real]
    
    if not exist_r_inv:
        point_operations += [-rot.T for rot in point_operations_real]
        
    return np.array(point_operations)

            
        
if __name__ == '__main__':
    import sys
    import h5py

    def read_kappa(filename):
        vals = []
        for line in open(filename):
            if line.strip()[0] == '#':
                continue
            vals.append([float(x) for x in line.split()])
        vals = np.array(vals)
        return vals[:, 0], vals[:, 1]

    def sum_partial_kappa(filenames):
        temps, kappa = read_kappa(filenames[0])
        sum_kappa = kappa.copy()
        for filename in filenames[1:]:
            temps, kappa = parse_kappa(filename)
            sum_kappa += kappa
        return temps, sum_kappa
    
    def sum_partial_kappa_hdf5(filenames):
        f = h5py.File(filenames[0], 'r')
        kappas = f['kappas'][:]
        temps = f['temperatures'][:]
        for filename in filenames[1:]:
            f = h5py.File(filename, 'r')
            kappas += f['kappas'][:]
        return temps, kappas

    temps, kappa = sum_partial_kappa(sys.argv[1:])
    for t, k in zip(temps, kappa):
        print "%8.2f %.5f" % (t, k)
    # temps, kappa = sum_partial_kappa_hdf5(sys.argv[1:])
    # for t, k in zip(temps, kappa.sum(axis=1)):
    #     print "%8.2f %.5f" % (t, k)


