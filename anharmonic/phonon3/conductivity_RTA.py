import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.units import THzToEv, THz
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from anharmonic.file_IO import write_kappa_to_hdf5, write_triplets, read_gamma_from_hdf5, write_grid_address
from anharmonic.phonon3.conductivity import Conductivity

def get_thermal_conductivity(
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
        print "-------------------- Lattice thermal conducitivity (RTA) --------------------"
    br = Conductivity_RTA(interaction,
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

    if read_gamma:
        _set_gamma_from_file(br, filename=input_filename)
        
    for i in br:
        if write_gamma:
            _write_gamma(br, interaction, i, filename=output_filename)
        if log_level > 1:
            _write_triplets(interaction)

    if grid_points is None:
        _write_kappa(br, filename=output_filename, log_level=log_level)

    return br
        
def _write_gamma(br, interaction, i, filename=None):
    grid_points = br.get_grid_points()
    group_velocities = br.get_group_velocities()
    mode_heat_capacities = br.get_mode_heat_capacities()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    temperatures = br.get_temperatures()
    gamma = br.get_gamma()
    sigmas = br.get_sigmas()
    
    gp = grid_points[i]
    gv = group_velocities[i]
    mode_cv = mode_heat_capacities[i]
    frequencies = interaction.get_phonons()[0][gp]
    # kappa = br.get_kappa()
    
    for j, sigma in enumerate(sigmas):
        write_kappa_to_hdf5(gamma[j, i],
                            temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            heat_capacity=mode_cv,
                            kappa=None, # kappa[j, i],
                            mesh_divisors=mesh_divisors,
                            grid_point=gp,
                            sigma=sigma,
                            filename=filename)

def _write_triplets(interaction, filename=None):
    triplets, weights = interaction.get_triplets_at_q()
    grid_address = interaction.get_grid_address()
    mesh = interaction.get_mesh_numbers()
    write_triplets(triplets,
                   weights,
                   mesh,
                   grid_address,
                   grid_point=triplets[0, 0],
                   filename=filename)
    write_grid_address(grid_address, mesh, filename=filename)

def _write_kappa(br, filename=None, log_level=0):
    temperatures = br.get_temperatures()
    sigmas = br.get_sigmas()
    gamma = br.get_gamma()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    frequencies = br.get_frequencies()
    gv = br.get_group_velocities()
    mode_cv = br.get_mode_heat_capacities()
    qpoints = br.get_qpoints()
    weights = br.get_grid_weights()
    num_sampling_points = br.get_number_of_sampling_points()
    
    assert num_sampling_points > 0
    
    kappa = br.get_kappa() / num_sampling_points
    
    for i, sigma in enumerate(sigmas):
        kappa_at_sigma = kappa[i].sum(axis=2).sum(axis=0)
        if log_level:
            print "----------- Thermal conductivity (W/m-k)",
            if sigma:
                print "for sigma=%s -----------" % sigma
            else:
                print "with tetrahedron method -----------"
            print ("#%6s     " + " %-9s" * 6) % ("T(K)", "xx", "yy", "zz",
                                                "yz", "xz", "xy")
            for t, k in zip(temperatures, kappa_at_sigma):
                print ("%7.1f" + " %9.3f" * 6) % ((t,) + tuple(k))
            print
        write_kappa_to_hdf5(gamma[i],
                            temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            heat_capacity=mode_cv,
                            kappa=kappa_at_sigma,
                            qpoint=qpoints,
                            weight=weights,
                            mesh_divisors=mesh_divisors,
                            sigma=sigma,
                            filename=filename)
               
def _set_gamma_from_file(br, filename=None):
    sigmas = br.get_sigmas()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    grid_points = br.get_grid_points()
    
    gamma = []
    for sigma in sigmas:
        gamma_at_sigma = read_gamma_from_hdf5(
            mesh,
            mesh_divisors=mesh_divisors,
            sigma=sigma,
            filename=filename)
        if gamma_at_sigma is False:
            gamma_at_sigma = []
            for i, gp in enumerate(grid_points):
                gamma_gp = read_gamma_from_hdf5(
                    mesh,
                    mesh_divisors=mesh_divisors,
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename)
                if gamma_gp is False:
                    print "Gamma at grid point %d doesn't exist." % gp
                gamma_at_sigma.append(gamma_gp)
        gamma.append(gamma_at_sigma)
    br.set_gamma(np.array(gamma, dtype='double', order='C'))

class Conductivity_RTA(Conductivity):
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
        self._ise = None
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

    def get_mode_heat_capacities(self):
        return self._cv

    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]
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
        for j in range(len(self._sigmas)):
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

        return np.array(gv_by_gv, dtype='double', order='C')

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
                    finite_t, get_mode_cv(
                        np.where(finite_t, self._temperatures, 10000),
                        f * THzToEv), 0)
        return cv

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
    
