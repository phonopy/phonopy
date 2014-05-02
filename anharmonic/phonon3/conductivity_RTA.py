import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.units import THzToEv, THz
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from anharmonic.file_IO import write_kappa_to_hdf5, write_triplets, read_gamma_from_hdf5, write_grid_address
from anharmonic.phonon3.conductivity import Conductivity
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.triplets import get_grid_points_by_rotations

def get_thermal_conductivity_RTA(
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
                          grid_points=grid_points,
                          temperatures=temperatures,
                          sigmas=sigmas,
                          mass_variances=mass_variances,
                          mesh_divisors=mesh_divisors,
                          coarse_mesh_shifts=coarse_mesh_shifts,
                          cutoff_lifetime=cutoff_lifetime,
                          no_kappa_stars=no_kappa_stars,
                          gv_delta_q=gv_delta_q,
                          log_level=log_level)

    if read_gamma:
        _set_gamma_from_file(br, filename=input_filename)
        
    for i in br:
        if write_gamma:
            _write_gamma(br, interaction, i, filename=output_filename)
        if log_level > 1:
            _write_triplets(interaction)

    if grid_points is None:
        br.set_kappa_at_sigmas()
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
    gamma_isotope = br.get_gamma_isotope()
    sigmas = br.get_sigmas()
    
    gp = grid_points[i]
    gv = group_velocities[i]
    mode_cv = mode_heat_capacities[i]
    frequencies = interaction.get_phonons()[0][gp]
    
    for j, sigma in enumerate(sigmas):
        if gamma_isotope is not None:
            gamma_isotope_at_sigma = gamma_isotope[j, i]
        else:
            gamma_isotope_at_sigma = None
        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            heat_capacity=mode_cv,
                            kappa=None,
                            gamma=gamma[j, :, i],
                            gamma_isotope=gamma_isotope_at_sigma,
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
    gamma_isotope = br.get_gamma_isotope()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    frequencies = br.get_frequencies()
    gv = br.get_group_velocities()
    mode_cv = br.get_mode_heat_capacities()
    qpoints = br.get_qpoints()
    weights = br.get_grid_weights()
    # num_sampling_points = br.get_number_of_sampling_points()
    
    kappa = br.get_kappa()
    
    for i, sigma in enumerate(sigmas):
        kappa_at_sigma = kappa[i]
        if gamma_isotope is not None:
            gamma_isotope_at_sigma = gamma_isotope[i]
        else:
            gamma_isotope_at_sigma = None
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
        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            heat_capacity=mode_cv,
                            kappa=kappa_at_sigma,
                            gamma=gamma[i],
                            gamma_isotope=gamma_isotope_at_sigma,
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
    temperatures = br.get_temperatures()
    num_band = br.get_frequencies().shape[1]

    gamma = np.zeros((len(sigmas),
                      len(temperatures),
                      len(grid_points),
                      num_band), dtype='double')

    for j, sigma in enumerate(sigmas):
        gamma_at_sigma = read_gamma_from_hdf5(
            mesh,
            mesh_divisors=mesh_divisors,
            sigma=sigma,
            filename=filename)
        if gamma_at_sigma is False:
            for i, gp in enumerate(grid_points):
                gamma_gp = read_gamma_from_hdf5(
                    mesh,
                    mesh_divisors=mesh_divisors,
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename)
                if gamma_gp is False:
                    print "Gamma at grid point %d doesn't exist." % gp
                else:
                    gamma[j, :, i] = gamma_gp
        else:
            gamma[j] = gamma_at_sigma
        
    br.set_gamma(gamma)

class Conductivity_RTA(Conductivity):
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

        self._gamma = None
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
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              mass_variances=mass_variances,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              cutoff_lifetime=cutoff_lifetime,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=log_level)

        self._cv = None
        self._allocate_values()

    def set_kappa_at_sigmas(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_sampling_points = 0
        
        for i, grid_point in enumerate(self._grid_points):
            cv = self._cv[i]
            
            # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
            gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i)
            num_sampling_points += order_kstar
    
            # Sum all vxv at k*
            gv_sum2 = np.zeros((6, num_band), dtype='double')
            for j, vxv in enumerate(
                ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
                gv_sum2[j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]
    
            # Kappa
            for j in range(len(self._sigmas)):
                for k, l in list(np.ndindex(len(self._temperatures), num_band)):
                    g_phph = self._gamma[j, k, i, l]
                    if g_phph < 0.5 / self._cutoff_lifetime / THz:
                        continue
                    if self._isotope is None:
                        g_sum = g_phph
                    else:
                        g_iso = self._gamma_iso[j, i, l]
                        g_sum = g_phph + g_iso
                    self._kappa[j, k] += (
                        gv_sum2[:, l] * cv[k, l] / (g_sum * 2) *
                        self._conversion_factor)

        self._kappa /= num_sampling_points

    def get_mode_heat_capacities(self):
        return self._cv

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
            self._set_gamma_at_sigmas(i)

        if self._isotope is not None:
            self._set_gamma_isotope_at_sigmas(i)

        self._cv[i] = self._get_cv(self._frequencies[grid_point])
        self._set_gv(i)
        
        if self._log_level:
            self._show_log(self._qpoints[i], i)

    def _allocate_values(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        self._kappa = np.zeros((len(self._sigmas),
                                len(self._temperatures),
                                6), dtype='double')
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    len(self._temperatures),
                                    num_grid_points,
                                    num_band), dtype='double')
        self._gv = np.zeros((num_grid_points,
                             num_band,
                             3), dtype='double')
        self._cv = np.zeros((num_grid_points,
                             len(self._temperatures),
                             num_band), dtype='double')
        if self._isotope is not None:
            self._gamma_iso = np.zeros((len(self._sigmas),
                                        num_grid_points,
                                        num_band), dtype='double')
        self._collision = ImagSelfEnergy(self._pp)
        
    def _get_gv_by_gv(self, i):
        rotation_map = get_grid_points_by_rotations(
            self._grid_points[i], self._point_operations, self._mesh)
        gv_by_gv = np.zeros((len(self._gv[i]), 3, 3), dtype='double')
        
        if self._no_kappa_stars:
            count = 0
            for r, r_gp in zip(self._rotations_cartesian, rotation_map):
                if r_gp == rotation_map[0]:
                    gvs_rot = np.dot(r, self._gv[i].T).T
                    gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
                    count += 1
            gv_by_gv /= count
            order_kstar = 1
        else:
            for r in self._rotations_cartesian:
                gvs_rot = np.dot(r, self._gv[i].T).T
                gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
            gv_by_gv /= len(rotation_map) / len(np.unique(rotation_map))
            order_kstar = len(np.unique(rotation_map))
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

        return gv_by_gv, order_kstar

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

    def _show_log(self, q, i):
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        gv = self._gv[i]
        
        print "Frequency     group velocity (x, y, z)     |gv|",
        if self._gv_delta_q is None:
            print
        else:
            print " (dq=%3.1e)" % self._gv_delta_q

        if self._log_level > 1:
            rotation_map = get_grid_points_by_rotations(
                gp, self._point_operations, self._mesh)
            for i, j in enumerate(np.unique(rotation_map)):
                for k, (rot, rot_c) in enumerate(zip(self._point_operations,
                                                     self._rotations_cartesian)):
                    if rotation_map[k] != j:
                        continue
    
                    print " k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) +
                                                           tuple(np.dot(rot, q)))
                    for f, v in zip(frequencies,
                                    np.dot(rot_c, gv.T).T):
                        print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                            f, v[0], v[1], v[2], np.linalg.norm(v))
            print
        else:
            for f, v in zip(frequencies, gv):
                print "%8.3f   (%8.3f %8.3f %8.3f) %8.3f" % (
                    f, v[0], v[1], v[2], np.linalg.norm(v))
    
