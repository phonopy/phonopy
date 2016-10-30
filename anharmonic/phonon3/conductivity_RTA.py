import numpy as np
from phonopy.phonon.group_velocity import get_group_velocity
from phonopy.units import THzToEv, THz, Angstrom
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from anharmonic.file_IO import (write_kappa_to_hdf5, write_triplets,
                                read_gamma_from_hdf5, write_grid_address)
from anharmonic.phonon3.conductivity import Conductivity, unit_to_WmK
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.triplets import get_grid_points_by_rotations

def get_thermal_conductivity_RTA(
        interaction,
        symmetry,
        temperatures=np.arange(0, 1001, 10, dtype='double'),
        sigmas=None,
        mass_variances=None,
        grid_points=None,
        is_isotope=False,
        boundary_mfp=None, # in micrometre
        use_ave_pp=False,
        gamma_unit_conversion=None,
        mesh_divisors=None,
        coarse_mesh_shifts=None,
        is_kappa_star=True,
        gv_delta_q=1e-4,
        run_with_g=True, # integration weights from gaussian smearing function
        is_full_pp=False,
        write_gamma=False,
        read_gamma=False,
        write_kappa=False,
        write_gamma_detail=False,
        input_filename=None,
        output_filename=None,
        log_level=0):

    if log_level:
        print("-------------------- Lattice thermal conducitivity (RTA) "
              "--------------------")
    br = Conductivity_RTA(
        interaction,
        symmetry,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=sigmas,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        gamma_unit_conversion=gamma_unit_conversion,
        mesh_divisors=mesh_divisors,
        coarse_mesh_shifts=coarse_mesh_shifts,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        run_with_g=run_with_g,
        is_full_pp=is_full_pp,
        with_gamma_detail=write_gamma_detail,
        log_level=log_level)

    if read_gamma:
        if not _set_gamma_from_file(br, filename=input_filename):
            print("Reading collisions failed.")
            return False

    for i in br:
        if write_gamma:
            _write_gamma(br,
                         interaction,
                         i,
                         filename=output_filename,
                         verbose=log_level)
        if log_level > 1 and read_gamma is False:
            _write_triplets(interaction)

    if write_kappa:
        if (grid_points is None and _all_bands_exist(interaction)):
            br.set_kappa_at_sigmas()
            _write_kappa(br,
                         interaction,
                         filename=output_filename,
                         log_level=log_level)

    return br

def _write_gamma(br, interaction, i, filename=None, verbose=True):
    grid_points = br.get_grid_points()
    group_velocities = br.get_group_velocities()
    gv_by_gv = br.get_gv_by_gv()
    mode_heat_capacities = br.get_mode_heat_capacities()
    ave_pp = br.get_averaged_pp_interaction()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    temperatures = br.get_temperatures()
    gamma = br.get_gamma()
    gamma_isotope = br.get_gamma_isotope()
    sigmas = br.get_sigmas()
    volume = interaction.get_primitive().get_volume()

    gp = grid_points[i]
    if _all_bands_exist(interaction):
        if ave_pp is None:
            ave_pp_i = None
        else:
            ave_pp_i = ave_pp[i]
        frequencies = interaction.get_phonons()[0][gp]
        for j, sigma in enumerate(sigmas):
            if gamma_isotope is not None:
                gamma_isotope_at_sigma = gamma_isotope[j, i]
            else:
                gamma_isotope_at_sigma = None
            write_kappa_to_hdf5(temperatures,
                                mesh,
                                frequency=frequencies,
                                group_velocity=group_velocities[i],
                                gv_by_gv=gv_by_gv[i],
                                heat_capacity=mode_heat_capacities[:, i],
                                gamma=gamma[j, :, i],
                                gamma_isotope=gamma_isotope_at_sigma,
                                averaged_pp_interaction=ave_pp_i,
                                mesh_divisors=mesh_divisors,
                                grid_point=gp,
                                sigma=sigma,
                                kappa_unit_conversion=unit_to_WmK / volume,
                                filename=filename,
                                verbose=verbose)
    else:
        for j, sigma in enumerate(sigmas):
            for k, bi in enumerate(interaction.get_band_indices()):
                if ave_pp is None:
                    ave_pp_ik = None
                else:
                    ave_pp_ik = ave_pp[i, k]
                frequencies = interaction.get_phonons()[0][gp, k]
                if gamma_isotope is not None:
                    gamma_isotope_at_sigma = gamma_isotope[j, i, k]
                else:
                    gamma_isotope_at_sigma = None
                    write_kappa_to_hdf5(
                        temperatures,
                        mesh,
                        frequency=frequencies,
                        group_velocity=group_velocities[i, k],
                        gv_by_gv=gv_by_gv[i, k],
                        heat_capacity=mode_heat_capacities[:, i, k],
                        gamma=gamma[j, :, i, k],
                        gamma_isotope=gamma_isotope_at_sigma,
                        averaged_pp_interaction=ave_pp_ik,
                        mesh_divisors=mesh_divisors,
                        grid_point=gp,
                        band_index=bi,
                        sigma=sigma,
                        kappa_unit_conversion=unit_to_WmK / volume,
                        filename=filename,
                        verbose=verbose)

def _all_bands_exist(interaction):
    band_indices = interaction.get_band_indices()
    num_band = interaction.get_primitive().get_number_of_atoms() * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False

def _write_triplets(interaction, filename=None):
    triplets, weights = interaction.get_triplets_at_q()[:2]
    grid_address = interaction.get_grid_address()
    mesh = interaction.get_mesh_numbers()
    write_triplets(triplets,
                   weights,
                   mesh,
                   grid_address,
                   grid_point=triplets[0, 0],
                   filename=filename)
    write_grid_address(grid_address, mesh, filename=filename)

def _write_kappa(br, interaction, filename=None, log_level=0):
    temperatures = br.get_temperatures()
    sigmas = br.get_sigmas()
    gamma = br.get_gamma()
    gamma_isotope = br.get_gamma_isotope()
    mesh = br.get_mesh_numbers()
    mesh_divisors = br.get_mesh_divisors()
    frequencies = br.get_frequencies()
    gv = br.get_group_velocities()
    gv_by_gv = br.get_gv_by_gv()
    mode_cv = br.get_mode_heat_capacities()
    ave_pp = br.get_averaged_pp_interaction()
    qpoints = br.get_qpoints()
    weights = br.get_grid_weights()
    kappa = br.get_kappa()
    mode_kappa = br.get_mode_kappa()
    num_ignored_phonon_modes = br.get_number_of_ignored_phonon_modes()
    num_band = br.get_frequencies().shape[1]
    num_phonon_modes = br.get_number_of_sampling_grid_points() * num_band
    volume = interaction.get_primitive().get_volume()

    for i, sigma in enumerate(sigmas):
        kappa_at_sigma = kappa[i]
        if gamma_isotope is not None:
            gamma_isotope_at_sigma = gamma_isotope[i]
        else:
            gamma_isotope_at_sigma = None
        if log_level:
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(("#%6s       " + " %-10s" * 6 + "#ipm") %
                      ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
                for j, (t, k) in enumerate(zip(temperatures, kappa_at_sigma)):
                    print(("%7.1f" + " %10.3f" * 6 + " %d/%d") %
                          ((t,) + tuple(k) +
                           (num_ignored_phonon_modes[i, j], num_phonon_modes)))
            else:
                print(("#%6s       " + " %-10s" * 6) %
                      ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy"))
                for j, (t, k) in enumerate(zip(temperatures, kappa_at_sigma)):
                    print(("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print('')

        write_kappa_to_hdf5(temperatures,
                            mesh,
                            frequency=frequencies,
                            group_velocity=gv,
                            gv_by_gv=gv_by_gv,
                            heat_capacity=mode_cv,
                            kappa=kappa_at_sigma,
                            mode_kappa=mode_kappa[i],
                            gamma=gamma[i],
                            gamma_isotope=gamma_isotope_at_sigma,
                            averaged_pp_interaction=ave_pp,
                            qpoint=qpoints,
                            weight=weights,
                            mesh_divisors=mesh_divisors,
                            sigma=sigma,
                            kappa_unit_conversion=unit_to_WmK / volume,
                            filename=filename,
                            verbose=log_level)

def _set_gamma_from_file(br, filename=None, verbose=True):
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
    gamma_iso = np.zeros((len(sigmas),
                          len(grid_points),
                          num_band), dtype='double')
    ave_pp = np.zeros((len(grid_points), num_band), dtype='double')

    is_isotope = False
    read_succeeded = True

    for j, sigma in enumerate(sigmas):
        collisions = read_gamma_from_hdf5(
            mesh,
            mesh_divisors=mesh_divisors,
            sigma=sigma,
            filename=filename,
            verbose=verbose)
        if collisions:
            gamma_at_sigma, gamma_iso_at_sigma, ave_pp = collisions
            gamma[j] = gamma_at_sigma
            if gamma_iso_at_sigma is not None:
                is_isotope = True
                gamma_iso[j] = gamma_iso_at_sigma
        else:
            for i, gp in enumerate(grid_points):
                collisions_gp = read_gamma_from_hdf5(
                    mesh,
                    mesh_divisors=mesh_divisors,
                    grid_point=gp,
                    sigma=sigma,
                    filename=filename,
                    verbose=verbose)
                if collisions_gp:
                    gamma_gp, gamma_iso_gp, ave_pp_gp = collisions_gp
                    gamma[j, :, i] = gamma_gp
                    if gamma_iso_gp is not None:
                        is_isotope = True
                        gamma_iso[j, i] = gamma_iso_gp
                    if ave_pp_gp is not None:
                        ave_pp[i] = ave_pp_gp
                else:
                    for bi in range(num_band):
                        collisions_band = read_gamma_from_hdf5(
                            mesh,
                            mesh_divisors=mesh_divisors,
                            grid_point=gp,
                            band_index=bi,
                            sigma=sigma,
                            filename=filename,
                            verbose=verbose)
                        if collisions_band:
                            gamma_bi, gamma_iso_bi, ave_pp_bi = collisions_band
                            gamma[j, :, i, bi] = gamma_bi
                            if gamma_iso_bi is not None:
                                is_isotope = True
                                gamma_iso[j, i, bi] = gamma_iso_bi
                            if ave_pp_bi is not None:
                                ave_pp[i, bi] = ave_pp_bi
                        else:
                            read_succeeded = False

    if read_succeeded:
        br.set_gamma(gamma)
        if ave_pp is not None:
            br.set_averaged_pp_interaction(ave_pp)
        return True
    else:
        return False

class Conductivity_RTA(Conductivity):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_points=None,
                 temperatures=np.arange(0, 1001, 10, dtype='double'),
                 sigmas=None,
                 is_isotope=False,
                 mass_variances=None,
                 boundary_mfp=None, # in micrometre
                 use_ave_pp=False,
                 gamma_unit_conversion=None,
                 mesh_divisors=None,
                 coarse_mesh_shifts=None,
                 is_kappa_star=True,
                 gv_delta_q=None,
                 run_with_g=True,
                 is_full_pp=False,
                 with_gamma_detail=False,
                 log_level=0):
        self._pp = None
        self._temperatures = None
        self._sigmas = None
        self._is_kappa_star = None
        self._gv_delta_q = None
        self._run_with_g = run_with_g
        self._is_full_pp = is_full_pp
        self._with_gamma_detail = with_gamma_detail
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

        self._read_gamma = False
        self._read_gamma_iso = False

        self._frequencies = None
        self._gv = None
        self._gv_sum2 = None
        self._gamma = None
        self._gamma_iso = None
        self._gamma_unit_conversion = gamma_unit_conversion
        self._use_ave_pp = use_ave_pp
        self._averaged_pp_interaction = None
        self._num_ignored_phonon_modes = None
        self._num_sampling_grid_points = None

        self._mesh = None
        self._mesh_divisors = None
        self._coarse_mesh = None
        self._coarse_mesh_shifts = None
        self._conversion_factor = None

        self._is_isotope = None
        self._isotope = None
        self._mass_variances = None
        self._grid_point_count = None

        Conductivity.__init__(self,
                              interaction,
                              symmetry,
                              grid_points=grid_points,
                              temperatures=temperatures,
                              sigmas=sigmas,
                              is_isotope=is_isotope,
                              mass_variances=mass_variances,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              boundary_mfp=boundary_mfp,
                              is_kappa_star=is_kappa_star,
                              gv_delta_q=gv_delta_q,
                              log_level=log_level)

        self._cv = None

        if self._temperatures is not None:
            self._allocate_values()

    def set_kappa_at_sigmas(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        self._num_sampling_grid_points = 0

        for i, grid_point in enumerate(self._grid_points):
            cv = self._cv[:, i, :]
            gp = self._grid_points[i]
            frequencies = self._frequencies[gp]

            # Outer product of group velocities (v x v) [num_k*, num_freqs, 3, 3]
            gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i)
            self._num_sampling_grid_points += order_kstar

            # Sum all vxv at k*
            gv_sum2 = np.zeros((6, num_band), dtype='double')
            for j, vxv in enumerate(
                ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
                gv_sum2[j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

            # Kappa
            for j in range(len(self._sigmas)):
                for k in range(len(self._temperatures)):
                    g_sum = self._get_main_diagonal(i, j, k)
                    for l in range(num_band):
                        if frequencies[l] < self._cutoff_frequency:
                            self._num_ignored_phonon_modes[j, k] += 1
                            continue

                        self._mode_kappa[j, k, i, l] = (
                            gv_sum2[:, l] * cv[k, l] / (g_sum[l] * 2) *
                            self._conversion_factor)

            self._gv_sum2[i] = gv_sum2.T

        self._mode_kappa /= self._num_sampling_grid_points
        self._kappa = self._mode_kappa.sum(axis=2).sum(axis=2)

    def get_mode_heat_capacities(self):
        return self._cv

    def get_gv_by_gv(self):
        return self._gv_sum2

    def get_number_of_ignored_phonon_modes(self):
        return self._num_ignored_phonon_modes

    def get_number_of_sampling_grid_points(self):
        return self._num_sampling_grid_points

    def get_averaged_pp_interaction(self):
        return self._averaged_pp_interaction

    def set_averaged_pp_interaction(self, ave_pp):
        self._averaged_pp_interaction = ave_pp

    def _run_at_grid_point(self):
        i = self._grid_point_count
        self._show_log_header(i)
        grid_point = self._grid_points[i]

        if self._read_gamma:
            if self._use_ave_pp:
                self._collision.set_grid_point(grid_point)
                self._collision.set_averaged_pp_interaction(
                    self._averaged_pp_interaction[i])
                self._set_gamma_at_sigmas(i)
        else:
            self._collision.set_grid_point(grid_point)
            if self._log_level:
                print("Number of triplets: %d" %
                      len(self._pp.get_triplets_at_q()[0]))
                print("Calculating interaction...")

            self._set_gamma_at_sigmas(i)

        if self._isotope is not None and not self._read_gamma_iso:
            self._set_gamma_isotope_at_sigmas(i)

        freqs = self._frequencies[grid_point][self._pp.get_band_indices()]
        self._cv[:, i, :] = self._get_cv(freqs)
        self._set_gv(i)

        if self._log_level:
            self._show_log(self._qpoints[i], i)

    def _allocate_values(self):
        num_band0 = len(self._pp.get_band_indices())
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid_points = len(self._grid_points)
        self._kappa = np.zeros((len(self._sigmas),
                                len(self._temperatures),
                                6), dtype='double')
        self._mode_kappa = np.zeros((len(self._sigmas),
                                     len(self._temperatures),
                                     num_grid_points,
                                     num_band0,
                                     6), dtype='double')
        if not self._read_gamma:
            self._gamma = np.zeros((len(self._sigmas),
                                    len(self._temperatures),
                                    num_grid_points,
                                    num_band0), dtype='double')
        self._gv = np.zeros((num_grid_points, num_band0, 3), dtype='double')
        self._gv_sum2 = np.zeros((num_grid_points, num_band0, 6), dtype='double')
        self._cv = np.zeros(
            (len(self._temperatures), num_grid_points, num_band0), dtype='double')
        if self._isotope is not None:
            self._gamma_iso = np.zeros(
                (len(self._sigmas), num_grid_points, num_band0), dtype='double')
        if self._is_full_pp or self._use_ave_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band0), dtype='double')
        self._num_ignored_phonon_modes = np.zeros(
            (len(self._sigmas), len(self._temperatures)), dtype='intc')
        self._collision = ImagSelfEnergy(
            self._pp,
            with_detail=self._with_gamma_detail,
            unit_conversion=self._gamma_unit_conversion)

    def _set_gamma_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating Gamma of ph-ph with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)

            self._collision.set_sigma(sigma)
            if not self._use_ave_pp:
                if sigma is None or self._run_with_g:
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

    def _get_gv_by_gv(self, i):
        rotation_map = get_grid_points_by_rotations(
            self._grid_address[self._grid_points[i]],
            self._point_operations,
            self._mesh)
        gv_by_gv = np.zeros((len(self._gv[i]), 3, 3), dtype='double')

        for r in self._rotations_cartesian:
            gvs_rot = np.dot(self._gv[i], r.T)
            gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv /= len(rotation_map) // len(np.unique(rotation_map))
        order_kstar = len(np.unique(rotation_map))

        if order_kstar != self._grid_weights[i]:
            if self._log_level:
                print("*" * 33  + "Warning" + "*" * 33)
                print(" Number of elements in k* is unequal "
                      "to number of equivalent grid-points.")
                print("*" * 73)

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
        frequencies = self._frequencies[gp][self._pp.get_band_indices()]
        gv = self._gv[i]

        if self._is_full_pp or self._use_ave_pp:
            ave_pp = self._averaged_pp_interaction[i]

        if self._is_full_pp or self._use_ave_pp:
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"
        if self._gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._gv_delta_q
        print(text)

        if self._log_level > 1:
            rotation_map = get_grid_points_by_rotations(
                self._grid_address[gp],
                self._point_operations,
                self._mesh)
            for i, j in enumerate(np.unique(rotation_map)):
                for k, (rot, rot_c) in enumerate(zip(
                        self._point_operations, self._rotations_cartesian)):
                    if rotation_map[k] != j:
                        continue

                    print(" k*%-2d (%5.2f %5.2f %5.2f)" %
                          ((i + 1,) + tuple(np.dot(rot, q))))
                    if self._is_full_pp or self._use_ave_pp:
                        for f, v, pp in zip(frequencies,
                                            np.dot(rot_c, gv.T).T,
                                            ave_pp):
                            print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                                  (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
                    else:
                        for f, v in zip(frequencies, np.dot(rot_c, gv.T).T):
                            print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                                  (f, v[0], v[1], v[2], np.linalg.norm(v)))
            print('')
        else:
            if self._is_full_pp or self._use_ave_pp:
                for f, v, pp in zip(frequencies, gv, ave_pp):
                    print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e" %
                          (f, v[0], v[1], v[2], np.linalg.norm(v), pp))
            else:
                for f, v in zip(frequencies, gv):
                    print("%8.3f   (%8.3f %8.3f %8.3f) %8.3f" %
                          (f, v[0], v[1], v[2], np.linalg.norm(v)))
