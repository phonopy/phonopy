import numpy as np
from phonopy.structure.symmetry import Symmetry
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.frequency_shift import FrequencyShift
from anharmonic.phonon3.interaction import Interaction
from anharmonic.phonon3.conductivity_RTA import conductivity_RTA
from anharmonic.phonon3.jointDOS import get_jointDOS
from anharmonic.phonon3.gruneisen import Gruneisen
from anharmonic.file_IO import write_kappa_to_hdf5
from anharmonic.file_IO import read_gamma_from_hdf5, write_damping_functions, write_linewidth, write_frequency_shift
from anharmonic.other.isotope import Isotope
from phonopy.units import VaspToTHz

class Phono3py:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 band_indices=None,
                 cutoff_frequency=1e-4,
                 frequency_factor_to_THz=VaspToTHz,
                 is_nosym=False,
                 symmetrize_fc3_q=False,
                 symprec=1e-5,
                 log_level=0,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        if band_indices is None:
            self._band_indices = [
                np.arange(primitive.get_number_of_atoms() * 3)]
        else:
            self._band_indices = band_indices
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._kappa = None
        self._gamma = None

        self._band_indices_flatten = np.intc(
            [x for bi in self._band_indices for x in bi])

        self._symmetry = Symmetry(primitive, symprec)
        
        self._interaction = Interaction(
            fc3,
            supercell,
            primitive,
            mesh,
            self._symmetry,
            band_indices=self._band_indices_flatten,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            cutoff_frequency=self._cutoff_frequency,
            is_nosym=self._is_nosym,
            symmetrize_fc3_q=self._symmetrize_fc3_q,
            lapack_zheev_uplo=lapack_zheev_uplo)
        
    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        self._interaction.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor)
        self._interaction.set_nac_q_direction(nac_q_direction=nac_q_direction)
                           
    def get_imag_self_energy(self,
                             grid_points,
                             frequency_step=1.0,
                             sigmas=[0.1],
                             temperatures=[0.0],
                             output_filename=None):
        ise = ImagSelfEnergy(self._interaction)
        for gp in grid_points:
            ise.set_grid_point(gp)
            ise.run_interaction()
            for sigma in sigmas:
                ise.set_sigma(sigma)
                for t in temperatures:
                    ise.set_temperature(t)
                    max_freq = (np.amax(self._interaction.get_phonons()[0]) * 2
                                + sigma * 4)
                    fpoints = np.arange(0, max_freq + frequency_step / 2,
                                        frequency_step)
                    ise.set_fpoints(fpoints)
                    ise.run()
                    gamma = ise.get_imag_self_energy()

                    for i, bi in enumerate(self._band_indices):
                        pos = 0
                        for j in range(i):
                            pos += len(self._band_indices[j])

                        write_damping_functions(
                            gp,
                            bi,
                            self._mesh,
                            fpoints,
                            gamma[:, pos:(pos + len(bi))].sum(axis=1) / len(bi),
                            sigma=sigma,
                            temperature=t,
                            filename=output_filename)

    def get_linewidth(self,
                      grid_points,
                      sigmas=[0.1],
                      t_max=1500,
                      t_min=0,
                      t_step=10,
                      output_filename=None):
        ise = ImagSelfEnergy(self._interaction)
        temperatures = np.arange(t_min, t_max + t_step / 2.0, t_step)
        for gp in grid_points:
            ise.set_grid_point(gp)
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Linewidth ------"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            ise.run_interaction()
            for sigma in sigmas:
                ise.set_sigma(sigma)
                gamma = np.zeros((len(temperatures),
                                  len(self._band_indices_flatten)),
                                 dtype='double')
                for i, t in enumerate(temperatures):
                    ise.set_temperature(t)
                    ise.run()
                    gamma[i] = ise.get_imag_self_energy()

                for i, bi in enumerate(self._band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(self._band_indices[j])

                    write_linewidth(gp,
                                    bi,
                                    temperatures,
                                    gamma[:, pos:(pos+len(bi))],
                                    self._mesh,
                                    sigma=sigma,
                                    filename=output_filename)

    def get_frequency_shift(self,
                            grid_points,
                            epsilon=0.1,
                            t_max=1500,
                            t_min=0,
                            t_step=10,
                            output_filename=None):
        fst = FrequencyShift(self._interaction)
        temperatures = np.arange(t_min, t_max + t_step / 2.0, t_step)
        for gp in grid_points:
            fst.set_grid_point(gp)
            if self._log_level:
                weights = self._interaction.get_triplets_at_q()[1]
                print "------ Frequency shift -o- ------"
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
            fst.run_interaction()
            fst.set_epsilon(epsilon)
            delta = np.zeros((len(temperatures),
                              len(self._band_indices_flatten)),
                             dtype='double')
            for i, t in enumerate(temperatures):
                fst.set_temperature(t)
                fst.run()
                delta[i] = fst.get_frequency_shift()

            for i, bi in enumerate(self._band_indices):
                pos = 0
                for j in range(i):
                    pos += len(self._band_indices[j])

                write_frequency_shift(gp,
                                      bi,
                                      temperatures,
                                      delta[:, pos:(pos+len(bi))],
                                      self._mesh,
                                      epsilon=epsilon,
                                      filename=output_filename)

    def get_thermal_conductivity(self,
                                 sigmas=[0.1],
                                 t_max=1500,
                                 t_min=0,
                                 t_step=10,
                                 mass_variances=None,
                                 grid_points=None,
                                 mesh_divisors=None,
                                 coarse_mesh_shifts=None,
                                 cutoff_lifetime=1e-4, # in second
                                 no_kappa_stars=False,
                                 gv_delta_q=1e-4, # for group velocity
                                 write_gamma=False,
                                 read_gamma=False,
                                 write_amplitude=False,
                                 read_amplitude=False,
                                 output_filename=None,
                                 input_filename=None):
        br = conductivity_RTA(self._interaction,
                              self._symmetry,
                              sigmas=sigmas,
                              t_max=t_max,
                              t_min=t_min,
                              t_step=t_step,
                              mass_variances=mass_variances,
                              mesh_divisors=mesh_divisors,
                              coarse_mesh_shifts=coarse_mesh_shifts,
                              cutoff_lifetime=cutoff_lifetime,
                              no_kappa_stars=no_kappa_stars,
                              gv_delta_q=gv_delta_q,
                              log_level=self._log_level,
                              filename=output_filename)
        br.set_grid_points(grid_points)

        if read_gamma:
            gamma = []
            for sigma in sigmas:
                gamma_at_sigma = read_gamma_from_hdf5(
                    br.get_mesh_numbers(),
                    mesh_divisors=br.get_mesh_divisors(),
                    sigma=sigma,
                    filename=input_filename)
                if gamma_at_sigma is False:
                    gamma_at_sigma = []
                    for i, gp in enumerate(br.get_grid_points()):
                        gamma_gp = read_gamma_from_hdf5(
                            br.get_mesh_numbers(),
                            mesh_divisors=br.get_mesh_divisors(),
                            grid_point=gp,
                            sigma=sigma,
                            filename=input_filename)
                        if gamma_gp is False:
                            print "Gamma at grid point %d doesn't exist." % gp
                        gamma_at_sigma.append(gamma_gp)
                gamma.append(gamma_at_sigma)
            br.set_gamma(np.array(gamma, dtype='double'))

        br.calculate_kappa(write_amplitude=write_amplitude,
                           read_amplitude=read_amplitude,
                           write_gamma=write_gamma)        
        mode_kappa = br.get_kappa()
        gamma = br.get_gamma()

        if grid_points is None:
            temperatures = br.get_temperatures()
            for i, sigma in enumerate(sigmas):
                kappa = mode_kappa[i].sum(axis=2).sum(axis=0)
                print "----------- Thermal conductivity (W/m-k) for",
                print "sigma=%s -----------" % sigma
                print ("#%6s     " + " %-9s" * 6) % ("T(K)", "xx", "yy", "zz",
                                                    "yz", "xz", "xy")
                for t, k in zip(temperatures, kappa):
                    print ("%7.1f" + " %9.3f" * 6) % ((t,) + tuple(k))
                print
                write_kappa_to_hdf5(gamma[i],
                                    temperatures,
                                    br.get_mesh_numbers(),
                                    frequency=br.get_frequencies(),
                                    group_velocity=br.get_group_velocities(),
                                    heat_capacity=br.get_mode_heat_capacities(),
                                    kappa=kappa,
                                    qpoint=br.get_qpoints(),
                                    weight=br.get_grid_weights(),
                                    mesh_divisors=br.get_mesh_divisors(),
                                    sigma=sigma,
                                    filename=output_filename)

        self._kappa = mode_kappa
        self._gamma = gamma

class IsotopeScattering:
    def __init__(self,
                 mesh,
                 mass_variances, # length of list is num_atom.
                 sigma=0.1,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._iso = Isotope(mesh,
                            mass_variances,
                            sigma=sigma,
                            frequency_factor_to_THz=frequency_factor_to_THz,
                            symprec=symprec,
                            cutoff_frequency=cutoff_frequency,
                            lapack_zheev_uplo=lapack_zheev_uplo)

    def run(self, grid_point, band_indices=None):
        if band_indices is None:
            bi = [np.arange(self._primitive.get_number_of_atoms() * 3)]
        else:
            bi = band_indices
        self._iso.run(grid_point, bi)
        g_iso = self._iso.get_gamma()
        return g_iso
    
    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._primitive = primitive
        self._iso.set_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals)

    def set_sigma(self, sigma):
        self._iso.set_sigma(sigma)
        

class JointDOS:
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 fc2,
                 nac_params=None,
                 sigma=None,
                 frequency_step=None,
                 factor=None,
                 frequency_factor=VaspToTHz,
                 frequency_scale_factor=None,
                 is_nosym=False,
                 symprec=1e-5,
                 log_level=0):
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc2 = fc2
        self._nac_params = nac_params
        self._sigma = sigma
        self._frequency_step = frequency_step
        self._factor = factor
        self._frequency_factor = frequency_factor
        self._frequency_scale_factor = frequency_scale_factor
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._log_level = log_level

    def get_jointDOS(self, grid_points, output_filename=None):
        get_jointDOS(grid_points,
                     self._mesh,
                     self._primitive,
                     self._supercell,
                     self._fc2,
                     nac_params=self._nac_params,
                     sigma=self._sigma,
                     frequency_step=self._frequency_step,
                     factor=self._factor,
                     frequency_factor=self._frequency_factor,
                     frequency_scale=self._frequency_scale_factor,
                     is_nosym=self._is_nosym,
                     symprec=self._symprec,
                     filename=output_filename,
                     log_level=self._log_level)


def get_gruneisen_parameters(fc2,
                             fc3,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             ion_clamped=False,
                             factor=None,
                             symprec=1e-5):
    return Gruneisen(fc2,
                     fc3,
                     supercell,
                     primitive,
                     nac_params=nac_params,
                     nac_q_direction=nac_q_direction,
                     ion_clamped=ion_clamped,
                     factor=factor,
                     symprec=symprec)
