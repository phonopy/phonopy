import numpy as np
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.interaction import Interaction
from anharmonic.BTE_RTA import BTE_RTA
from anharmonic.jointDOS import get_jointDOS
from anharmonic.gruneisen import Gruneisen
import anharmonic.triplets as triplets
from anharmonic.file_IO import write_kappa_to_hdf5, read_gamma_from_hdf5, write_damping_functions

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
                 frequency_factor=None,
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

    def get_jointDOS(self, grid_points, filename=None):
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
                     filename=filename,
                     log_level=self._log_level)

class Phono3py:
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 fc3,
                 band_indices=None,
                 frequency_factor_to_THz=None,
                 is_nosym=False,
                 symmetrize_fc3=False,
                 symprec=1e-5,
                 log_level=0,
                 lapack_zheev_uplo='L'):

        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc3 = fc3
        if band_indices is None:
            self._band_indices = np.arange(
                primitive.get_number_of_atoms() * 3, dtype='intc')
        else:
            self._band_indices = np.intc(band_indices)
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._symmetrize_fc3 = symmetrize_fc3
        self._symprec = symprec
        self._log_level = log_level
        self._kappa = None
        self._gamma = None

        self._interaction = Interaction(
            fc3,
            supercell,
            primitive,
            mesh,
            band_indices=self._band_indices,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._symprec,
            symmetrize_fc3=self._symmetrize_fc3,
            is_nosym=self._is_nosym,
            log_level=self._log_level,
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
                             filename=None):
        for gp in grid_points:
            self._interaction.set_triplets_at_q(grid_points)
            for sigma in sigmas:
                for t in temperatures:
                    imag_self_energy = ImagSelfEnergy(
                        self._interaction,
                        temperature=t,
                        sigma=sigma)
                    imag_self_energy.run_interaction()
                    max_freq = (np.amax(self._interaction.get_phonons()[0]) * 2
                                + sigma * 4)
                    fpoints = np.linspace(0, max_freq,
                                          int(max_freq / frequency_step))
                    imag_self_energy.set_fpoints(fpoints)
                    imag_self_energy.run()
                    gamma = imag_self_energy.get_imag_self_energy()
                    fpoints = imag_self_energy.get_fpoints()

                    for bi in self._band_indices:
                        write_damping_functions(gp,
                                                [bi + 1],
                                                self._mesh,
                                                fpoints,
                                                gamma[:, bi],
                                                sigma=sigma,
                                                temperature=t,
                                                filename=filename)
                    write_damping_functions(
                        gp,
                        self._band_indices + 1,
                        self._mesh,
                        fpoints,
                        (gamma[:, self._band_indices].sum(axis=1) /
                         len(self._band_indices)),
                        sigma=sigma,
                        temperature=t,
                        filename=filename)

    def get_linewidth(self,
                      grid_points,
                      sets_of_band_indices,
                      sigmas=[0.1],
                      t_max=1500,
                      t_min=0,
                      t_step=10,
                      gamma_option=0,
                      filename=None):

        lw = Linewidth(self._pp,
                       sigmas=sigmas,
                       t_max=t_max,
                       t_min=t_min,
                       t_step=t_step)
        
        if grid_points is None:
            print "Grid points are not specified."
            return False

        if sets_of_band_indices is None:
            print "Band indices are not specified."
            return False

        for gp in grid_points:
            self._pp.set_triplets_at_q(gp)
            for band_indices in sets_of_band_indices:
                self._pp.set_interaction_strength(band_indices)
                fwhms_sigmas, freqs_sigmas = lw.get_linewidth(
                    filename=filename)
                temps = lw.get_temperatures()

                for sigma, fwhms, freqs in zip(
                    sigmas, fwhms_sigmas, freqs_sigmas):
                    print "# Grid point:", gp
                    print "# Sigma:", sigma
                    print "# Frequencies:", freqs
                    for fwhm, t in zip(fwhms.T, temps):
                        print t, fwhm
                    print

    def get_thermal_conductivity(self,
                                 sigmas=[0.1],
                                 t_max=1500,
                                 t_min=0,
                                 t_step=10,
                                 max_freepath=0.01, # in meter
                                 grid_points=None,
                                 mesh_divisors=None,
                                 coarse_mesh_shifts=None,
                                 no_kappa_stars=False,
                                 write_gamma=False,
                                 read_gamma=False,
                                 write_amplitude=False,
                                 read_amplitude=False,
                                 gamma_option=0,
                                 filename=None):
        br = BTE_RTA(self._pp,
                     sigmas=sigmas,
                     t_max=t_max,
                     t_min=t_min,
                     t_step=t_step,
                     max_freepath=max_freepath,
                     mesh_divisors=mesh_divisors,
                     coarse_mesh_shifts=coarse_mesh_shifts,
                     no_kappa_stars=no_kappa_stars,
                     gamma_option=gamma_option,
                     log_level=self._log_level,
                     filename=filename)
        br.set_grid_points(grid_points)

        if read_gamma:
            gamma = []
            for sigma in sigmas:
                gamma_at_sigma = []
                for i, gp in enumerate(br.get_grid_points()):
                    gamma_at_sigma.append(read_gamma_from_hdf5(
                        br.get_mesh_numbers(),
                        mesh_divisors=br.get_mesh_divisors(),
                        grid_point=gp,
                        sigma=sigma,
                        filename=filename))
                gamma.append(gamma_at_sigma)
            br.set_gamma(np.double(gamma))

        br.calculate_kappa(write_amplitude=write_amplitude,
                           read_amplitude=read_amplitude,
                           write_gamma=write_gamma)        
        mode_kappa = br.get_kappa()
        gamma = br.get_gamma()

        if grid_points is None:
            temperatures = br.get_temperatures()
            for i, sigma in enumerate(sigmas):
                print
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
                                    filename=filename)

        self._kappa = mode_kappa
        self._gamma = gamma

    def solve_dynamical_matrix(self, q):
        """Only for test phonopy zheev wrapper"""
        import anharmonic._phono3py as phono3c
        dm = self._pp.get_dynamical_matrix()
        dm.set_dynamical_matrix(q)
        dynmat = dm.get_dynamical_matrix()
        eigvals = np.zeros(len(dynmat), dtype=float)
        phono3c.zheev(dynmat, eigvals)
        
        for f, row in zip(np.sqrt(abs(eigvals)) * self._factor *
                          np.sign(eigvals), dynmat.T):
            print f
            print row
        

    
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

def get_ir_grid_points(mesh, primitive, is_shift=[0, 0, 0]):
    return triplets.get_ir_grid_points(mesh, primitive, is_shift=is_shift)
