import numpy as np
from phonopy.structure.symmetry import Symmetry
from anharmonic.phonon3.imag_self_energy import get_imag_self_energy, write_imag_self_energy, get_linewidth, write_linewidth
from anharmonic.phonon3.frequency_shift import FrequencyShift
from anharmonic.phonon3.interaction import Interaction
from anharmonic.phonon3.conductivity_RTA import get_thermal_conductivity
from anharmonic.phonon3.joint_dos import JointDos
from anharmonic.phonon3.gruneisen import Gruneisen
from anharmonic.file_IO import write_frequency_shift, write_joint_dos
from anharmonic.other.isotope import Isotope
from phonopy.units import VaspToTHz

class Phono3py:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 tetrahedron_method=False,
                 sigmas = [],
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
            
        self._tetrahedron_method = tetrahedron_method
        if tetrahedron_method:
            self._sigmas = [None] + list(sigmas)
        else:
            self._sigmas = list(sigmas)
        
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._band_indices_flatten = np.hstack(self._band_indices).astype('intc')
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

        # Thermal conductivity
        self._thermal_conductivity = None

        # Imaginary part of self energy at frequency points
        self._imag_self_energy = None

        # Linewidth (Imaginary part of self energy x 2) at temperatures
        self._linewidth = None

        self._grid_points = None
        self._frequency_points = None
        self._temperatures = None
                
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
                           
    def run_imag_self_energy(self,
                             grid_points,
                             frequency_step=0.1,
                             temperatures=[0.0, 300.0],
                             output_filename=None):
        self._grid_points = grid_points
        self._temperatures = temperatures
        self._imag_self_energy, self._frequency_points = get_imag_self_energy(
            self._interaction,
            grid_points,
            self._sigmas,
            frequency_step=frequency_step,
            temperatures=temperatures,
            log_level=self._log_level)
            
    def write_imag_self_energy(self, filename=None):
        write_imag_self_energy(self._imag_self_energy,
                               self._mesh,
                               self._grid_points,
                               self._band_indices,
                               self._frequency_points,
                               self._temperatures,
                               self._sigmas,
                               filename=filename)
        
    def run_linewidth(self,
                      grid_points,
                      temperatures=np.arange(0, 1001, 10, dtype='double')):
        self._grid_points = grid_points
        self._temperatures = temperatures
        self._linewidth = get_linewidth(self._interaction,
                                        grid_points,
                                        self._sigmas,
                                        temperatures=temperatures,
                                        log_level=self._log_level)

    def write_linewidth(self, filename=None):
        write_linewidth(self._linewidth,
                        self._band_indices,
                        self._mesh,
                        self._grid_points,
                        self._sigmas,
                        self._temperatures,
                        filename=filename)

    def run_thermal_conductivity(
            self,
            temperatures=np.arange(0, 1001, 10, dtype='double'),
            sigmas=[],
            mass_variances=None,
            grid_points=None,
            mesh_divisors=None,
            coarse_mesh_shifts=None,
            cutoff_lifetime=1e-4, # in second
            no_kappa_stars=False,
            gv_delta_q=None, # for group velocity
            write_gamma=False,
            read_gamma=False,
            write_amplitude=False,
            read_amplitude=False,
            input_filename=None,
            output_filename=None):

        self._thermal_conductivity = get_thermal_conductivity(
                self._interaction,
                self._symmetry,
                temperatures=temperatures,
                sigmas=self._sigmas,
                mass_variances=mass_variances,
                grid_points=grid_points,
                mesh_divisors=mesh_divisors,
                coarse_mesh_shifts=coarse_mesh_shifts,
                cutoff_lifetime=cutoff_lifetime,
                no_kappa_stars=no_kappa_stars,
                gv_delta_q=gv_delta_q,
                write_gamma=write_gamma,
                read_gamma=read_gamma,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=self._log_level)        

    def get_frequency_shift(self,
                            grid_points,
                            epsilon=0.1,
                            temperatures=np.arange(0, 1001, 10, dtype='double'),
                            output_filename=None):
        fst = FrequencyShift(self._interaction)
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
        

class Phono3pyJointDos:
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 fc2,
                 nac_params=None,
                 sigmas=[],
                 tetrahedron_method=False,
                 frequency_step=None,
                 frequency_factor_to_THz=VaspToTHz,
                 frequency_scale_factor=None,
                 is_nosym=False,
                 symprec=1e-5,
                 output_filename=None,
                 log_level=0):
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._fc2 = fc2
        self._nac_params = nac_params
        self._sigmas = sigmas
        self._tetrahedron_method = tetrahedron_method
        self._frequency_step = frequency_step
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._filename = output_filename
        self._log_level = log_level

        self._jdos = JointDos(
            self._mesh,
            self._primitive,
            self._supercell,
            self._fc2,
            nac_params=self._nac_params,
            tetrahedron_method=self._tetrahedron_method,
            frequency_step=self._frequency_step,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            frequency_scale_factor=self._frequency_scale_factor,
            is_nosym=self._is_nosym,
            symprec=self._symprec,
            filename=output_filename,
            log_level=self._log_level)

    def run(self, grid_points):
        for gp in grid_points:
            self._jdos.set_grid_point(gp)
            
            if self._log_level:
                weights = self._jdos.get_triplets_at_q()[1]
                print "------ Joint DOS ------"
                print "Grid point: %d" % gp
                print "Number of ir-triplets:",
                print "%d / %d" % (len(weights), weights.sum())
                adrs = self._jdos.get_grid_address()[gp]
                q = adrs.astype('double') / self._mesh
                print "q-point:", q
                print "Phonon frequency:"
                frequencies = self._jdos.get_phonons()[0]
                print frequencies[gp]
            
            if self._tetrahedron_method:
                print "Tetrahedron method"
                self._jdos.set_sigma(None)
                self._jdos.run()
                self._write(gp, sigma=None)
            if self._sigmas:
                for sigma in self._sigmas:
                    print "Sigma:", sigma
                    self._jdos.set_sigma(sigma)
                    self._jdos.run()
                    self._write(gp, sigma)
            else:
                print "sigma or tetrahedron method has to be set."

    def _write(self, gp, sigma=None):
        write_joint_dos(gp,
                        self._mesh,
                        self._jdos.get_frequency_points(),
                        self._jdos.get_joint_dos(),
                        sigma=sigma,
                        filename=self._filename,
                        is_nosym=self._is_nosym)
        
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
