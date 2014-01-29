import numpy as np
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, get_primitive
from anharmonic.phonon3.imag_self_energy import get_imag_self_energy, write_imag_self_energy, get_linewidth, write_linewidth
from anharmonic.phonon3.frequency_shift import FrequencyShift
from anharmonic.phonon3.interaction import Interaction
from anharmonic.phonon3.conductivity_RTA import get_thermal_conductivity
from anharmonic.phonon3.joint_dos import JointDos
from anharmonic.phonon3.gruneisen import Gruneisen
from anharmonic.phonon3.displacement_fc3 import get_third_order_displacements, direction_to_displacement
from anharmonic.file_IO import write_frequency_shift, write_joint_dos
from anharmonic.other.isotope import Isotope
from phonopy.harmonic.force_constants import get_fc2, set_permutation_symmetry, \
     set_translational_invariance
from anharmonic.phonon3.fc3 import get_fc3, set_permutation_symmetry_fc3, \
     set_translational_invariance_fc3, cutoff_fc3_by_zero
from phonopy.units import VaspToTHz

class Phono3py:
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 primitive_matrix=None,
                 phonon_supercell_matrix=None,
                 mesh=None,
                 band_indices=None,
                 sigmas=[],
                 cutoff_frequency=1e-4,
                 frequency_factor_to_THz=VaspToTHz,
                 is_symmetry=True,
                 is_nosym=False,
                 symmetrize_fc3_q=False,
                 symprec=1e-5,
                 log_level=0,
                 lapack_zheev_uplo='L'):
        self._symprec = symprec
        self._sigmas = sigmas
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_nosym = is_nosym
        self._lapack_zheev_uplo =  lapack_zheev_uplo
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level

        # Create supercell and primitive cell
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._phonon_supercell_matrix = phonon_supercell_matrix # optional
        self._supercell = None
        self._primitive = None
        self._phonon_supercell = None
        self._phonon_primitive = None
        self._build_supercell()
        self._build_primitive_cell()
        self._build_phonon_supercell()
        self._build_phonon_primitive_cell()

        # Set supercell, primitive, and phonon supercell symmetries
        self._symmetry = None
        self._primitive_symmetry = None
        self._phonon_supercell_symmetry = None
        self._search_symmetry()
        self._search_primitive_symmetry()
        self._search_phonon_supercell_symmetry()

        # Thermal conductivity
        self._thermal_conductivity = None # conductivity_RTA object

        # Imaginary part of self energy at frequency points
        self._imag_self_energy = None

        # Linewidth (Imaginary part of self energy x 2) at temperatures
        self._linewidth = None

        self._grid_points = None
        self._frequency_points = None
        self._temperatures = None

        # Other variables
        self._fc2 = None
        self._fc3 = None
        
        # Setup interaction
        self._interaction = None
        self._mesh = None
        self._band_indices = None
        self._band_indices_flatten = None
        if mesh is not None:
            self._mesh = np.array(mesh, dtype='intc')
        if band_indices is None:
            num_band = self._primitive.get_number_of_atoms() * 3
            self._band_indices = [np.arange(num_band)]
        else:
            self._band_indices = band_indices
        self._band_indices_flatten = np.hstack(self._band_indices).astype('intc')

    def set_phph_interaction(self,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        self._interaction = Interaction(
            self._supercell,
            self._primitive,
            self._mesh,
            self._primitive_symmetry,
            fc3=self._fc3,
            band_indices=self._band_indices_flatten,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            cutoff_frequency=self._cutoff_frequency,
            is_nosym=self._is_nosym,
            symmetrize_fc3_q=self._symmetrize_fc3_q,
            lapack_zheev_uplo=self._lapack_zheev_uplo)
        self._interaction.set_dynamical_matrix(
            self._fc2,
            self._phonon_supercell,
            self._phonon_primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor)
        self._interaction.set_nac_q_direction(nac_q_direction=nac_q_direction)

    def get_interaction_strength(self):
        return self._interaction
        
    def produce_fc2(self,
                    forces_fc2,
                    disp_dataset,
                    is_permutation_symmetry=False,
                    is_translational_symmetry=False):
        for forces, disp1 in zip(forces_fc2, disp_dataset['first_atoms']):
            disp1['forces'] = forces
        self._fc2 = get_fc2(self._phonon_supercell,
                            self._phonon_supercell_symmetry,
                            disp_dataset)
        if is_permutation_symmetry:
            set_permutation_symmetry(self._fc2)
        if is_translational_symmetry:
            set_translational_invariance(self._fc2)

    def produce_fc3(self,
                    forces_fc3,
                    disp_dataset,
                    cutoff_distance=None, # set fc3 zero
                    is_translational_symmetry=False,
                    is_permutation_symmetry=False):

        if self._log_level:
            print "Solving fc2"
        for forces, disp1 in zip(forces_fc3, disp_dataset['first_atoms']):
            disp1['forces'] = forces
        self._fc2 = get_fc2(self._supercell, self._symmetry, disp_dataset)
        if is_permutation_symmetry:
            set_permutation_symmetry(self._fc2)
        if is_translational_symmetry:
            set_translational_invariance(self._fc2)
        
        if self._log_level:
            print "Solving fc3:"
        count = len(disp_dataset['first_atoms'])
        for disp1 in disp_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                disp2['delta_forces'] = forces_fc3[count] - disp1['forces']
                count += 1
        self._fc3 = get_fc3(
            self._supercell,
            disp_dataset,
            self._fc2,
            self._symmetry,
            is_translational_symmetry=is_translational_symmetry,
            is_permutation_symmetry=is_permutation_symmetry,
            verbose=self._log_level)

        # Set fc3 elements zero beyond cutoff_distance
        if cutoff_distance:
            if self._log_level:
                print ("Cutting-off fc3 by zero (cut-off distance: %f)" %
                       cutoff_distance)
            self.cutoff_fc3_by_zero(cutoff_distance)

    def cutoff_fc3_by_zero(self, cutoff_distance):
        cutoff_fc3_by_zero(self._fc3,
                           self._supercell,
                           cutoff_distance,
                           self._symprec)
            
    def set_permutation_symmetry(self):
        if self._fc2 is not None:
            set_permutation_symmetry(self._fc2)
        if self._fc3 is not None:
            set_permutation_symmetry_fc3(self._fc3)

    def set_translational_invariance(self):
        if self._fc2 is not None:
            set_translational_invariance(self._fc2)
        if self._fc3 is not None:
            set_translational_invariance_fc3(self._fc3)
        
    def get_fc2(self):
        return self._fc2

    def set_fc2(self, fc2):
        self._fc2 = fc2

    def get_fc3(self):
        return self._fc3

    def set_fc3(self, fc3):
        self._fc3 = fc3

    def get_primitive(self):
        return self._primitive

    def get_unitcell(self):
        return self._unitcell

    def get_supercell(self):
        return self._supercell

    def get_phonon_supercell(self):
        return self._phonon_supercell

    def get_phonon_primitive(self):
        return self._phonon_primitive

    def get_symmetry(self):
        """return symmetry of supercell"""
        return self._symmetry

    def get_primitive_symmetry(self):
        return self._primitive_symmetry

    def get_phonon_supercell_symmetry(self):
        return self._phonon_supercell_symmetry
        
    def generate_displacements(self,
                               distance=0.01,
                               cutoff_pair_distance=None,
                               is_plusminus='auto',
                               is_diagonal=True):
        direction_dataset = get_third_order_displacements(
            self._supercell,
            self._symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal)
        self._displacement_dataset = direction_to_displacement(
            direction_dataset,
            distance,
            self._supercell,
            cutoff_distance=cutoff_pair_distance)

    def set_displacement_dataset(self, dataset):
        self._displacement_dataset = dataset
        
    def get_displacement_dataset(self):
        return self._displacement_dataset
        
    def run_imag_self_energy(self,
                             grid_points,
                             frequency_step=0.1,
                             temperatures=[0.0, 300.0]):
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
                self._primitive_symmetry,
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

    def get_thermal_conductivity(self):
        return self._thermal_conductivity

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

    def _search_symmetry(self):
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)

    def _search_primitive_symmetry(self):
        self._primitive_symmetry = Symmetry(self._primitive,
                                            self._symprec,
                                            self._is_symmetry)
        if (len(self._symmetry.get_pointgroup_operations()) !=
            len(self._primitive_symmetry.get_pointgroup_operations())):
            print ("Warning: point group symmetries of supercell and primitive"
                   "cell are different.")
        
    def _search_phonon_supercell_symmetry(self):
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell_symmetry = self._symmetry
        else:
            self._phonon_supercell_symmetry = Symmetry(self._phonon_supercell,
                                                       self._symprec,
                                                       self._is_symmetry)

    def _build_supercell(self):
        self._supercell = get_supercell(self._unitcell,
                                        self._supercell_matrix,
                                        self._symprec)

    def _build_supercells_with_displacements(self):
        supercells = []
        for disp in self._displacements:
            positions = self._supercell.get_positions()
            positions[disp[0]] += disp[1:4]
            supercells.append(Atoms(
                    numbers=self._supercell.get_atomic_numbers(),
                    masses=self._supercell.get_masses(),
                    magmoms=self._supercell.get_magnetic_moments(),
                    positions=positions,
                    cell=self._supercell.get_cell(),
                    pbc=True))

        self._supercells_with_displacements = supercells

    def _build_primitive_cell(self):
        """
        primitive_matrix:
          Relative axes of primitive cell to the input unit cell.
          Relative axes to the supercell is calculated by:
             supercell_matrix^-1 * primitive_matrix
          Therefore primitive cell lattice is finally calculated by:
             (supercell_lattice * (supercell_matrix)^-1 * primitive_matrix)^T
        """
        self._primitive = self._get_primitive_cell(
            self._supercell, self._supercell_matrix, self._primitive_matrix)

    def _build_phonon_supercell(self):
        """
        phonon_supercell:
          This supercell is used for harmonic phonons (frequencies,
          eigenvectors, group velocities, ...)
        phonon_supercell_matrix:
          Different supercell size can be specified.
        """
        if self._phonon_supercell_matrix is None:
            self._phonon_supercell = self._supercell
        else:
            self._phonon_supercell = get_supercell(
                self._unitcell, self._phonon_supercell_matrix, self._symprec)

    def _build_phonon_primitive_cell(self):
        if self._phonon_supercell_matrix is None:
            self._phonon_primitive = self._primitive
        else:
            self._phonon_primitive = self._get_primitive_cell(
                self._phonon_supercell,
                self._phonon_supercell_matrix,
                self._primitive_matrix)

    def _get_primitive_cell(self, supercell, supercell_matrix, primitive_matrix):
        inv_supercell_matrix = np.linalg.inv(supercell_matrix)
        if primitive_matrix is None:
            t_mat = inv_supercell_matrix
        else:
            t_mat = np.dot(inv_supercell_matrix, primitive_matrix)
            
        return get_primitive(supercell, t_mat, self._symprec)
        
        

class IsotopeScattering:
    def __init__(self,
                 mesh,
                 mass_variances, # length of list is num_atom.
                 band_indices=None,
                 sigma=0.1,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._iso = Isotope(mesh,
                            mass_variances,
                            band_indices=band_indices,
                            sigma=sigma,
                            frequency_factor_to_THz=frequency_factor_to_THz,
                            symprec=symprec,
                            cutoff_frequency=cutoff_frequency,
                            lapack_zheev_uplo=lapack_zheev_uplo)

    def run(self, grid_point):
        self._iso.run(grid_point)
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
            
            if self._sigmas:
                for sigma in self._sigmas:
                    if sigma is None:
                        print "Tetrahedron method"
                    else:
                        print "Sigma:", sigma
                    self._jdos.set_sigma(sigma)
                    self._jdos.run()
                    self._write(gp, sigma=sigma)
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
