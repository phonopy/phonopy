import numpy as np
from anharmonic.phonon4.frequency_shift import FrequencyShift
from phonopy.structure.atoms import Atoms
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import set_translational_invariance, set_permutation_symmetry, get_fc2
from anharmonic.phonon3.fc3 import get_fc3
from anharmonic.phonon4.fc4 import get_fc4
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.cells import get_supercell, get_primitive
from anharmonic.phonon4.displacement_fc4 import get_fourth_order_displacements
from anharmonic.phonon4.displacement_fc4 import direction_to_displacement
from anharmonic.file_IO import write_frequency_shift

class Phono4py:
    def __init__(self,
                 unitcell,
                 supercell_matrix,
                 primitive_matrix=None,
                 mesh=None,
                 band_indices=None,
                 frequency_factor_to_THz=VaspToTHz,
                 is_symmetry=True,
                 is_nosym=False,
                 symprec=1e-3,
                 cutoff_frequency=1e-4,
                 log_level=False,
                 lapack_zheev_uplo='L'):
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        if mesh is None:
            self._mesh = None
        else:
            self._mesh = np.array(mesh, dtype='intc')
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._supercell = None
        self._primitive = None
        self._build_supercell()
        self._build_primitive_cell()
        self._supercells_with_displacements = None
        self._displacement_dataset = None

        if band_indices is None:
            num_band = self._primitive.get_number_of_atoms() * 3
            self._band_indices = [np.arange(num_band, dtype='intc')]
        else:
            self._band_indices = band_indices
        self._band_indices_flatten = np.array(
            [x for bi in self._band_indices for x in bi],
            dtype='intc')
        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)
        self._frequency_shifts = None

    def get_fc2(self):
        return self._fc2

    def set_fc2(self, fc2):
        self._fc2 = fc2

    def get_fc3(self):
        return self._fc3

    def set_fc3(self, fc3):
        self._fc3 = fc3

    def get_fc4(self):
        return self._fc4

    def set_fc4(self, fc4):
        self._fc4 = fc4

    def get_primitive(self):
        return self._primitive

    def get_unitcell(self):
        return self._unitcell

    def get_supercell(self):
        return self._supercell

    def get_symmetry(self):
        return self._symmetry

    def get_displacement_dataset(self):
        return self._displacement_dataset

    def get_supercells_with_displacements(self):
        if self._supercells_with_displacements is None:
            self._build_supercells_with_displacements()
        return self._supercells_with_displacements

    def generate_displacements(self,
                               distance=0.03,
                               is_plusminus='auto',
                               is_diagonal=True):
        direction_dataset = get_fourth_order_displacements(
            self._supercell,
            self._symmetry,
            is_plusminus=is_plusminus,
            is_diagonal=is_diagonal)
        self._displacement_dataset = direction_to_displacement(
            direction_dataset,
            distance,
            self._supercell)

    def produce_fc4(self,
                    forces_fc4,
                    displacement_dataset,
                    translational_symmetry_type=0,
                    is_permutation_symmetry=False,
                    is_permutation_symmetry_fc3=False,
                    is_permutation_symmetry_fc2=False):
        disp_dataset = displacement_dataset
        file_count = 0
        for disp1 in disp_dataset['first_atoms']:
            disp1['forces'] = forces_fc4[file_count]
            file_count += 1
        self._fc2 = get_fc2(self._supercell, self._symmetry, disp_dataset)
        if is_permutation_symmetry_fc2:
            set_permutation_symmetry(self._fc2)
        if translational_symmetry_type:
            set_translational_invariance(
                self._fc2,
                translational_symmetry_type=translational_symmetry_type)
        
        for disp1 in disp_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                disp2['forces'] = forces_fc4[file_count]
                disp2['delta_forces'] = disp2['forces'] - disp1['forces']
                file_count += 1

        self._fc3 = get_fc3(
            self._supercell,
            disp_dataset,
            self._fc2,
            self._symmetry,
            translational_symmetry_type=translational_symmetry_type,
            is_permutation_symmetry=is_permutation_symmetry_fc3,
            verbose=self._log_level)

        for disp1 in disp_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                for disp3 in disp2['third_atoms']:
                    disp3['delta_forces'] = (forces_fc4[file_count] -
                                             disp2['forces'])
                    file_count += 1
        
        self._fc4 = get_fc4(
            self._supercell,
            disp_dataset,
            self._fc3,
            self._symmetry,
            translational_symmetry_type=translational_symmetry_type,
            is_permutation_symmetry=is_permutation_symmetry,
            verbose=self._log_level)

    def set_frequency_shift(self, temperatures=None):
        self._interaction = FrequencyShift(
            self._fc4,
            self._supercell,
            self._primitive,
            self._mesh,
            temperatures=temperatures,
            band_indices=self._band_indices_flatten,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            is_nosym=self._is_nosym,
            symprec=self._symprec,
            cutoff_frequency=self._cutoff_frequency,
            log_level=self._log_level,
            lapack_zheev_uplo=self._lapack_zheev_uplo)
        self._grid_address = self._interaction.get_grid_address()
        self._frequencies = self._interaction.get_phonons()[0]
        self._temperatures = temperatures

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

    def run_frequency_shift(self, grid_points):
        if self._log_level:
            print "------------------------",
            print "Frequency shifts of fc4 ",
            print "------------------------"
        
        freq_shifts = []
        num_band = len(self._band_indices_flatten)
        for i, gp in enumerate(grid_points):
            qpoint = self._grid_address[gp].astype(float) / self._mesh
            if self._log_level:
                print "=========================",
                print "Grid point %d (%d/%d)" % (gp, i + 1, len(grid_points)),
                print "========================="
                print "q-point:", qpoint
            self._interaction.set_grid_point(gp)
            self._interaction.run()
            f_shifts_at_temps = self._interaction.get_frequency_shifts()
            if self._log_level:
                print "Harmonic phonon frequencies:"
                freqs = self._frequencies[gp][self._band_indices_flatten]
                print "%7s " % "",
                print ("%8.4f " * num_band) % tuple(freqs)
                print "Frequency shifts:" 
                for t, f_shift in zip(self._temperatures, f_shifts_at_temps):
                    print "%7.1f " % t,
                    print ("%8.4f " * num_band) % tuple(f_shift)
            freq_shifts.append(f_shifts_at_temps)

        self._frequency_shifts = np.array(freq_shifts, dtype='double')

        for i, gp in enumerate(grid_points):
            for j, bi in enumerate(self._band_indices):
                pos = 0
                for k in range(j):
                    pos += len(self._band_indices[k])

                write_frequency_shift(gp,
                                      bi,
                                      self._temperatures,
                                      freq_shifts[i][:, pos:(pos+len(bi))],
                                      self._mesh)

    def get_frequency_shift(self):
        return self._frequency_shifts
        
    def _build_supercells_with_displacements(self):
        supercells = []
        magmoms = self._supercell.get_magnetic_moments()
        masses = self._supercell.get_masses()
        numbers = self._supercell.get_atomic_numbers()
        lattice = self._supercell.get_cell()
        
        for disp1 in self._displacement_dataset['first_atoms']:
            disp_cart1 = disp1['displacement']
            positions = self._supercell.get_positions()
            positions[disp1['number']] += disp_cart1
            supercells.append(Atoms(numbers=numbers,
                                    masses=masses,
                                    magmoms=magmoms,
                                    positions=positions,
                                    cell=lattice,
                                    pbc=True))

        for disp1 in self._displacement_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                positions = self._supercell.get_positions()
                positions[disp1['number']] += disp_cart1
                positions[disp2['number']] += disp2['displacement']
                supercells.append(Atoms(numbers=numbers,
                                        masses=masses,
                                        magmoms=magmoms,
                                        positions=positions,
                                        cell=lattice,
                                        pbc=True))

        for disp1 in self._displacement_dataset['first_atoms']:
            for disp2 in disp1['second_atoms']:
                for disp3 in disp2['third_atoms']:
                    positions = self._supercell.get_positions()
                    positions[disp1['number']] += disp_cart1
                    positions[disp2['number']] += disp2['displacement']
                    positions[disp3['number']] += disp3['displacement']
                    supercells.append(Atoms(numbers=numbers,
                                            masses=masses,
                                            magmoms=magmoms,
                                            positions=positions,
                                            cell=lattice,
                                            pbc=True))
                
        self._supercells_with_displacements = supercells
            
    def _build_supercell(self):
        self._supercell = get_supercell(self._unitcell,
                                        self._supercell_matrix,
                                        self._symprec)

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

    def _get_primitive_cell(self, supercell, supercell_matrix, primitive_matrix):
        inv_supercell_matrix = np.linalg.inv(supercell_matrix)
        if primitive_matrix is None:
            t_mat = inv_supercell_matrix
        else:
            t_mat = np.dot(inv_supercell_matrix, primitive_matrix)
            
        return get_primitive(supercell, t_mat, self._symprec)
