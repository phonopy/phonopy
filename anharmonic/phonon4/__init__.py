import numpy as np
from anharmonic.phonon4.frequency_shift import FrequencyShift
from phonopy.units import VaspToTHz
from phonopy.harmonic.force_constants import set_translational_invariance, set_permutation_symmetry, get_fc2
from anharmonic.phonon3.fc3 import get_fc3
from anharmonic.phonon4.fc4 import get_fc4
from phonopy.structure.symmetry import Symmetry

class Phono4py:
    def __init__(self,
                 unitcell,
                 supercell,
                 primitive,
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
        self._supercell = supercell
        self._primitive = primitive
        
        self._mesh = None
        if mesh is not None:
            self._mesh = np.array(mesh, dtype='intc')

        if band_indices is None:
            self._band_indices = [
                np.arange(primitive.get_number_of_atoms() * 3,
                          dtype='intc')]
        else:
            self._band_indices = band_indices
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_symmetry = is_symmetry
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._band_indices_flatten = np.array(
            [x for bi in self._band_indices for x in bi],
            dtype='intc')

        self._frequency_shifts = None

        self._symmetry = Symmetry(self._supercell,
                                  self._symprec,
                                  self._is_symmetry)

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

    def get_symmetry(self):
        return self._symmetry

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
            print "----- Frequency shifts of fc4 -----"
        
        freq_shifts = []
        for i, gp in enumerate(grid_points):
            if self._log_level:
                print "=====================",
                print "Grid point %d (%d/%d)" % (gp, i + 1, len(grid_points)),
                print "====================="
            self._interaction.set_grid_point(gp)
            self._interaction.run()
            freq_shifts.append(self._interaction.get_frequency_shifts())

        self._frequency_shifts = np.array(freq_shifts, dtype='double')
        print self._frequency_shifts

    def get_frequency_shift(self):
        return self._frequency_shifts
        
