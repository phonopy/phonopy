import numpy as np
from anharmonic.phonon4.frequency_shift import FrequencyShift
from phonopy.units import VaspToTHz

class Phono4py:
    def __init__(self,
                 fc4,
                 supercell,
                 primitive,
                 mesh,
                 band_indices=None,
                 frequency_factor_to_THz=VaspToTHz,
                 is_nosym=False,
                 symprec=1e-3,
                 cutoff_frequency=1e-4,
                 log_level=False,
                 lapack_zheev_uplo='L'):
        self._fc4 = fc4
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.intc(mesh)
        if band_indices is None:
            self._band_indices = [
                np.arange(primitive.get_number_of_atoms() * 3)]
        else:
            self._band_indices = band_indices
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._band_indices_flatten = np.intc(
            [x for bi in self._band_indices for x in bi])

        self._frequency_shifts = None
        
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

        self._frequency_shifts = np.double(freq_shifts)
        print self._frequency_shifts

    def get_frequency_shift(self):
        return self._frequency_shifts
        
