import numpy as np
from anharmonic.phonon3.interaction import get_dynamical_matrix, set_phonon_c, set_phonon_py
from anharmonic.phonon3.triplets import get_bz_grid_address
from anharmonic.phonon3.imag_self_energy import gaussian
import phonopy.structure.spglib as spg
from phonopy.units import VaspToTHz

class Isotope:
    def __init__(self,
                 mesh,
                 mass_variances, # length of list is num_atom.
                 sigma=0.1,
                 frequency_factor_to_THz=VaspToTHz,
                 symprec=1e-5,
                 lapack_zheev_uplo='L'):
        self._mesh = mesh
        self._mass_variances = mass_variances
        self._sigma = sigma
        self._symprec = symprec
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._lapack_zheev_uplo = lapack_zheev_uplo
        self._nac_q_direction = None
        
        self._grid_address = None
        self._grid_points = None

        self._frequencies = None
        self._eigenvectors = None
        self._phonon_done = None
        self._dm = None
        self._band_indices = None
        self._grid_point = None

    def run(self, grid_point, band_indices):
        self._grid_point = grid_point
        self._band_indices = band_indices
        primitive = self._dm.get_primitive()
        num_grid = np.prod(self._mesh)
        self._grid_points = np.arange(num_grid, dtype='intc')
        
        if self._phonon_done is None:
            num_band = primitive.get_number_of_atoms() * 3
            self._phonon_done = np.zeros(num_grid, dtype='byte')
            self._frequencies = np.zeros((num_grid, num_band), dtype='double')
            self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                          dtype='complex128')

        if self._grid_address is None:
            primitive_lattice = np.linalg.inv(primitive.get_cell())
            self._grid_address = get_bz_grid_address(self._mesh,
                                                     primitive_lattice)
        return self._run_py()

    def set_phonons(self, frequencies, eigenvectors, phonon_done, dm=None):
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._phonon_done = phonon_done
        if dm is not None:
            self._dm = dm

    def get_phonons(self):
        return (self._frequencies,
                self._eigenvectors,
                self._phonon_done)

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def _run_c(self):
        self._set_phonon_c()

    def _run_py(self):
        for gp in self._grid_points:
            self._set_phonon_py(gp)

        t_inv = []
        for bi in self._band_indices:
            vec0 = self._eigenvectors[self._grid_point][:, bi].reshape(-1, 3)
            f0 = self._frequencies[self._grid_point][bi]
            ti_sum = 0.0
            for freqs, eigvecs in zip(self._frequencies, self._eigenvectors):
                for f, vec in zip(freqs, eigvecs.T):
                    ti_sum_band = 0.0
                    for v, v0, g in zip(
                        vec.reshape(-1, 3), vec0, self._mass_variances):
                        ti_sum_band += g * np.abs(np.vdot(v, v0)) ** 2
                    ti_sum += ti_sum_band * gaussian(f0 - f, self._sigma)
            t_inv.append(np.pi ** 2 / np.prod(self._mesh) * f0 ** 2 * ti_sum)

        return np.array(t_inv, dtype='double')
            
    def _set_phonon_c(self):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._phonon_done,
                     self._grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)

    def _set_phonon_py(self, grid_point):
        set_phonon_py(grid_point,
                      self._phonon_done,
                      self._frequencies,
                      self._eigenvectors,
                      self._grid_address,
                      self._mesh,
                      self._dm,
                      self._frequency_factor_to_THz,                  
                      self._lapack_zheev_uplo)
