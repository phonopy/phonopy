import numpy as np
from phonopy.structure.symmetry import Symmetry
import phonopy.structure.spglib as spg
from anharmonic.phonon3.triplets import get_grid_address, invert_grid_point
from anharmonic.phonon3.imag_self_energy import occupation as be_func
from anharmonic.phonon4.real_to_reciprocal import RealToReciprocal
from phonopy.units import VaspToTHz
from phonopy.units import Hbar, EV, Angstrom, THz, AMU
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC

class FrequencyShift:
    def __init__(self,
                 fc4,
                 supercell,
                 primitive,
                 mesh,
                 temperatures=None,
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
        if temperatures is None:
            self._temperatures = np.double([0])
        else:
            self._temperatures = np.double(temperatures)
        num_band = primitive.get_number_of_atoms() * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.intc(band_indices)
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._is_nosym = is_nosym
        self._symprec = symprec
        self._cutoff_frequency = cutoff_frequency
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        symmetry = Symmetry(primitive, symprec=symprec)
        self._point_group_operations = symmetry.get_pointgroup_operations()

        self._grid_address = get_grid_address(mesh)
        self._grid_point = None
        self._quartets_at_q = None
        self._weights_at_q = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_q_direction = None

        # Unit to THz of Gamma
        self._unit_conversion = (EV / Angstrom ** 4 / AMU ** 2
                                 / (2 * np.pi * THz) ** 2
                                 * Hbar * EV / (2 * np.pi * THz) / 8
                                 / len(self._grid_address))

    def run(self):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = np.prod(self._mesh)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype='complex128')
        self._run_py()

    def set_grid_point(self, grid_point):
        if self._is_nosym:
            quartets_at_q = np.arange(len(self._grid_address), dtype='intc')
            weights_at_q = np.ones(len(self._grid_address), dtype='intc')
        else:
            q = self._grid_address[grid_point].astype('double') / self._mesh
            (grid_mapping_table,
             grid_address) = spg.get_stabilized_reciprocal_mesh(
                self._mesh,
                self._point_group_operations,
                is_shift=np.zeros(3, dtype='intc'),
                is_time_reversal=True,
                qpoints=np.double([q]))

            quartets_at_q = np.intc(np.unique(grid_mapping_table))
            weights_at_q_all = np.zeros_like(grid_mapping_table)
            for g in grid_mapping_table:
                weights_at_q[g] += 1
            weights_at_q = weights_at_q_all[quartets_at_q]

        self._grid_point = grid_point
        self._quartets_at_q = quartets_at_q
        self._weights_at_q = weights_at_q

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             frequency_scale_factor=None,
                             decimals=None):
        if nac_params is None:
            self._dm = DynamicalMatrix(
                supercell,
                primitive,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                decimals=decimals,
                symprec=self._symprec)
        else:
            self._dm = DynamicalMatrixNAC(
                supercell,
                primitive,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                decimals=decimals,
                symprec=self._symprec)
            self._dm.set_nac_params(nac_params)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.double(nac_q_direction)

    def _run_py(self):
        r2r = RealToReciprocal(self._fc4,
                               self._supercell,
                               self._primitive,
                               self._mesh,
                               symprec=self._symprec)
        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors,
                                 cutoff_frequency=self._cutoff_frequency)
        
        gp = self._grid_point
        self._set_phonon_py(gp)
        igp = invert_grid_point(gp, self._grid_address, self._mesh)

        for t in self._temperatures:
            for band_index in self._band_indices:
                q = self._grid_address[gp].astype('double') / self._mesh
                print "q-point:", q, "band index:", band_index + 1
                fc4_normal = np.zeros((len(self._quartets_at_q),
                                       len(self._frequencies[0])),
                                      dtype='complex128')
                
                if self._frequencies[gp][band_index] < self._cutoff_frequency:
                    continue
    
                for i, (gp1, w) in enumerate(zip(self._quartets_at_q,
                                                 self._weights_at_q)):
                    igp1 = invert_grid_point(gp, self._grid_address, self._mesh)
                    r2r.run(self._grid_address[[igp, gp, gp1, igp1]])
                    fc4_reciprocal = r2r.get_fc4_reciprocal()
                    self._set_phonon_py(gp1)
                    r2n.run(fc4_reciprocal, gp, band_index, gp1)
                    occupations = be_func(self._frequencies[gp1], t)
                    fc4_normal[i] = (r2n.get_reciprocal_to_normal() *
                                     self._unit_conversion *
                                     (2 * occupations + 1))
                print fc4_normal.sum()

    def _set_phonon_py(self, grid_point):
        gp = grid_point
        if self._phonon_done[gp] == 0:
            self._phonon_done[gp] = 1
            q = self._grid_address[gp].astype('double') / self._mesh
            self._dm.set_dynamical_matrix(q)
            dm = self._dm.get_dynamical_matrix()
            eigvals, eigvecs = np.linalg.eigh(dm, UPLO=self._lapack_zheev_uplo)
            eigvals = eigvals.real
            self._frequencies[gp] = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                                     * self._frequency_factor_to_THz)
            self._eigenvectors[gp] = eigvecs

class ReciprocalToNormal:
    def __init__(self,
                 primitive,
                 frequencies,
                 eigenvectors,
                 cutoff_frequency=1e-4):
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._cutoff_frequency = cutoff_frequency

        self._masses = self._primitive.get_masses()

        self._fc4_normal = None
        self._fc4_reciprocal = None

    def run(self, fc4_reciprocal, gp, band_index, gp1):
        num_band = self._primitive.get_number_of_atoms() * 3
        self._fc4_reciprocal = fc4_reciprocal
        self._fc4_normal = np.zeros(num_band, dtype='complex128')
        self._reciprocal_to_normal(gp, band_index, gp1)

    def get_reciprocal_to_normal(self):
        return self._fc4_normal

    def _reciprocal_to_normal(self, gp, band_index, gp1):
        e1 = self._eigenvectors[gp][:, band_index]
        e2 = self._eigenvectors[gp1]
        f1 = self._frequencies[gp][band_index]
        f2 = self._frequencies[gp1]
        num_band = len(f2)
        if f1 > self._cutoff_frequency:
            for i in range(num_band):
                if f2[i] > self._cutoff_frequency:
                    fc4_elem = self._sum_in_atoms(e1, e2[:, i])
                    self._fc4_normal[i] = fc4_elem / f1 / f2[i]

    def _sum_in_atoms(self, e1, e2):
        num_atom = self._primitive.get_number_of_atoms()
        sum_fc4 = 0j
        for (i, j, k, l) in list(np.ndindex((num_atom,) * 4)):
            sum_fc4_cart = 0
            for (m, n, p, q) in list(np.ndindex((3, 3, 3, 3))):
                sum_fc4_cart += (e1[i * 3 + m].conj() * e1[j * 3 + n] *
                                 e2[k * 3 + p] * e2[l * 3 + q].conj() *
                                 self._fc4_reciprocal[i, j, k, l, m, n, p, q])
            mass_sqrt = np.sqrt(np.prod(self._masses[[i, j, k, l]]))
            sum_fc4 += sum_fc4_cart / mass_sqrt

        return sum_fc4
