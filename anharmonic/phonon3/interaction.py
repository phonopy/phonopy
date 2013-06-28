import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
from anharmonic.phonon3.real_to_reciprocal import RealToReciprocal
from anharmonic.phonon3.reciprocal_to_normal import ReciprocalToNormal
from anharmonic.triplets import get_triplets_at_q

class Phonon3:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 factor=VaspToTHz,
                 frequency_factor=1.0,
                 symprec=1e-3,
                 read_triplets=False,
                 r2q_TI_index=None,
                 symmetrize_fc3_q=False,
                 is_Peierls=False,
                 is_nosym=False,
                 log_level=False,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3 
        self._supercell_fc3 = supercell
        self._primitive_fc3 = primitive
        self._mesh = mesh
        self._factor = factor
        self._frequency_factor = frequency_factor
        self._symprec = symprec
        self._r2q_TI_index = r2q_TI_index
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._is_Peierls = is_Peierls
        self._is_nosym = is_nosym
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._decimals = None

        symmetry = Symmetry(primitive, symprec=symprec)
        self._point_group = symmetry.get_pointgroup_operations()

        
        self._triplets_at_q = None
        self._weights_at_q = None
        self._grid_address = None
        self._triplets_address = None

    def run(self):
        num_grid = np.prod(self._mesh)
        num_band = self._primitive_fc3.get_number_of_atoms() * 3
        self._q_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype='complex128')

        r2r = RealToReciprocal(self._fc3,
                               self._supercell_fc3,
                               self._primitive_fc3,
                               self._triplets_address,
                               self._mesh,
                               symprec=self._symprec)
        
        r2n = ReciprocalToNormal(self._primitive_fc3,
                                 self._frequencies,
                                 self._eigenvectors)

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print "%d / %d" % (i + 1, len(self._triplets_at_q))
            r2r.run(self._grid_address[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            print fc3_reciprocal
            for gp in grid_triplet:
                self._set_phonon(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            print r2n.get_reciprocal_to_normal()

        print self._q_done

    def set_triplets_at_q(self, grid_point):
        if self._is_nosym:
            triplets_at_q, weights_at_q, grid_address = get_nosym_triplets(
                self._mesh,
                grid_point)
        else:
            triplets_at_q, weights_at_q, grid_address = get_triplets_at_q(
                grid_point,
                self._mesh,
                self._point_group)

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._grid_address = grid_address
        self._triplets_address = grid_address[triplets_at_q]

    def set_dynamical_matrix(self,
                             fc2,
                             supercell,
                             primitive,
                             nac_params=None,
                             nac_q_direction=None,
                             frequency_scale_factor=None):
        self._supercell_fc2 = supercell
        self._primitive_fc2 = primitive

        if nac_params is None:
            self._dm = DynamicalMatrix(
                self._supercell_fc2,
                self._primitive_fc2,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                decimals=self._decimals,
                symprec=self._symprec)
        else:
            self._dm = DynamicalMatrixNAC(
                self._supercell_fc2,
                self._primitive_fc2,
                fc2,
                frequency_scale_factor=frequency_scale_factor,
                decimals=self._decimals,
                symprec=self._symprec)
            self._dm.set_nac_params(nac_params)
            self._nac_q_direction = nac_q_direction

    def _set_phonon(self, grid_point):
        gp = grid_point
        if self._q_done[gp] == 0:
            self._q_done[gp] = 1
            q = self._grid_address[gp].astype('double') / self._mesh
            self._dm.set_dynamical_matrix(q)
            dm = self._dm.get_dynamical_matrix()
            eigvals, eigvecs = np.linalg.eigh(dm)
            eigvals = eigvals.real
            self._frequencies[gp] = (np.sqrt(np.abs(eigvals)) *
                                     np.sign(eigvals) * self._factor)
            self._eigenvectors[gp] = eigvecs

