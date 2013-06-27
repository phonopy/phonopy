import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
from anharmonic.phonon3.real_to_reciprocal import RealToReciprocal
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
        self._qpoint_triplets = None

    def run(self):
        r2r = RealToReciprocal(self._fc3,
                               self._qpoint_triplets,
                               self._supercell_fc3,
                               self._primitive_fc3,
                               symprec=self._symprec)
        r2r.run()
        print r2r.get_fc3_reciprocal()


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
        self._qpoint_triplets = (grid_address[triplets_at_q].astype('double') /
                                 self._mesh)

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
