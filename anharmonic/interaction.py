import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC, get_smallest_vectors
from phonopy.structure.symmetry import Symmetry
from phonopy.units import VaspToTHz
from anharmonic.phonon3.real_to_reciprocal import RealToReciprocal
from anharmonic.phonon3.reciprocal_to_normal import ReciprocalToNormal
from anharmonic.triplets import get_triplets_at_q

class Interaction:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 band_indices=None,
                 frequency_factor_to_THz=VaspToTHz,
                 symmetrize_fc3=False,
                 is_nosym=False,
                 symprec=1e-3,
                 log_level=False,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3 
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.intc(mesh)
        
        num_band = primitive.get_number_of_atoms() * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.intc(band_indices)
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._symprec = symprec
        self._symmetrize_fc3 = symmetrize_fc3
        self._is_nosym = is_nosym
        self._log_level = log_level
        self._lapack_zheev_uplo = lapack_zheev_uplo

        symmetry = Symmetry(primitive, symprec=symprec)
        self._point_group_operations = symmetry.get_pointgroup_operations()

        self._triplets_at_q = None
        self._weights_at_q = None
        self._grid_address = None
        self._triplets_address = None
        self._interaction_strength = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_q_direction = None

    def run(self, lang='C'):
        num_band = self._primitive.get_number_of_atoms() * 3

        num_grid = np.prod(self._mesh)
        num_triplets = len(self._triplets_at_q)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype='complex128')

        self._interaction_strength = np.zeros(
            (num_triplets, len(self._band_indices), num_band, num_band),
            dtype='double')

        if lang == 'C':
            self._run_c()
        else:
            self._run_py()

    def get_interaction_strength(self):
        return self._interaction_strength

    def get_mesh_numbers(self):
        return self._mesh
    
    def get_phonons(self):
        return (self._frequencies,
                self._eigenvectors,
                self._phonon_done)

    def get_dynamical_matrix(self):
        return self._dm

    def get_primitive(self):
        return self._primitive

    def get_point_group_operations(self):
        return self._point_group_operations
    
    def get_triplets_at_q(self):
        return self._triplets_at_q, self._weights_at_q

    def get_band_indices(self):
        return self._band_indices

    def get_frequency_factor_to_THz(self):
        return self._frequency_factor_to_THz

    def get_lapack_zheev_uplo(self):
        return self._lapack_zheev_uplo
        
    def set_grid_point(self, grid_point):
        if self._is_nosym:
            triplets_at_q, weights_at_q, grid_address = get_nosym_triplets(
                self._mesh,
                grid_point)
        else:
            triplets_at_q, weights_at_q, grid_address = get_triplets_at_q(
                grid_point,
                self._mesh,
                self._point_group_operations)

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._grid_address = grid_address
        self._triplets_address = grid_address[triplets_at_q]

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

    def _run_c(self):
        import anharmonic._phono3py as phono3c
        
        # for i, grid_triplet in enumerate(self._triplets_at_q):
        #     for gp in grid_triplet:
        #         self._set_phonon_py(gp)

        self._set_phonon_c()

        num_band = self._primitive.get_number_of_atoms() * 3
        svecs, multiplicity = get_smallest_vectors(self._supercell,
                                                   self._primitive,
                                                   self._symprec)
        masses = np.double(self._primitive.get_masses())
        p2s = np.intc(self._primitive.get_primitive_to_supercell_map())
        s2p = np.intc(self._primitive.get_supercell_to_primitive_map())

        phono3c.interaction(self._interaction_strength,
                            self._frequencies,
                            self._eigenvectors,
                            self._triplets_at_q,
                            self._grid_address,
                            self._mesh,
                            self._fc3,
                            svecs,
                            multiplicity,
                            np.double(masses),
                            p2s,
                            s2p,
                            self._band_indices)

    def _set_phonon_c(self):
        import anharmonic._phono3py as phono3c
        
        svecs, multiplicity = self._dm.get_shortest_vectors()
        masses = np.double(self._dm.get_primitive().get_masses())
        rec_lattice = np.double(
            np.linalg.inv(self._dm.get_primitive().get_cell())).copy()
        if self._dm.is_nac():
            born = self._dm.get_born_effective_charges()
            nac_factor = self._dm.get_nac_factor()
            dielectric = self._dm.get_dielectric_constant()
        else:
            born = None
            nac_factor = 0
            dielectric = None

        phono3c.phonon_triplets(self._frequencies,
                                self._eigenvectors,
                                self._phonon_done,
                                self._triplets_at_q,
                                self._grid_address,
                                self._mesh,
                                self._dm.get_force_constants(),
                                svecs,
                                multiplicity,
                                masses,
                                self._dm.get_primitive_to_supercell_map(),
                                self._dm.get_supercell_to_primitive_map(),
                                self._frequency_factor_to_THz,
                                born,
                                dielectric,
                                rec_lattice,
                                self._nac_q_direction,
                                nac_factor,
                                self._lapack_zheev_uplo)
        
    def _run_py(self):
        r2r = RealToReciprocal(self._fc3,
                               self._supercell,
                               self._primitive,
                               self._triplets_address,
                               self._mesh,
                               symprec=self._symprec)
        
        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors)

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print "%d / %d" % (i + 1, len(self._triplets_at_q))
            r2r.run(self._grid_address[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._set_phonon_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            self._interaction_strength[i] = r2n.get_reciprocal_to_normal()

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
