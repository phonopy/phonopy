import numpy as np
from phonopy.phonon.solver import set_phonon_c, set_phonon_py
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors, get_dynamical_matrix
from phonopy.units import VaspToTHz, Hbar, EV, Angstrom, THz, AMU, PlanckConstant
from anharmonic.phonon3.real_to_reciprocal import RealToReciprocal
from anharmonic.phonon3.reciprocal_to_normal import ReciprocalToNormal
from anharmonic.phonon3.triplets import get_triplets_at_q, get_nosym_triplets_at_q, get_bz_grid_address

class Interaction:
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 symmetry,
                 fc3=None,
                 band_indices=None,
                 constant_averaged_interaction=None,
                 frequency_factor_to_THz=VaspToTHz,
                 unit_conversion=None,
                 is_nosym=False,
                 symmetrize_fc3_q=False,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._fc3 = fc3 
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype='intc')
        self._symmetry = symmetry

        self._band_indices = None
        self.set_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        self._frequency_factor_to_THz = frequency_factor_to_THz

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self._mesh)
            self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                     * EV ** 2 / Angstrom ** 6
                                     / (2 * np.pi * THz) ** 3
                                     / AMU ** 3 / num_grid
                                     / EV ** 2)
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_nosym = is_nosym
        self._symmetrize_fc3_q = symmetrize_fc3_q
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._symprec = symmetry.get_symmetry_tolerance()

        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._grid_address = None
        self._bz_map = None
        self._interaction_strength = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_q_direction = None
        
        self._allocate_phonon()
        
    def run(self, lang='C'):
        num_band = self._primitive.get_number_of_atoms() * 3
        num_triplets = len(self._triplets_at_q)

        if self._constant_averaged_interaction is None:
            self._interaction_strength = np.zeros(
                (num_triplets, len(self._band_indices), num_band, num_band),
                dtype='double')
            if lang == 'C':
                self._run_c()
            else:
                self._run_py()
        else:
            num_grid = np.prod(self._mesh)
            self._interaction_strength = np.ones(
                (num_triplets, len(self._band_indices), num_band, num_band),
                dtype='double') * self._constant_averaged_interaction / num_grid
            self.set_phonon(self._triplets_at_q.ravel())

    def get_interaction_strength(self):
        return self._interaction_strength

    def get_mesh_numbers(self):
        return self._mesh
    
    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done

    def get_dynamical_matrix(self):
        return self._dm

    def get_primitive(self):
        return self._primitive

    def get_triplets_at_q(self):
        return (self._triplets_at_q,
                self._weights_at_q,
                self._triplets_map_at_q,
                self._ir_map_at_q)

    def get_grid_address(self):
        return self._grid_address

    def get_bz_map(self):
        return self._bz_map
    
    def get_band_indices(self):
        return self._band_indices

    def set_band_indices(self, band_indices):
        num_band = self._primitive.get_number_of_atoms() * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(band_indices, dtype='intc')

    def get_frequency_factor_to_THz(self):
        return self._frequency_factor_to_THz

    def get_lapack_zheev_uplo(self):
        return self._lapack_zheev_uplo

    def is_nosym(self):
        return self._is_nosym

    def get_cutoff_frequency(self):
        return self._cutoff_frequency
        
    def set_grid_point(self, grid_point, stores_triplets_map=False):
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        if self._is_nosym:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q) = get_nosym_triplets_at_q(
                 grid_point,
                 self._mesh,
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)
        else:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q)= get_triplets_at_q(
                 grid_point,
                 self._mesh,
                 self._symmetry.get_pointgroup_operations(),
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)

        for triplet in triplets_at_q:
            sum_q = (grid_address[triplet]).sum(axis=0)
            if (sum_q % self._mesh != 0).any():
                print("============= Warning ==================")
                print("%s" % triplet)
                for tp in triplet:
                    print("%s %s" %
                          (grid_address[tp],
                           np.linalg.norm(
                               np.dot(reciprocal_lattice,
                                      grid_address[tp] /
                                      self._mesh.astype('double')))))
                print("%s" % sum_q)
                print("============= Warning ==================")

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._triplets_map_at_q = triplets_map_at_q
        self._grid_address = grid_address
        self._bz_map = bz_map
        self._ir_map_at_q = ir_map_at_q

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

    def set_phonon(self, grid_points):
        # for i, grid_triplet in enumerate(self._triplets_at_q):
        #     for gp in grid_triplet:
        #         self._set_phonon_py(gp)
        self._set_phonon_c(grid_points)

    def get_averaged_interaction(self):
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = v.sum(axis=2).sum(axis=2)
        num_band = self._primitive.get_number_of_atoms() * 3
        return np.dot(w, v_sum) / num_band ** 2
            
    def _run_c(self):
        import anharmonic._phono3py as phono3c
        
        self.set_phonon(self._triplets_at_q.ravel())
        num_band = self._primitive.get_number_of_atoms() * 3
        svecs, multiplicity = get_smallest_vectors(self._supercell,
                                                   self._primitive,
                                                   self._symprec)
        masses = np.array(self._primitive.get_masses(), dtype='double')
        p2s = self._primitive.get_primitive_to_supercell_map()
        s2p = self._primitive.get_supercell_to_primitive_map()

        phono3c.interaction(self._interaction_strength,
                            self._frequencies,
                            self._eigenvectors,
                            self._triplets_at_q,
                            self._grid_address,
                            self._mesh,
                            self._fc3,
                            svecs,
                            multiplicity,
                            masses,
                            p2s,
                            s2p,
                            self._band_indices,
                            self._symmetrize_fc3_q,
                            self._cutoff_frequency)
        self._interaction_strength *= self._unit_conversion

    def _set_phonon_c(self, grid_points):
        set_phonon_c(self._dm,
                     self._frequencies,
                     self._eigenvectors,
                     self._phonon_done,
                     grid_points,
                     self._grid_address,
                     self._mesh,
                     self._frequency_factor_to_THz,
                     self._nac_q_direction,
                     self._lapack_zheev_uplo)
        
    def _run_py(self):
        r2r = RealToReciprocal(self._fc3,
                               self._supercell,
                               self._primitive,
                               self._mesh,
                               symprec=self._symprec)
        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors,
                                 self._band_indices,
                                 cutoff_frequency=self._cutoff_frequency)

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)))
            r2r.run(self._grid_address[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._set_phonon_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            self._interaction_strength[i] = np.abs(
                r2n.get_reciprocal_to_normal()) ** 2 * self._unit_conversion

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

    def _allocate_phonon(self):
        primitive_lattice = np.linalg.inv(self._primitive.get_cell())
        self._grid_address, self._bz_map = get_bz_grid_address(
            self._mesh, primitive_lattice, with_boundary=True)
        num_band = self._primitive.get_number_of_atoms() * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype=("c%d" % (itemsize * 2)))
