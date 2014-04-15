import numpy as np
import phonopy.structure.spglib as spg
from phonopy.units import Hbar, THz, Kb
from phonopy.harmonic.force_constants import similarity_transformation
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.triplets import get_triplets_integration_weights, get_grid_point_from_address

class CollisionMatrix(ImagSelfEnergy):
    def __init__(self,
                 interaction,
                 symmetry,
                 grid_point=None,
                 frequency_points=None,
                 temperature=None,
                 sigma=None,
                 lang='C'):
        self._interaction = None
        self._sigma = None
        self._frequency_points = None
        self._temperature = None
        self._grid_point = None
        self._lang = None
        self._imag_self_energy = None
        self._collision_matrix = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._triplets_at_q = None
        self._triplets_map_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = None
        self._g = None
        self._mesh = None
        self._is_collision_matrix = None
        self._unit_conversion = None
        
        ImagSelfEnergy.__init__(self,
                                interaction,
                                grid_point=grid_point,
                                frequency_points=frequency_points,
                                temperature=temperature,
                                sigma=sigma,
                                lang=lang)

        self._is_collision_matrix = True
        self._symmetry = symmetry
        self._point_operations = symmetry.get_reciprocal_operations()
        self._primitive = self._interaction.get_primitive()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = [similarity_transformation(rec_lat, r)
                                     for r in self._point_operations]
        
    def run_collision_matrix(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        # num_band0 is supposed to be equal to num_band.
        num_band0 = self._fc3_normal_squared.shape[1]
        num_band = self._fc3_normal_squared.shape[2]
        num_triplets = len(self._triplets_at_q)
        self._imag_self_energy = np.zeros(num_band0, dtype='double')
        self._collision_matrix = np.zeros((num_band0, num_triplets, num_band),
                                          dtype='double')
        self._run_collision_matrix()

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point,
                                             stores_triplets_map=True)
            self._fc3_normal_squared = None
            (self._triplets_at_q,
             self._weights_at_q,
             self._triplets_map_at_q) = self._interaction.get_triplets_at_q()
            self._grid_address = self._interaction.get_grid_address()
            self._grid_point = grid_point

            self._gp2tpindex = {}
            for i, j in enumerate(np.unique(self._triplets_map_at_q)):
                self._gp2tpindex[j] = i
            
    def _run_collision_matrix(self):
        self._run_with_band_indices() # for Gamma
        self._run_py_collision_matrix() # for Omega

    def _run_py_collision_matrix(self):
        if self._temperature > 0:
            self._set_collision_matrix()
        else:
            self._set_collision_matrix_0K()
        
    def _set_collision_matrix(self):
        ir_address = self._grid_address[self._grid_point]
        r_address = np.dot(self._point_operations.reshape(-1, 3),
                           ir_address).reshape(-1, 3)
        r_gps = get_grid_point_from_address(r_address.T, self._mesh)
        unique_gps = np.unique(r_gps)
        col_mat = np.zeros(self._fc3_normal_squared.shape[:3], dtype='double')
        for i, gp in enumerate(unique_gps):
            ti = self._gp2tpindex[self._triplets_map_at_q[gp]]
            tp = self._triplets_at_q[ti]
            for j, k in list(np.ndindex(col_mat.shape[1:])):
                col_mat[i, j, k] = (
                    self._fc3_normal_squared[ti, j, k]
                    / np.sinc(THz * Hbar * self._frequencies[tp[2]]
                              / (2 * Kb * self._temperature))
                    * self._g[2, ti, j, k]).sum()

        sum_rots = np.zeros((len(unique_gps), 3, 3), dtype='double')
        count = 0
        for i, gp in enumerate(unique_gps):
            for r, r_gp in zip(self._rotations_cartesian, r_gps):
                if gp == r_gp:
                    sum_rots[i] += r
                    print count
                    count += 1
                    
                

    def _set_collision_matrix_0K(self):
        pass
