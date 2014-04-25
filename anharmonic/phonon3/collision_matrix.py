import sys
import numpy as np
import phonopy.structure.spglib as spg
from phonopy.units import THzToEv, Kb
from phonopy.harmonic.force_constants import similarity_transformation
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.triplets import get_grid_point_from_address, get_ir_grid_points

class CollisionMatrix(ImagSelfEnergy):
    def __init__(self,
                 interaction,
                 symmetry,
                 ir_grid_points,
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

        self._ir_grid_points = ir_grid_points
        self._is_collision_matrix = True
        self._gamma_iso = None
        self._symmetry = symmetry
        self._point_operations = symmetry.get_reciprocal_operations()
        self._primitive = self._interaction.get_primitive()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._point_operations], dtype='double')
        
    def run(self, gamma_iso=None):
        self._gamma_iso = gamma_iso
        
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        # num_band0 is supposed to be equal to num_band.
        num_band0 = self._fc3_normal_squared.shape[1]
        num_band = self._fc3_normal_squared.shape[2]

        if num_band0 != num_band:
            print "--bi option is not allowed to use with collision matrix."
            sys.exit(1)
        
        num_triplets = len(self._triplets_at_q)
        self._imag_self_energy = np.zeros(num_band, dtype='double')
        self._collision_matrix = np.zeros(
            (num_band, 3, len(self._ir_grid_points), num_band, 3),
            dtype='double')
        self._run_with_band_indices()
        self._run_collision_matrix()

    def get_collision_matrix(self):
        return self._collision_matrix

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point,
                                             stores_triplets_map=True)
            self._fc3_normal_squared = None
            (self._triplets_at_q,
             self._weights_at_q,
             self._triplets_map_at_q,
             self._ir_map_at_q) = self._interaction.get_triplets_at_q()
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
        num_band = self._fc3_normal_squared.shape[1]
        for i, ir_gp in enumerate(self._ir_grid_points):
            ir_address = self._grid_address[ir_gp]
            r_address = np.dot(self._point_operations.reshape(-1, 3),
                               ir_address).reshape(-1, 3)
            r_gps = get_grid_point_from_address(r_address.T, self._mesh)
            
            order_r_gp = np.sqrt(len(r_gps) / len(np.unique(r_gps)))
            
            for r, r_gp in zip(self._rotations_cartesian, r_gps):
                ti = self._gp2tpindex[self._triplets_map_at_q[r_gp]]
                tp = self._triplets_at_q[ti]
                if self._triplets_map_at_q[ir_gp] == self._ir_map_at_q[ir_gp]:
                    gp2 = tp[2]
                else:
                    gp2 = tp[1]
                sinh = np.sinh(THzToEv * self._frequencies[gp2]
                               / (2 * Kb * self._temperature))
                for j, k in list(np.ndindex((num_band, num_band))):
                    collision = (
                        self._fc3_normal_squared[ti, j, k]
                        / sinh
                        * self._g[2, ti, j, k]).sum()
                    self._collision_matrix[j, :, i, k, :] += collision * r

            self._collision_matrix[:, :, i, :, :] *= (
                self._unit_conversion / order_r_gp)

            multi = 0
            collision_r = np.zeros((num_band, 3, 3), dtype='double')
            for r, r_gp in zip(self._rotations_cartesian, r_gps):
                if r_gp == self._grid_point:
                    multi += 1
                    for j in range(num_band):
                        collision = self._imag_self_energy[j]
                        if self._gamma_iso is not None:
                            collision += self._gamma_iso[j]
    
                        collision_r[j] += collision * r

            if multi > 0:
                for j in range(num_band):
                    self._collision_matrix[j, :, i, j, :] += collision_r[j] / order_r_gp

    def _set_collision_matrix_0K(self):
        """Collision matrix is zero."""
        pass
