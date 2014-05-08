import sys
import numpy as np
from phonopy.units import THzToEv, Kb
from phonopy.harmonic.force_constants import similarity_transformation
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy
from anharmonic.phonon3.triplets import get_triplets_third_q_list

class CollisionMatrix(ImagSelfEnergy):
    """
    Main diagonal part (imag-self-energy) and
    the other part are separately stored.
    """
    def __init__(self,
                 interaction,
                 point_operations,
                 ir_grid_points,
                 rotated_grid_points,
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
                                temperature=temperature,
                                sigma=sigma,
                                lang=lang)

        self._ir_grid_points = ir_grid_points
        self._rot_grid_points = rotated_grid_points
        self._is_collision_matrix = True
        self._point_operations = point_operations
        self._primitive = self._interaction.get_primitive()
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._point_operations], dtype='double', order='C')
        
    def run(self):
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
            self._third_q_list = get_triplets_third_q_list(
                grid_point,
                self._grid_address,
                self._interaction.get_bz_map(),
                self._mesh)
            
    def _run_collision_matrix(self):
        self._run_with_band_indices() # for Gamma
        if self._temperature > 0:
            if self._lang == 'C':
                self._run_c_collision_matrix() # for Omega
            else:
                self._run_py_collision_matrix() # for Omega

    def _run_c_collision_matrix(self):
        import anharmonic._phono3py as phono3c
        phono3c.collision_matrix(self._collision_matrix,
                                 self._fc3_normal_squared,
                                 self._frequencies,
                                 self._g,
                                 self._triplets_at_q,
                                 self._triplets_map_at_q,
                                 self._ir_map_at_q,
                                 self._ir_grid_points,
                                 self._rot_grid_points,
                                 self._rotations_cartesian,
                                 self._temperature,
                                 self._unit_conversion,
                                 self._cutoff_frequency)

    def _run_py_collision_matrix(self):
        gp2tp_map = {}
        for i, j in enumerate(self._triplets_at_q[:, 1]):
            gp2tp_map[j] = i

        num_band = self._fc3_normal_squared.shape[1]
        for i, ir_gp in enumerate(self._ir_grid_points):
            r_gps = self._rot_grid_points[i]
            for r, r_gp in zip(self._rotations_cartesian, r_gps):
                ti = gp2tp_map[self._triplets_map_at_q[r_gp]]
                tp = self._triplets_at_q[ti]
                if self._triplets_map_at_q[r_gp] == self._ir_map_at_q[r_gp]:
                    gp2 = tp[2]
                else:
                    gp2 = tp[1]
                freqs = self._frequencies[gp2]
                sinh = np.where(
                    freqs > self._cutoff_frequency,
                    np.sinh(freqs * THzToEv / (2 * Kb * self._temperature)),
                    -1)
                inv_sinh = np.where(sinh > 0, 1 / sinh, 0)
                for j, k in list(np.ndindex((num_band, num_band))):
                    collision = (self._fc3_normal_squared[ti, j, k]
                                 * inv_sinh
                                 * self._g[2, ti, j, k]).sum()
                    collision *= self._unit_conversion
                    self._collision_matrix[j, :, i, k, :] += collision * r


