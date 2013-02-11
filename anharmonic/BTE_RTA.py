import numpy as np
import phonopy.structure.spglib as spg
from anharmonic.im_self_energy import get_gamma
from phonopy.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV

class BTE_RTA:
    def __init__(self,
                 interaction_strength,
                 sigma=0.2,
                 t_max=1000,
                 t_min=0,
                 t_step=10,
                 is_nosym=False):
        self._pp = interaction_strength
        self._sigma = sigma
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)

        self._primitive = self._pp.get_primitive()
        self._mesh = None
        self._grid_points = None
        self.set_mesh_sampling(self._pp.get_mesh_numbers(),
                               is_nosym=is_nosym)

        self._point_operations = np.array(
            [rot.T
             for rot in self._pp.get_symmetry().get_pointgroup_operations()])

    def set_mesh_sampling(self, mesh, is_nosym=False):
        self._mesh = np.array(mesh)
        if is_nosym:
            self._grid_points = range(np.prod(self._mesh))
        else:
            (grid_mapping_table,
             self._grid_address) = spg.get_ir_reciprocal_mesh(mesh,
                                                              self._primitive)
            self._grid_points = np.unique(grid_mapping_table)
            # self._ir_grid_weights = [np.sum(grid_mapping_table == g)
            #                          for g in self._grid_points]
        
    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures
        
    def get_kappa(self,
                  gamma_option=0,
                  filename=None,
                  verbose=True):
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        partial_k = np.zeros((len(self._grid_points),
                              len(self._temperatures)), dtype=float)
        volume = self._primitive.get_volume()
        num_grid = np.prod(self._mesh)
        unit_to_WmK = 1e22 * EV / volume / num_grid
        for i, gp in enumerate(self._grid_points):
            if verbose:
                print ("================= %d/%d =================" %
                       (i + 1, len(self._grid_points)))

            self._pp.set_triplets_at_q(gp)
            self._pp.set_interaction_strength()
            gv = get_group_velocity(self._pp.get_qpoint(),
                                    self._pp.get_dynamical_matrix(),
                                    reciprocal_lattice,
                                    eigenvectors=self._pp.get_eigenvectors(),
                                    frequencies=self._pp.get_frequencies())
            rot_unit_n = self._get_rotated_unit_directions([1, 0, 0], gp)
            print np.array(rot_unit_n)
            gv_sum2 = np.zeros(len(self._pp.get_frequencies()), dtype=float)
            for unit_n in rot_unit_n:
                gv_sum2 += np.dot(unit_n, gv) ** 2
            print gv_sum2
            lt_cv = self._get_lifetime_by_cv(gamma_option=gamma_option,
                                             verbose=verbose)
            partial_k[i] = np.dot(lt_cv, gv_sum2) * unit_to_WmK

            w = open("partial-k-%d%d%d-%d.dat" %
                     (tuple(self._mesh) + (gp,)), 'w')
            for t, g in zip(self._temperatures, partial_k[i]):
                w.write("%6.1f %f\n" % (t, g.sum()))
            w.close()

        return partial_k

    def _get_lifetime_by_cv(self,
                            gamma_option=0,
                            verbose=True):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = self._pp.get_frequencies()
        cutoff_freq = self._pp.get_cutoff_frequency()

        lt_cv = np.zeros((len(self._temperatures), len(freqs)), dtype=float)

        for i, t in enumerate(self._temperatures):
            if t > 0:
                for j, f in enumerate(freqs):
                    if f > cutoff_freq:
                        g = get_gamma(amplitude_at_q,
                                      np.array([f], dtype=float),
                                      weights_at_q,
                                      frequencies_at_q,
                                      j,
                                      t,
                                      self._sigma,
                                      freq_conv_factor,
                                      gamma_option)[0] * unit_conversion
                        lt_cv[i, j] = get_cv(f, t) / g

        return lt_cv

    def _get_rotated_unit_directions(self,
                                     directon,
                                     grid_point):
        rec_lat = np.linalg.inv(self._primitive.get_cell())
        inv_rec_lat = self._primitive.get_cell()
        unit_n = np.array(directon) / np.linalg.norm(directon)
        orig_address = self._grid_address[grid_point]
        orbits = []
        rot_unit_n = []
        for rot in self._point_operations:
            rot_address = np.dot(rot, orig_address) % self._mesh
            in_orbits = False
            for orbit in orbits:
                if (rot_address == orbit).all():
                    in_orbits = True
                    break
            if not in_orbits:
                orbits.append(rot_address)
                rot_cart = np.dot(rec_lat, np.dot(rot, inv_rec_lat))
                rot_unit_n.append(np.dot(rot_cart.T, unit_n))

        return rot_unit_n

        
            
        
def get_cv(freqs, t):
    x = freqs * THzToEv / Kb / t
    expVal = np.exp(x)
    return Kb * x ** 2 * expVal / (expVal - 1.0) ** 2 # eV/K
