import numpy as np
import phonopy.structure.spglib as spg
from anharmonic.im_self_energy import get_gamma
from phonopy.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from anharmonic.file_IO import parse_kappa
from anharmonic.triplets import get_grid_address

class BTE_RTA:
    def __init__(self,
                 interaction_strength,
                 sigma=0.2,
                 t_max=1000,
                 t_min=0,
                 t_step=10,
                 no_kappa_stars=False,
                 log_level=0):
        self._pp = interaction_strength
        self._sigma = sigma
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._no_kappa_stars = no_kappa_stars
        print "No kappa", self._no_kappa_stars
        self._log_level = log_level
        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)
        self._primitive = self._pp.get_primitive()
        self._mesh = None
        self._grid_points = None
        self._point_operations = np.array(
            [rot.T for rot in
             self._pp.get_symmetry().get_pointgroup_operations()])

    def set_mesh_sampling(self, mesh=None):
        if mesh is None:
            self._mesh = self._pp.get_mesh_numbers()
        else:
            self._mesh = np.array(mesh)
        
    def set_grid_points(self, grid_points=None):
        if grid_points is None:
            if self._no_kappa_stars:
                self._grid_address = get_grid_address(self._mesh)
                self._grid_points = range(np.prod(self._mesh))
                self._grid_weights = [0] * len(self._grid_points)
            else:
                (self._grid_mapping_table,
                 self._grid_address) = spg.get_ir_reciprocal_mesh(
                    self._mesh,
                    self._primitive)
                self._grid_points = np.unique(self._grid_mapping_table)
                self._grid_weights = [np.sum(self._grid_mapping_table == g)
                                      for g in self._grid_points]
        else:
            self._grid_address = get_grid_address(self._mesh)
            self._grid_points = grid_points
            self._grid_weights = [0] * len(self._grid_points)

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures
        
    def get_kappa(self,
                  gamma_option=0,
                  filename=None):
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        kappa = np.zeros((len(self._grid_points),
                              len(self._temperatures)), dtype=float)
        volume = self._primitive.get_volume()
        num_grid = np.prod(self._mesh)
        unit_to_WmK = (THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz / 2 / np.pi
        conversion_factor = unit_to_WmK / volume / num_grid
        for i, gp in enumerate(self._grid_points):
            if self._log_level:
                print ("================= %d/%d =================" %
                       (i + 1, len(self._grid_points)))

            self._pp.set_triplets_at_q(gp)
            self._pp.set_interaction_strength()
            gv = get_group_velocity(
                self._pp.get_qpoint(),
                self._pp.get_dynamical_matrix(),
                reciprocal_lattice,
                eigenvectors=self._pp.get_eigenvectors(),
                frequencies=self._pp.get_frequencies())

            direction = [1, 0, 0]
            if self._no_kappa_stars:
                rot_unit_n = [np.array(direction)]
            else:
                rot_unit_n = self._get_rotated_unit_directions(direction, gp)
                # check if the number of rotations is correct.
                if self._grid_weights[i] > 0:
                    assert len(rot_unit_n) == self._grid_weights[i]
                
            gv_sum2 = np.zeros(len(self._pp.get_frequencies()), dtype=float)
            for unit_n in rot_unit_n:
                gv_sum2 += np.dot(unit_n, gv) ** 2

            if self._log_level:
                filename = "kappa-m%d%d%d-%d.dat" % (tuple(self._mesh) + (gp,))
                print "----- Partial kappa at grid address %d -----" % gp

                print "Kappa at temperatures at grid adress %d are written into %s" % (gp, filename)
                print "Frequency, Group velocity (x y z):"
                for f, v in zip(self._pp.get_frequencies(), gv.T):
                    print "%8.3f (%8.3f %8.3f %8.3f)" % ((f,) + tuple(v))
                print "Frequency, projected group velocity (GV), and GV squared"
                for unit_n in rot_unit_n:
                    print "Direction:", unit_n
                    for f, v in zip(self._pp.get_frequencies(),
                                    np.dot(unit_n, gv)):
                        print "%8.3f %8.3f %12.3f" % (f, v, v**2)
                print "Sigma for delta function smearing: %f" % self._sigma
                
            gamma, cv = self._get_gamma_and_cv(gamma_option=gamma_option)
            lt_cv = np.zeros_like(gamma)
            for j in range(len(self._temperatures)):
                for k in range(len(self._pp.get_frequencies())):
                    if gamma[j, k] > 0:
                        lt_cv[j, k] = cv[j, k] / gamma[j, k] / 2
            kappa[i] = np.dot(lt_cv, gv_sum2) * conversion_factor

            if self._log_level:
                w = open(filename, 'w')
                for t, g in zip(self._temperatures, kappa[i]):
                    w.write("%6.1f %.5f\n" % (t, g.sum()))
                w.close()

        return kappa

    def _get_gamma_and_cv(self,
                          gamma_option=0):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = self._pp.get_frequencies()
        cutoff_freq = self._pp.get_cutoff_frequency()

        gamma = -1 * np.ones((len(self._temperatures), len(freqs)), dtype=float)
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype=float)

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
                                      cutoff_freq,
                                      gamma_option)[0] * unit_conversion
                        cv[i, j] = get_cv(f / freq_conv_factor, t)
                        gamma[i, j] = g
        return gamma, cv

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

def sum_partial_kappa(filenames):
    temps, kappa = parse_kappa(filenames[0])
    sum_kappa = np.array(kappa)
    for filename in filenames[1:]:
        temps, kappa = parse_kappa(filename)
        sum_kappa += kappa

    return temps, sum_kappa
        

if __name__ == '__main__':
    import sys
    temps, kappa = sum_partial_kappa(sys.argv[1:])
    for t, k in zip(temps, kappa):
        print "%8.2f %.5f" % (t, k)

