import numpy as np
import phonopy.structure.spglib as spg
from anharmonic.im_self_energy import get_gamma
from phonopy.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv, EV, THz, Angstrom
from anharmonic.file_IO import parse_kappa, write_kappa
from anharmonic.triplets import get_grid_address

class BTE_RTA:
    def __init__(self,
                 interaction_strength,
                 sigmas=[0.1],
                 t_max=1500,
                 t_min=0,
                 t_step=10,
                 no_kappa_stars=False,
                 gamma_option=0,
                 log_level=0,
                 filename=None):
        self._pp = interaction_strength
        self._sigmas = sigmas
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._no_kappa_stars = no_kappa_stars
        print "No kappa", self._no_kappa_stars
        self._gamma_option = gamma_option
        self._log_level = log_level
        self._filename = filename

        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)
        self._primitive = self._pp.get_primitive()
        self._mesh = None
        self._grid_points = None
        self._grid_weights = None
        self._grid_address = None
        self._point_operations = np.array(
            [rot.T for rot in
             self._pp.get_symmetry().get_pointgroup_operations()])

        self._gamma = None

    def set_mesh_numbers(self, mesh=None):
        if mesh is None:
            self._mesh = self._pp.get_mesh_numbers()
        else:
            self._mesh = np.array(mesh)

    def get_mesh_numbers(self):
        return self._mesh
        
    def set_grid_points(self, grid_points=None):
        if grid_points is None:
            if self._no_kappa_stars:
                self._grid_points = range(np.prod(self._mesh))
            else:
                (grid_mapping_table,
                 self._grid_address) = spg.get_ir_reciprocal_mesh(
                    self._mesh,
                    self._primitive)
                self._grid_points = np.unique(grid_mapping_table)
                self._grid_weights = [np.sum(grid_mapping_table == g)
                                      for g in self._grid_points]
        else:
            self._grid_address = get_grid_address(self._mesh)
            self._grid_points = grid_points

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures

    def get_temperatures(self):
        return self._temperatures

    def set_gamma(self, gamma):
        self._gamma = gamma
    
    def get_kappa(self):
        unit_to_WmK = (THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz / 2 / np.pi
        volume = self._primitive.get_volume()
        num_grid = np.prod(self._mesh)
        conversion_factor = unit_to_WmK / volume / num_grid

        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        cutoff_freq = self._pp.get_cutoff_frequency()
        reciprocal_lattice = np.linalg.inv(self._primitive.get_cell())
        num_atom = self._primitive.get_number_of_atoms()
        kappa = np.zeros((len(self._sigmas),
                          len(self._grid_points),
                          len(self._temperatures),
                          num_atom * 3), dtype=float)
        # if gamma is not set.
        if self._gamma is None:
            gamma = np.zeros_like(kappa)
        else:
            gamma = self._gamma

        for i, grid_point in enumerate(self._grid_points):
            if self._log_level:
                print ("================= %d/%d =================" %
                       (i + 1, len(self._grid_points)))

            self._pp.set_triplets_at_q(grid_point)
            self._pp.set_interaction_strength()

            # Group velocity
            gv = get_group_velocity(
                self._pp.get_qpoint(),
                self._pp.get_dynamical_matrix(),
                reciprocal_lattice,
                eigenvectors=self._pp.get_eigenvectors(),
                frequencies=self._pp.get_frequencies())

            # Sum group velocities at symmetrically equivalent q-points
            direction = [1, 0, 0]
            if self._no_kappa_stars:
                rot_unit_n = [np.array(direction)]
            else:
                rot_unit_n = self._get_rotated_unit_directions(direction,
                                                               grid_point)
                # check if the number of rotations is correct.
                if self._grid_weights is not None:
                    assert len(rot_unit_n) == self._grid_weights[i]
                
            gv_sum2 = np.zeros(len(self._pp.get_frequencies()), dtype=float)
            for unit_n in rot_unit_n:
                gv_sum2 += np.dot(unit_n, gv) ** 2

            self._show_log(grid_point, gv, rot_unit_n)
    
            # Heat capacity
            cv = self._get_cv(freq_conv_factor, cutoff_freq)

            # Kappa and Gamma
            self._get_kappa_at_sigmas(i,
                                      kappa,
                                      gamma,
                                      gv_sum2,
                                      cv,
                                      conversion_factor)
            if self._log_level:
                for j, sigma in enumerate(self._sigmas):
                    write_kappa(kappa[j, i].sum(axis=1),
                                self._temperatures,
                                self._mesh,
                                gamma=gamma[j, i],
                                grid_point=grid_point,
                                sigma=sigma,
                                filename=self._filename)

        if self._log_level:
            if self._grid_weights is not None:
                print "-------------- Total kappa --------------"
                for sigma, kappa_at_sigma in zip(self._sigmas, kappa):
                    write_kappa(kappa_at_sigma.sum(axis=0).sum(axis=1),
                                self._temperatures,
                                self._mesh,
                                sigma=sigma,
                                filename=self._filename)

        return kappa, gamma

    def _get_kappa_at_sigmas(self,
                             i,
                             kappa,
                             gamma,
                             gv_sum2,
                             cv,
                             conversion_factor):
        for j, sigma in enumerate(self._sigmas):
            if self._log_level > 0:
                print "Sigma used to approximate delta function by gaussian: %f" % sigma

            if self._gamma is None:
                gamma[j, i] = self._get_gamma(sigma)
            
            for k in range(len(self._temperatures)):
                for l in range(len(self._pp.get_frequencies())):
                    if gamma[j, i, k, l] > 0:
                        kappa[j, i, k, l] = (gv_sum2[l] * cv[k, l] /
                                             gamma[j, i, k, l] / 2 *
                                             conversion_factor)

    def _get_cv(self, freq_conv_factor, cutoff_frequency):
        def get_cv(freqs, t):
            x = freqs * THzToEv / Kb / t
            expVal = np.exp(x)
            return Kb * x ** 2 * expVal / (expVal - 1.0) ** 2 # eV/K

        freqs = self._pp.get_frequencies()
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype=float)
        for i, t in enumerate(self._temperatures):
            if t > 0:
                for j, f in enumerate(freqs):
                    if f > cutoff_frequency:
                        cv[i, j] = get_cv(f / freq_conv_factor, t)

        return cv

    def _get_gamma(self, sigma):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = self._pp.get_frequencies()
        cutoff_freq = self._pp.get_cutoff_frequency()

        gamma = -1 * np.ones((len(self._temperatures), len(freqs)), dtype=float)

        for i, t in enumerate(self._temperatures):
            if t > 0:
                for j, f in enumerate(freqs):
                    if f > cutoff_freq:
                        g = get_gamma(
                            amplitude_at_q,
                            np.array([f], dtype=float),
                            weights_at_q,
                            frequencies_at_q,
                            j,
                            t,
                            sigma,
                            freq_conv_factor,
                            cutoff_frequency=cutoff_freq,
                            gamma_option=self._gamma_option
                            )[0] * unit_conversion
                        gamma[i, j] = g
        return gamma

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

    def _show_log(self, grid_point, group_velocity, rot_unit_n):
        if self._log_level:
            print "----- Partial kappa at grid address %d -----" % grid_point
            print "Frequency, Group velocity (x y z):"
            for f, v in zip(self._pp.get_frequencies(), group_velocity.T):
                print "%8.3f (%8.3f %8.3f %8.3f)" % ((f,) + tuple(v))
            print "Frequency, projected group velocity (GV), and GV squared"
            for unit_n in rot_unit_n:
                print "Direction:", unit_n
                for f, v in zip(self._pp.get_frequencies(),
                                np.dot(unit_n, group_velocity)):
                    print "%8.3f %8.3f %12.3f" % (f, v, v**2)

        
            
        
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

