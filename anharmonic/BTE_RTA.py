import numpy as np
from anharmonic.im_self_energy import get_gamma
from phonopy.group_velocity import get_group_velocity
from phonopy.units import Kb, THzToEv

class BTE_RTA:
    def __init__(self,
                 interaction_strength,
                 sigma=0.2,
                 t_max=1000,
                 t_min=0,
                 t_step=10):
        self._pp = interaction_strength
        self._sigma = sigma
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step
        self._temperatures = np.arange(self._t_min,
                                       self._t_max + float(self._t_step) / 2,
                                       self._t_step)

        print "Lifetime is long"
        
    def set_grid_points(self, grid_points):
        self._grid_points = grid_points

    def set_temperatures(self, temperatures):
        self._temperatures = temperatures
        
    def get_kappa(self,
                  gamma_option=0,
                  filename=None,
                  verbose=True):
        partial_k = []
        for i, grid_point in enumerate(self._grid_points):
            if verbose:
                print ("============== %d/%d ===============" %
                       (i + 1, len(self._grid_points)))
            # lifetimes.append(self._get_gamma(grid_point,
            #                                  gamma_option=gamma_option,
            #                                  verbose=verbose))
            partial_k.append(self._get_gamma(grid_point,
                                             gamma_option=gamma_option,
                                             verbose=verbose))
        return partial_k
            

    def _get_gamma(self, grid_point, gamma_option=0, verbose=True):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()
        reciprocal_lattice = np.linalg.inv(self._pp.get_primitive().get_cell())

        self._pp.set_triplets_at_q(grid_point)
        self._pp.set_interaction_strength()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = self._pp.get_frequencies()

        gv = get_group_velocity(self._pp.get_qpoint(),
                                [1, 0, 0], # direction
                                self._pp.get_dynamical_matrix(),
                                reciprocal_lattice,
                                eigenvectors=self._pp.get_eigenvectors(),
                                frequencies=freqs)

        gammas = np.zeros((len(self._temperatures), len(freqs)), dtype=float)

        for i, t in enumerate(self._temperatures):
            if t > 0:
                for j, f in enumerate(freqs):
                    if f > 0:
                        g = get_gamma(amplitude_at_q,
                                      np.array([f], dtype=float),
                                      weights_at_q,
                                      frequencies_at_q,
                                      j,
                                      t,
                                      self._sigma,
                                      freq_conv_factor,
                                      gamma_option)[0] * unit_conversion
                        gammas[i, j] = get_cv(f, t) / g

        partial_k = np.dot(gammas, (gv ** 2).sum(axis=0))
 
        w = open("partial-k-%d.dat" % grid_point, 'w')
        for t, g in zip(self._temperatures, partial_k):
            w.write("%6.1f %f\n" % (t, g.sum()))
        w.close()

        return partial_k
        
def get_cv(freqs, t):
    x = freqs * THzToEv / Kb / t
    expVal = np.exp(x)
    return Kb * x ** 2 * expVal / (expVal - 1.0) ** 2 / THzToEv # THz/K
