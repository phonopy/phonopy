import numpy as np
from anharmonic.im_self_energy import get_gamma

class Lifetime:
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
        
    def get_lifetime(self,
                     gamma_option=0,
                     filename=None,
                     verbose=True):
        lifetimes = []
        for i, grid_point in enumerate(self._grid_points):
            if verbose:
                print ("============== %d/%d ===============" %
                       (i + 1, len(self._grid_points)))
            lifetimes.append(self._get_gamma(grid_point,
                                             gamma_option=gamma_option))
        return lifetimes
            
    def _get_gamma(self, grid_point, gamma_option=0):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()

        self._pp.set_triplets_at_q(grid_point)
        self._pp.set_interaction_strength()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = self._pp.get_frequencies()
        gammas = np.zeros((len(freqs), len(self._temperatures)), dtype=float)

        for i, f in enumerate(freqs):
            for j, t in enumerate(self._temperatures):
                g = get_gamma(amplitude_at_q,
                              np.array([f], dtype=float),
                              weights_at_q,
                              frequencies_at_q,
                              i,
                              t,
                              self._sigma,
                              freq_conv_factor,
                              gamma_option)[0] * unit_conversion
                gammas[i, j] = g

        return gammas
