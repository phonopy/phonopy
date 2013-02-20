import numpy as np
from anharmonic.file_IO import write_fwhm
from anharmonic.im_self_energy import get_gamma

class Linewidth:
    def __init__(self,
                 interaction_strength,
                 sigmas=[0.2],
                 t_max=1000,
                 t_min=0,
                 t_step=10):
        self._pp = interaction_strength
        self._sigmas = sigmas
        self._t_max = t_max
        self._t_min = t_min
        self._t_step = t_step

    def get_linewidth(self,
                      gamma_option=0,
                      filename=None):
        band_indices = self._pp.get_band_indices()
        for sigma in self._sigmas:
            fwhms, temps, freqs = self._get_linewidth_at_sigma(sigma)
            mesh = self._pp.get_mesh_numbers()
            is_nosym = self._pp.is_nosym()
            grid_point = self._pp.get_grid_point()
            for i in range(len(band_indices)):
                write_fwhm(grid_point,
                           [band_indices[i] + 1],
                           temps,
                           fwhms[i],
                           mesh,
                           sigma=sigma,
                           is_nosym=is_nosym,
                           filename=filename)
    
            write_fwhm(grid_point,
                       band_indices + 1,
                       temps,
                       fwhms.sum(axis=0),
                       mesh,
                       sigma=sigma,
                       is_nosym=is_nosym,
                       filename=filename)

        
    def _get_linewidth_at_sigma(self, sigma, filename=None):
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        unit_conversion = self._pp.get_unit_conversion_factor()

        # After pp.set_interaction_strength()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        freqs = [self._pp.get_frequencies()[x]
                 for x in self._pp.get_band_indices()]

        temps = np.arange(self._t_min,
                          self._t_max + float(self._t_step) / 2,
                          self._t_step)
        gammas = np.zeros((len(freqs), len(temps)), dtype=float)

        for i, f in enumerate(freqs):
            for j, t in enumerate(temps):
                g = get_gamma(amplitude_at_q,
                              np.array([f], dtype=float),
                              weights_at_q,
                              frequencies_at_q,
                              i,
                              t,
                              sigma,
                              freq_conv_factor,
                              cutoff_frequency=self._pp.get_cutoff_frequency(),
                              gamma_option=gamma_option)[0] * unit_conversion
                gammas[i, j] = g

        fwhms = gammas * 2

        return fwhms, temps, freqs

