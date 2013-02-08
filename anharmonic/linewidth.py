import numpy as np
from anharmonic.file_IO import write_fwhm
from anharmonic.im_self_energy import get_gamma

class Linewidth:
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

    def get_linewidth(self,
                      gamma_option=0,
                      filename=None):
        num_atom = self._pp.get_primitive().get_number_of_atoms()
        mesh = self._pp.get_mesh_numbers()
        q_direction = self._pp.get_q_direction()
        freq_scale_factor = self._pp.get_frequency_scale_factor()
        freq_conv_factor = self._pp.get_frequency_unit_conversion_factor()
        freq_factor_to_THz = self._pp.get_frequency_factor_to_THz()
        factor = freq_conv_factor * freq_scale_factor * freq_factor_to_THz
        unit_conversion = self._pp.get_unit_conversion_factor()
        is_nosym = self._pp.is_nosym()

        # After pp.set_interaction_strength()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        band_indices = self._pp.get_band_indices()
        grid_point = self._pp.get_grid_point()
        grid_address = self._pp.get_grid_address()

        # After pp.set_dynamical_matrix()
        q = grid_address[grid_point].astype(float) / mesh
        dm = self._pp.get_dynamical_matrix()
        if (not q_direction is not None) and grid_point == 0:
            dm.set_dynamical_matrix(q, q_direction)
        else:
            dm.set_dynamical_matrix(q)
        vals = np.linalg.eigvalsh(dm.get_dynamical_matrix()).real
        freqs = np.sqrt(abs(vals)) * np.sign(vals) * factor
        omegas = [freqs[x] for x in band_indices]


        temps = np.arange(self._t_min,
                          self._t_max + float(self._t_step) / 2,
                          self._t_step)
        gammas = np.zeros((len(omegas), len(temps)), dtype=float)

        for i in range(len(band_indices)):
            for j, t in enumerate(temps):
                g = get_gamma(amplitude_at_q,
                              np.array([omegas[i]], dtype=float),
                              weights_at_q,
                              frequencies_at_q,
                              i,
                              t,
                              self._sigma,
                              freq_conv_factor,
                              gamma_option)[0] * unit_conversion
                gammas[i, j] = g


        fwhms = gammas * 2
        for i in range(len(band_indices)):
            write_fwhm(grid_point,
                       [band_indices[i] + 1],
                       temps,
                       fwhms[i],
                       mesh,
                       is_nosym=is_nosym,
                       filename=filename)

        write_fwhm(grid_point,
                   band_indices + 1,
                   temps,
                   fwhms.sum(axis=0),
                   mesh,
                   is_nosym=is_nosym,
                   filename=filename)

        return fwhms, temps, omegas

