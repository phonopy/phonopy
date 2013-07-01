import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1) 
    

class Gamma:
    def __init__(self,
                 interaction_strength,
                 frequencies,
                 eigenvectors,
                 triplets,
                 triplet_weights,
                 sigma=0.1):
        self._interaction_strength = interaction_strength
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._triplets = triplets
        self._triplet_weights = triplet_weights
        self._sigma = sigma

        self._temperature = -1
        self._fpoints = None
        num_band = len(frequencies[0])
        self._band_indices = np.arange(num_band, dtype='intc')
        self._gamma = None

        # Unit to THz of Gamma
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 * np.pi / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2)

    def run(self):
        num_band = len(self._band_indices)
        num_fpoints = len(self._fpoints)
        self._gamma = np.zeros((num_band, num_fpoints), dtype='double')
        self._run_gamma()
        self._gamma *= self._unit_conversion / len(self._frequencies)

    def get_gamma(self):
        return self._gamma

    def set_temperature(self, t):
        self._temperature = t
    
    def set_frequency_points(self, fpoints):
        self._fpoints = fpoints

    def set_grid_point(self, grid_point):
        self._grid_point = grid_point

    def set_band_indices(self, band_indices):
        """
        This is used when specific band indices are calculated."
        Band index starts by 0.
        The default is to calculate all bands.
        """
        self._band_indices = np.intc(band_indices)

    def _run_gamma(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._triplets,
                self._triplet_weights,
                self._interaction_strength)):
            print "%d / %d" % (i + 1, len(self._triplets))

            interaction_band = interaction[self._band_indices]
            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._gamma_at_bands(freqs, interaction, w)
            else:
                self._gamma_at_bands_0K(freqs, interaction, w)

    def _gamma_at_bands(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            if freqs[0][j] > 0 and freqs[1][k] > 0:
                n2 = occupation(freqs[0][j], self._temperature)
                n3 = occupation(freqs[1][k], self._temperature)
                g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                              self._sigma)
                g2 = gaussian(self._fpoints + freqs[0][j] - freqs[1][k],
                              self._sigma)
                g3 = gaussian(self._fpoints - freqs[0][j] + freqs[1][k],
                              self._sigma)
                self._gamma[i] += (
                    (n2 + n3 + 1) * g1 +
                    (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight

    def _gamma_at_bands_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                          self._sigma)
            self._gamma[i] += g1 * interaction[i, j, k] * weight
        
