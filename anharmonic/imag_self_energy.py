import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1) 
    

class ImagSelfEnergy:
    def __init__(self,
                 interaction,
                 grid_point=None,
                 fpoints=None,
                 temperature=None,
                 sigma=0.1,
                 lang='C'):
        self._interaction = interaction
        self.set_sigma(sigma)
        self.set_temperature(temperature)
        self.set_fpoints(fpoints)
        self.set_grid_point(grid_point=grid_point)

        self._lang = lang
        self._imag_self_energy = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._grid_point_triplets = None
        self._triplet_weights = None
        self._unit_conversion = None

    def run(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        num_band0 = self._fc3_normal_squared.shape[1]
        if self._fpoints is None:
            self._imag_self_energy = np.zeros(num_band0, dtype='double')
            if self._lang == 'C':
                self._run_c_with_band_indices()
            else:
                self._run_py_with_band_indices()
        else:
            self._imag_self_energy = np.zeros((len(self._fpoints), num_band0),
                                              dtype='double')
            if self._lang == 'C':
                self._run_c_with_fpoints()
            else:
                self._run_py_with_fpoints()

    def run_interaction(self):
        self._interaction.run(lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        (self._grid_point_triplets,
         self._triplet_weights) = self._interaction.get_triplets_at_q()
        
        # Unit to THz of Gamma
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 * np.pi / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / len(self._frequencies))

    def get_imag_self_energy(self):
        return self._imag_self_energy

    def get_phonon_at_grid_point(self):
        return (self._frequencies[self._grid_point],
                self._eigenvectors[self._grid_point])

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._grid_point = grid_point
            self._interaction.set_grid_point(grid_point)
            self._fc3_normal_squared = None
        
    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    def set_fpoints(self, fpoints):
        if fpoints is None:
            self._fpoints = None
        else:
            self._fpoints = np.double(fpoints)

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)
        
    def _run_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        band_indices = self._interaction.get_band_indices()
        phono3c.imag_self_energy_at_bands(self._imag_self_energy,
                                          self._fc3_normal_squared,
                                          self._grid_point_triplets,
                                          self._triplet_weights,
                                          self._frequencies,
                                          band_indices,
                                          self._temperature,
                                          self._sigma,
                                          self._unit_conversion)

    def _run_c_with_fpoints(self):
        import anharmonic._phono3py as phono3c
        for i, fpoint in enumerate(self._fpoints):
            phono3c.imag_self_energy(self._imag_self_energy[i],
                                     self._fc3_normal_squared,
                                     self._grid_point_triplets,
                                     self._triplet_weights,
                                     self._frequencies,
                                     fpoint,
                                     self._temperature,
                                     self._sigma,
                                     self._unit_conversion)

    def _run_py_with_band_indices(self):
        band_indices = self._interaction.get_band_indices()
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            freqs = self._frequencies[triplet]

            for j, bi in enumerate(band_indices):
                if self._temperature > 0:
                    self._imag_self_energy[j] = (
                        self._imag_self_energy_at_bands(
                            bi, freqs, interaction, w))
                else:
                    self._imag_self_energy[j] = (
                        self._imag_self_energy_at_bands_0K(
                            bi, freqs, interaction, w))

        self._imag_self_energy *= self._unit_conversion

    def _imag_self_energy_at_bands(self, i, freqs, interaction, weight):
        sum_g = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if freqs[1][j] > 0 and freqs[2][k] > 0:
                n2 = occupation(freqs[1][j], self._temperature)
                n3 = occupation(freqs[2][k], self._temperature)
                g1 = gaussian(freqs[0, i] - freqs[1, j] - freqs[2, k],
                              self._sigma)
                g2 = gaussian(freqs[0, i] + freqs[1, j] - freqs[2, k],
                              self._sigma)
                g3 = gaussian(freqs[0, i] - freqs[1, j] + freqs[2, k],
                              self._sigma)
                sum_g += ((n2 + n3 + 1) * g1 +
                          (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight
        return sum_g

    def _imag_self_energy_at_bands_0K(self, i, freqs, interaction, weight):
        sum_g = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            g1 = gaussian(freqs[0, i] - freqs[1, j] - freqs[2, k],
                          self._sigma)
            sum_g += g1 * interaction[i, j, k] * weight

        return sum_g


    def _run_py_with_fpoints(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._imag_self_energy_with_fpoints(freqs, interaction, w)
            else:
                self._imag_self_energy_with_fpoints_0K(freqs, interaction, w)

        self._imag_self_energy *= self._unit_conversion

    def _imag_self_energy_with_fpoints(self, freqs, interaction, weight):
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
                self._imag_self_energy[:, i] += (
                    (n2 + n3 + 1) * g1 +
                    (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight

    def _imag_self_energy_with_fpoints_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                          self._sigma)
            self._imag_self_energy[:, i] += g1 * interaction[i, j, k] * weight
        
