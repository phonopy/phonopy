import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1) 
    

class ImagSelfEnergy:
    def __init__(self,
                 interaction,
                 grid_point=0,
                 fpoints=None,
                 temperature=-1,
                 sigma=0.1,
                 lang='C'):
        self._interaction = interaction
        self._temperature = temperature
        self._sigma = sigma
        self._fpoints = fpoints
        self._grid_point = grid_point

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

        if self._fpoints is not None:
            num_band0 = self._fc3_normal_squared.shape[1]
            self._imag_self_energy = np.zeros((len(self._fpoints), num_band0),
                                              dtype='double')
            if self._lang == 'C':
                self._run_c()
            else:
                self._run_py()

    def run_interaction(self):
        self._interaction.set_triplets_at_q(self._grid_point)
        self._interaction.run(lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:1]
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

    def set_grid_point(self, grid_point):
        self._grid_point = grid_point
        self._fc3_normal_squared = None
        
    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_fpoints(self, fpoints):
        self._fpoints = fpoints

    def set_temperature(self, temperature):
        self._temperature = temperature
        
    def _run_c(self):
        import anharmonic._phono3py as phono3c
        num_band0 = self._imag_self_energy.shape[1]
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

    def _run_py(self):
        self._run_imag_self_energy()
        self._imag_self_energy *= self._unit_conversion

    def _run_imag_self_energy(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._grid_point_triplets,
                self._triplet_weights,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._grid_point_triplets))

            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._imag_self_energy_at_bands(freqs, interaction, w)
            else:
                self._imag_self_energy_at_bands_0K(freqs, interaction, w)

    def _imag_self_energy_at_bands(self, freqs, interaction, weight):
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

    def _imag_self_energy__at_bands_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            g1 = gaussian(self._fpoints - freqs[0][j] - freqs[1][k],
                          self._sigma)
            self._imag_self_energy[:, i] += g1 * interaction[i, j, k] * weight
        
