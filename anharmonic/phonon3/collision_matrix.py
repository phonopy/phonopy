import numpy as np
from anharmonic.phonon3.imag_self_energy import ImagSelfEnergy

class CollisionMatrix(ImagSelfEnergy):
    def __init__(self,
                 interaction,
                 grid_point=None,
                 frequency_points=None,
                 temperature=None,
                 sigma=None,
                 lang='C'):

        self._interaction = None
        self._sigma = None
        self._frequency_points = None
        self._temperature = None
        self._grid_point = None
        self._lang = None
        self._imag_self_energy = None
        self._collision_matrix = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = None
        self._g = None
        self._mesh = None
        self._unit_conversion = None
        
        ImagSelfEnergy.__init__(self,
                                interaction,
                                grid_point=None,
                                frequency_points=None,
                                temperature=None,
                                sigma=None,
                                lang='C')
        
    def run(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        # num_band0 is supposed to be equal to num_band.
        num_band0 = self._fc3_normal_squared.shape[1]
        num_band = self._fc3_normal_squared.shape[2]
        num_triplets = len(self._triplets_at_q)
        self._imag_self_energy = np.zeros(num_band0, dtype='double')
        self._collision_matrix = np.zeros((num_band0, num_triplets, num_band),
                                          dtype='double')
        self._run_collision_matrix()

    def _run_collision_matrix(self):
        self._run_with_band_indices()
        self._run_py_collision_matrix()

    def _run_py_collision_matrix(self):
        g = np.zeros((2,) + self._fc3_normal_squared.shape, dtype='double')
        if self._temperature > 0:
            self._set_collision_matrix()
        else:
            self._set_collision_matrix_0K()
        
    def _set_collision_matrix(self):
        freqs = self._frequencies[self._triplets_at_q[:, [1, 2]]]
        freqs = np.where(freqs > self._cutoff_frequency, freqs, 1)
        n = occupation(freqs, self._temperature)
        for i, (tp, w, interaction) in enumerate(zip(self._triplets_at_q,
                                                     self._weights_at_q,
                                                     self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = n[i, 0, j]
                    n3 = n[i, 1, k]
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    self._imag_self_energy[:] += (
                        (n2 + n3 + 1) * g1 +
                        (n2 - n3) * (g2_g3)) * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _set_collision_matrix_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion
