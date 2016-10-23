import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.degeneracy import degenerate_sets
from anharmonic.phonon3.triplets import occupation
from anharmonic.file_IO import write_frequency_shift

def get_frequency_shift(interaction,
                        grid_points,
                        band_indices,
                        epsilons,
                        temperatures=None,
                        output_filename=None,
                        log_level=0):
    if temperatures is None:
        temperatures = [0.0, 300.0]
    fst = FrequencyShift(interaction)
    band_indices_flatten = interaction.get_band_indices()
    mesh = interaction.get_mesh_numbers()
    for gp in grid_points:
        fst.set_grid_point(gp)
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            print("------ Frequency shift -o- ------")
            print("Number of ir-triplets: "
                  "%d / %d" % (len(weights), weights.sum()))
        fst.run_interaction()

        for epsilon in epsilons:
            fst.set_epsilon(epsilon)
            delta = np.zeros((len(temperatures),
                              len(band_indices_flatten)),
                             dtype='double')
            for i, t in enumerate(temperatures):
                fst.set_temperature(t)
                fst.run()
                delta[i] = fst.get_frequency_shift()
    
            for i, bi in enumerate(band_indices):
                pos = 0
                for j in range(i):
                    pos += len(band_indices[j])
    
                write_frequency_shift(gp,
                                      bi,
                                      temperatures,
                                      delta[:, pos:(pos+len(bi))],
                                      mesh,
                                      epsilon=epsilon,
                                      filename=output_filename)

class FrequencyShift(object):
    def __init__(self,
                 interaction,
                 grid_point=None,
                 temperature=None,
                 epsilon=0.1,
                 lang='C'):
        self._pp = interaction
        self.set_epsilon(epsilon)
        self.set_temperature(temperature)
        self.set_grid_point(grid_point=grid_point)

        self._lang = lang
        self._frequency_ = None
        self._pp_strength = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

        self._frequency_shifts = None
        self.set_epsilon(epsilon)

    def run(self):
        if self._pp_strength is None:        
            self.run_interaction()

        num_band0 = self._pp_strength.shape[1]
        self._frequency_shifts = np.zeros(num_band0, dtype='double')
        self._run_with_band_indices()

    def run_interaction(self):
        self._pp.run(lang=self._lang)
        self._pp_strength = self._pp.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._pp.get_phonons()[:2]
        (self._triplets_at_q,
         self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
        self._band_indices = self._pp.get_band_indices()
        
        mesh = self._pp.get_mesh_numbers()
        num_grid = np.prod(mesh)

        # Unit to THz of Delta
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / num_grid)

    def get_frequency_shift(self):
        if self._cutoff_frequency is None:
            return self._frequency_shifts
        else: # Averaging frequency shifts by degenerate bands
            shifts = np.zeros_like(self._frequency_shifts)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs) # such like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        shifts[i] = (self._frequency_shifts[bi_set].sum() /
                                     len(bi_set))
            return shifts

    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            (self._triplets_at_q,
             self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
            self._grid_point = self._triplets_at_q[0, 0]
        
    def set_epsilon(self, epsilon):
        if epsilon is None:
            self._epsilon = None
        else:
            self._epsilon = float(epsilon)

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)
        
    def _run_with_band_indices(self):
        if self._lang == 'C':
            self._run_c_with_band_indices()
        else:
            self._run_py_with_band_indices()
    
    def _run_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        phono3c.frequency_shift_at_bands(self._frequency_shifts,
                                         self._pp_strength,
                                         self._triplets_at_q,
                                         self._weights_at_q,
                                         self._frequencies,
                                         self._band_indices,
                                         self._temperature,
                                         self._epsilon,
                                         self._unit_conversion,
                                         self._cutoff_frequency)

    def _run_py_with_band_indices(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._triplets_at_q,
                self._weights_at_q,
                self._pp_strength)):

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if self._temperature > 0:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands(
                            j, bi, freqs, interaction, w))
                else:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands_0K(
                            j, bi, freqs, interaction, w))

        self._frequency_shifts *= self._unit_conversion

    def _frequency_shifts_at_bands(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                n2 = occupation(freqs[1, j], self._temperature)
                n3 = occupation(freqs[2, k], self._temperature)
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]
                f3 = freqs[0, bi] - freqs[1, j] + freqs[2, k]
                f4 = freqs[0, bi] + freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= (n2 + n3 + 1) / f1
                # if abs(f2) > self._epsilon:
                #     d += (n2 + n3 + 1) / f2
                # if abs(f3) > self._epsilon:
                #     d -= (n2 - n3) / f3
                # if abs(f4) > self._epsilon:
                #     d += (n2 - n3) / f4
                d -= (n2 + n3 + 1) * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += (n2 + n3 + 1) * f2 / (f2 ** 2 + self._epsilon ** 2)
                d -= (n2 - n3) * f3 / (f3 ** 2 + self._epsilon ** 2)
                d += (n2 - n3) * f4 / (f4 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d

    def _frequency_shifts_at_bands_0K(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= 1.0 / f1
                # if abs(f2) > self._epsilon:
                #     d += 1.0 / f2
                d -= 1.0 * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += 1.0 * f2 / (f2 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d
