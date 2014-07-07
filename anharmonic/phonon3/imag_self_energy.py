import numpy as np
from phonopy.units import VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.structure.spglib import get_neighboring_grid_points
from anharmonic.phonon3.triplets import get_triplets_integration_weights, gaussian, occupation
import anharmonic.file_IO as file_IO

def get_imag_self_energy(interaction,
                         grid_points,
                         sigmas,
                         frequency_step=None,
                         num_frequency_points=None,
                         temperatures=[0.0, 300.0],
                         log_level=0):
    if temperatures is None:
        print "Temperatures have to be set."
        return False

    mesh = interaction.get_mesh_numbers()
    ise = ImagSelfEnergy(interaction)
    imag_self_energy = []
    frequency_points = []
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            print "------ Imaginary part of self energy ------"
            print "Grid point: %d" % gp
            print "Number of ir-triplets:",
            print "%d / %d" % (len(weights), weights.sum())
        ise.run_interaction()
        frequencies = interaction.get_phonons()[0]
        max_phonon_freq = np.amax(frequencies)

        if log_level:
            adrs = interaction.get_grid_address()[gp]
            q = adrs.astype('double') / mesh
            print "q-point:", q
            print "Phonon frequency:"
            print "[",
            for i, freq in enumerate(frequencies[gp]):
                if i % 6 == 0 and i != 0:
                    print
                print "%8.4f" % freq,
            print "]"

        ise_sigmas = []
        fp_sigmas = []
        for j, sigma in enumerate(sigmas):
            if log_level:
                if sigma:
                    print "Sigma:", sigma
                else:
                    print "Tetrahedron method"
            ise.set_sigma(sigma)
            if sigma:
                fmax = max_phonon_freq * 2 + sigma * 4
            else:
                fmax = max_phonon_freq * 2
            fmax *= 1.005
            fmin = 0
            frequency_points_at_sigma = get_frequency_points(
                fmin,
                fmax,
                frequency_step=frequency_step,
                num_frequency_points=num_frequency_points)
            fp_sigmas.append(frequency_points_at_sigma)
            ise_temperatures = np.zeros(
                (len(temperatures), len(frequency_points_at_sigma),
                 len(interaction.get_band_indices())), dtype='double')
                 
            for k, freq_point in enumerate(frequency_points_at_sigma):
                ise.set_frequency_points([freq_point])
                if not sigma:
                    ise.set_integration_weights()
    
                for l, t in enumerate(temperatures):
                    ise.set_temperature(t)
                    ise.run()
                    ise_temperatures[l, k] = ise.get_imag_self_energy()[0]
                    
                ise_sigmas.append(ise_temperatures)
            
        imag_self_energy.append(ise_sigmas)
        frequency_points.append(fp_sigmas)
                
    return imag_self_energy, frequency_points

def get_frequency_points(f_min,
                         f_max,
                         frequency_step=None,
                         num_frequency_points=None):
    if num_frequency_points is None:
        if frequency_step is not None:
            frequency_points = np.arange(
                f_min, f_max, frequency_step, dtype='double')
        else:
            frequency_points = np.array(np.linspace(
                f_min, f_max, 201), dtype='double')
    else:
        frequency_points = np.array(np.linspace(
            f_min, f_max, num_frequency_points), dtype='double')
        
    return frequency_points
    
def get_linewidth(interaction,
                  grid_points,
                  sigmas,
                  temperatures=np.arange(0, 1001, 10, dtype='double'),
                  log_level=0):
    ise = ImagSelfEnergy(interaction)
    band_indices = interaction.get_band_indices()
    mesh = interaction.get_mesh_numbers()
    gamma = np.zeros(
        (len(grid_points), len(sigmas), len(temperatures), len(band_indices)),
        dtype='double')
    
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            print "------ Linewidth ------"
            print "Grid point: %d" % gp
            print "Number of ir-triplets:",
            print "%d / %d" % (len(weights), weights.sum())
        ise.run_interaction()
        frequencies = interaction.get_phonons()[0]
        if log_level:
            adrs = interaction.get_grid_address()[gp]
            q = adrs.astype('double') / mesh
            print "q-point:", q
            print "Phonon frequency:"
            print frequencies[gp]
        
        for j, sigma in enumerate(sigmas):
            if log_level:
                if sigma:
                    print "Sigma:", sigma
                else:
                    print "Tetrahedron method"
            ise.set_sigma(sigma)
            if not sigma:
                ise.set_integration_weights()
            
            for k, t in enumerate(temperatures):
                ise.set_temperature(t)
                ise.run()
                gamma[i, j, k] = ise.get_imag_self_energy()

    return gamma

def write_linewidth(linewidth,
                    band_indices,
                    mesh,
                    grid_points,
                    sigmas,
                    temperatures,
                    filename=None):
    for i, gp in enumerate(grid_points):
        for j, sigma in enumerate(sigmas):
            for k, bi in enumerate(band_indices):
                pos = 0
                for l in range(k):
                    pos += len(band_indices[l])
            file_IO.write_linewidth(
                gp,
                bi,
                temperatures,
                linewidth[i, j, :, pos:(pos+len(bi))],
                mesh,
                sigma=sigma,
                filename=filename)

def write_imag_self_energy(imag_self_energy,
                           mesh,
                           grid_points,
                           band_indices,
                           frequency_points,
                           temperatures,
                           sigmas,
                           filename=None):
    for gp, ise_sigmas, fp_sigmas in zip(grid_points,
                                         imag_self_energy,
                                         frequency_points):
        for sigma, ise_temps, fp in zip(sigmas, ise_sigmas, fp_sigmas):
            for t, ise in zip(temperatures, ise_temps):
                 for i, bi in enumerate(band_indices):
                     pos = 0
                     for j in range(i):
                         pos += len(band_indices[j])
                     file_IO.write_damping_functions(
                         gp,
                         bi,
                         mesh,
                         fp,
                         ise[:, pos:(pos + len(bi))].sum(axis=1) / len(bi),
                         sigma=sigma,
                         temperature=t,
                         filename=filename)
    
class ImagSelfEnergy:
    def __init__(self,
                 interaction,
                 grid_point=None,
                 frequency_points=None,
                 temperature=None,
                 sigma=None,
                 lang='C'):
        self._interaction = interaction
        self._sigma = None
        self.set_sigma(sigma)
        self._temperature = None
        self.set_temperature(temperature)
        self._frequency_points = None
        self.set_frequency_points(frequency_points)
        self._grid_point = None
        self.set_grid_point(grid_point)

        self._lang = lang
        self._imag_self_energy = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

        self._g = None # integration weights
        self._mesh = self._interaction.get_mesh_numbers()
        self._is_collision_matrix = False

        # Unit to THz of Gamma
        num_grid = np.prod(self._mesh)
        self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                 * EV ** 2 / Angstrom ** 6
                                 / (2 * np.pi * THz) ** 3
                                 / AMU ** 3
                                 * 18 * np.pi / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 / num_grid)

    def run(self):
        if self._fc3_normal_squared is None:        
            self.run_interaction()

        num_band0 = self._fc3_normal_squared.shape[1]
        if self._frequency_points is None:
            self._imag_self_energy = np.zeros(num_band0, dtype='double')
            self._run_with_band_indices()
        else:
            self._imag_self_energy = np.zeros(
                (len(self._frequency_points), num_band0), dtype='double')
            self._run_with_frequency_points()

    def run_interaction(self):
        self._interaction.run(lang=self._lang)
        self._fc3_normal_squared = self._interaction.get_interaction_strength()
        (self._frequencies,
         self._eigenvectors) = self._interaction.get_phonons()[:2]
        self._band_indices = self._interaction.get_band_indices()

    def set_integration_weights(self):
        if self._frequency_points is None:
            f_points = self._frequencies[self._grid_point][self._band_indices]
        else:
            f_points = self._frequency_points

        self._g = get_triplets_integration_weights(
            self._interaction,
            f_points,
            self._sigma,
            is_collision_matrix=self._is_collision_matrix)
        
    def get_imag_self_energy(self):
        if self._cutoff_frequency is None:
            return self._imag_self_energy

        # Averaging imag-self-energies by degenerate bands
        imag_se = np.zeros_like(self._imag_self_energy)
        freqs = self._frequencies[self._grid_point]
        deg_sets = degenerate_sets(freqs)
        for dset in deg_sets:
            bi_set = []
            for i, bi in enumerate(self._band_indices):
                if bi in dset:
                    bi_set.append(i)
            for i in bi_set:
                if self._frequency_points is None:
                    imag_se[i] = (self._imag_self_energy[bi_set].sum() /
                                  len(bi_set))
                else:
                    imag_se[:, i] = (
                        self._imag_self_energy[:, bi_set].sum(axis=1) /
                        len(bi_set))
        return imag_se
            
    def set_grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._interaction.set_grid_point(grid_point)
            self._fc3_normal_squared = None
            (self._triplets_at_q,
             self._weights_at_q) = self._interaction.get_triplets_at_q()
            self._grid_point = grid_point
            
    def set_sigma(self, sigma):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

    def set_frequency_points(self, frequency_points):
        if frequency_points is None:
            self._frequency_points = None
        else:
            self._frequency_points = np.array(frequency_points, dtype='double')

    def set_temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)
        
    def _run_with_band_indices(self):
        if self._sigma is None:
            if self._lang == 'C':
                self._run_thm_c_with_band_indices()
            else:
                self._run_thm_py_with_band_indices()
        else:
            if self._lang == 'C':
                self._run_c_with_band_indices()
            else:
                self._run_py_with_band_indices()
    
    def _run_with_frequency_points(self):
        if self._sigma is None:
            if self._lang == 'C':
                self._run_thm_c_with_frequency_points()
            else:
                self._run_thm_py_with_frequency_points()
        else:
            if self._lang == 'C':
                self._run_c_with_frequency_points()
            else:
                self._run_py_with_frequency_points()

    def _run_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        phono3c.imag_self_energy_at_bands(self._imag_self_energy,
                                          self._fc3_normal_squared,
                                          self._triplets_at_q,
                                          self._weights_at_q,
                                          self._frequencies,
                                          self._band_indices,
                                          self._temperature,
                                          self._sigma,
                                          self._unit_conversion,
                                          self._cutoff_frequency)

    def _run_thm_c_with_band_indices(self):
        import anharmonic._phono3py as phono3c
        phono3c.thm_imag_self_energy_at_bands(self._imag_self_energy,
                                              self._fc3_normal_squared,
                                              self._triplets_at_q,
                                              self._weights_at_q,
                                              self._frequencies,
                                              self._temperature,
                                              self._g,
                                              self._unit_conversion,
                                              self._cutoff_frequency)
        
    def _run_c_with_frequency_points(self):
        import anharmonic._phono3py as phono3c
        for i, fpoint in enumerate(self._frequency_points):
            phono3c.imag_self_energy(self._imag_self_energy[i],
                                     self._fc3_normal_squared,
                                     self._triplets_at_q,
                                     self._weights_at_q,
                                     self._frequencies,
                                     fpoint,
                                     self._temperature,
                                     self._sigma,
                                     self._unit_conversion,
                                     self._cutoff_frequency)

    def _run_thm_c_with_frequency_points(self):
        import anharmonic._phono3py as phono3c
        g = np.zeros((2,) + self._fc3_normal_squared.shape, dtype='double')
        for i in range(len(self._frequency_points)):
            for j in range(g.shape[2]):
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
            phono3c.thm_imag_self_energy_at_bands(self._imag_self_energy[i],
                                                  self._fc3_normal_squared,
                                                  self._triplets_at_q,
                                                  self._weights_at_q,
                                                  self._frequencies,
                                                  self._temperature,
                                                  g,
                                                  self._unit_conversion,
                                                  self._cutoff_frequency)
        
    def _run_py_with_band_indices(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._triplets_at_q,
                self._weights_at_q,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._triplets_at_q))

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if self._temperature > 0:
                    self._imag_self_energy[j] += (
                        self._ise_at_bands(j, bi, freqs, interaction, w))
                else:
                    self._imag_self_energy[j] += (
                        self._ise_at_bands_0K(j, bi, freqs, interaction, w))

        self._imag_self_energy *= self._unit_conversion

    def _ise_at_bands(self, i, bi, freqs, interaction, weight):
        sum_g = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1][j] > self._cutoff_frequency and
                freqs[2][k] > self._cutoff_frequency):
                n2 = occupation(freqs[1][j], self._temperature)
                n3 = occupation(freqs[2][k], self._temperature)
                g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                              self._sigma)
                g2 = gaussian(freqs[0, bi] + freqs[1, j] - freqs[2, k],
                              self._sigma)
                g3 = gaussian(freqs[0, bi] - freqs[1, j] + freqs[2, k],
                              self._sigma)
                sum_g += ((n2 + n3 + 1) * g1 +
                          (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight
        return sum_g

    def _ise_at_bands_0K(self, i, bi, freqs, interaction, weight):
        sum_g = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            g1 = gaussian(freqs[0, bi] - freqs[1, j] - freqs[2, k],
                          self._sigma)
            sum_g += g1 * interaction[i, j, k] * weight

        return sum_g

    def _run_thm_py_with_band_indices(self):
        if self._temperature > 0:
            self._ise_thm_with_band_indices()
        else:
            self._ise_thm_with_band_indices_0K()

    def _ise_thm_with_band_indices(self):
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

    def _ise_thm_with_band_indices_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_py_with_frequency_points(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._triplets_at_q,
                self._weights_at_q,
                self._fc3_normal_squared)):
            print "%d / %d" % (i + 1, len(self._triplets_at_q))

            # freqs[2, num_band]
            freqs = self._frequencies[triplet[1:]]
            if self._temperature > 0:
                self._ise_with_frequency_points(freqs, interaction, w)
            else:
                self._ise_with_frequency_points_0K(freqs, interaction, w)

        self._imag_self_energy *= self._unit_conversion

    def _ise_with_frequency_points(self, freqs, interaction, weight):
        for j, k in list(np.ndindex(interaction.shape[1:])):
            if (freqs[0][j] > self._cutoff_frequency and
                freqs[1][k] > self._cutoff_frequency):
                n2 = occupation(freqs[0][j], self._temperature)
                n3 = occupation(freqs[1][k], self._temperature)
                g1 = gaussian(self._frequency_points
                              - freqs[0][j] - freqs[1][k], self._sigma)
                g2 = gaussian(self._frequency_points
                              + freqs[0][j] - freqs[1][k], self._sigma)
                g3 = gaussian(self._frequency_points
                              - freqs[0][j] + freqs[1][k], self._sigma)
            else:
                continue
            
            for i in range(len(interaction)):
                self._imag_self_energy[:, i] += (
                    (n2 + n3 + 1) * g1 +
                    (n2 - n3) * (g2 - g3)) * interaction[i, j, k] * weight

    def _ise_with_frequency_points_0K(self, freqs, interaction, weight):
        for (i, j, k) in list(np.ndindex(interaction.shape)):
            g1 = gaussian(self._frequency_points - freqs[0][j] - freqs[1][k],
                          self._sigma)
            self._imag_self_energy[:, i] += g1 * interaction[i, j, k] * weight
        
    def _run_thm_py_with_frequency_points(self):
        if self._temperature > 0:
            self._ise_thm_with_frequency_points()
        else:
            self._ise_thm_with_frequency_points_0K()
            
    def _ise_thm_with_frequency_points(self):
        for i, (tp, w, interaction) in enumerate(zip(self._triplets_at_q,
                                                     self._weights_at_q,
                                                     self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if (f1 > self._cutoff_frequency and
                    f2 > self._cutoff_frequency):
                    n2 = occupation(f1, self._temperature)
                    n3 = occupation(f2, self._temperature)
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k] # g2 - g3
                    for l in range(len(interaction)):
                        self._imag_self_energy[:, l] += (
                            (n2 + n3 + 1) * g1 +
                            (n2 - n3) * (g2_g3)) * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion
                
    def _ise_thm_with_frequency_points_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                for l in range(len(interaction)):
                    self._imag_self_energy[:, l] += g1 * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion


