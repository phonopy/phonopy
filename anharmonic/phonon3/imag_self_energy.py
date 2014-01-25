import numpy as np
from phonopy.units import THzToEv, Kb, VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phonopy.phonon.group_velocity import degenerate_sets
from phonopy.structure.tetrahedron_method import TetrahedronMethod
from phonopy.structure.spglib import get_neighboring_grid_points
from anharmonic.phonon3.triplets import get_tetrahedra_vertices
from anharmonic.phonon3.interaction import set_phonon_c
import anharmonic.file_IO as file_IO

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1)

def get_imag_self_energy(interaction,
                         grid_points,
                         sigmas,
                         frequency_step=0.1,
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
            print frequencies[gp]

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
                fmax = max_phonon_freq * 2 + sigma * 4 + frequency_step / 10
            else:
                fmax = max_phonon_freq * 2 + frequency_step / 10
            fmin = 0
            frequency_points_at_sigma = np.arange(fmin, fmax, frequency_step)
            ise.set_frequency_points(frequency_points_at_sigma)
            fp_sigmas.append(frequency_points_at_sigma)

            if not sigma:
                ise.set_integration_weights()

            ise_temperatures = []
            for k, t in enumerate(temperatures):
                if log_level:
                    print "Temperature:", t
                ise.set_temperature(t)
                ise.run()
                ise_temperatures.append(ise.get_imag_self_energy())
                
            ise_sigmas.append(ise_temperatures)
            
        imag_self_energy.append(ise_sigmas)
        frequency_points.append(fp_sigmas)
                
    return imag_self_energy, frequency_points

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
        self.set_sigma(sigma)
        self.set_temperature(temperature)
        self.set_frequency_points(frequency_points)
        self.set_grid_point(grid_point=grid_point)

        self._lang = lang
        self._imag_self_energy = None
        self._fc3_normal_squared = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.get_cutoff_frequency()

        self._g = None # integration weights of tetrahedron method
        self._mesh = self._interaction.get_mesh_numbers()

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
        self._set_integration_weights()
        
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
            self._frequency_points = np.double(frequency_points)

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
        g = np.zeros(self._fc3_normal_squared.shape + (2,), dtype='double')
        for i in range(len(self._frequency_points)):
            shape = self._fc3_normal_squared.shape
            for j in range(g.shape[1]):
                g[:, j, :, :, :] = self._g[:, i, :, :, :]
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
        n = occupation(self._frequencies[self._triplets_at_q[:, [1, 2]]],
                       self._temperature)
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
                    g1 = self._g[i, :, j, k, 0]
                    g2_g3 = self._g[i, :, j, k, 1] # g2 - g3
                    self._imag_self_energy[:] += (
                        (n2 + n3 + 1) * g1 +
                        (n2 - n3) * (g2_g3)) * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_band_indices_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[i, :, j, k, 0]
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
                    g1 = self._g[i, :, j, k, 0]
                    g2_g3 = self._g[i, :, j, k, 1] # g2 - g3
                    for l in range(len(interaction)):
                        self._imag_self_energy[:, l] += (
                            (n2 + n3 + 1) * g1 +
                            (n2 - n3) * (g2_g3)) * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion
                
    def _ise_thm_with_frequency_points_0K(self):
        for i, (w, interaction) in enumerate(zip(self._weights_at_q,
                                                 self._fc3_normal_squared)):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[i, :, j, k, 0]
                for l in range(len(interaction)):
                    self._imag_self_energy[:, l] += g1 * interaction[l, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _set_integration_weights(self):
        reciprocal_lattice = np.linalg.inv(
            self._interaction.get_primitive().get_cell())
        thm = TetrahedronMethod(reciprocal_lattice, mesh=self._mesh)
        grid_address = self._interaction.get_grid_address()
        bz_map = self._interaction.get_bz_map()
        self._set_triplets_integration_weights_c(thm, grid_address, bz_map)
        
    def _set_triplets_integration_weights_c(self, thm, grid_address, bz_map):
        import anharmonic._phono3py as phono3c
        unique_vertices = thm.get_unique_tetrahedra_vertices()
        for i, j in zip((1, 2), (1, -1)):
            neighboring_grid_points = np.zeros(
                len(unique_vertices) * len(self._triplets_at_q), dtype='intc')
            phono3c.neighboring_grid_points(
                neighboring_grid_points,
                self._triplets_at_q[:, i].flatten(),
                j * unique_vertices,
                self._mesh,
                grid_address,
                bz_map)
            self._interaction.set_phonon(np.unique(neighboring_grid_points))

        if self._frequency_points is None:
            gp = self._grid_point
            frequency_points = self._frequencies[gp][self._band_indices]
        else:
            frequency_points = self._frequency_points
        shape = self._fc3_normal_squared.shape
        self._g = np.zeros(
            (shape[0], len(frequency_points), shape[2], shape[3], 2),
            dtype='double')

        phono3c.triplets_integration_weights(
            self._g,
            np.array(frequency_points, dtype='double'),
            thm.get_tetrahedra(),
            self._mesh,
            self._triplets_at_q,
            self._frequencies,
            grid_address,
            bz_map)
        
    def _set_triplets_integration_weights_py(self, thm, grid_address, bz_map):
        tetrahedra_vertices = get_tetrahedra_vertices(thm.get_tetrahedra(),
                                                      self._mesh,
                                                      self._triplets_at_q,
                                                      grid_address,
                                                      bz_map)
        self._interaction.set_phonon(np.unique(tetrahedra_vertices))
        if self._frequency_points is None:
            gp = self._grid_point
            frequency_points = self._frequencies[gp][self._band_indices]
        else:
            frequency_points = self._frequency_points
        shape = self._fc3_normal_squared.shape
        self._g = np.zeros(
            (shape[0], len(frequency_points), shape[2], shape[3], 2),
            dtype='double')

        for i, vertices in enumerate(tetrahedra_vertices):
            for j, k in list(np.ndindex(self._fc3_normal_squared.shape[2:])):
                f1_v = self._frequencies[vertices[0], j]
                f2_v = self._frequencies[vertices[1], k]
                thm.set_tetrahedra_omegas(f1_v + f2_v)
                thm.run(frequency_points)
                self._g[i, :, j, k, 0] = thm.get_integration_weight()
                thm.set_tetrahedra_omegas(-f1_v + f2_v)
                thm.run(frequency_points)
                self._g[i, :, j, k, 1] = thm.get_integration_weight()
                thm.set_tetrahedra_omegas(f1_v - f2_v)
                thm.run(frequency_points)
                self._g[i, :, j, k, 1] -= thm.get_integration_weight()
