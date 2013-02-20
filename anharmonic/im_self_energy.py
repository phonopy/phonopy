import numpy as np
from anharmonic.file_IO import write_damping_functions
from phonopy.units import VaspToTHz, PlanckConstant, Kb

class ImSelfEnergy:
    def __init__(self,
                 interaction_strength,
                 sigmas=[0.1],
                 frequency_step=0.1,
                 temperatures=[None],
                 gamma_option=0,
                 filename=None,
                 log_level=False):

        self._pp = interaction_strength
        self._sigmas = sigmas
        self._frequency_step = frequency_step
        self._temperatures = temperatures
        self._gamma_option = gamma_option
        self._filename = filename
        self._log_level = log_level
        
    def get_damping_function(self):
        for t in self._temperatures:
            self._get_damping_function_at_temperature(t)
        
    def _get_damping_function_at_temperature(self,
                                             temperature=None):

        for sigma in self._sigmas:
            gammas_bands, frequencies = self._get_damping_function_at_sigma(
                sigma,
                temperature=temperature)

            band_indices = []
            sum_gammas = None
            for band_index, gammas in zip(self._pp.get_band_indices(),
                                          gammas_bands):
                if gammas is None:
                    continue

                if sum_gammas is None:
                    sum_gammas = np.zeros_like(gammas)
                else:
                    sum_gammas += gammas

                band_indices.append(band_index)
                write_damping_functions(self._pp.get_grid_point(),
                                        [band_index + 1],
                                        self._pp.get_mesh_numbers(),
                                        frequencies,
                                        gammas,
                                        sigma=sigma,
                                        temperature=temperature,
                                        filename=self._filename,
                                        is_nosym=self._pp.is_nosym())

            write_damping_functions(self._pp.get_grid_point(),
                                    np.array(band_indices) + 1,
                                    self._pp.get_mesh_numbers(),
                                    frequencies,
                                    sum_gammas / len(band_indices),
                                    sigma=sigma,
                                    temperature=temperature,
                                    filename=self._filename,
                                    is_nosym=self._pp.is_nosym())

    def _get_damping_function_at_sigma(self,
                                       sigma,
                                       temperature=None):
        """
        Units of inputs are supposed:
          energy: eV
          distance: Angstrom
          frequency: THz
        """

        conversion_factor = self._pp.get_unit_conversion_factor()
        (amplitude_at_q,
         weights_at_q,
         frequencies_at_q) = self._pp.get_amplitude()
        frequencies = get_frequencies(np.max(frequencies_at_q),
                                      self._frequency_step,
                                      sigma)
        grid_point = self._pp.get_grid_point()
        num_atom = self._pp.get_primitive().get_number_of_atoms()
        freq_factor = self._pp.get_frequency_unit_conversion_factor()

        # Calculate damping function at each band
        gammas_bands = []
        for i, band_index in enumerate(self._pp.get_band_indices()):
            if ((grid_point == 0 and band_index < 3) or
                band_index < 0 or band_index > num_atom * 3 - 1):
                if self._log_level:
                    print "The band index %d is not calculated.\n" % band_index
                gammas_bands.append(None)
                continue
    
            # Unit: frequency^{-1} 
            #   frequency THz
            # 18\pi / \hbar^2 to be multiplied
            gammas = get_gamma(
                amplitude_at_q,
                frequencies,
                weights_at_q,
                frequencies_at_q,
                i,
                temperature,
                sigma,
                freq_factor,
                cutoff_frequency=self._pp.get_cutoff_frequency(),
                gamma_option=self._gamma_option) * conversion_factor

            gammas_bands.append(gammas)

        return gammas_bands, frequencies

def get_frequencies(max_freq, freq_step, sigma):
    return np.array(range(int((max_freq * 2 + sigma * 4) / freq_step + 1)),
                    dtype=float) * freq_step

def get_gamma(amplitudes,
              frequencies,
              weights_at_q,
              frequencies_at_q,
              band_index,
              temperature,
              sigma,
              freq_factor,
              cutoff_frequency=0,
              gamma_option=0):
    
    gammas = np.zeros(len(frequencies), dtype=float)
    if temperature is None:
        t = -1 # Means 0 K
    else:
        t = temperature

    try:
        import anharmonic._phono3py as phono3c
        phono3c.gamma(gammas,
                      frequencies,
                      amplitudes,
                      weights_at_q,
                      frequencies_at_q,
                      band_index,
                      float(sigma),
                      freq_factor,
                      float(t),
                      float(cutoff_frequency),
                      gamma_option)

    except ImportError:
        get_py_gamma(gammas,
                     frequencies,
                     amplitudes,
                     weights_at_q,
                     frequencies_at_q,
                     band_index,
                     sigma,
                     freq_factor,
                     t)

    return gammas

def get_py_gamma(gammas,
                 frequencies,
                 amplitudes,
                 weights_at_q,
                 frequencies_at_q,
                 band_index,
                 sigma,
                 freq_factor,
                 t):

    num_band = frequencies_at_q.shape[2]
    sum_ir_triplet = np.zeros((len(weights_at_q), len(frequencies)),
                              dtype=float)

    for i, omega in enumerate(frequencies):
        for l, (a, w, freqs) in enumerate(
            zip(amplitudes, weights_at_q, frequencies_at_q)):
            sum_local = 0.0
            for j in range(num_band):
                for k in range(num_band):
                    f2 = freqs[1, j]
                    f3 = freqs[2, k]
                    vv = a[band_index, j, k]

                    if t > 0:
                        n2 = bs(f2 / freq_factor, t)
                        n3 = bs(f3 / freq_factor, t)
                        sum_local += ((1 + n2 + n3) *
                                      gauss(f2 + f3 - omega, sigma) +
                                      2 * (n3 - n2) *
                                      gauss(f2 - f3 - omega, sigma)) * vv * w
                    else:
                        sum_local += gauss(f2 + f3 - omega, sigma) * vv * w

            sum_ir_triplet[l, i] = sum_local
            gammas[i] += sum_local



def gauss(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x ** 2 / 2.0 / sigma ** 2)

def bs(x, t): # Bose Einstein distribution (For frequency THz)
    return 1.0 / (np.exp(PlanckConstant * 1e12 * x / (Kb * t)) - 1)
    
