import numpy as np
from anharmonic.file_IO import write_damping_functions
from phonopy.units import VaspToTHz

class ImSelfEnergy:
    def __init__(self,
                 interaction_strength,
                 sigma=0.2,
                 omega_step=0.1,
                 verbose=False):

        self._pp = interaction_strength
        self._sigma = sigma
        self._omega_step = omega_step
        self._verbose = verbose

    def get_damping_function(self,
                             temperature=None,
                             filename=None,
                             gamma_option=0):
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
        grid_point = self._pp.get_grid_point()
        num_atom = self._pp.get_primitive().get_number_of_atoms()
        freq_factor = self._pp.get_frequency_unit_conversion_factor()
        band_indices = self._pp.get_band_indices()
        
        omegas = get_omegas(np.max(frequencies_at_q),
                            self._omega_step,
                            self._sigma)
    
        # Calculate damping function at each band
        damps_bands = []
        for i, band_index in enumerate(band_indices):
            if ((grid_point == 0 and band_index < 3) or
                band_index < 0 or band_index > num_atom * 3 - 1):
                if self._verbose:
                    print "The band index %d is not calculated.\n" % band_index
                continue
    
            # Unit: frequency^{-1} 
            #   frequency THz
            # 18\pi / \hbar^2 to be multiplied
            dampings = get_gamma(amplitude_at_q,
                                 omegas,
                                 weights_at_q,
                                 frequencies_at_q,
                                 i,
                                 temperature,
                                 self._sigma,
                                 freq_factor,
                                 gamma_option) * conversion_factor
    
            write_damping_functions(grid_point,
                                    band_index + 1,
                                    self._pp.get_mesh_numbers(),
                                    omegas,
                                    dampings,
                                    filename=filename,
                                    is_nosym=self._pp.is_nosym())

def get_omegas(max_omega, omega_step, sigma):
    return np.array(range(int((max_omega * 2 + sigma * 4) / omega_step + 1)),
                    dtype=float) * omega_step

def get_gamma(amplitudes,
              omegas,
              weights,
              frequencies,
              band_index,
              temperature,
              sigma,
              freq_factor,
              cutoff_frequency=0,
              gamma_option=0):
    
    gammas = np.zeros(len(omegas), dtype=float)
    if temperature==None:
        t = -1 # Means 0 K
    else:
        t = temperature

    try:
        import anharmonic._phono3py as phono3c
        phono3c.gamma(gammas,
                      omegas,
                      amplitudes,
                      weights,
                      frequencies,
                      band_index,
                      float(sigma),
                      freq_factor,
                      float(t),
                      float(cutoff_frequency),
                      gamma_option)

    except ImportError:
        get_py_gamma(gammas,
                     omegas,
                     amplitudes,
                     weights,
                     frequencies,
                     band_index,
                     sigma,
                     freq_factor,
                     t)

    return gammas

def get_py_gamma(gammas,
                 omegas,
                 amplitudes,
                 weights,
                 frequencies,
                 band_index,
                 sigma,
                 freq_factor,
                 t):

    num_band = frequencies.shape[2]
    sum_ir_triplet = np.zeros((len(weights), len(omegas)), dtype=float)

    for i, omega in enumerate(omegas):
        for l, (a, w, freqs) in enumerate(zip(amplitudes, weights, frequencies)):
            sum_local = 0.0
            for j in range(num_band):
                for k in range(num_band):
                    f2 = freqs[1, j]
                    f3 = freqs[2, k]
                    vv = a[band_index, j, k]

                    if t > 0:
                        n2 = bs(f2 / freq_factor, t)
                        n3 = bs(f3 / freq_factor, t)
                        sum_local += ((1 + n2 + n3) * gauss(f2 + f3 - omega, sigma) +
                                      2 * (n3 - n2) * gauss(f2 - f3 - omega, sigma)
                                      ) * vv * w
                    else:
                        sum_local += gauss(f2 + f3 - omega, sigma) * vv * w

            sum_ir_triplet[l, i] = sum_local
            gammas[i] += sum_local
    gamma /= weights.sum()


def get_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    
    try:
        import anharmonic._phono3py as phono3c
        return get_c_sum_in_primitive(fc3_q, e1.copy(), e2.copy(), e3.copy(), primitive)
               
    except ImportError:
        return get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive)
               


def get_c_sum_in_primitive(fc3_q, e1, e2, e3, primitive):
    import anharmonic._phono3py as phono3c
    return phono3c.sum_in_primitive(fc3_q,
                                    e1, e2, e3,
                                    primitive.get_masses())

def get_py_sum_in_primitive(fc3_q, e1, e2, e3, primitive):

    num_atom = primitive.get_number_of_atoms()
    m = primitive.get_masses()
    sum = 0

    for i1 in range(num_atom):
        for i2 in range(num_atom):
            for i3 in range(num_atom):
                sum += get_sum_in_cartesian(fc3_q[i1, i2, i3],
                                            e1[i1*3:(i1+1)*3],
                                            e2[i2*3:(i2+1)*3],
                                            e3[i3*3:(i3+1)*3]) \
                                            / np.sqrt(m[i1] * m[i2] * m[i3])
                
    return abs(sum)**2



def get_sum_in_cartesian(f3, e1, e2, e3):
    """
    i1, i2, i3 are the atom indices.
    """
    sum = 0
    for a in (0, 1, 2):
        for b in (0, 1, 2):
            for c in (0, 1, 2):
                sum += f3[a, b, c] * e1[a] * e2[b] * e3[c]

    return sum

def gauss(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2.0 / sigma**2)

def bs(x, t): # Bose Einstein distribution (For frequency THz)
    return 1.0 / (np.exp(PlanckConstant * 1e12 * x / (Kb * t)) - 1)
    
