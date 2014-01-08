# Copyright (C) 2011 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np

class NormalDistribution:
    def __init__(self, sigma):
        self._sigma = sigma

    def calc(self, x):
        return 1.0 / np.sqrt(2 * np.pi) / self._sigma * \
            np.exp(-x**2 / 2.0 / self._sigma**2)

class CauchyDistribution:
    def __init__(self, gamma):
        self._gamma = gamma

    def calc(self, x):
        return self._gamma / np.pi / (x**2 + self._gamma**2)


class Dos:
    def __init__(self, frequencies, weights, sigma=None):
        self._frequencies = frequencies
        self._weights = weights

        self._freq_pitch = None
        if sigma:
            self._sigma = sigma
        else:
            self._sigma = (self._frequencies.max() -
                           self._frequencies.min()) / 100
        self.set_draw_area()
        # Default smearing 
        self.set_smearing_function('Normal')

    def set_smearing_function(self, function_name):
        """
        function_name ==
        'Normal': smearing is done by normal distribution.
        'Cauchy': smearing is done by Cauchy distribution.
        """
        if function_name == 'Cauchy':
            self._smearing_function = CauchyDistribution(self._sigma)
        else:
            self._smearing_function = NormalDistribution(self._sigma)

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_draw_area(self,
                      freq_min=None,
                      freq_max=None,
                      freq_pitch=None):

        if freq_pitch == None:
            self._freq_pitch = (self._frequencies.max() -
                                self._frequencies.min()) / 200
        else:
            self._freq_pitch = freq_pitch

        if freq_min == None:
            self._freq_min = self._frequencies.min() - self._sigma * 10
        else:
            self._freq_min = freq_min

        if freq_max == None:
            self._freq_max = self._frequencies.max() + self._sigma * 10
        else:
            self._freq_max = freq_max
                    

class TotalDos(Dos):
    def __init__(self, frequencies, weights, sigma=None):
        Dos.__init__(self, frequencies, weights, sigma)
        self._freq_Debye = None
        self._freqs = None
        self._dos = None

    def calculate(self):
        freq = self._freq_min
        dos = []
        while freq < self._freq_max + self._freq_pitch/10 :
            dos.append([freq, self._get_density_of_states_at_freq(freq)])
            freq += self._freq_pitch

        dos = np.array(dos)
        self._freqs = dos[:,0]
        self._dos = dos[:,1]

    def get_dos(self):
        """
        Return freqs and total dos
        """
        return self._freqs, self._dos

    def get_Debye_frequency(self):
        return self._freq_Debye

    def set_Debye_frequency(self, num_atoms, freq_max_fit=None):
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print "You need to install python-scipy."
            exit(1)

        def Debye_dos(freq, a):
            return a * freq**2

        freqs_min = self._freqs.min()
        freqs_max = self._freqs.max()
        
        if freq_max_fit is None:
            N_fit = len(self._freqs) / 4 # Hard coded
        else:
            N_fit = int(freqs_max_fit / (freqs_max - freqs_min) *
                        len(self._freqs.size))
        popt, pcov = curve_fit(Debye_dos,
                               self._freqs[0:N_fit],
                               self._dos[0:N_fit])
        a2 = popt[0]
        self._freq_Debye = (3 * 3 * num_atoms / a2)**(1.0 / 3)
        self._Debye_fit_coef = a2

    def plot_dos(self):
        import matplotlib.pyplot as plt
        plt.plot(self._freqs, self._dos, 'r-')
        if self._freq_Debye:
            num_points = int(self._freq_Debye / self._freq_pitch)
            freqs = np.linspace(0, self._freq_Debye, num_points + 1)
            plt.plot(np.append(freqs, self._freq_Debye),
                     np.append(self._Debye_fit_coef * freqs**2, 0), 'b-')
        plt.grid(True)
        plt.xlabel('Frequency')
        plt.ylabel('Density of states')
        
        return plt
    
    def write(self):
        file = open('total_dos.dat', 'w')
        file.write("# Sigma = %f\n" % self._sigma)
        for freq, dos in zip(self._freqs, self._dos):
            file.write("%20.10f%20.10f" % (freq, dos))
            file.write("\n")

    def _get_density_of_states_at_freq(self, freq):
        return np.sum(np.dot(
                self._weights,
                self._smearing_function.calc(self._frequencies - freq))
                      ) /  np.sum(self._weights)


class PartialDos(Dos):
    def __init__(self,
                 frequencies,
                 weights,
                 eigenvectors,
                 sigma=None,
                 direction=None):
        Dos.__init__(self, frequencies, weights, sigma)
        self._eigenvectors = eigenvectors

        num_atom = self._frequencies.shape[1] / 3
        i_x = np.arange(num_atom, dtype='int') * 3
        i_y = np.arange(num_atom, dtype='int') * 3 + 1
        i_z = np.arange(num_atom, dtype='int') * 3 + 2
        if direction is not None:
            self._direction = np.array(
                direction, dtype='double') / np.linalg.norm(direction)
            proj_eigvecs = self._eigenvectors[:, i_x, :] * self._direction[0]
            proj_eigvecs += self._eigenvectors[:, i_y, :] * self._direction[1]
            proj_eigvecs += self._eigenvectors[:, i_z, :] * self._direction[2]
            self._eigvecs2 = np.abs(proj_eigvecs) ** 2
        else:
            self._direction = None
            self._eigvecs2 = np.abs(self._eigenvectors[:, i_x, :]) ** 2
            self._eigvecs2 += np.abs(self._eigenvectors[:, i_y, :]) ** 2
            self._eigvecs2 += np.abs(self._eigenvectors[:, i_z, :]) ** 2
        self._partial_dos = None
        self._freqs = None

    def calculate(self):
        freq = self._freq_min
        pdos = []
        freqs = []
        weights = self._weights / float(np.sum(self._weights))
        while freq < self._freq_max + self._freq_pitch/10 :
            freqs.append(freq)
            amplitudes = self._smearing_function.calc(self._frequencies - freq)
            pdos_at_freq = self._get_partial_dos_at_freq(amplitudes, weights)
            freq += self._freq_pitch
            pdos.append(pdos_at_freq)

        self._partial_dos = np.array(pdos).T
        self._freqs = np.array(freqs)

    def get_partial_dos(self):
        """
        freqs: Sampling frequencies
        partial_dos:
          [[atom1-band1, atom1-band2, ... ],
           [atom2-band1, atom2-band2, ... ],
           ... ]
        """
        return self._freqs, self._partial_dos

    def plot_pdos(self, indices=None, legend=None):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.xlim(self._freq_min, self._freq_max)
        plt.xlabel('Frequency')
        plt.ylabel('Partial density of states')
        plots = []

        num_atom = self._frequencies.shape[1] / 3

        if indices == None:
            indices = []
            for i in range(num_atom):
                indices.append([i])

        for set_for_sum in indices:
            pdos_sum = np.zeros(self._freqs.shape, dtype=float)
            for i in set_for_sum:
                if i > num_atom - 1 or i < 0:
                    print "Your specified atom number is out of range."
                    raise ValueError
                pdos_sum += self._partial_dos[i]
            plots.append(plt.plot(self._freqs, pdos_sum))

        if not legend==None:
            plt.legend(legend)

        return plt


    def write(self):
        file = open('partial_dos.dat', 'w')
        file.write("# Sigma = %f\n" % self._sigma)
        for freq, pdos in zip(self._freqs, self._partial_dos.transpose()):
            file.write("%20.10f" % freq)
            file.write(("%20.10f" * len(pdos)) % tuple(pdos))
            file.write("\n")

    def _get_partial_dos_at_freq(self, amplitudes, weights):
        num_band = self._frequencies.shape[1]
        pdos = [(np.dot(weights, self._eigvecs2[:, i, :] * amplitudes)).sum()
                for i in range(num_band / 3)]
        return pdos
