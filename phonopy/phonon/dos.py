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

        self._omega_pitch = None
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
                      omega_min=None,
                      omega_max=None,
                      omega_pitch=None):

        if omega_pitch == None:
            self._omega_pitch = (self._frequencies.max() -
                                 self._frequencies.min()) / 200
        else:
            self._omega_pitch = omega_pitch

        if omega_min == None:
            self._omega_min = self._frequencies.min() - self._sigma * 10
        else:
            self._omega_min = omega_min

        if omega_max == None:
            self._omega_max = self._frequencies.max() + self._sigma * 10
        else:
            self._omega_max = omega_max
                    

class TotalDos(Dos):
    def __init__(self, frequencies, weights, sigma=None):
        Dos.__init__(self, frequencies, weights, sigma)
        self._freq_Debye = None

    def calculate(self):
        omega = self._omega_min
        dos = []
        while omega < self._omega_max + self._omega_pitch/10 :
            dos.append([omega, self._get_density_of_states_at_omega(omega)])
            omega += self._omega_pitch

        dos = np.array(dos)
        self._omegas = dos[:,0]
        self._dos = dos[:,1]

    def get_dos(self):
        """
        Return omegas and total dos
        """
        return self._omegas, self._dos

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

        freqs_min = self._omegas.min()
        freqs_max = self._omegas.max()
        
        if freq_max_fit is None:
            N_fit = len(self._omegas) / 4 # Hard coded
        else:
            N_fit = int(freqs_max_fit / (freqs_max - freqs_min) *
                        len(self._omegas.size))
        popt, pcov = curve_fit(Debye_dos,
                               self._omegas[0:N_fit],
                               self._dos[0:N_fit])
        a2 = popt[0]
        self._freq_Debye = (3 * 3 * num_atoms / a2)**(1.0 / 3)
        self._Debye_fit_coef = a2

    def plot_dos(self):
        import matplotlib.pyplot as plt
        plt.plot(self._omegas, self._dos, 'r-')
        if self._freq_Debye:
            num_points = int(self._freq_Debye / self._omega_pitch)
            omegas = np.linspace(0, self._freq_Debye, num_points + 1)
            plt.plot(np.append(omegas, self._freq_Debye),
                     np.append(self._Debye_fit_coef * omegas**2, 0), 'b-')
        plt.grid(True)
        plt.xlabel('Frequency')
        plt.ylabel('Density of states')
        
        return plt
    
    def write(self):
        file = open('total_dos.dat', 'w')
        file.write("# Sigma = %f\n" % self._sigma)
        for omega, dos in zip(self._omegas, self._dos):
            file.write("%20.10f%20.10f" % (omega, dos))
            file.write("\n")

    def _get_density_of_states_at_omega(self, omega):
        return np.sum(np.dot(
                self._weights,
                self._smearing_function.calc(self._frequencies - omega))
                      ) /  np.sum(self._weights)


class PartialDos(Dos):
    def __init__(self, frequencies, weights, eigenvectors, sigma=None):
        Dos.__init__(self, frequencies, weights, sigma)
        self._eigenvectors2 = (np.abs(eigenvectors))**2

    def get_partial_density_of_states_at_omega(self, index, amplitudes):
        eigvec2 = self._eigenvectors2
        return np.sum(
            np.dot(self._weights,
                   eigvec2[:,index,:] * amplitudes)) / np.sum(self._weights)

    def calculate(self):
        omega = self._omega_min
        pdos = []
        omegas = []
        while omega < self._omega_max + self._omega_pitch/10 :
            omegas.append(omega)
            axis_dos = []
            amplitudes = self._smearing_function.calc(self._frequencies - omega)
            for i in range(self._frequencies.shape[1]):
                axis_dos.append(
                    self.get_partial_density_of_states_at_omega(i, amplitudes))
            omega += self._omega_pitch
            pdos.append(axis_dos)

        self._partial_dos = np.array(pdos).T
        self._omegas = np.array(omegas)

    def get_partial_dos(self):
        """
        omegas: Sampling frequencies
        partial_dos:
          [[elem1-freq1, elem1-freq2, ... ],
           [elem2-freq1, elem2-freq2, ... ],
           ... ]

          where
           elem1: atom1-x compornent
           elem2: atom1-y compornent
           elem3: atom1-z compornent
           elem4: atom2-x compornent
           ...
        """
        return self._omegas, self._partial_dos

    def plot_pdos(self, indices=None, legend=None):
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.xlim(self._omega_min, self._omega_max)
        plt.xlabel('Frequency')
        plt.ylabel('Partial density of states')
        plots = []

        num_atom = self._frequencies.shape[1] / 3

        if indices == None:
            indices = []
            for i in range(num_atom):
                indices.append([i])

        for set_for_sum in indices:
            pdos_sum = np.zeros(self._omegas.shape, dtype=float)
            for i in set_for_sum:
                if i > num_atom - 1 or i < 0:
                    print "Your specified atom number is out of range."
                    raise ValueError
                pdos_sum += self._partial_dos[i*3:(i+1)*3].sum(axis=0)
            plots.append(plt.plot(self._omegas, pdos_sum))

        if not legend==None:
            plt.legend(legend)

        return plt


    def write(self):
        file = open('partial_dos.dat', 'w')
        file.write("# Sigma = %f\n" % self._sigma)
        num_mode = self._frequencies.shape[1]
        for omega, pdos in zip(self._omegas, self._partial_dos.transpose()):
            file.write("%20.10f" % omega)
            file.write(("%20.10f" * num_mode) % tuple(pdos))
            file.write("\n")



