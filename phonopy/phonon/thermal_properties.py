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
from phonopy.units import *

def mode_cv(temp, freqs): # freqs (eV)
    x = freqs / Kb / temp
    expVal = np.exp(x)
    return Kb * x ** 2 * expVal / (expVal - 1.0) ** 2

class ThermalPropertiesBase:
    def __init__(self, frequencies, weights=None):
        self._temperature = 0
        self._frequencies = frequencies
        if weights == None:
            self._weights = np.ones(frequencies.shape[0], dtype='int32')
        else:
            self._weights = weights
        self._nqpoint = frequencies.shape[0]

    def set_temperature(self, temperature):
        self._temperature = temperature

    def get_free_energy(self):

        def func(temp, omega):
            return Kb * temp * np.log(1.0 - np.exp((- omega) / (Kb * temp)))

        free_energy = self.get_thermal_property(func)
        return free_energy / np.sum(self._weights) * EvTokJmol + self._zero_point_energy

    def get_free_energy2(self):

        if self._temperature > 0:
            def func(temp, omega):
                return  Kb * temp * np.log(2.0 * np.sinh(omega / (2 * Kb * temp)))

            free_energy = self.get_thermal_property(func)
            return free_energy / np.sum(self._weights) * EvTokJmol
        else:
            return self._zero_point_energy

    def get_heat_capacity_v(self):
        func = mode_cv
        
        cv = self.get_thermal_property(mode_cv)
        return cv / np.sum(self._weights) * EvTokJmol

    def get_entropy(self):
        
        def func(temp, omega):
            val = omega / (2 * Kb * temp)
            return 1. / (2 * temp) * omega * np.cosh(val) / np.sinh(val) - Kb * np.log(2 * np.sinh(val))

        entropy = self.get_thermal_property(func)
        return entropy / np.sum(self._weights) * EvTokJmol

    def get_entropy2(self):

        def func(temp, omega):
            val = omega / (Kb * temp)
            return -Kb * np.log(1 - np.exp( -val )) + 1.0 / temp * omega / (np.exp( val ) - 1)

        entropy = self.get_thermal_property(func)
        return entropy / np.sum(self._weights) * EvTokJmol


class ThermalProperties(ThermalPropertiesBase):
    def __init__(self,
                 frequencies,
                 weights=None,
                 cutoff_frequency=None):
        ThermalPropertiesBase.__init__(self, frequencies, weights)
        if cutoff_frequency:
            self._cutoff_frequency = cutoff_frequency
        else:
            self._cutoff_frequency = 0.0
        self._frequencies = np.where(frequencies > self._cutoff_frequency,
                                     frequencies, 0) * THzToEv
        self._set_zero_point_energy()
        self._set_high_T_entropy()
        
    def get_zero_point_energy(self):
        return self._zero_point_energy

    def get_high_T_entropy(self):
        return self._high_T_entropy

    def get_thermal_property(self, func):
        t_property = 0.0

        if self._temperature > 0:
            temp = self._temperature
            for i, freqs in enumerate(self._frequencies):
                t_property += np.sum(func(temp, freqs)) * self._weights[i]

        return t_property

    def get_c_thermal_properties(self):
        import phonopy._phonopy as phonoc

        if self._temperature > 0:
            return phonoc.thermal_properties(self._temperature,
                                             self._frequencies,
                                             self._weights)
        else:
            return (0.0, 0.0, 0.0)

    def plot_thermal_properties(self):
        import matplotlib.pyplot as plt
        
        temps, fe, entropy, cv = self._thermal_properties

        plt.plot(temps, fe, 'r-')
        plt.plot(temps, entropy, 'b-')
        plt.plot(temps, cv, 'g-')
        plt.legend(('Free energy [kJ/mol]', 'Entropy [J/K/mol]',
                    r'C$_\mathrm{V}$ [J/K/mol]'),
                   'best', shadow=True)
        plt.grid(True)
        plt.xlabel('Temperature [K]')

        return plt

    def set_thermal_properties(self, t_step=10, t_max=1000, t_min=0):
        t = t_min
        temps = []
        fe = []
        entropy = []
        cv = []
        energy = []
        while t < t_max + t_step / 2.0:
            self.set_temperature(t)
            temps.append(t)


            try:
                import phonopy._phonopy as phonoc
                # C implementation, but not so faster than Numpy
                props = self.get_c_thermal_properties()
                fe.append(props[0] * EvTokJmol + self._zero_point_energy)
                entropy.append(props[1] * EvTokJmol * 1000)
                cv.append(props[2] * EvTokJmol * 1000)
            except ImportError:
                # Numpy implementation, but not so bad
                fe.append(self.get_free_energy())
                entropy.append(self.get_entropy()*1000)
                cv.append(self.get_heat_capacity_v()*1000)

            t += t_step

        self._thermal_properties = [np.array(temps),
                                    np.array(fe),
                                    np.array(entropy),
                                    np.array(cv)]

    def get_thermal_properties( self ):
        return self._thermal_properties

    def write_yaml(self, filename='thermal_properties.yaml'):
        file = open(filename, 'w')
        file.write("# Thermal properties / unit cell (natom)\n")
        file.write("\n")
        file.write("unit:\n")
        file.write("  temperature:   K\n")
        file.write("  free_energy:   kJ/mol\n")
        file.write("  entropy:       J/K/mol\n")
        file.write("  heat_capacity: J/K/mol\n")
        file.write("\n")
        file.write("natom: %5d\n" % ((self._frequencies[0].shape)[0]/3))
        file.write("zero_point_energy: %15.7f\n" % self._zero_point_energy)
        file.write("high_T_entropy:    %15.7f\n" % (self._high_T_entropy * 1000))
        file.write("\n")
        file.write("thermal_properties:\n")
        temperatures, fe, entropy, cv = self._thermal_properties
        for i, t in enumerate(temperatures):
            file.write("- temperature:   %15.7f\n" % t)
            file.write("  free_energy:   %15.7f\n" % fe[i])
            file.write("  entropy:       %15.7f\n" % entropy[i])
            # Sometimes 'nan' of C_V is returned at low temperature.
            if np.isnan( cv[i] ):
                file.write("  heat_capacity: %15.7f\n" % 0 )
            else:
                file.write("  heat_capacity: %15.7f\n" % cv[i])
            file.write("  energy:        %15.7f\n" % (fe[i]+entropy[i]*t/1000))
            file.write("\n")

    def _set_zero_point_energy(self):
        z_energy = np.sum(1.0 / 2 * np.dot(self._weights, self._frequencies))
        self._zero_point_energy = z_energy / np.sum(self._weights) * EvTokJmol

    def _set_high_T_entropy(self):
        entropy = 0.0
        for i, freqs in enumerate(self._frequencies):
            entropy -= np.sum(
                np.log(np.extract(freqs > 0.0, freqs))) * self._weights[i]
        self._high_T_entropy = entropy * Kb / np.sum(self._weights) * EvTokJmol
