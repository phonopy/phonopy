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

def mode_F(temp, freqs):
    return Kb * temp * np.log(1.0 - np.exp((- freqs) / (Kb * temp))) + freqs / 2

def mode_S(temp, freqs):
    val = freqs / (2 * Kb * temp)
    return (1 / (2 * temp) * freqs * np.cosh(val) / np.sinh(val) -
            Kb * np.log(2 * np.sinh(val)))

def mode_ZPE(temp, freqs):
    return freqs / 2

def mode_zero(temp, freqs):
    return 0

class ThermalPropertiesBase:
    def __init__(self,
                 frequencies,
                 weights=None,
                 eigenvectors=None,
                 is_projection=False,
                 band_indices=None,
                 cutoff_frequency=None):
        self._band_indices = None
        self._frequencies = None
        self._eigenvectors = None
        self._weights = None
        self._is_projection = is_projection
        self._cutoff_frequency = cutoff_frequency
        
        if band_indices is not None:
            bi = np.hstack(band_indices).astype('intc')
            self._band_indices = bi
            self._frequencies = np.array(frequencies[:, bi],
                                         dtype='double', order='C')
            if eigenvectors is not None:
                self._eigenvectors = np.array(eigenvectors[:, :, bi],
                                              dtype='double', order='C')
        else:
            self._frequencies = frequencies
            self._eigenvectors = eigenvectors

        if cutoff_frequency is not None:
            self._frequencies = np.where(self._frequencies > cutoff_frequency,
                                         self._frequencies, -1)
        self._frequencies = np.array(self._frequencies,
                                     dtype='double', order='C') * THzToEv
        if weights is None:
            self._weights = np.ones(frequencies.shape[0], dtype='intc')
        else:
            self._weights = weights

        self._num_modes = self._frequencies.shape[1] * self._weights.sum()
        self._num_integrated_modes = np.sum(
            self._weights * (self._frequencies > 0).sum(axis=1))

    def get_free_energy(self, t):
        if t > 0:
            free_energy = self._calculate_thermal_property(mode_F, t)
        else:
            free_energy = self._calculate_thermal_property(mode_ZPE, None)
        return free_energy / np.sum(self._weights) * EvTokJmol

    def get_heat_capacity_v(self, t):
        if t > 0:
            cv = self._calculate_thermal_property(mode_cv, t)
        else:
            cv = self._calculate_thermal_property(mode_zero, None)
        return cv / np.sum(self._weights) * EvTokJmol

    def get_entropy(self, t):
        if t > 0:
            entropy = self._calculate_thermal_property(mode_S, t)
        else:
            entropy = self._calculate_thermal_property(mode_zero, None)
        return entropy / np.sum(self._weights) * EvTokJmol

    def _calculate_thermal_property(self, func, t):
        if not self._is_projection:
            t_property = 0.0
            for freqs, w in zip(self._frequencies, self._weights):
                t_property += np.sum(
                    func(t, np.extract(freqs > 0, freqs))) * w
            return t_property
        else:
            t_property = np.zeros(len(self._frequencies[0]), dtype='double')
            for freqs, eigvecs2, w in zip(self._frequencies,
                                          np.abs(self._eigenvectors) ** 2,
                                          self._weights):
                for f, fracs in zip(freqs, eigvecs2.T):
                    if f > 0:
                        t_property += func(t, f) * w * fracs
    
            return t_property

class ThermalProperties(ThermalPropertiesBase):
    def __init__(self,
                 frequencies,
                 weights=None,
                 eigenvectors=None,
                 is_projection=False,
                 band_indices=None,
                 cutoff_frequency=None):
        ThermalPropertiesBase.__init__(self,
                                       frequencies,
                                       weights=weights,
                                       eigenvectors=eigenvectors,
                                       is_projection=is_projection,
                                       band_indices=band_indices,
                                       cutoff_frequency=cutoff_frequency)
        self._set_high_T_entropy_and_zero_point_energy()

    def get_number_of_integrated_modes(self):
        """Number of phonon modes used for integration on sampling mesh"""
        return self._num_integrated_modes

    def get_number_of_modes(self):
        """Number of phonon modes on sampling mesh"""
        return self._num_modes
        
    def get_zero_point_energy(self):
        return self._zero_point_energy

    def get_high_T_entropy(self):
        return self._high_T_entropy

    def plot(self, pyplot):
        temps, fe, entropy, cv = self._thermal_properties

        pyplot.plot(temps, fe, 'r-')
        pyplot.plot(temps, entropy, 'b-')
        pyplot.plot(temps, cv, 'g-')
        pyplot.legend(('Free energy [kJ/mol]', 'Entropy [J/K/mol]',
                       r'C$_\mathrm{V}$ [J/K/mol]'),
                      loc='best')
        pyplot.grid(True)
        pyplot.xlabel('Temperature [K]')

    def set_thermal_properties(self, t_step=10, t_max=1000, t_min=0):
        import warnings
        warnings.warn("\'set_thermal_properties\' method is depreciated. "
                      "Use \'run\' method instead.")
        self.run(t_step=t_step, t_max=t_max, t_min=t_min)

    def run(self, t_step=10, t_max=1000, t_min=0):
        temperatures = np.arange(t_min, t_max + t_step / 2.0, t_step,
                                 dtype='double')
        fe = []
        entropy = []
        cv = []
        energy = []
        try:
            import phonopy._phonopy as phonoc
            for t in temperatures:
                props = self._get_c_thermal_properties(t)
                fe.append(props[0] * EvTokJmol + self._zero_point_energy)
                entropy.append(props[1] * EvTokJmol * 1000)
                cv.append(props[2] * EvTokJmol * 1000)
        except ImportError:
            for t in temperatures:
                props = self._get_py_thermal_properties(t)
                fe.append(props[0])
                entropy.append(props[1] * 1000,)
                cv.append(props[2] * 1000)

        self._thermal_properties = [temperatures,
                                    np.array(fe, dtype='double', order='C'),
                                    np.array(entropy, dtype='double', order='C'),
                                    np.array(cv, dtype='double', order='C')]

        if self._is_projection:
            fe = []
            entropy = []
            cv = []
            energy = []
            for t in temperatures:
                fe.append(self.get_free_energy(t))
                entropy.append(self.get_entropy(t) * 1000,)
                cv.append(self.get_heat_capacity_v(t) * 1000)

            self._projected_thermal_properties = [
                temperatures,
                np.array(fe, dtype='double'),
                np.array(entropy, dtype='double'),
                np.array(cv, dtype='double')]

    def get_thermal_properties(self):
        return self._thermal_properties

    def write_yaml(self, filename='thermal_properties.yaml'):
        lines = self._get_tp_yaml_lines()
        if self._is_projection:
            lines += self._get_projected_tp_yaml_lines()
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        
    def _get_tp_yaml_lines(self):
        lines = []
        lines.append("# Thermal properties / unit cell (natom)")
        lines.append("")
        lines.append("unit:")
        lines.append("  temperature:   K")
        lines.append("  free_energy:   kJ/mol")
        lines.append("  entropy:       J/K/mol")
        lines.append("  heat_capacity: J/K/mol")
        lines.append("")
        lines.append("natom: %5d" % (self._frequencies[0].shape[0] // 3))
        if self._cutoff_frequency:
            lines.append("cutoff_frequency: %8.3f" % self._cutoff_frequency)
        lines.append("num_modes: %d" % self._num_modes)
        lines.append("num_integrated_modes: %d" % self._num_integrated_modes)
        if self._band_indices is not None:
            bi = self._band_indices + 1
            lines.append("band_index: [ " + ("%d, " * (len(bi) - 1)) %
                    tuple(bi[:-1]) + ("%d ]" % bi[-1]))
        lines.append("")
        lines.append("zero_point_energy: %15.7f" % self._zero_point_energy)
        lines.append("high_T_entropy:    %15.7f" % (self._high_T_entropy * 1000))
        lines.append("")
        lines.append("thermal_properties:")
        temperatures, fe, entropy, cv = self._thermal_properties
        for i, t in enumerate(temperatures):
            lines.append("- temperature:   %15.7f" % t)
            lines.append("  free_energy:   %15.7f" % fe[i])
            lines.append("  entropy:       %15.7f" % entropy[i])
            # Sometimes 'nan' of C_V is returned at low temperature.
            if np.isnan(cv[i]):
                lines.append("  heat_capacity: %15.7f" % 0 )
            else:
                lines.append("  heat_capacity: %15.7f" % cv[i])
            lines.append("  energy:        %15.7f" %
                         (fe[i] + entropy[i] * t / 1000))
            lines.append("")
        return lines

    def _get_projected_tp_yaml_lines(self):
        lines = []
        lines.append("projected_thermal_properties:")
        temperatures, fe, entropy, cv = self._projected_thermal_properties
        for i, t in enumerate(temperatures):
            lines.append("- temperature:   %13.7f" % t)
            line = "  free_energy:   [ "
            line += ", ".join(["%13.7f" % x for x in fe[i]])
            line += " ] # %13.7f" % np.sum(fe[i])
            lines.append(line)
            line = "  entropy:       [ "
            line += ", ".join(["%13.7f" % x for x in entropy[i]])
            line += " ] # %13.7f" % np.sum(entropy[i])
            lines.append(line)
            # Sometimes 'nan' of C_V is returned at low temperature.
            line = "  heat_capacity: [ "
            sum_cv = 0.0
            for j, cv_i in enumerate(cv[i]):
                if np.isnan(cv_i):
                    line += "%13.7f" % 0
                else:
                    sum_cv += cv_i
                    line += "%13.7f" % cv_i
                if j < len(cv[i]) - 1:
                    line += ", "
                else:
                    line += " ]"
            line += " # %13.7f" % sum_cv
            lines.append(line)
            energy = fe[i] + entropy[i] * t / 1000
            line = "  energy:        [ "
            line += ", ".join(["%13.7f" % x for x in energy])
            line += " ] # %13.7f" % np.sum(energy)
            lines.append(line)
        return lines
            
    def _get_c_thermal_properties(self, t):
        import phonopy._phonopy as phonoc

        if t > 0:
            return phonoc.thermal_properties(t,
                                             self._frequencies,
                                             self._weights)
        else:
            return (0.0, 0.0, 0.0)

    def _get_py_thermal_properties(self, t):
        return (self.get_free_energy(t),
                self.get_entropy(t),
                self.get_heat_capacity_v(t))

    def _set_high_T_entropy_and_zero_point_energy(self):
        zp_energy = 0.0
        entropy = 0.0
        for freqs, w in zip(self._frequencies, self._weights):
            positive_fs = np.extract(freqs > 0.0, freqs)
            entropy -= np.sum(np.log(positive_fs)) * w
            zp_energy += np.sum(positive_fs) * w / 2
        self._high_T_entropy = entropy * Kb / np.sum(self._weights) * EvTokJmol
        self._zero_point_energy = zp_energy / np.sum(self._weights) * EvTokJmol
