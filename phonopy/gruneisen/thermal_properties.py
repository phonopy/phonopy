# Copyright (C) 2015 Atsushi Togo
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
from phonopy.phonon.thermal_properties import ThermalProperties as PhononThermalProperties

class ThermalProperties(object):
    def __init__(self,
                 gruneisen_mesh,
                 volumes,
                 t_step=2,
                 t_max=2004,
                 t_min=0,
                 cutoff_frequency=None):
        phonon = gruneisen_mesh.get_phonon()
        self._cutoff_frequency = cutoff_frequency
        self._factor = phonon.get_unit_conversion_factor(),
        self._V0 = phonon.get_primitive().get_volume()
        self._gamma = gruneisen_mesh.get_gruneisen()
        self._gamma_prime = gruneisen_mesh.get_gamma_prime()
        self._weights = gruneisen_mesh.get_weights()
        self._eigenvalues = gruneisen_mesh.get_eigenvalues()
        self._frequencies = gruneisen_mesh.get_frequencies()
        self._thermal_properties = []
        for V in volumes:
            tp = self._get_thermal_properties_at_V(V)
            tp.run(t_step=t_step, t_max=t_max, t_min=t_min)
            self._thermal_properties.append(tp)

    def get_thermal_properties(self):
        """Return a set of phonopy.phonon::ThermalProperties object"""
        return self._thermal_properties

    def write_yaml(self, filename='thermal_properties'):
        for i, tp in enumerate(self._thermal_properties):
            tp.write_yaml(filename="%s-%02d.yaml" % (filename, i))

    def _get_thermal_properties_at_V(self, V):
        frequencies = self._get_frequencies_at_V(V)
        tp = PhononThermalProperties(frequencies,
                                     weights=self._weights,
                                     cutoff_frequency=self._cutoff_frequency)
        return tp

    def _get_frequencies_at_V(self, V):
        return self._get_frequencies_at_V_analytical_solution(V)

    def _get_frequencies_at_V_analytical_solution(self, V):
        eigvals = self._eigenvalues * np.exp(-2 * self._gamma *
                                             np.log(V / self._V0))
        return np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor

    def _get_frequencies_at_V_analytical_solution_with_1st_correction(self, V):
        V0 = self._V0
        g_prime = self._gamma_prime / V0
        eigvals = self._eigenvalues * np.exp(
            -2 * ((self._gamma - g_prime * V0) * np.log(V / V0)
                  + g_prime * (V - V0)))
        return np.sqrt(abs(eigvals)) * np.sign(eigvals) * self._factor
                                    
    def _get_frequencies_at_V_Taylor_expansion_to_1st_order(self, V):
        return self._frequencies * (
            1.0
            - self._gamma * (V - self._V0) / self._V0)

    def _get_frequencies_at_V_Taylor_expansion_to_2nd_order(self, V):
        return self._frequencies * (
            1.0
            - self._gamma * (V - self._V0) / self._V0
            - self._gamma_prime * ((V - self._V0) / self._V0) ** 2 / 2)
