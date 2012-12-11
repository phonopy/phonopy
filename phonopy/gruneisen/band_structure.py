# Copyright (C) 2012 Atsushi Togo
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
from phonopy.gruneisen import Gruneisen

class BandStructure:
    def __init__(self,
                 phonon,
                 phonon_plus,
                 phonon_minus,
                 paths,
                 num_points):
        self._num_points = num_points
        
        primitive = phonon.get_primitive()
        gruneisen = Gruneisen(phonon.get_dynamical_matrix(),
                              phonon_plus.get_dynamical_matrix(),
                              phonon_minus.get_dynamical_matrix(),
                              primitive.get_volume(),
                              phonon_plus.get_primitive().get_volume(),
                              phonon_minus.get_primitive().get_volume(),
                              is_band_connection=True)
        rec_vectors = np.linalg.inv(primitive.get_cell())
        factor = phonon.get_unit_conversion_factor(),
        distance_shift = 0.0

        self._paths = []
        
        for path in paths:
            qpoints, distances = _get_band_qpoints(path[0],
                                                   path[1],
                                                   rec_vectors,
                                                   num_points=num_points)
            gruneisen.set_qpoints(qpoints)
            gamma = gruneisen.get_gruneisen()
            eigenvalues = gruneisen.get_eigenvalues()
            frequencies = np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues) * factor
            
            distances_with_shift = distances + distance_shift

            self._paths.append([qpoints,
                                distances,
                                gamma,
                                eigenvalues,
                                frequencies,
                                distances_with_shift])

            distance_shift = distances_with_shift[-1]

    def write_yaml(self):
        f = open("gruneisen.yaml", 'w')
        f.write("path:\n\n")
        for band_structure in self._paths:
            (qpoints,
             distances,
             gamma,
             eigenvalues,
             frequencies,
             distances_with_shift) = band_structure
            
            f.write("- nqpoint: %d\n" % self._num_points)
            f.write("  phonon:\n")
            for q, d, gs, freqs in zip(qpoints, distances, gamma, frequencies):
                f.write("  - q-position: [ %10.7f, %10.7f, %10.7f ]\n" %
                        tuple(q))
                f.write("    distance: %10.7f\n" % d)
                f.write("    band:\n")
                for i, (g, freq) in enumerate(zip(gs, freqs)):
                    f.write("    - # %d\n" % (i + 1))
                    f.write("      gruneisen: %15.10f\n" % g)
                    f.write("      frequency: %15.10f\n" % freq)
                f.write("\n")
                    
        f.close()

    def plot(self, epsilon=1e-4):
        import matplotlib.pyplot as plt
        for band_structure in self._paths:
            (qpoints,
             distances,
             gamma,
             eigenvalues,
             frequencies,
             distances_with_shift) = band_structure
            _bandplot(plt,
                      gamma,
                      frequencies,
                      qpoints,
                      distances_with_shift,
                      epsilon)
        return plt

def _get_band_qpoints(q_start, q_end, rec_lattice, num_points=51):
    qpoints = []
    distances = []
    distance = 0.0
    q_start_ = np.array(q_start)
    q_end_ = np.array(q_end)
    dq = (q_end_ - q_start_) / (num_points - 1)
    delta = np.linalg.norm(np.dot(rec_lattice, dq))
    
    for i in range(num_points):
        distances.append(distance)
        qpoints.append(q_start_+ dq * i)
        distance += delta
        
    return np.array(qpoints), np.array(distances)

def _bandplot(plt,
              gamma,
              freqencies,
              qpoints,
              distances_with_shift,
              epsilon=1e-4):
    plt.subplot(2, 1, 1)
    for curve in gamma.T.copy():
        if (abs(qpoints[0]) < epsilon).all():
            curve[0] = curve[1]   # To avoid divergence at Gamma
        if (abs(qpoints[-1]) < epsilon).all():
            curve[-1] = curve[-2] # To avoid divergence at Gamma
        plt.plot(distances_with_shift, curve)
    plt.xlim(0, distances_with_shift[-1])

    plt.subplot(2, 1, 2)
    for freqs in freqencies.T:
        plt.plot(distances_with_shift, freqs)
    plt.xlim(0, distances_with_shift[-1])


