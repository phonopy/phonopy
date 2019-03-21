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
from .core import GruneisenBase
from phonopy.units import VaspToTHz


class GruneisenBandStructure(GruneisenBase):
    def __init__(self,
                 paths,
                 dynmat,
                 dynmat_plus,
                 dynmat_minus,
                 delta_strain=None,
                 factor=VaspToTHz):
        GruneisenBase.__init__(self,
                               dynmat,
                               dynmat_plus,
                               dynmat_minus,
                               delta_strain=delta_strain,
                               is_band_connection=True)
        primitive = dynmat.get_primitive()
        rec_lattice = np.linalg.inv(primitive.get_cell())
        distance_shift = 0.0

        self._paths = []
        for qpoints_ in paths:
            qpoints = np.array(qpoints_)
            distances = np.zeros(len(qpoints))
            delta_qpoints = qpoints[1:] - qpoints[:-1]
            delta_distances = np.sqrt(
                (np.dot(delta_qpoints, rec_lattice) ** 2).sum(axis=1))
            for i, dd in enumerate(delta_distances):
                distances[i + 1] = distances[i] + dd

            self.set_qpoints(qpoints)
            eigenvalues = self._eigenvalues
            frequencies = np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues) * factor
            distances_with_shift = distances + distance_shift

            self._paths.append([qpoints,
                                distances,
                                self._gruneisen,
                                eigenvalues,
                                self._eigenvectors,
                                frequencies,
                                distances_with_shift])

            distance_shift = distances_with_shift[-1]

    def get_qpoints(self):
        return [path[0] for path in self._paths]

    def get_distances(self):
        return [path[6] for path in self._paths]

    def get_gruneisen(self):
        return [path[2] for path in self._paths]

    def get_eigenvalues(self):
        return [path[3] for path in self._paths]

    def get_eigenvectors(self):
        return [path[4] for path in self._paths]

    def get_frequencies(self):
        return [path[5] for path in self._paths]

    def write_yaml(self):
        f = open("gruneisen.yaml", 'w')
        f.write("path:\n\n")
        for band_structure in self._paths:
            (qpoints,
             distances,
             gamma,
             eigenvalues,
             _,
             frequencies,
             distances_with_shift) = band_structure

            f.write("- nqpoint: %d\n" % len(qpoints))
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

    def plot(self,
             axarr,
             epsilon=None,
             color_scheme=None):
        for band_structure in self._paths:
            self._plot(axarr, band_structure, epsilon, color_scheme)

    def _plot(self, axarr, band_structure, epsilon, color_scheme):
        (qpoints,
         distances,
         gamma,
         eigenvalues,
         _,
         frequencies,
         distances_with_shift) = band_structure

        n = len(gamma.T) - 1
        ax1, ax2 = axarr

        for i, (curve, freqs) in enumerate(zip(gamma.T.copy(), frequencies.T)):
            if epsilon is not None:
                if np.linalg.norm(qpoints[0]) < epsilon:
                    cutoff_index = 0
                    for j, q in enumerate(qpoints):
                        if not np.linalg.norm(q) < epsilon:
                            cutoff_index = j
                            break
                    for j in range(cutoff_index):
                        if abs(freqs[j]) < abs(max(freqs)) / 10:
                            curve[j] = curve[cutoff_index]

                if np.linalg.norm(qpoints[-1]) < epsilon:
                    cutoff_index = len(qpoints) - 1
                    for j in reversed(range(len(qpoints))):
                        q = qpoints[j]
                        if not np.linalg.norm(q) < epsilon:
                            cutoff_index = j
                            break
                    for j in reversed(range(len(qpoints))):
                        if j == cutoff_index:
                            break
                        if abs(freqs[j]) < abs(max(freqs)) / 10:
                            curve[j] = curve[cutoff_index]

            self._plot_a_band(ax1, curve, distances_with_shift, i, n,
                              color_scheme)
        ax1.set_xlim(0, distances_with_shift[-1])

        for i, freqs in enumerate(frequencies.T):
            self._plot_a_band(ax2, freqs, distances_with_shift, i, n,
                              color_scheme)
        ax2.set_xlim(0, distances_with_shift[-1])

    def _plot_a_band(self, ax, curve, distances_with_shift, i, n, color_scheme):
        color = None
        if color_scheme == 'RB':
            color = (1. / n * i, 0, 1./ n * (n - i))
        elif color_scheme == 'RG':
            color = (1. / n * i, 1./ n * (n - i), 0)
        elif color_scheme == 'RGB':
            color = (max(2./ n * (i - n / 2.), 0),
                     min(2./ n * i, 2./ n * (n - i)),
                     max(2./ n * (n / 2. - i), 0))
        if color:
            ax.plot(distances_with_shift, curve, color=color)
        else:
            ax.plot(distances_with_shift, curve)
