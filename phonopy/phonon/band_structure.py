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
from phonopy.units import VaspToTHz

def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order):
    metric = np.abs(np.dot(prev_eigvecs.conjugate().T, eigvecs))
    connection_order = []
    indices = range(len(metric))
    indices.reverse()
    for overlaps in metric:
        maxval = 0
        for i in indices:
            val = overlaps[i]
            if i in connection_order:
                continue
            if val > maxval:
                maxval = val
                maxindex = i
        connection_order.append(maxindex)

    band_order = [connection_order[x] for x in prev_band_order]
        
    return band_order


class BandStructure:
    def __init__(self,
                 paths,
                 dynamical_matrix,
                 cell,
                 is_eigenvectors=False,
                 is_band_connection=False,
                 group_velocity=None,
                 factor=VaspToTHz,
                 verbose=False):
        self._dynamical_matrix = dynamical_matrix
        self._cell = cell
        self._factor = factor
        self._is_eigenvectors = is_eigenvectors
        self._is_band_connection = is_band_connection
        if is_band_connection:
            self._is_eigenvectors = True
        self._group_velocity = group_velocity
            
        self._paths = [np.array(path) for path in paths]
        self._distances = []
        self._distance = 0.
        self._special_point = [0.]
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._set_band(verbose=verbose)

    def get_distances(self):
        return self._distances

    def get_qpoints(self):
        return self._paths

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_eigenvectors(self):
        return self._eigenvectors

    def get_frequencies(self):
        return self._frequencies

    def get_group_velocities(self):
        return self._group_velocities
    
    def get_unit_conversion_factor(self):
        return self._factor
    
    def plot_band(self, symbols=None):
        import matplotlib.pyplot as plt
        if symbols:
            from matplotlib import rc
            rc('text', usetex=True)

        for distances, frequencies in zip(self._distances,
                                          self._frequencies):
            for freqs in frequencies.T:
                if self._is_band_connection:
                    plt.plot(distances, freqs, '-')
                else:
                    plt.plot(distances, freqs, 'r-')

        plt.ylabel('Frequency')
        plt.xlabel('Wave vector')
        if symbols and len(symbols)==len(self._special_point):
            plt.xticks(self._special_point, symbols)
        else:
            plt.xticks(self._special_point, [''] * len(self._special_point))
        plt.xlim(0, self._distance)
        plt.axhline(y=0, linestyle=':', linewidth=0.5, color='b')
        return plt

    def write_yaml(self):
        f = open('band.yaml', 'w')
        natom = self._cell.get_number_of_atoms()
        nqpoint = 0
        for qpoints in self._paths:
            nqpoint += len(qpoints)
        f.write("nqpoint: %-7d\n" % nqpoint)
        f.write("npath: %-7d\n" % len(self._paths))
        f.write("natom: %-7d\n" % (natom))
        f.write("phonon:\n")
        for i, (qpoints, distances, frequencies) in enumerate(zip(
            self._paths,
            self._distances,
            self._frequencies)):
             for j, q in enumerate(qpoints):
                f.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % tuple(q))
                f.write("  distance: %12.7f\n" % distances[j])
                f.write("  band:\n")
                for k, freq in enumerate(frequencies[j]):
                    f.write("  - # %d\n" % (k + 1))
                    f.write("    frequency: %15.10f\n" % freq)
    
                    if self._group_velocity is not None:
                        gv = self._group_velocities[i][j, k]
                        f.write("    group_velocity: ")
                        f.write("[ %13.7f, %13.7f, %13.7f ]\n" % tuple(gv))
                        
                    if self._is_eigenvectors:
                        eigenvectors = self._eigenvectors[i]
                        f.write("    eigenvector:\n")
                        for l in range(natom):
                            f.write("    - # atom %d\n" % (l + 1))
                            for m in (0, 1, 2):
                                f.write("      - [ %17.14f, %17.14f ]\n" %
                                        (eigenvectors[j, l * 3 + m, k].real,
                                         eigenvectors[j, l * 3 + m, k].imag))

                        
                f.write("\n")

    def _set_initial_point(self, qpoint):
        self._lastq = qpoint.copy()

    def _shift_point(self, qpoint):
        self._distance += np.linalg.norm(
            np.dot(qpoint - self._lastq,
                   np.linalg.inv(self._cell.get_cell()).T))
        self._lastq = qpoint.copy()

    def _set_band(self, verbose=False):
        eigvals = []
        eigvecs = []
        group_velocities = []
        distances = []
        is_nac = self._dynamical_matrix.is_nac()

        for path in self._paths:
            self._set_initial_point(path[0])

            q_direction = None # used for NAC only
            if is_nac:
                # One of end points has to be Gamma point.
                if (np.linalg.norm(path[0]) < 0.0001 or 
                    np.linalg.norm(path[-1]) < 0.0001):
                    q_direction = path[0] - path[-1]
            
            (distances_on_path,
             eigvals_on_path,
             eigvecs_on_path,
             gv_on_path) = self._solve_dm_on_path(path,
                                                  q_direction,
                                                  verbose)

            eigvals.append(np.array(eigvals_on_path))
            if self._is_eigenvectors:
                eigvecs.append(np.array(eigvecs_on_path))
            if self._group_velocity is not None:
                group_velocities.append(np.array(gv_on_path))
            distances.append(np.array(distances_on_path))
            self._special_point.append(self._distance)

        self._eigenvalues = eigvals
        if self._is_eigenvectors:
            self._eigenvectors = eigvecs
        if self._group_velocity is not None:
            self._group_velocities = group_velocities
        self._distances = distances
        
        self._set_frequencies()

    def _solve_dm_on_path(self, path, q_direction, verbose):
        is_nac = self._dynamical_matrix.is_nac()
        distances_on_path = []
        eigvals_on_path = []
        eigvecs_on_path = []
        gv_on_path = []

        if self._group_velocity is not None:
            self._group_velocity.set_q_points(path)
            gv = self._group_velocity.get_group_velocity()
        
        for i, q in enumerate(path):
            self._shift_point(q)
            distances_on_path.append(self._distance)
            
            if is_nac:
                self._dynamical_matrix.set_dynamical_matrix(
                    q, q_direction=q_direction, verbose=verbose)
            else:
                self._dynamical_matrix.set_dynamical_matrix(
                    q, verbose=verbose)
            dm = self._dynamical_matrix.get_dynamical_matrix()

            if self._is_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(dm)
                eigvals = eigvals.real
            else:
                eigvals = np.linalg.eigvalsh(dm).real

            if self._is_band_connection:
                if i == 0:
                    band_order = range(len(eigvals))
                else:
                    band_order = estimate_band_connection(prev_eigvecs,
                                                          eigvecs,
                                                          band_order)
                eigvals_on_path.append(eigvals[band_order])
                eigvecs_on_path.append((eigvecs.T)[band_order].T)

                if self._group_velocity is not None:
                    gv_on_path.append(gv[i][band_order])
                prev_eigvecs = eigvecs
            else:
                eigvals_on_path.append(eigvals)
                if self._is_eigenvectors:
                    eigvecs_on_path.append(eigvecs)
                if self._group_velocity is not None:
                    gv_on_path.append(gv[i])

        return distances_on_path, eigvals_on_path, eigvecs_on_path, gv_on_path

    def _set_frequencies(self):
        frequencies = []
        for eigs_path in self._eigenvalues:
            frequencies.append(np.sqrt(abs(eigs_path)) * np.sign(eigs_path)
                               * self._factor)
            ## This expression may not be supported in old python versions.
            # freqs_on_path = []
            # for eigs in eigs_path:
            #     freqs = [np.sqrt(x) if x > 0 else -np.sqrt(-x) for x in eigs])
            # frequencies.append(np.array(freqs_on_path) * self._factor)
        self._frequencies = frequencies
