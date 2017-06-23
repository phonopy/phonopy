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
    for overlaps in metric:
        maxval = 0
        for i in reversed(range(len(metric))):
            val = overlaps[i]
            if i in connection_order:
                continue
            if val > maxval:
                maxval = val
                maxindex = i
        connection_order.append(maxindex)

    band_order = [connection_order[x] for x in prev_band_order]

    return band_order


class BandStructure(object):
    def __init__(self,
                 paths,
                 dynamical_matrix,
                 is_eigenvectors=False,
                 is_band_connection=False,
                 group_velocity=None,
                 factor=VaspToTHz):
        self._dynamical_matrix = dynamical_matrix
        self._cell = dynamical_matrix.get_primitive()
        self._supercell = dynamical_matrix.get_supercell()
        self._factor = factor
        self._is_eigenvectors = is_eigenvectors
        self._is_band_connection = is_band_connection
        if is_band_connection:
            self._is_eigenvectors = True
        self._group_velocity = group_velocity

        self._paths = [np.array(path) for path in paths]
        self._distances = []
        self._distance = 0.
        self._special_points = [0.]
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._group_velocities = None
        self._set_band()

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

    def plot(self, pyplot, labels=None):
        for distances, frequencies in zip(self._distances,
                                          self._frequencies):
            for freqs in frequencies.T:
                if self._is_band_connection:
                    pyplot.plot(distances, freqs, '-')
                else:
                    pyplot.plot(distances, freqs, 'r-')

        pyplot.ylabel('Frequency')
        pyplot.xlabel('Wave vector')
        if labels and len(labels) == len(self._special_points):
            pyplot.xticks(self._special_points, labels)
        else:
            pyplot.xticks(self._special_points,
                          [''] * len(self._special_points))
        pyplot.xlim(0, self._distance)
        pyplot.axhline(y=0, linestyle=':', linewidth=0.5, color='b')

    def write_yaml(self, labels=None, comment=None, filename="band.yaml"):
        with open(filename, 'w') as w:
            natom = self._cell.get_number_of_atoms()
            rec_lattice = np.linalg.inv(self._cell.get_cell()) # column vectors
            smat = self._supercell.get_supercell_matrix()
            pmat = self._cell.get_primitive_matrix()
            tmat = np.rint(np.dot(np.linalg.inv(pmat), smat)).astype(int)
            nq_paths = []
            for qpoints in self._paths:
                nq_paths.append(len(qpoints))
            text = []
            if comment is not None:
                try:
                    import yaml
                    text.append(
                        yaml.dump(comment, default_flow_style=False).rstrip())
                except ImportError:
                    print("You need to install python-yaml.")
                    print("Additional comments were not written in %s." %
                          filename)
            text.append("nqpoint: %-7d" % np.sum(nq_paths))
            text.append("npath: %-7d" % len(self._paths))
            text.append("segment_nqpoint:")
            text += ["- %d" % nq for nq in nq_paths]
            text.append("reciprocal_lattice:")
            for vec, axis in zip(rec_lattice.T, ('a*', 'b*', 'c*')):
                text.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" %
                            (tuple(vec) + (axis,)))
            text.append("natom: %-7d" % (natom))
            text.append(str(self._cell))
            text.append("supercell_matrix:")
            for v in tmat:
                text.append("- [ %4d, %4d, %4d ]" % tuple(v))
            text.append('')
            text.append("phonon:")
            text.append('')
            w.write("\n".join(text))

            for i in range(len(self._paths)):
                qpoints = self._paths[i]
                distances = self._distances[i]
                frequencies = self._frequencies[i]
                if self._group_velocities is None:
                    group_velocities = None
                else:
                    group_velocities = self._group_velocities[i]
                if self._eigenvectors is None:
                    eigenvectors = None
                else:
                    eigenvectors = self._eigenvectors[i]
                _labels = None
                if labels is not None:
                    if len(labels) == len(self._paths) + 1:
                        _labels = (labels[i], labels[i + 1])

                w.write("\n".join(self._get_q_segment_yaml(qpoints,
                                                           distances,
                                                           frequencies,
                                                           eigenvectors,
                                                           group_velocities,
                                                           _labels)))

    def _get_q_segment_yaml(self,
                            qpoints,
                            distances,
                            frequencies,
                            eigenvectors,
                            group_velocities,
                            labels):
        natom = self._cell.get_number_of_atoms()
        text = []
        for j in range(len(qpoints)):
            q = qpoints[j]
            text.append("- q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(q))
            text.append("  distance: %12.7f" % distances[j])
            if labels is not None:
                if j == 0:
                    text.append("  label: \'%s\'" % labels[0])
                elif j == len(qpoints) - 1:
                    text.append("  label: \'%s\'" % labels[1])
            text.append("  band:")
            for k, freq in enumerate(frequencies[j]):
                text.append("  - # %d" % (k + 1))
                text.append("    frequency: %15.10f" % freq)

                if group_velocities is not None:
                    gv = group_velocities[j, k]
                    text.append("    group_velocity: "
                                "[ %13.7f, %13.7f, %13.7f ]" % tuple(gv))

                if eigenvectors is not None:
                    text.append("    eigenvector:")
                    for l in range(natom):
                        text.append("    - # atom %d" % (l + 1))
                        for m in (0, 1, 2):
                            text.append("      - [ %17.14f, %17.14f ]" %
                                        (eigenvectors[j, l * 3 + m, k].real,
                                         eigenvectors[j, l * 3 + m, k].imag))
            text.append('')
        text.append('')

        return text

    def _set_initial_point(self, qpoint):
        self._lastq = qpoint.copy()

    def _shift_point(self, qpoint):
        self._distance += np.linalg.norm(
            np.dot(qpoint - self._lastq,
                   np.linalg.inv(self._cell.get_cell()).T))
        self._lastq = qpoint.copy()

    def _set_band(self):
        eigvals = []
        eigvecs = []
        group_velocities = []
        distances = []
        is_nac = self._dynamical_matrix.is_nac()

        for path in self._paths:
            self._set_initial_point(path[0])

            (distances_on_path,
             eigvals_on_path,
             eigvecs_on_path,
             gv_on_path) = self._solve_dm_on_path(path)

            eigvals.append(np.array(eigvals_on_path))
            if self._is_eigenvectors:
                eigvecs.append(np.array(eigvecs_on_path))
            if self._group_velocity is not None:
                group_velocities.append(np.array(gv_on_path))
            distances.append(np.array(distances_on_path))
            self._special_points.append(self._distance)

        self._eigenvalues = eigvals
        if self._is_eigenvectors:
            self._eigenvectors = eigvecs
        if self._group_velocity is not None:
            self._group_velocities = group_velocities
        self._distances = distances

        self._set_frequencies()

    def _solve_dm_on_path(self, path):
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
                q_direction = None
                if (np.abs(q) < 0.0001).all(): # For Gamma point
                    q_direction = path[0] - path[-1]
                self._dynamical_matrix.set_dynamical_matrix(
                    q, q_direction=q_direction)
            else:
                self._dynamical_matrix.set_dynamical_matrix(q)
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
        self._frequencies = frequencies
