"""Mode Grueneisen parameter band structure calculation."""

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

import gzip
import lzma
import sys

import numpy as np
import yaml

from phonopy.gruneisen.core import GruneisenBase
from phonopy.units import VaspToTHz


class GruneisenBandStructure(GruneisenBase):
    """Class to calculate mode Grueneisen parameter along band structure paths."""

    def __init__(
        self,
        paths,
        dynmat,
        dynmat_plus,
        dynmat_minus,
        delta_strain=None,
        path_connections=None,
        labels=None,
        factor=VaspToTHz,
    ):
        """Init method."""
        super().__init__(
            dynmat,
            dynmat_plus,
            dynmat_minus,
            delta_strain=delta_strain,
            is_band_connection=True,
        )
        self._cell = dynmat.primitive
        rec_lattice = np.linalg.inv(self._cell.cell)
        distance_shift = 0.0

        self._paths = []
        for qpoints_ in paths:
            qpoints = np.array(qpoints_)
            distances = np.zeros(len(qpoints))
            delta_qpoints = qpoints[1:] - qpoints[:-1]
            delta_distances = np.sqrt(
                (np.dot(delta_qpoints, rec_lattice) ** 2).sum(axis=1)
            )
            for i, dd in enumerate(delta_distances):
                distances[i + 1] = distances[i] + dd

            self.set_qpoints(qpoints)
            eigenvalues = self._eigenvalues
            frequencies = np.sqrt(abs(eigenvalues)) * np.sign(eigenvalues) * factor
            distances_with_shift = distances + distance_shift

            self._paths.append(
                [
                    qpoints,
                    distances,
                    self._gruneisen,
                    eigenvalues,
                    self._eigenvectors,
                    frequencies,
                    distances_with_shift,
                ]
            )

            distance_shift = distances_with_shift[-1]
        self._labels = None
        self._path_connections = None
        if path_connections is None:
            self._path_connections = [
                True,
            ] * len(self._paths)
            self._path_connections[-1] = False
        else:
            self._path_connections = path_connections
        if (
            labels is not None
            and len(labels) == (2 - np.array(self._path_connections)).sum()
        ):
            self._labels = labels

    def get_qpoints(self):
        """Return q-points."""
        return [path[0] for path in self._paths]

    def get_distances(self):
        """Return distances."""
        return [path[6] for path in self._paths]

    def get_gruneisen(self):
        """Return mode Gruneisen parameters."""
        return [path[2] for path in self._paths]

    def get_eigenvalues(self):
        """Return eigenvalues."""
        return [path[3] for path in self._paths]

    def get_eigenvectors(self):
        """Return eigenvectors."""
        return [path[4] for path in self._paths]

    def get_frequencies(self):
        """Return frequencies."""
        return [path[5] for path in self._paths]

    def write_yaml(self, comment=None, filename=None, compression=None):
        """Write results to file in yaml."""
        if filename is not None:
            _filename = filename

        if compression is None:
            if filename is None:
                _filename = "gruneisen.yaml"
            with open(_filename, "w") as w:
                self._write_yaml(w, comment)
        elif compression == "gzip":
            if filename is None:
                _filename = "gruneisen.yaml.gz"
            with gzip.open(_filename, "wb") as w:
                self._write_yaml(w, comment, is_binary=True)
        elif compression == "lzma":
            if filename is None:
                _filename = "gruneisen.yaml.xz"
            with lzma.open(_filename, "w") as w:
                self._write_yaml(w, comment, is_binary=True)

    def _write_yaml(self, w, comment, is_binary=False):
        natom = self._cell.get_number_of_atoms()
        rec_lattice = np.linalg.inv(self._cell.cell)  # column vecs
        nq_paths = []
        for qpoints in self._paths:
            nq_paths.append(len(qpoints))
        text = []
        if comment is not None:
            text.append(yaml.dump(comment, default_flow_style=False).rstrip())
        text.append("nqpoint: %-7d" % np.sum(nq_paths))
        text.append("npath: %-7d" % len(self._paths))
        text.append("segment_nqpoint:")
        text += ["- %d" % nq for nq in nq_paths]
        if self._labels:
            text.append("labels:")
            if self._is_legacy_plot:
                for i in range(len(self._paths)):
                    text.append(
                        "- [ '%s', '%s' ]" % (self._labels[i], self._labels[i + 1])
                    )
            else:
                i = 0
                for c in self._path_connections:
                    text.append(
                        "- [ '%s', '%s' ]" % (self._labels[i], self._labels[i + 1])
                    )
                    if c:
                        i += 1
                    else:
                        i += 2
        text.append("reciprocal_lattice:")
        for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*")):
            text.append("- [ %12.8f, %12.8f, %12.8f ] # %2s" % (tuple(vec) + (axis,)))
        text.append("natom: %-7d" % (natom))
        text.append(str(self._cell))
        text.append("")
        text.append("path:")
        text.append("")

        for band_structure in self._paths:
            (
                qpoints,
                distances,
                gamma,
                eigenvalues,
                _,
                frequencies,
                distances_with_shift,
            ) = band_structure

            text.append("- nqpoint: %d" % len(qpoints))
            text.append("  phonon:")
            for q, d, gs, freqs in zip(qpoints, distances, gamma, frequencies):
                text.append("  - q-position: [ %10.7f, %10.7f, %10.7f ]" % tuple(q))
                text.append("    distance: %10.7f" % d)
                text.append("    band:")
                for i, (g, freq) in enumerate(zip(gs, freqs)):
                    text.append("    - # %d" % (i + 1))
                    text.append("      gruneisen: %15.10f" % g)
                    text.append("      frequency: %15.10f" % freq)
                text.append("")

        self._write_lines(w, text, is_binary)

    def _write_lines(self, w, lines, is_binary):
        text = "\n".join(lines)
        if is_binary:
            if sys.version_info < (3, 0):
                w.write(bytes(text))
            else:
                w.write(bytes(text, "utf8"))
        else:
            w.write(text)

    def plot(self, axarr, epsilon=None, color_scheme=None):
        """Return pyplot of band structure calculation results."""
        for band_structure in self._paths:
            self._plot(axarr, band_structure, epsilon, color_scheme)

    def _plot(self, axarr, band_structure, epsilon, color_scheme):
        (
            qpoints,
            distances,
            gamma,
            eigenvalues,
            _,
            frequencies,
            distances_with_shift,
        ) = band_structure

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

            self._plot_a_band(ax1, curve, distances_with_shift, i, n, color_scheme)
        ax1.set_xlim(0, distances_with_shift[-1])

        for i, freqs in enumerate(frequencies.T):
            self._plot_a_band(ax2, freqs, distances_with_shift, i, n, color_scheme)
        ax2.set_xlim(0, distances_with_shift[-1])

    def _plot_a_band(self, ax, curve, distances_with_shift, i, n, color_scheme):
        color = None
        if color_scheme == "RB":
            color = (1.0 / n * i, 0, 1.0 / n * (n - i))
        elif color_scheme == "RG":
            color = (1.0 / n * i, 1.0 / n * (n - i), 0)
        elif color_scheme == "RGB":
            color = (
                max(2.0 / n * (i - n / 2.0), 0),
                min(2.0 / n * i, 2.0 / n * (n - i)),
                max(2.0 / n * (n / 2.0 - i), 0),
            )
        if color:
            ax.plot(distances_with_shift, curve, color=color)
        else:
            ax.plot(distances_with_shift, curve)
