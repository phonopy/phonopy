"""Phonon band structure module."""

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

from __future__ import annotations

import gzip
import lzma
import sys
import warnings
from typing import Optional, Union

import numpy as np
import yaml

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.units import VaspToTHz


class BandPlot:
    """Band structure plotting class.

    This class adds band structure plots to Matplotlib axes.

    Attributes
    ----------
    xscale : float
        This is used to scale the plot shape to be nicer. The value
        can be computed using set_xscale as default, which is simply:

            xscale = max_freq / max_dist * 1.5

    """

    def __init__(self, axs):
        """Init method.

        Parameters
        ----------
        axs : Matplotlib axes of ImageGrid, optional
            axs = ImageGrid(fig, 111, nrows_ncols=(1, n), ...)

        """
        self._axs = axs
        self.xscale = None
        self._decorated = False

    def plot(self, distances, frequencies, path_connections, fmt=None, label=None):
        """Plot one band structure.

        If ``labels`` is given, decoration such as horizontal line at freq=0,
        x-label, y-label, and tics are set, which should be done only once.

        distances : list of ndarray
            Distances in reciprocal space.
            See the detail in docstring of Phonopy.get_band_structure_dict.
        frequencies : list of ndarray
            Phonon frequencies.
            See the detail in docstring of Phonopy.get_band_structure_dict.
        path_connections : list of ndarray
            This describes band segments are connected or not.
            See the detail in docstring of Phonopy.run_band_structure.
        fmt : str, optional
            Matplotlib format strings. Default is None, which is equivalent to
            'r-'.
        label : str, optional
            Label attached to band structure.

        """
        if fmt is None:
            _fmt = "r-"
        else:
            _fmt = fmt

        if self.xscale is None:
            self.set_xscale_from_data(frequencies, distances)

        count = 0
        distances_scaled = [d * self.xscale for d in distances]
        for i, (d, f, c) in enumerate(
            zip(distances_scaled, frequencies, path_connections)
        ):
            ax = self._axs[count]
            if i == 0 and label is not None:
                curves = ax.plot(d, f, _fmt, linewidth=1)
                curves[0].set_label(label)
                ax.legend()
            else:
                ax.plot(d, f, _fmt, linewidth=1)
            if not c:
                count += 1

    def set_xscale_from_data(self, frequencies, distances):
        """Set xscale from data."""
        max_freq = max([np.max(fq) for fq in frequencies])
        max_dist = distances[-1][-1]
        self.xscale = max_freq / max_dist * 1.5

    def decorate(self, labels, path_connections, frequencies, distances):
        """Decorate plots.

        Parameters
        ----------
        labels : List of str, optional
            Labels of special points.
            See the detail in docstring of Phonopy.run_band_structure.

        """
        if self._decorated:
            raise RuntimeError("Already BandPlot instance is decorated.")
        else:
            self._decorated = True

        if self.xscale is None:
            self.set_xscale_from_data(frequencies, distances)

        distances_scaled = [d * self.xscale for d in distances]

        # T T T F F -> [[0, 3], [4, 4]]
        lefts = [0]
        rights = []
        for i, c in enumerate(path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)
        seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
        special_points = []
        for indices in seg_indices:
            pts = [distances_scaled[i][0] for i in indices]
            pts.append(distances_scaled[indices[-1]][-1])
            special_points.append(pts)

        self._axs[0].set_ylabel("Frequency")
        l_count = 0
        for ax, spts in zip(self._axs, special_points):
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in")
            ax.yaxis.set_tick_params(which="both", direction="in")
            ax.set_xlim(spts[0], spts[-1])
            ax.set_xticks(spts)
            if labels is None:
                ax.set_xticklabels(
                    [
                        "",
                    ]
                    * len(spts)
                )
            else:
                ax.set_xticklabels(labels[l_count : (l_count + len(spts))])
                l_count += len(spts)
            ax.plot(
                [spts[0], spts[-1]], [0, 0], linestyle=":", linewidth=0.5, color="b"
            )


class BandStructure:
    """Class for phonons of q-poitns along reciprocal space paths.

    Note
    ----
    Numbers of qpoints on paths can be different, therefore qpoints of
    paths are stored in a list.

    Attributes
    ----------
    distances: list of ndarray
        Distances in reciprocal space made by summing up distances of
        neighboring q-points except for end points. This is useful to plot
        the band structure diagram.
        Each ndarray corresponding to each q-path has
            dtype='double'
            shape=(qpoints on a path, )
    qpoitns: list of ndarray
        q-points along reciprocal space paths.
        Each ndarray corresponding to each q-path has
            dtype='double'
            shape=(qpoints on a path, 3)
    frequencies: list of ndarray
        Phonon frequencies. Imaginary frequenies are represented by negative
        real numbers.
        Each ndarray corresponding to each q-path has
            dtype='double'
            shape=(qpoints, bands)
    eigenvectors: list of ndarray
        Phonon eigenvectors. See the data structure at np.linalg.eigh.
        Each ndarray corresponding to each q-path has
            dtype=complex of "c%d" % (np.dtype('double').itemsize * 2)
            shape=(qpoints, bands, bands)
            order='C'
    group_velocities: list of ndarray
        Phonon group velocities.
        Each ndarray corresponding to each q-path has
            dtype='double'
            shape=(qpoints, bands, 3)
    path_connections : List of bool, optional
        This gives whether each path is connected to the next path or not,
        i.e., if False, there is a jump of q-points. Number of elements is
        the same at that of paths. Default is None.
    labels : List of str, optional
        This is only used in graphical plot of band structure and gives
        labels of end points of each path. The number of labels is equal
        to (2 - np.array(path_connections)).sum().
        For non-legacy plot, for example:

            band_paths = [
                [[0, 0, 0], [0.5, 0.5, 0.5]],
                [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]],
            ]
            labels = ["G", "L", "X", "G", "W"]
            path_connections = [False, True, False]
                G -> L  False
                X -> G  True
                G -> W  False (last one has to be False)

    """

    def __init__(
        self,
        paths: list,
        dynamical_matrix: Union[DynamicalMatrix, DynamicalMatrixNAC],
        with_eigenvectors: bool = False,
        is_band_connection: bool = False,
        group_velocity: Optional[GroupVelocity] = None,
        path_connections: Optional[Union[list, bool]] = None,
        labels: Optional[list[str]] = None,
        is_legacy_plot: bool = False,
        factor: float = VaspToTHz,
    ):
        """Init method.

        Parameters
        ----------
        paths : List of array_like
            Sets of qpoints that can be passed to phonopy.set_band_structure().
            Numbers of qpoints can be different.
            shape of each array_like : (qpoints, 3)
        dynamical_matrix : DynamicalMatrix or DynamicalMatrixNAC
            Dynamical matrix calculator.
        with_eigenvectors : bool, optional
            Flag whether eigenvectors are calculated or not. Default is False.
        is_band_connection : bool, optional
            Flag whether each band is connected or not. This is achieved by
            comparing similarity of eigenvectors of neghboring poins. Sometimes
            this fails. Default is False.
        group_velocity : GroupVelocity, optional
            Group velocity calculator. Default is None.
        path_connections : List of bool, optional
            This is only used in graphical plot of band structure and gives
            whether each path is connected to the next path or not,
            i.e., if False, there is a jump of q-points. Number of elements is
            the same at that of paths. Default is None.
        labels : List of str, optional
            This is only used in graphical plot of band structure and gives
            labels of end points of each path. The number of labels is equal
            to (2 - np.array(path_connections)).sum().
        is_legacy_plot: bool, optional
            This makes the old style band structure plot. Default is False.

        """
        self._dynamical_matrix = dynamical_matrix
        self._cell = dynamical_matrix.primitive
        self._supercell = dynamical_matrix.supercell
        self._factor = factor
        self._with_eigenvectors = with_eigenvectors
        self._is_band_connection = is_band_connection
        if is_band_connection:
            self._with_eigenvectors = True
        self._group_velocity = group_velocity

        self._paths = [np.array(path) for path in paths]
        self._is_legacy_plot = is_legacy_plot
        self._labels = None
        self._path_connections = None
        if self._is_legacy_plot:
            if labels is not None and len(labels) == len(self._paths) + 1:
                self._labels = labels
        else:
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
        self._distances = []
        self._distance = 0.0
        self._special_points = [0.0]
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        self._group_velocities = None
        self._set_band()

    @property
    def distances(self) -> list:
        """Return distances of band segments."""
        return self._distances

    def get_distances(self):
        """Return distances of band segments."""
        warnings.warn(
            "BandStructure.get_distances() is deprecated."
            "Use BandStructure.distances attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.distances

    @property
    def qpoints(self) -> list:
        """Return qpoints of band segments."""
        return self._paths

    def get_qpoints(self):
        """Return qpoints of band segments."""
        warnings.warn(
            "BandStructure.get_qpoints() is deprecated."
            "Use BandStructure.qpoints attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.qpoints

    @property
    def eigenvectors(self) -> Optional[list]:
        """Return phonon eigenvectors of band segments."""
        return self._eigenvectors

    def get_eigenvectors(self):
        """Return phonon eigenvectors of band segments."""
        warnings.warn(
            "BandStructure.get_eigenvectors() is deprecated."
            "Use BandStructure.eigenvectors attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.eigenvectors

    @property
    def frequencies(self) -> Optional[list]:
        """Return phonon frequencies of band segments."""
        return self._frequencies

    def get_frequencies(self):
        """Return phonon frequencies of band segments."""
        warnings.warn(
            "BandStructure.get_frequencies() is deprecated."
            "Use BandStructure.frequencies attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.frequencies

    @property
    def group_velocities(self) -> Optional[list]:
        """Return phonon group velocities of band segments."""
        return self._group_velocities

    def get_group_velocities(self):
        """Return phonon group velocities of band segments."""
        warnings.warn(
            "BandStructure.get_group_velocities() is deprecated."
            "Use BandStructure.group_velocities attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.group_velocities

    def get_eigenvalues(self):
        """Return phonon eigenvalues of band segments."""
        warnings.warn(
            "Bandstructure.get_engenvalues is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._eigenvalues

    def get_unit_conversion_factor(self):
        """Return frequency unit conversion factor of band segments."""
        warnings.warn(
            "Bandstructure.get_unit_conversion_factor is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._factor

    @property
    def labels(self) -> Optional[list]:
        """Return special point symbols."""
        return self._labels

    @property
    def path_connections(self):
        """Return band segment connections."""
        return self._path_connections

    @property
    def is_legacy_plot(self) -> bool:
        """Identify legacy plot or not."""
        return self._is_legacy_plot

    def plot(self, ax):
        """Plot band structure.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis

        """
        if self._is_legacy_plot:
            self._plot_legacy(ax)
        else:
            self._plot(ax)

    def _plot(self, axs):
        if self._is_band_connection:
            fmt = "-"
        else:
            fmt = None
        band_plot(
            axs,
            self._frequencies,
            self._distances,
            self._path_connections,
            self._labels,
            fmt=fmt,
        )

    def _plot_legacy(self, axs):
        _plot_legacy(
            axs,
            self._distances,
            self._frequencies,
            self._labels,
            self._special_points,
            self._is_band_connection,
        )

    def write_hdf5(self, comment=None, filename="band.hdf5"):
        """Write band structure in hdf5 format."""
        import h5py

        with h5py.File(filename, "w") as w:
            w.create_dataset("path", data=self._paths)
            w.create_dataset("distance", data=self._distances)
            w.create_dataset("frequency", data=self._frequencies)
            if self._eigenvectors is not None:
                w.create_dataset("eigenvector", data=self._eigenvectors)
            if self._group_velocities is not None:
                w.create_dataset("group_velocity", data=self._group_velocities)
            if comment:
                for key in comment:
                    if key not in (
                        "path",
                        "distance",
                        "frequency",
                        "eigenvector",
                        "group_velocity",
                    ):
                        w.create_dataset(key, data=comment[key])

            path_labels = []
            if self._labels:
                if self._is_legacy_plot:
                    for i in range(len(self._paths)):
                        path_labels.append(
                            [
                                self._labels[i],
                                self._labels[i + 1],
                            ]
                        )
                else:
                    i = 0
                    for c in self._path_connections:
                        path_labels.append(
                            [
                                self._labels[i],
                                self._labels[i + 1],
                            ]
                        )
                        if c:
                            i += 1
                        else:
                            i += 2
            w.create_dataset("label", data=path_labels)

            nq_paths = []
            for qpoints in self._paths:
                nq_paths.append(len(qpoints))
            w.create_dataset("nqpoint", data=[np.sum(nq_paths)])
            w.create_dataset("segment_nqpoint", data=nq_paths)

    def write_yaml(self, comment=None, filename=None, compression=None):
        """Write band structure in yaml format.

        Parameters
        ----------
        comment : dict
            Data structure dumped in YAML and the dumped YAML text is put
            at the beggining of the file.
        filename : str
            Default filename is 'band.yaml' when compression=None.
            With compression, an extention of filename is added such as
            'band.yaml.xz'.
        compression : None, 'gzip', or 'lzma'
            None gives usual text file. 'gzip and 'lzma' compresse yaml
            text in respective compression methods.

        """
        if filename is not None:
            _filename = filename

        if compression is None:
            if filename is None:
                _filename = "band.yaml"
            with open(_filename, "w") as w:
                self._write_yaml(w, comment)
        elif compression == "gzip":
            if filename is None:
                _filename = "band.yaml.gz"
            with gzip.open(_filename, "wb") as w:
                self._write_yaml(w, comment, is_binary=True)
        elif compression == "lzma":
            if filename is None:
                _filename = "band.yaml.xz"
            with lzma.open(_filename, "w") as w:
                self._write_yaml(w, comment, is_binary=True)

    def _write_yaml(self, w, comment, is_binary=False):
        natom = len(self._cell)
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
                    text.append(f"- [ '{self._labels[i]}', '{self._labels[i + 1]}' ]")
            else:
                i = 0
                for c in self._path_connections:
                    text.append(f"- [ '{self._labels[i]}', '{self._labels[i + 1]}' ]")
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
        text.append("phonon:")
        text.append("")
        self._write_lines(w, text, is_binary)

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

            text = self._get_q_segment_yaml(
                qpoints, distances, frequencies, eigenvectors, group_velocities
            )
            self._write_lines(w, text, is_binary)

    def _get_q_segment_yaml(
        self, qpoints, distances, frequencies, eigenvectors, group_velocities
    ):
        natom = len(self._cell)
        text = []
        for j in range(len(qpoints)):
            q = qpoints[j]
            text.append("- q-position: [ %12.7f, %12.7f, %12.7f ]" % tuple(q))
            text.append("  distance: %12.7f" % distances[j])
            text.append("  band:")
            for k, freq in enumerate(frequencies[j]):
                text.append("  - # %d" % (k + 1))
                text.append("    frequency: %15.10f" % freq)

                if group_velocities is not None:
                    gv = group_velocities[j, k]
                    text.append(
                        "    group_velocity: " "[ %13.7f, %13.7f, %13.7f ]" % tuple(gv)
                    )

                if eigenvectors is not None:
                    text.append("    eigenvector:")
                    for ll in range(natom):
                        text.append("    - # atom %d" % (ll + 1))
                        for m in (0, 1, 2):
                            text.append(
                                "      - [ %17.14f, %17.14f ]"
                                % (
                                    eigenvectors[j, ll * 3 + m, k].real,
                                    eigenvectors[j, ll * 3 + m, k].imag,
                                )
                            )
            text.append("")
        text.append("")

        return text

    def _write_lines(self, w, lines, is_binary):
        text = "\n".join(lines)
        if is_binary:
            if sys.version_info < (3, 0):
                w.write(bytes(text))
            else:
                w.write(bytes(text, "utf8"))
        else:
            w.write(text)

    def _set_initial_point(self, qpoint):
        self._lastq = qpoint.copy()

    def _shift_point(self, qpoint):
        self._distance += np.linalg.norm(
            np.dot(qpoint - self._lastq, np.linalg.inv(self._cell.cell).T)
        )
        self._lastq = qpoint.copy()

    def _set_band(self):
        eigvals = []
        eigvecs = []
        group_velocities = []
        distances = []

        for path in self._paths:
            self._set_initial_point(path[0])

            (
                distances_on_path,
                eigvals_on_path,
                eigvecs_on_path,
                gv_on_path,
            ) = self._solve_dm_on_path(path)

            eigvals.append(np.array(eigvals_on_path))
            if self._with_eigenvectors:
                eigvecs.append(np.array(eigvecs_on_path))
            if self._group_velocity is not None:
                group_velocities.append(np.array(gv_on_path))
            distances.append(np.array(distances_on_path))
            self._special_points.append(self._distance)

        self._eigenvalues = eigvals
        if self._with_eigenvectors:
            self._eigenvectors = eigvecs
        if self._group_velocity is not None:
            self._group_velocities = group_velocities
        self._distances = distances

        self._set_frequencies()

    def _solve_dm_on_path(self, path):
        distances_on_path = []
        eigvals_on_path = []
        eigvecs_on_path = []
        gv_on_path = []
        prev_eigvecs = None

        if self._group_velocity is not None:
            self._group_velocity.run(path)
            gv = self._group_velocity.group_velocities

        if isinstance(self._dynamical_matrix, DynamicalMatrixNAC):
            q_direction = None
            # A cross product close to 0 indicates a path crossing or ending at Gamma
            rec_lat = np.linalg.inv(self._dynamical_matrix.primitive.cell)
            dist_from_Gamma = np.linalg.norm(
                np.cross(rec_lat @ path[0], rec_lat @ path[-1])
            )
            if dist_from_Gamma < DynamicalMatrixNAC.Q_DIRECTION_TOLERANCE:
                q_direction = path[0] - path[-1]

        for i, q in enumerate(path):
            self._shift_point(q)
            distances_on_path.append(self._distance)

            if isinstance(self._dynamical_matrix, DynamicalMatrixNAC):
                self._dynamical_matrix.run(q, q_direction=q_direction)
            else:
                self._dynamical_matrix.run(q)
            dm = self._dynamical_matrix.dynamical_matrix

            if self._with_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(dm)
                eigvals = eigvals.real
            else:
                eigvals = np.linalg.eigvalsh(dm).real

            if self._is_band_connection:
                if i == 0:
                    band_order = range(len(eigvals))
                else:
                    band_order = estimate_band_connection(
                        prev_eigvecs, eigvecs, band_order
                    )
                eigvals_on_path.append(eigvals[band_order])
                eigvecs_on_path.append((eigvecs.T)[band_order].T)

                if self._group_velocity is not None:
                    gv_on_path.append(gv[i][band_order])
                prev_eigvecs = eigvecs
            else:
                eigvals_on_path.append(eigvals)
                if self._with_eigenvectors:
                    eigvecs_on_path.append(eigvecs)
                if self._group_velocity is not None:
                    gv_on_path.append(gv[i])

        return distances_on_path, eigvals_on_path, eigvecs_on_path, gv_on_path

    def _set_frequencies(self):
        frequencies = []
        for eigs_path in self._eigenvalues:
            frequencies.append(
                np.sqrt(abs(eigs_path)) * np.sign(eigs_path) * self._factor
            )
        self._frequencies = frequencies


def estimate_band_connection(prev_eigvecs, eigvecs, prev_band_order):
    """Connect neighboring qpoints by eigenvector similarity."""
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


def get_band_qpoints_and_path_connections(band_paths, npoints=51, rec_lattice=None):
    """Return qpoints and connections of paths."""
    path_connections = []
    for paths in band_paths:
        path_connections += [
            True,
        ] * (len(paths) - 2)
        path_connections.append(False)
    return (
        get_band_qpoints(band_paths, npoints=npoints, rec_lattice=rec_lattice),
        path_connections,
    )


def get_band_qpoints(band_paths, npoints=51, rec_lattice=None):
    """Generate qpoints for band structure path.

    Note
    ----

    Behavior changes with and without rec_lattice given.

    Parameters
    ----------
    band_paths: list of array_likes
        Sets of end points of paths
        dtype='double'
        shape=(sets of paths, paths, 3)

        example:
            [[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
             [[0.5, 0.25, 0.75], [0, 0, 0]]]

    npoints: int, optional
        Number of q-points in each path including end points. Default is 51.

    rec_lattice: array_like, optional
        When given, q-points are sampled in a similar interval. The longest
        path length divided by npoints including end points is used as the
        reference interval. Reciprocal basis vectors given in column vectors.
        dtype='double'
        shape=(3, 3)

    """
    npts = _get_npts(band_paths, npoints, rec_lattice)
    qpoints_of_paths = []
    c = 0
    for band_path in band_paths:
        nd = len(band_path)
        for i in range(nd - 1):
            delta = np.subtract(band_path[i + 1], band_path[i]) / (npts[c] - 1)
            qpoints = [delta * j for j in range(npts[c])]
            qpoints_of_paths.append(np.array(qpoints) + band_path[i])
            c += 1

    return qpoints_of_paths


def get_band_qpoints_by_seekpath(primitive, npoints, is_const_interval=False):
    """q-points along BZ high symmetry paths are generated using seekpath.

    Parameters
    ----------
    primitive : PhonopyAtoms
        Primitive cell.
    npoints : int
        Number of q-points sampled along a path including end points.
    is_const_interval : bool, optional
        When True, q-points are sampled in a similar interval. The longest
        path length divided by npoints including end points is used as the
        reference interval. Default is False.

    Returns
    -------
    bands : List of ndarray
        Sets of qpoints that can be passed to phonopy.set_band_structure().
        shape of each ndarray : (npoints, 3)
    labels : List of pairs of str
        Symbols of end points of paths.
    connections : List of bool
        This gives one path is connected to the next path, i.e., if False,
        there is a jump of q-points. Number of elements is the same at
        that of paths.

    """
    try:
        import seekpath
    except ImportError as exc:
        raise ModuleNotFoundError("You need to install seekpath.") from exc

    band_path = seekpath.get_path(primitive.totuple())
    point_coords = band_path["point_coords"]
    qpoints_of_paths = []
    if is_const_interval:
        reclat = np.linalg.inv(primitive.cell)
    else:
        reclat = None
    band_paths = [
        [point_coords[path[0]], point_coords[path[1]]] for path in band_path["path"]
    ]
    npts = _get_npts(band_paths, npoints, reclat)
    for c, path in enumerate(band_path["path"]):
        q_s = np.array(point_coords[path[0]])
        q_e = np.array(point_coords[path[1]])
        band = [q_s + (q_e - q_s) / (npts[c] - 1) * i for i in range(npts[c])]
        qpoints_of_paths.append(band)
    labels, path_connections = _get_labels(band_path["path"])

    return qpoints_of_paths, labels, path_connections


def band_plot(axs, frequencies, distances, path_connections, labels, fmt="r-"):
    """Return band structure plot."""
    bp = BandPlot(axs)
    bp.decorate(labels, path_connections, frequencies, distances)
    bp.plot(distances, frequencies, path_connections, fmt=fmt)


def _plot_legacy(
    ax, all_distances, all_frequencies, labels, special_points, is_band_connection
):
    """Plot band structure in legacy style."""
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    for distances, frequencies in zip(all_distances, all_frequencies):
        for freqs in frequencies.T:
            if is_band_connection:
                ax.plot(distances, freqs, "-")
            else:
                ax.plot(distances, freqs, "r-")

    ax.set_ylabel("Frequency")
    ax.set_xlabel("Wave vector")

    if labels and len(labels) == len(special_points):
        ax.set_xticks(special_points)
        ax.set_xticklabels(labels)
    else:
        ax.set_xticks(special_points)
        ax.set_xticklabels(
            [
                "",
            ]
            * len(special_points)
        )

    ax.set_xlim(0, all_distances[-1][-1])
    ax.axhline(y=0, linestyle=":", linewidth=0.5, color="b")


def _get_npts(band_paths, npoints, rec_lattice):
    """Return numbers of qpoints of band segments."""
    if rec_lattice is not None:
        path_lengths = []
        for band_path in band_paths:
            nd = len(band_path)
            for i in range(nd - 1):
                vector = np.subtract(band_path[i + 1], band_path[i])
                length = np.linalg.norm(np.dot(rec_lattice, vector))
                path_lengths.append(length)
        max_length = max(path_lengths)
        npts = [np.rint(pl / max_length * npoints).astype(int) for pl in path_lengths]
    else:
        npts = [
            npoints,
        ] * np.sum([len(paths) for paths in band_paths])

    for i, npt in enumerate(npts):
        if npt < 2:
            npts[i] = 2

    return npts


def _get_labels(pairs_of_symbols):
    path_connections = []
    labels = []

    for i, pairs in enumerate(pairs_of_symbols[:-1]):
        if pairs[1] != pairs_of_symbols[i + 1][0]:
            path_connections.append(False)
            labels += list(pairs)
        else:
            path_connections.append(True)
            labels.append(pairs[0])
    path_connections.append(False)
    labels += list(pairs_of_symbols[-1])

    for i, ll in enumerate(labels):
        if "GAMMA" in ll:
            labels[i] = "$" + ll.replace("GAMMA", r"\Gamma") + "$"
        elif "SIGMA" in ll:
            labels[i] = "$" + ll.replace("SIGMA", r"\Sigma") + "$"
        elif "DELTA" in ll:
            labels[i] = "$" + ll.replace("DELTA", r"\Delta") + "$"
        elif "LAMBDA" in ll:
            labels[i] = "$" + ll.replace("LAMBDA", r"\Lambda") + "$"
        else:
            labels[i] = r"$\mathrm{%s}$" % ll

    return labels, path_connections
