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

import argparse
import dataclasses
import lzma
import os
import sys
from collections.abc import Sequence

import h5py
import numpy as np

from phonopy.phonon.band_structure import BandPlot
from phonopy.phonon.dos import plot_projected_dos, plot_total_dos

try:
    import yaml
except ImportError:
    print("You need to install python-yaml.")
    sys.exit(1)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


from numpy.typing import NDArray


def _get_label_for_latex(label: str) -> str:
    return label.replace("_", r"\_")


def _get_max_frequency(frequencies: list[NDArray]) -> float:
    return max([np.max(fq) for fq in frequencies])


def _find_wrong_path_connections(all_path_connections: list[list[bool]]) -> int:
    for i, path_connections in enumerate(all_path_connections):
        if path_connections != all_path_connections[0]:
            return i
    return 0


def _arrange_band_data(
    distances: NDArray,
    frequencies: NDArray,
    qpoints: NDArray,
    segment_nqpoints: NDArray,
    label_pairs,
) -> tuple[list[str] | None, list[bool], list[NDArray], list[NDArray]]:
    i = 0
    freq_list = []
    dist_list = []
    qpt_list = []
    for nq in segment_nqpoints:
        freq_list.append(frequencies[i : (i + nq)])
        dist_list.append(distances[i : (i + nq)])
        qpt_list.append(qpoints[i : (i + nq)])
        i += nq

    if not label_pairs:
        labels = None
        path_connections = []
        if len(qpt_list) > 1:
            for i, qpts in enumerate(qpt_list[1:]):
                if (np.abs(qpt_list[i][-1] - qpts[0]) < 1e-5).all():
                    path_connections.append(True)
                else:
                    path_connections.append(False)
        path_connections += [
            False,
        ]
    else:
        labels = []
        path_connections = []
        if len(label_pairs) > 1:
            for i, pair in enumerate(label_pairs[1:]):
                labels.append(label_pairs[i][0])
                if label_pairs[i][1] != pair[0]:
                    labels.append(label_pairs[i][1])
                    path_connections.append(False)
                else:
                    path_connections.append(True)
            if label_pairs[-2][1] != label_pairs[-1][1]:
                labels += label_pairs[-1]
            else:
                labels.append(label_pairs[-1][1])
        else:
            labels += label_pairs[0]
        path_connections += [
            False,
        ]

    return labels, path_connections, freq_list, dist_list


def _savefig(plt, file, fonttype=42, family="serif"):
    plt.rcParams["pdf.fonttype"] = fonttype
    plt.rcParams["font.family"] = family
    plt.savefig(file)


def _get_dos(d: NDArray, f: NDArray, dmax: float) -> NDArray:
    """Cut DOS at dmax and dmin.

    Assume f is ordered.

    """
    pdos = []
    for f1, f2, d1, d2 in zip(f[:-1], f[1:], d[:-1], d[1:], strict=True):
        pdos += _cut_dos((f1, d1), (f2, d2), dmax)
    return np.transpose(pdos)


def _cut_dos(p1: tuple[float, float], p2: tuple[float, float], dmax: float):
    """Cut DOS at dmax.

    Parameters
    ----------
    p1 : tuple[float, float]
        (f1, d1)
    p2 : tuple[float, float]
        (f2, d2)

    When d1 < dmax and d2 < dmax
        p1: (f1, d1)
        p2: (f2, d2)
    When d1 < dmax and d2 > dmax
        p1: (f1, d1)
        pi: (fi, dmax), f1 < fi < f2
        p2: (f2, dmax)
    When d1 > dmax and d2 < dmax,
        p1: (f1, dmax)
        pi: (fi, dmax), f1 < fi < f2
        p2: (f2, d2)
    When d2 > dmax and d1 > dmax
        p1: (f1, dmax)
        p2: (f2, dmax)

    """

    def _get_fi(p1, p2, dmax):
        df = p2[0] - p1[0]
        dd = p2[1] - p1[1]
        fi = (dmax - p1[1]) / dd * df + p1[0]
        return fi

    if p1[1] < dmax and p2[1] < dmax:
        return [p1, p2]
    elif p1[1] < dmax and p2[1] > dmax:
        fi = _get_fi(p1, p2, dmax)
        return [p1, [fi, dmax], [p2[0], dmax]]
    elif p1[1] > dmax and p2[1] < dmax:
        fi = _get_fi(p1, p2, dmax)
        return [[p1[0], dmax], [fi, dmax], p2]
    else:
        return [[p1[0], dmax], [p1[0], dmax]]


def _read_band_yaml(
    filename: str | os.PathLike,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[list[str]]]:
    _, ext = os.path.splitext(filename)
    if ext == ".xz" or ext == ".lzma":
        with lzma.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    elif ext == ".gz":
        import gzip

        with gzip.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    else:
        with open(filename, "r") as f:
            data = yaml.load(f, Loader=Loader)

    frequencies = []
    distances = []
    qpoints = []
    labels = []
    for v in data["phonon"]:
        frequencies.append([f["frequency"] for f in v["band"]])
        qpoints.append(v["q-position"])
        distances.append(v["distance"])

    if "labels" in data:
        labels = data["labels"]

    return (
        np.array(distances),
        np.array(frequencies),
        np.array(qpoints),
        np.array(data["segment_nqpoint"]),
        labels,
    )


def _read_band_hdf5(
    filename: str | os.PathLike,
) -> tuple[NDArray, NDArray, NDArray, NDArray, list[list[str]]]:
    with h5py.File(filename, "r") as data:
        f2: NDArray = data["frequency"][:]  # type: ignore
        d1: NDArray = data["distance"][:]  # type: ignore
        # lbl = data['label'][:]

        labels_path = []
        for x in data["label"][:]:  # type: ignore
            labels_path.append([y.decode("utf-8") for y in x])

        frequencies = f2.reshape((f2.shape[0] * f2.shape[1], f2.shape[2]))
        distances = []
        for x in d1:
            for y in x:
                distances.append(y)
        qpoints = [i for j in data["path"][:] for i in j]  # type: ignore

        seg_pt = data["segment_nqpoint"][:]  # type: ignore

        # nqpt = data['nqpoint'][0]

    return (
        np.array(distances),
        np.array(frequencies),
        np.array(qpoints),
        np.array(seg_pt),
        labels_path,
    )


def _read_dos_dat(
    filename: str | os.PathLike,
    pdos_indices: str | None = None,
    dos_factor: float | None = None,
) -> tuple[NDArray, NDArray]:
    """Read DOS data.

    When pdos_indices is given, sum up projected DOSs of specified indices.

    Total DOS is appended at the last column.

    """
    dos_from_file = []
    frequencies_from_file = []
    for line in open(filename):
        if line.strip()[0] == "#":
            continue
        ary = [float(x) for x in line.split()]
        frequencies_from_file.append(ary.pop(0))
        dos_from_file.append(ary)

    dos = np.array(dos_from_file)
    frequencies = np.array(frequencies_from_file)

    if pdos_indices is None:
        pi = [[i] for i in range(dos.shape[1])]
    else:
        pi = []
        for nums in pdos_indices.split(","):
            pi.append([int(x) - 1 for x in nums.split()])

    dos_sum = []
    for indices in pi:
        dos_sum.append(dos[:, indices].sum(axis=1))
    dos_sum.append(dos.sum(axis=1))
    dos = np.transpose(dos_sum)

    if dos_factor:
        dos *= dos_factor

    ind = np.argsort(frequencies)

    return frequencies[ind], dos[ind]


def get_options():
    """Parse options."""
    parser = argparse.ArgumentParser(description="Phonopy bandplot command-line-tool")
    default_vals = PhonopyBandplotMockArgs()
    parser.add_argument(
        "--hdf5",
        dest="is_hdf5",
        action="store_true",
        default=default_vals.is_hdf5,
        help="Read HDF5 format",
    )
    parser.add_argument(
        "--dmax",
        dest="dos_max",
        type=float,
        default=default_vals.dos_max,
        help="Maximum DOS plotted",
    )
    parser.add_argument(
        "--dos",
        dest="dos_filename",
        default=default_vals.dos_filename,
        help="Filename of dos.dat file to plot alongside the band structure",
    )
    parser.add_argument(
        "--dos-factor",
        dest="dos_factor",
        type=float,
        default=default_vals.dos_factor,
        help="Factor to be multiplied with DOS",
    )
    parser.add_argument(
        "--dos-xlabel",
        dest="dos_xlabel",
        default=default_vals.dos_xlabel,
        help="Specify x-label of DOS",
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        type=float,
        default=default_vals.factor,
        help="Conversion factor to favorite frequency unit",
    )
    parser.add_argument(
        "--fmax",
        dest="f_max",
        type=float,
        default=default_vals.f_max,
        help="Maximum frequency plotted",
    )
    parser.add_argument(
        "--fmin",
        dest="f_min",
        type=float,
        default=default_vals.f_min,
        help="Minimum frequency plotted",
    )
    parser.add_argument(
        "--gnuplot",
        dest="is_gnuplot",
        action="store_true",
        default=default_vals.is_gnuplot,
        help="Output in gnuplot data style",
    )
    parser.add_argument(
        "-i",
        "--indices",
        dest="pdos_indices",
        default=default_vals.pdos_indices,
        help="Indices like 1 2, 3 4 5 6...",
    )
    parser.add_argument(
        "--legend",
        dest="show_legend",
        action="store_true",
        default=default_vals.show_legend,
        help="Show legend",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        default=default_vals.output_filename,
        help="Output filename of PDF plot",
    )
    parser.add_argument(
        "--ylabel",
        dest="ylabel",
        default=default_vals.ylabel,
        help="Specify y-label of band structure",
    )
    parser.add_argument("-t", "--title", dest="title", help="Title of plot")
    parser.add_argument(
        "filenames",
        nargs="*",
        help=(
            "Filenames of phonon band structure result: band.yaml "
            "or band.hdf5(with --hdf5)"
        ),
    )
    args = parser.parse_args()
    return args


def _plot(args: argparse.Namespace | PhonopyBandplotMockArgs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    if args.is_hdf5:
        if len(args.filenames) == 0:
            filenames = [
                "band.hdf5",
            ]
        else:
            filenames = args.filenames

        bands_data = [_read_band_hdf5(fname) for fname in filenames]
    else:
        if len(args.filenames) == 0:
            filenames = [
                "band.yaml",
            ]
        else:
            filenames = args.filenames
        bands_data = [_read_band_yaml(fname) for fname in filenames]

    plots_data = [_arrange_band_data(*band_data) for band_data in bands_data]
    # plots_data = [[labels, path_connections, freq_list, dist_list], ...]
    # Check consistency of input band structures
    all_path_connections = [data[1] for data in plots_data]
    wrong_file_i = _find_wrong_path_connections(all_path_connections)
    if wrong_file_i > 0:
        raise RuntimeError(
            "Band path of %s is inconsistent with %s."
            % (filenames[wrong_file_i], filenames[0])
        )

    # Decoration of figure
    max_frequencies = [_get_max_frequency(data[2]) for data in plots_data]
    plot_data = plots_data[np.argmax(max_frequencies)]
    _, path_connections, _, _ = plot_data
    n = len([x for x in path_connections if not x])

    if args.dos_filename:
        n += 1

    fig = plt.figure()
    axs = list(
        ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(1, n),
            axes_pad=0.11,
            label_mode="L",
        )  # type: ignore
    )

    if args.dos_filename:
        band_plot = BandPlot(axs[:-1])
    else:
        band_plot = BandPlot(axs)
    band_plot.set_xscale_from_data(plot_data[2], plot_data[3])
    band_plot.xscale = band_plot.xscale * args.factor
    band_plot.decorate(*plot_data, ylabel=args.ylabel)

    # Plot band structures
    fmts = [
        "r-",
        "b-",
        "g-",
        "c-",
        "m-",
        "y-",
        "k-",
        "r--",
        "b--",
        "g--",
        "c--",
        "m--",
        "y--",
        "k--",
    ]
    for i, label in enumerate(filenames):
        _, p, f, d = plots_data[i]
        fmt = fmts[i % len(fmts)]
        _f = [f_seg * args.factor for f_seg in f]
        if args.show_legend:
            band_plot.plot(d, _f, p, fmt=fmt, label=_get_label_for_latex(str(label)))
        else:
            band_plot.plot(d, _f, p, fmt=fmt)

    # dos
    if args.dos_filename:
        _plot_dos(args, axs, max_frequencies)

    for ax in axs:
        ax.set_ylim(args.f_min, args.f_max)

    # Bring legend in front.
    if args.show_legend:
        axs[0].set_zorder(1)

    if args.title is not None:
        plt.suptitle(args.title)

    if args.output_filename is not None:
        _savefig(plt, args.output_filename)
    else:
        plt.show()


def _plot_dos(
    args: argparse.Namespace | PhonopyBandplotMockArgs,
    axs,
    max_frequencies: list[float],
):
    assert args.dos_filename is not None
    dos_frequencies, dos = _read_dos_dat(
        args.dos_filename,
        pdos_indices=args.pdos_indices,
        dos_factor=args.dos_factor,
    )
    arg_fmax = None
    if args.f_max is None:
        max_freq = max(max_frequencies) * 1.01
    else:
        max_freq = args.f_max
    for i, f in enumerate(dos_frequencies):
        if f > max_freq:
            arg_fmax = i
            break

    arg_fmin = None
    if args.f_min is not None:
        for i, f in enumerate(dos_frequencies):
            if f > args.f_min:
                if i > 0:
                    arg_fmin = i - 1
                break

    axs[-1].xaxis.set_ticks_position("both")
    axs[-1].yaxis.set_ticks_position("both")
    axs[-1].xaxis.set_tick_params(which="both", direction="in")
    axs[-1].yaxis.set_tick_params(which="both", direction="in")

    pdos_plot_data = []

    for pdos in dos.T:
        if args.dos_max is not None:
            _pdos = _get_dos(
                pdos[arg_fmin:arg_fmax],
                dos_frequencies[arg_fmin:arg_fmax],
                args.dos_max,
            )
            pdos_plot_data.append(_pdos[1])
            freqs_plot_data = _pdos[0]
        else:
            pdos_plot_data.append(pdos[arg_fmin:arg_fmax])
            freqs_plot_data = dos_frequencies[arg_fmin:arg_fmax]

    plot_projected_dos(
        axs[-1],
        freqs_plot_data,
        pdos_plot_data[:-1],
        draw_grid=False,
        flip_xy=True,
        xlabel=args.dos_xlabel,
    )

    if len(pdos_plot_data) > 1:
        plot_total_dos(
            axs[-1],
            freqs_plot_data,
            pdos_plot_data[-1],
            draw_grid=False,
            flip_xy=True,
            linestyle="dotted",
            color="black",
            linewidth=0.5,
        )

    axs[-1].set_xlim(left=0, right=max([np.max(p) * 1.1 for p in pdos_plot_data[:-1]]))
    xlim = axs[-1].get_xlim()
    ylim = axs[-1].get_ylim()
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
    axs[-1].set_aspect(aspect)


def _write_gnuplot_data(args: argparse.Namespace | PhonopyBandplotMockArgs):
    if args.is_hdf5:
        if len(args.filenames) == 0:
            filenames = [
                "band.hdf5",
            ]
        else:
            filenames = args.filenames

        bands_data = [_read_band_hdf5(fname) for fname in filenames]
    else:
        if len(args.filenames) == 0:
            filenames = [
                "band.yaml",
            ]
        else:
            filenames = args.filenames
        bands_data = [_read_band_yaml(fname) for fname in filenames]

    if args.is_gnuplot:
        distances = bands_data[0][0]
        frequencies = bands_data[0][1]
        segment_nqpoint = bands_data[0][3]

        end_points = [
            0,
        ]
        for nq in segment_nqpoint:
            end_points.append(nq + end_points[-1])
        end_points[-1] -= 1
        segment_positions = distances[end_points]

        print("# End points of segments: ")
        print("#   " + "%10.8f " * len(segment_positions) % tuple(segment_positions))
        for freqs in frequencies.T:
            q = 0
            for nq in segment_nqpoint:
                for d, f in zip(
                    distances[q : (q + nq)],
                    freqs[q : (q + nq)] * args.factor,
                    strict=True,
                ):
                    print("%f %f" % (d, f))
                q += nq
                print("")
            print("")


@dataclasses.dataclass
class PhonopyBandplotMockArgs:
    """Mock args of ArgumentParser."""

    is_hdf5: bool | None = None
    band_labels: str | None = None
    dos_max: float | None = None
    dos_factor: float | None = None
    dos_xlabel: str | None = None
    factor: float = 1.0
    f_max: float | None = None
    f_min: float | None = None
    is_gnuplot: bool = False
    is_points: bool = False
    output_filename: str | os.PathLike | None = None
    pdos_indices: str | None = None
    ylabel: str | None = None
    show_legend: bool = False
    title: str | None = None
    dos_filename: str | os.PathLike | None = None

    filenames: Sequence[str | os.PathLike] = ()

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in dataclasses.fields(self))


def main(**argparse_control: PhonopyBandplotMockArgs):
    """Run phonopy-bandplot."""
    if argparse_control:
        args = argparse_control["args"]
    else:
        args = get_options()

    if args.is_gnuplot:
        _write_gnuplot_data(args)
        sys.exit(1)

    if args.output_filename:
        import matplotlib

        matplotlib.use("Agg")

    _plot(args)
