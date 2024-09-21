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

import lzma
import os
import sys

import h5py
import numpy as np

from phonopy.phonon.band_structure import BandPlot

try:
    import yaml
except ImportError:
    print("You need to install python-yaml.")
    sys.exit(1)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def _get_label_for_latex(label):
    return label.replace("_", r"\_")


def _get_max_frequency(frequencies):
    return max([np.max(fq) for fq in frequencies])


def _find_wrong_path_connections(all_path_connections):
    for i, path_connections in enumerate(all_path_connections):
        if path_connections != all_path_connections[0]:
            return i
    return 0


def _arrange_band_data(distances, frequencies, qpoints, segment_nqpoints, label_pairs):
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


def _get_dos(d, f, dmax):
    pdos = []
    for f1, f2, d1, d2 in zip(f[:-1], f[1:], d[:-1], d[1:]):
        pdos += _cut_dos([f1, d1], [f2, d2], dmax)
    return np.transpose(pdos)


def _cut_dos(p1: list[float, float], p2: list[float, float], dmax: float):
    """Cut DOS at dmax.

    ** Cutting by fmin and fmax should be implemented someday. **

    Parameters
    ----------
    p1 : list[float, float]
        (f1, d1)
    p2 : list[float, float]
        (f2, d2)

    When d1 < dmax and d2 < dmax
        p1: (f1, d1)
        p2: (f2, d2)
    When d1 < dmax and d2 > dmax
        p1: (f1, d1)
        pi: (fi, dmax), f1 < fi < f2
        p2: (f2, dmax)
    Whend1 > dmax and d2 < dmax,
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


def _read_band_yaml(filename):
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
        if "label" in v:
            labels.append(v["label"])
        else:
            labels.append(None)
        frequencies.append([f["frequency"] for f in v["band"]])
        qpoints.append(v["q-position"])
        distances.append(v["distance"])

    if "labels" in data:
        labels = data["labels"]
    elif all(x is None for x in labels):
        labels = []

    return (
        np.array(distances),
        np.array(frequencies),
        np.array(qpoints),
        data["segment_nqpoint"],
        labels,
    )


def _read_band_hdf5(filename):
    with h5py.File(filename, "r") as data:
        f2 = data["frequency"][:]
        d1 = data["distance"][:]
        # lbl = data['label'][:]

        labels_path = []
        for x in data["label"][:]:
            labels_path.append([y.decode("utf-8") for y in x])

        frequencies = f2.reshape((f2.shape[0] * f2.shape[1], f2.shape[2]))
        distances = []
        for x in d1:
            for y in x:
                distances.append(y)
        qpoints = [i for j in data["path"][:] for i in j]

        seg_pt = data["segment_nqpoint"][:]

        # nqpt = data['nqpoint'][0]

    return (
        np.array(distances),
        np.array(frequencies),
        np.array(qpoints),
        seg_pt,
        labels_path,
    )


def _read_dos_dat(filename, pdos_indices=None, dos_factor=None):
    dos = []
    frequencies = []
    for line in open(filename):
        if line.strip()[0] == "#":
            continue
        ary = [float(x) for x in line.split()]
        frequencies.append(ary.pop(0))
        dos.append(ary)
    dos = np.array(dos)
    frequencies = np.array(frequencies)

    if pdos_indices:
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

    return frequencies, dos


def get_options():
    """Parse options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy bandplot command-line-tool")
    parser.set_defaults(
        hdf5=None,
        band_labels=None,
        dos=None,
        dos_max=None,
        dos_min=None,
        dos_factor=None,
        factor=1.0,
        f_max=None,
        f_min=None,
        is_gnuplot=False,
        is_legacy_plot=False,
        is_points=False,
        is_vertical_line=False,
        output_filename=None,
        pdos_indices=None,
        xlabel=None,
        ylabel=None,
        show_legend=False,
        title=None,
    )
    parser.add_argument(
        "--hdf5", dest="is_hdf5", action="store_true", help="Read HDF5 format"
    )
    parser.add_argument(
        "--dmax",
        dest="dos_max",
        type=float,
        help="Maximum DOS plotted (legacy plot only)",
    )
    parser.add_argument(
        "--dmin",
        dest="dos_min",
        type=float,
        help="Minimum DOS plotted (legacy plot only)",
    )
    parser.add_argument(
        "--dos", dest="dos", help="Read dos.dat type file and plot with band structure"
    )
    parser.add_argument(
        "--dos-factor",
        dest="dos_factor",
        type=float,
        help="Factor to be multiplied with DOS (legacy plot only)",
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        type=float,
        help="Conversion factor to favorite frequency unit",
    )
    parser.add_argument(
        "--fmax", dest="f_max", type=float, help="Maximum frequency plotted"
    )
    parser.add_argument(
        "--fmin", dest="f_min", type=float, help="Minimum frequency plotted"
    )
    parser.add_argument(
        "--gnuplot",
        dest="is_gnuplot",
        action="store_true",
        help="Output in gnuplot data style",
    )
    parser.add_argument(
        "-i",
        "--indices",
        dest="pdos_indices",
        help="Indices like 1 2, 3 4 5 6... (legacy plot only)",
    )
    parser.add_argument(
        "--legend", dest="show_legend", action="store_true", help="Show legend"
    )
    parser.add_argument(
        "--legacy",
        dest="is_legacy_plot",
        action="store_true",
        help="Plot in legacy style",
    )
    parser.add_argument(
        "--line",
        "-l",
        dest="is_vertical_line",
        action="store_true",
        help="Vertical line is drawn at between paths (legacy plot only)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_filename",
        action="store",
        help="Output filename of PDF plot",
    )
    parser.add_argument(
        "--xlabel", dest="xlabel", help="Specify x-label (legacy plot only)"
    )
    parser.add_argument(
        "--ylabel", dest="ylabel", help="Specify y-label (legacy plot only)"
    )
    parser.add_argument(
        "--points",
        dest="points",
        help="Draw points (o, '*', v, ^, x, p, d etc) (legacy plot only)",
    )
    parser.add_argument(
        "-t", "--title", dest="title", help="Title of plot (legacy plot only)"
    )
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


def _old_plot(args):
    import matplotlib.pyplot as plt

    if args.dos:
        import matplotlib.gridspec as gridspec

        plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0, 0])
        ax1.xaxis.set_ticks_position("both")
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_tick_params(which="both", direction="in")
        ax1.yaxis.set_tick_params(which="both", direction="in")
    else:
        fig, ax = plt.subplots()
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in")
        ax.yaxis.set_tick_params(which="both", direction="in")

    colors = [
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

    if args.points is None:
        marker = None
    else:
        marker = args.points

    if args.is_hdf5:
        if len(args.filenames) == 0:
            filenames = [
                "band.hdf5",
            ]
        else:
            filenames = args.filenames

    else:
        if len(args.filenames) == 0:
            filenames = ["band.yaml"]
        else:
            filenames = args.filenames

    if args.dos:
        dos_frequencies, dos = _read_dos_dat(
            args.dos, pdos_indices=args.pdos_indices, dos_factor=args.dos_factor
        )

    curves = []
    for i, filename in enumerate(filenames):
        if args.is_hdf5:
            (
                distances,
                frequencies,
                qpoints,
                segment_nqpoint,
                labels,
            ) = _read_band_hdf5(filename)
        else:
            (
                distances,
                frequencies,
                qpoints,
                segment_nqpoint,
                labels,
            ) = _read_band_yaml(filename)

        end_points = [
            0,
        ]
        for nq in segment_nqpoint:
            end_points.append(nq + end_points[-1])
        end_points[-1] -= 1
        segment_positions = distances[end_points]

        if not labels:
            labels_at_ends = None
        elif isinstance(labels[0], list):
            labels_at_ends = [
                labels[0][0],
            ]
            for j, pair in enumerate(labels[1:]):
                if labels[j][1] != pair[0]:
                    labels_at_ends.append("|".join([labels[j][1], pair[0]]))
                else:
                    labels_at_ends.append(pair[0])
            labels_at_ends.append(labels[-1][1])
        else:
            labels_at_ends = [labels[n] for n in end_points]

        if args.is_vertical_line and len(filenames) == 1:
            for v in segment_positions[1:-1]:
                plt.axvline(x=v, linewidth=0.5, color="b")

        q = 0
        for j, nq in enumerate(segment_nqpoint):
            if j == 0:
                curves.append(
                    plt.plot(
                        distances[q : (q + nq)],
                        frequencies[q : (q + nq)] * args.factor,
                        color=colors[i % len(colors)][0],
                        linestyle=colors[i % len(colors)][1:],
                        marker=marker,
                        label=filename,
                    )[0]
                )
            else:
                plt.plot(
                    distances[q : (q + nq)],
                    frequencies[q : (q + nq)] * args.factor,
                    color=colors[i % len(colors)][0],
                    marker=marker,
                    linestyle=colors[i % len(colors)][1:],
                )
            q += nq

    if args.xlabel is None:
        plt.xlabel("Wave vector")
    else:
        plt.xlabel(args.xlabel)
    if args.ylabel is None:
        plt.ylabel("Frequency")
    else:
        plt.ylabel(args.ylabel)

    plt.xlim(distances[0], distances[-1])
    if args.f_max is not None:
        plt.ylim(top=args.f_max)
    if args.f_min is not None:
        plt.ylim(bottom=args.f_min)
    plt.axhline(y=0, linestyle=":", linewidth=0.5, color="b")
    if len(filenames) == 1:
        xticks = segment_positions
        if args.band_labels:
            band_labels = [x for x in args.band_labels.split()]
            if len(band_labels) == len(xticks):
                plt.xticks(xticks, band_labels)
            else:
                print("Numbers of labels and band segments don't match.")
                sys.exit(1)
        elif labels_at_ends:
            plt.xticks(xticks, labels_at_ends)
        else:
            plt.xticks(xticks, [""] * len(xticks))
    else:
        plt.xticks([])

    if args.title is not None:
        plt.title(args.title)

    if args.show_legend:
        plt.legend(handles=curves)

    if args.dos:
        arg_fmax = len(dos_frequencies)
        if args.f_max is not None:
            for i, f in enumerate(dos_frequencies):
                if f > args.f_max:
                    arg_fmax = i
                    break
        arg_fmin = 0
        if args.f_min is not None:
            for i, f in enumerate(dos_frequencies):
                if f > args.f_min:
                    if i > 0:
                        arg_fmin = i - 1
                    break
        ax2 = plt.subplot(gs[0, 1], sharey=ax1)
        ax2.xaxis.set_ticks_position("both")
        ax2.yaxis.set_ticks_position("both")
        ax2.xaxis.set_tick_params(which="both", direction="in")
        ax2.yaxis.set_tick_params(which="both", direction="in")

        plt.subplots_adjust(wspace=0.03)
        plt.setp(ax2.get_yticklabels(), visible=False)

        for pdos in dos.T:
            if args.dos_max:
                _pdos = _get_dos(
                    pdos[arg_fmin:arg_fmax],
                    dos_frequencies[arg_fmin:arg_fmax],
                    args.dos_max,
                )
                plt.plot(_pdos[1], _pdos[0])
            else:
                plt.plot(pdos[arg_fmin:arg_fmax], dos_frequencies[arg_fmin:arg_fmax])
        plt.xlabel("DOS")

        ax2.set_xlim((0, None))

        if args.f_max is not None:
            plt.ylim(top=args.f_max)
        if args.f_min is not None:
            plt.ylim(bottom=args.f_min)
        if args.dos_max is not None:
            plt.xlim(right=args.dos_max)
        if args.dos_min is not None:
            plt.xlim(left=args.dos_min)

    if args.output_filename is not None:
        _savefig(plt, args.output_filename)
    else:
        plt.show()


def _plot(args):
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

    if args.dos:
        dos_frequencies, dos = _read_dos_dat(
            args.dos, pdos_indices=args.pdos_indices, dos_factor=args.dos_factor
        )

    plots_data = [_arrange_band_data(*band_data) for band_data in bands_data]
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

    if args.dos:
        n += 1

    fig = plt.figure()

    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, n),
        axes_pad=0.11,
        label_mode="L",
    )
    for ax in axs[:-1]:
        if args.f_min:
            ax.set_ylim(ymin=args.f_min)
        if args.f_max:
            ax.set_ylim(ymax=args.f_max)

    band_plot = BandPlot(axs)
    band_plot.set_xscale_from_data(plot_data[2], plot_data[3])
    band_plot.xscale = band_plot.xscale * args.factor
    band_plot.decorate(*plot_data)

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
            band_plot.plot(d, _f, p, fmt=fmt, label=_get_label_for_latex(label))
        else:
            band_plot.plot(d, _f, p, fmt=fmt)

    # dos
    if args.dos:
        arg_fmax = len(dos_frequencies)
        if args.f_max is not None:
            for i, f in enumerate(dos_frequencies):
                if f > args.f_max:
                    arg_fmax = i
                    break
        arg_fmin = 0
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

        for pdos in dos.T:
            if args.dos_max:
                _pdos = _get_dos(
                    pdos[arg_fmin:arg_fmax],
                    dos_frequencies[arg_fmin:arg_fmax],
                    args.dos_max,
                )
                axs[-1].plot(_pdos[1], _pdos[0])
            else:
                axs[-1].plot(
                    pdos[arg_fmin:arg_fmax], dos_frequencies[arg_fmin:arg_fmax]
                )
        axs[-1].set_xlabel("DOS")

        xlim = axs[-1].get_xlim()
        ylim = axs[-1].get_ylim()
        aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
        axs[-1].set_aspect(aspect)

    # Bring legend in front.
    if args.show_legend:
        axs[0].set_zorder(1)

    if args.title is not None:
        plt.suptitle(args.title)

    if args.output_filename is not None:
        _savefig(plt, args.output_filename)
    else:
        plt.show()


def _write_gnuplot_data(args):
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
                    distances[q : (q + nq)], freqs[q : (q + nq)] * args.factor
                ):
                    print("%f %f" % (d, f))
                q += nq
                print("")
            print("")


def run():
    """Run phonopy-bandplot."""
    args = get_options()
    if args.is_gnuplot:
        _write_gnuplot_data(args)
        sys.exit(1)

    if args.output_filename:
        import matplotlib

        matplotlib.use("Agg")

    if args.is_legacy_plot:
        _old_plot(args)
    else:
        _plot(args)
