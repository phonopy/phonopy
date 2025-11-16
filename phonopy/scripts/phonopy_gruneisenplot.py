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

import sys

import numpy as np

try:
    import yaml
except ImportError:
    print("You need to install python-yaml.")
    sys.exit(1)

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def print_band(data):
    """Print band structure in gnuplot style."""
    distance = 0
    text = "#%-9s %-10s %-10s    " % ("    q_a", "    q_b", "    q_c")
    text += "%-12s    " % "   Distance"
    text += "%-12s %-12s     ..." % ("   Gruneisen", "   Frequency")
    lines = [
        text,
    ]

    for path in data["path"]:
        for q in path["phonon"]:
            text = "%10.7f %10.7f %10.7f    " % (tuple(q["q-position"]))
            text += "%12.7f    " % (q["distance"] + distance)
            for band in q["band"]:
                text += "%12.7f %12.7f  " % (band["gruneisen"], band["frequency"])
            lines.append(text)
        distance += path["phonon"][-1]["distance"]

    print("\n".join(lines))


def print_mesh(data):
    """Print mesh information."""
    text = "#%-9s %-10s %-10s  " % ("    q_a", "    q_b", "    q_c")
    text += "%-12s  " % "Multiplicity"
    text += "%-12s %-12s     ..." % ("   Gruneisen", "   Frequency")
    lines = [
        text,
    ]

    for qpt in data["phonon"]:
        text = "%10.7f %10.7f %10.7f  " % (tuple(qpt["q-position"]))
        text += "%12d  " % qpt["multiplicity"]
        for band in qpt["band"]:
            text += "%12.7f %12.7f  " % (band["gruneisen"], band["frequency"])
        lines.append(text)

    print("\n".join(lines))


def plot_band(data, is_fg, g_max, g_min):
    """Plot band structure."""
    import matplotlib.pyplot as plt

    d = []
    g = []
    f = []
    distance = 0.0

    for path in data["path"]:
        for q in path["phonon"]:
            d.append(q["distance"] + distance)
            g.append([band["gruneisen"] for band in q["band"]])
            f.append([band["frequency"] for band in q["band"]])
        distance += path["phonon"][-1]["distance"]

    if is_fg:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(d, g, "-")
        ax2.plot(d, np.array(g) * np.array(f), "-")
        ax3.plot(d, f, "-")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(d, g, "-")
        ax2.plot(d, f, "-")

    if g_max is not None:
        ax1.set_ylim(ymax=g_max)
    if g_min is not None:
        ax1.set_ylim(ymin=g_min)

    return plt


def plot_yaml_mesh(data, is_fg, cutoff_frequency=None):
    """Plot band structure using data from yaml."""
    x = []
    y = []
    for qpt in data["phonon"]:
        x.append([band["frequency"] for band in qpt["band"]])
        y.append([band["gruneisen"] for band in qpt["band"]])

    return plot_mesh(x, y, is_fg, cutoff_frequency=cutoff_frequency)


def plot_hdf5_mesh(data, is_fg, cutoff_frequency=None):
    """Plot band structure using data from hdf5."""
    x = data["frequency"]
    y = data["gruneisen"][:]
    return plot_mesh(x, y, is_fg, cutoff_frequency=cutoff_frequency)


def plot_mesh(x, y, is_fg, cutoff_frequency=None):
    """Plot mesh data."""
    import matplotlib.pyplot as plt

    for g, freqs in zip(np.transpose(y), np.transpose(x)):
        if cutoff_frequency:
            g = np.extract(freqs > cutoff_frequency, g)
            freqs = np.extract(freqs > cutoff_frequency, freqs)

        if is_fg:
            plt.plot(freqs, np.array(freqs) * np.array(g), "o")
        else:
            plt.plot(freqs, g, "o")

    return plt


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phonopy gruneisenplot command-line-tool"
    )
    parser.set_defaults(
        f_max=None,
        f_min=None,
        g_max=None,
        g_min=None,
        is_hdf5=False,
        is_fg=False,
        is_gnuplot=False,
        cutoff_frequency=None,
        output_filename=None,
        save_graph=None,
        title=None,
    )
    parser.add_argument(
        "--gnuplot",
        dest="is_gnuplot",
        action="store_true",
        help="Output in gnuplot data style",
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", help="Output filename of PDF plot"
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save_graph",
        action="store_true",
        help="Save plot data in pdf",
    )
    parser.add_argument(
        "--fg", dest="is_fg", action="store_true", help="Plot omega x gamma"
    )
    parser.add_argument("-t", "--title", dest="title", help="Title of plot")
    parser.add_argument(
        "--cutoff",
        dest="cutoff_frequency",
        type=float,
        help="Plot above this cutoff frequency for mesh sampling mode.",
    )
    parser.add_argument(
        "--fmax", dest="f_max", type=float, help="Maximum frequency plotted"
    )
    parser.add_argument(
        "--fmin", dest="f_min", type=float, help="Minimum frequency plotted"
    )
    parser.add_argument(
        "--gmax", dest="g_max", type=float, help="Maximum Gruneisen params plotted"
    )
    parser.add_argument(
        "--gmin", dest="g_min", type=float, help="Minimum Gruneisen params plotted"
    )
    parser.add_argument(
        "--hdf5", dest="is_hdf5", action="store_true", help="Use hdf5 to read results"
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help="Filename of phonopy-gruneisen result (gruneisen.yaml)",
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-gruneisenplot."""
    args = get_options()

    if args.is_hdf5:
        try:
            import h5py
        except ImportError:
            print("You need to install python-h5py.")
            sys.exit(1)

        if len(args.filename) == 0:
            filename = "gruneisen.hdf5"
        else:
            filename = args.filename[0]
        with h5py.File(filename, "r") as f:
            data = {key: f[key][:] for key in list(f)}
    else:
        if len(args.filename) == 0:
            filename = "gruneisen.yaml"
        else:
            filename = args.filename[0]
        with open(filename) as f:
            data = yaml.load(f.read(), Loader=Loader)

    if args.save_graph:
        import matplotlib as mpl

        mpl.use("Agg")

    if args.output_filename:
        pdffile = args.output_filename
    else:
        pdffile = "gruneisenplot.pdf"

    if "path" in data:
        if args.is_gnuplot:
            print_band(data)
        else:
            plt = plot_band(data, args.is_fg, args.g_max, args.g_min)
            if args.title is not None:
                plt.subtitle(args.title)
            if args.save_graph:
                plt.savefig(pdffile)
            else:
                plt.show()

    if "mesh" in data:
        if args.is_hdf5:
            plt = plot_hdf5_mesh(
                data, args.is_fg, cutoff_frequency=args.cutoff_frequency
            )
        else:
            if args.is_gnuplot:
                print_mesh(data)
            else:
                plt = plot_yaml_mesh(
                    data, args.is_fg, cutoff_frequency=args.cutoff_frequency
                )

        if args.title is not None:
            plt.subtitle(args.title)

        if args.f_max is not None:
            plt.xlim(xmax=args.f_max)
        if args.f_min is not None:
            plt.xlim(xmin=args.f_min)
        if args.g_max is not None:
            plt.ylim(ymax=args.g_max)
        if args.g_min is not None:
            plt.ylim(ymin=args.g_min)

        if args.save_graph:
            plt.savefig(pdffile)
        else:
            plt.show()
