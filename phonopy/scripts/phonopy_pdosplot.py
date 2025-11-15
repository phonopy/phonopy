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

# PDOS plot (pdosplot)
#
# Usage:
#   pdosplot -i "1 2, 4 5" -o "pdos.pdf"
#
# The axis resolved PDOS is summed up with the successive
# indices separated by ",". In this example, indices 1 and
# 2, 3 and 4 are summed respectively, and then they are
# ploted respectively.
#
# The indices are defined like:
# 1 2 3 : X Y Z of the 1st atom,
# 4 5 6 : X Y Z of the 2nd atom,
# ...

import matplotlib.pyplot as plt
import numpy as np


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy pdosplot command-line-tool")
    parser.set_defaults(
        output_filename=None,
        factor=1.0,
        legend_labels=None,
        xlabel=None,
        ylabel=None,
        show_legend=False,
        pdos_indices=None,
        ymax=None,
        ymin=None,
        title=None,
        f_max=None,
        f_min=None,
    )
    parser.add_argument(
        "--factor", dest="factor", type=float, help="Factor is multiplied with DOS."
    )
    parser.add_argument(
        "-l", "--legend", dest="show_legend", action="store_true", help="Show legend"
    )
    parser.add_argument(
        "--legend_labels", dest="legend_labels", help="Set legend labels"
    )
    parser.add_argument("--xlabel", dest="xlabel", help="Set x label")
    parser.add_argument("--ylabel", dest="ylabel", help="Set y label")
    parser.add_argument(
        "-i", "--indices", dest="pdos_indices", help="Indices like 1 2, 3 4 5 6..."
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", help="Output filename"
    )
    parser.add_argument("-t", "--title", dest="title", help="Title of plot")
    parser.add_argument(
        "--ymax", dest="ymax", type=float, help="Maximum value of y axis"
    )
    parser.add_argument(
        "--ymin", dest="ymin", type=float, help="Minimum value of y axis"
    )
    parser.add_argument(
        "--fmax", dest="f_max", type=float, help="Maximum frequency plotted"
    )
    parser.add_argument(
        "--fmin", dest="f_min", type=float, help="Minimum frequency plotted"
    )
    parser.add_argument(
        "filename", nargs="*", help="Filename of phonon DOS result (partial_dos.dat)"
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-pdosplot."""
    args = get_options()

    frequencies = []
    dos = []
    filename = "partial_dos.dat"
    if len(args.filename) > 0:
        filename = args.filename[0]
    for line in open(filename):
        if line.strip().split()[0] == "#" or line.strip().split() == "":
            continue

        tmp_array = [float(x) for x in line.split()]
        frequencies.append(tmp_array.pop(0))
        dos.append(tmp_array)

    frequencies = np.array(frequencies)
    dos = np.array(dos).transpose()

    # Extract indices
    indices = []
    if args.pdos_indices is None:
        indices.append(range(1, dos.shape[0] + 1))
    else:
        for v in args.pdos_indices.split(","):
            indices.append([int(x) for x in v.split()])

    # Set plot range in frequency axis
    if args.f_max is None:
        max_freq = max(frequencies)
    else:
        max_freq = args.f_max
    if args.f_min is None:
        min_freq = min(frequencies)
    else:
        min_freq = args.f_min

    min_i = 0
    max_i = len(frequencies)

    for i, f in enumerate(frequencies):
        if f > max_freq + (frequencies[1] - frequencies[0]) / 10:
            max_i = i + 1
            break

    for i, f in enumerate(frequencies):
        if f > min_freq - (frequencies[1] - frequencies[0]) / 10:
            min_i = i
            break

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    plots = []
    for nums in indices:
        pdos = np.zeros(frequencies.shape[0], dtype="double")
        for v in nums:
            pdos += dos[v - 1]
        (curve,) = plt.plot(frequencies[min_i:max_i], pdos[min_i:max_i] * args.factor)
        plots.append(curve)

    # plt.grid(True)

    ax.set_ylim((0, None))
    plt.xlim(min_freq, max_freq)

    if (args.ymin is not None) and (args.ymax is not None):
        plt.ylim(args.ymin, args.ymax)
    elif (args.ymin is None) and (args.ymax is not None):
        plt.ylim(ymax=args.ymax)
    elif (args.ymin is not None) and (args.ymax is None):
        plt.ylim(ymin=args.ymin)

    if args.xlabel is None:
        plt.xlabel("Frequency")
    else:
        plt.xlabel(args.xlabel)
    if args.ylabel is None:
        plt.ylabel("Partial density of states")
    else:
        plt.ylabel(args.ylabel)

    if args.show_legend:
        if args.legend_labels is not None:
            if len(args.legend_labels.split()) == len(plots):
                labels = args.legend_labels.split()
            else:
                print("Number of labels is not same as number of plots.")
                labels = indices
        else:
            labels = indices
        plt.legend(plots, labels, loc="upper left")

    if args.title is not None:
        plt.title(args.title)

    if args.output_filename is not None:
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["font.family"] = "serif"
        plt.savefig(args.output_filename)
    else:
        plt.show()
