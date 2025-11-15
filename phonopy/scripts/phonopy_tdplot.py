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

# Thermal displacement plot (tdplot)
#
# Usage:
#   tdplot -i "1 2, 4 5" -o "td.pdf"
#
# The axis resolved thermal displacements are averaged by root mean
# square with the successive indices separated by ",". In this
# example, values at a temperature with the indices 1 and 2, 3 and 4
# are averaged by root mean square respectively as follwos:
#   sqrt( ( x_1 ** 2 + x_2 ** 2 + x_3 ** 2 + x_4 ** 2 ) / 4 ),
# and then they are ploted as a function of temperature.
#
# The definition is that indices correspond to those as follows:
# 1 2 3 : X Y Z of the 1st atom,
# 4 5 6 : X Y Z of the 2nd atom,
# ...

import sys

import matplotlib.pyplot as plt
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


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy tdplot command-line-tool")
    parser.set_defaults(
        output_filename=None,
        factor=1.0,
        td_indices=None,
        show_legend=False,
        is_distance=False,
        is_gnuplot=False,
        ymax=None,
        ymin=None,
        tmax=None,
        tmin=None,
    )
    parser.add_argument(
        "-i", "--indices", dest="td_indices", help="Indices like 1 2, 3 4 5 6..."
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        type=float,
        help="Conversion factor of energy unit to internal electronic energy",
    )
    parser.add_argument(
        "--ymax", dest="ymax", type=float, help="Maximum value of y axis"
    )
    parser.add_argument(
        "--ymin", dest="ymin", type=float, help="Minimum value of y axis"
    )
    parser.add_argument(
        "--tmax", dest="tmax", type=float, help="Maximum value of temperature"
    )
    parser.add_argument(
        "--tmin", dest="tmin", type=float, help="Minimum value of temperature"
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", help="Output filename"
    )
    parser.add_argument(
        "-l", "--legend", dest="show_legend", action="store_true", help="Show legend"
    )
    parser.add_argument(
        "--distance",
        dest="is_distance",
        action="store_true",
        help="Plot thermal distance",
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help=(
            "Filename of phonon thermal displacement result "
            "(thermal_displacements.yaml)"
        ),
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-tdplot."""
    args = get_options()

    if not args.is_distance:
        if len(args.filename) == 0:
            filename = "thermal_displacements.yaml"
        else:
            filename = args.filename[0]
        yamldata = yaml.load(open(filename).read(), Loader=Loader)

        td = yamldata["thermal_displacements"]
        temperatures = [v["temperature"] for v in td]
        displacements = [v["displacements"] for v in td]
        displacements = np.array(displacements).reshape(len(temperatures), -1)
    else:
        if len(args.filename) == 0:
            filename = "thermal_distances.yaml"
        else:
            filename = args.filename[0]
        yamldata = yaml.load(open(filename).read(), Loader=Loader)

        td = yamldata["thermal_distances"]
        temperatures = [v["temperature"] for v in td]
        distances = [v["distance"] for v in td]
        for t, dists in zip(temperatures, distances):
            print(("%14.7f" * (len(dists) + 1)) % ((t,) + tuple(dists)))
            print("")
        print("")
        print("")
        sys.exit(0)

    # Set temperature range
    tmin_index = 0
    tmax_index = len(temperatures)
    if args.tmin is not None:
        for i, t in enumerate(temperatures):
            if t > args.tmin - (temperatures[1] - temperatures[0]) * 0.1:
                tmin_index = i
                break

    if args.tmax is not None:
        for i, t in enumerate(temperatures):
            if t > args.tmax + (temperatures[1] - temperatures[0]) * 0.1:
                tmax_index = i
                break

    # Extract indices
    set_of_indices = []
    if args.td_indices is None:
        set_of_indices.append(range(1, displacements.shape[1] + 1))
    else:
        for v in args.td_indices.split(","):
            set_of_indices.append([int(x) for x in v.split()])

    # Plot
    fig, ax = plt.subplots()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")

    plots = []
    for indices in set_of_indices:
        d = np.zeros(len(temperatures), dtype="double")
        for i in indices:
            d += displacements[:, i - 1]
        (curve,) = plt.plot(
            temperatures[tmin_index:tmax_index], d[tmin_index:tmax_index] / len(indices)
        )
        plots.append(curve)

    ax.set_xlim((0, temperatures[tmax_index - 1]))

    # Set y range
    if (args.ymin is not None) and (args.ymax is not None):
        plt.ylim(args.ymin, args.ymax)
    elif (args.ymin is None) and (args.ymax is not None):
        plt.ylim(ymax=args.ymax)
    elif (args.ymin is not None) and (args.ymax is None):
        plt.ylim(ymin=args.ymin)

    if args.show_legend:
        plt.legend(plots, set_of_indices, loc="upper left")

    # Set output device and show
    if args.output_filename is not None:
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["font.family"] = "serif"
        plt.savefig(args.output_filename)
    else:
        plt.show()
