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

# Thermal properties plot (propplot)
#
# Usage:
#   propplot

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


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy propplot command-line-tool")
    parser.set_defaults(
        output_filename=None,
        factor=1.0,
        is_heat_capacity=False,
        is_entropy=False,
        is_free_energy=False,
        is_diff=False,
        is_gnuplot=False,
        ymax=None,
        ymin=None,
        tmax=None,
        tmin=None,
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
        "--cv",
        "--heat_capacity",
        dest="is_heat_capacity",
        action="store_true",
        help="Plot heat capacity",
    )
    parser.add_argument(
        "-s", "--entropy", dest="is_entropy", action="store_true", help="Plot entropy"
    )
    parser.add_argument(
        "--fe",
        "--free_energy",
        dest="is_free_energy",
        action="store_true",
        help="Plot free energy",
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", help="Output filename"
    )
    parser.add_argument(
        "-d", "--diff", dest="is_diff", action="store_true", help="Calculate difference"
    )
    parser.add_argument(
        "--gnuplot",
        dest="is_gnuplot",
        action="store_true",
        help="Damp thermal properties in gnuplot data style",
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help=(
            "Filename of phonon thermal properties result " "(thermal_properties.yaml)"
        ),
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-propplot."""
    args = get_options()

    if args.is_gnuplot:
        print(
            "# temperature[K] Free energy [kJ/mol] Entropy [J/K/mol] "
            "Heat capacity [J/K/mol] Energy [kJ/mol]"
        )
    else:
        import matplotlib.pyplot as plt

        if (
            (not args.is_heat_capacity)
            and (not args.is_entropy)
            and (not args.is_free_energy)
        ):
            print("Set --cv, --entropy or --fe")
            sys.exit(0)

    if len(args.filename) == 0:
        filenames = ["thermal_properties.yaml"]
    else:
        filenames = args.filename

    if args.is_heat_capacity:
        prop_target = "heat_capacity"
    elif args.is_entropy:
        prop_target = "entropy"
    elif args.is_free_energy:
        prop_target = "free_energy"

    thermal_properties_0 = yaml.load(open(filenames[0]).read(), Loader=Loader)[
        "thermal_properties"
    ]
    temperatures = [v["temperature"] for v in thermal_properties_0]

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

    if args.is_diff:
        props_0 = [v[prop_target] for v in thermal_properties_0]
        props_0 = np.array(props_0) * args.factor

    for filename in filenames:
        thermal_properties = yaml.load(open(filename).read(), Loader=Loader)[
            "thermal_properties"
        ]

        if args.is_gnuplot:
            props = []
            for name in ("temperature", "free_energy", "entropy", "heat_capacity"):
                props.append([v[name] for v in thermal_properties])

            for t, f, e, h in zip(
                props[0][tmin_index:tmax_index],
                props[1][tmin_index:tmax_index],
                props[2][tmin_index:tmax_index],
                props[3][tmin_index:tmax_index],
            ):
                print(("%14.7f " * 5) % (t, f, e, h, f + e * t / 1000))
            print("")
            print("")
        else:
            temperatures = [v["temperature"] for v in thermal_properties]
            props = [v[prop_target] for v in thermal_properties]
            props = np.array(props) * args.factor
            if args.is_diff:
                props -= props_0
            plt.plot(temperatures[tmin_index:tmax_index], props[tmin_index:tmax_index])

    if not args.is_gnuplot:
        if (args.ymin is not None) and (args.ymax is not None):
            plt.ylim(args.ymin, args.ymax)
        elif (args.ymin is None) and (args.ymax is not None):
            plt.ylim(ymax=args.ymax)
        elif (args.ymin is not None) and (args.ymax is None):
            plt.ylim(ymin=args.ymin)

        if args.output_filename is not None:
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["font.family"] = "serif"
            plt.savefig(args.output_filename)
        else:
            plt.show()
