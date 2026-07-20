# SPDX-License-Identifier: BSD-3-Clause
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
        help=("Filename of phonon thermal properties result (thermal_properties.yaml)"),
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
                strict=True,
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
