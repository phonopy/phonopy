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

import sys

import numpy as np

from phonopy import PhonopyQHA
from phonopy.file_IO import read_efe, read_thermal_properties_yaml, read_v_e
from phonopy.units import EVAngstromToGPa


def get_options():
    """Parse command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Phonopy-QHA command-line-tool")
    parser.set_defaults(
        pressure=None,
        is_graph_plot=False,
        is_graph_save=False,
        is_bulk_modulus_only=False,
        efe_file=None,
        eos="vinet",
        thin_number=10,
        tmax=1000.0,
    )
    parser.add_argument(
        "-b",
        dest="is_bulk_modulus_only",
        action="store_true",
        help="Just show Bulk modulus from v-e data",
    )
    parser.add_argument(
        "--eos",
        dest="eos",
        help="Choise of EOS among vinet, birch_murnaghan, and murnaghan",
    )
    parser.add_argument(
        "--exclude_imaginary",
        dest="exclude_imaginary",
        action="store_true",
        help="Exclude volumes that show imaginary modes",
    )
    parser.add_argument(
        "-p", "--plot", dest="is_graph_plot", action="store_true", help="Plot data"
    )
    parser.add_argument(
        "--pressure", dest="pressure", type=float, help="Pressure in GPa"
    )
    parser.add_argument(
        "--efe",
        "--electronic-free-energy",
        dest="efe_file",
        nargs=1,
        help="Read electronic free energies at temperatures and volumes",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="is_graph_save",
        action="store_true",
        help="Save plot data in pdf",
    )
    parser.add_argument(
        "--sparse",
        dest="thin_number",
        type=int,
        help=(
            "Thin out the F-V plots of temperature. The value is "
            "used as deviser of number of temperature points."
        ),
    )
    parser.add_argument(
        "--tmax", dest="tmax", type=float, help="Maximum calculated temperature"
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Filenames of e-v.dat and thermal_properties.yaml's",
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-qha."""
    args = get_options()

    if args.is_graph_save:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rc("pdf", fonttype=42)

    # Choose EOS
    if args.eos == "birch_murnaghan":
        print("# Third-order Birch-Murnaghan EOS")
    elif args.eos == "murnaghan":
        print("# Murnaghan EOS")
    else:
        print("# Vinet EOS")

    ########################
    # Read data from files #
    ########################
    volumes, electronic_energies = read_v_e(args.filenames[0])
    if args.efe_file:
        _temperatures, electronic_energies = read_efe(args.efe_file[0])
    # electronic_energies += volumes * args.pressure / EVAngstromToGPa

    # Show bulk modulus of v-e data
    if args.is_bulk_modulus_only:
        # if args.efe_file:
        #     print("--efe optin can't be used with -b option")
        #     sys.exit(1)
        bulk_modulus = PhonopyQHA(
            volumes,
            electronic_energies=electronic_energies,
            pressure=args.pressure,
            eos=args.eos,
        )
        parameters = bulk_modulus.get_bulk_modulus_parameters()
        if args.efe_file:
            print(f"Volume: {parameters[3]}")
            print(f"Energy: {parameters[0]}")
            print(f"Bulk modulus: {parameters[1] * EVAngstromToGPa}")
            print(
                f"Parameters: {parameters[0]} {parameters[1]} "
                f"{parameters[2]} {parameters[3]}"
            )
        else:
            print(f"Volume: {parameters[3]:.7f}")
            print(f"Energy: {parameters[0]:.7f}")
            print(f"Bulk modulus: {parameters[1] * EVAngstromToGPa:.7f}")
            print(
                f"Parameters: {parameters[0]:.7f} {parameters[1]:.7f} "
                f"{parameters[2]:.7f} {parameters[3]:.7f}"
            )

        if args.is_graph_plot:
            bulk_modulus.plot_bulk_modulus(thin_number=args.thin_number).show()

        # These lines commented out are to print E-V points on the curve fitted.
        # def eos(v):
        #     params = bulk_modulus._bulk_modulus.get_parameters()
        #     _eos = bulk_modulus._bulk_modulus.get_eos()
        #     return _eos(v, *params)

        # print("E-V values")
        # for v in volumes:
        #     print(v, eos(v))

        sys.exit(0)

    # Check number of files in e-v.dat case
    if len(volumes) != len(args.filenames[1:]):
        print(
            "The number of thermal_properites.yaml files (%d) "
            "is inconsisten with" % len(args.filenames[1:])
        )
        print("the number of e-v data (%d)." % len(volumes))
        sys.exit(1)

    if args.efe_file:
        if len(volumes) != electronic_energies.shape[1]:
            print(
                "%s and %s are inconsistent for the volume points."
                % (args.filenames[0], args.efe_file[0])
            )
            sys.exit(1)

    (
        temperatures,
        cv,
        entropy,
        fe_phonon,
        num_modes,
        num_integrated_modes,
    ) = read_thermal_properties_yaml(args.filenames[1:])

    if args.efe_file:
        if (
            len(temperatures) >= len(_temperatures)
            and (
                np.abs(temperatures[: len(_temperatures)] - _temperatures) > 1e-5
            ).any()
        ) or (
            len(temperatures) < len(_temperatures)
            and (np.abs(temperatures - _temperatures[: len(temperatures)]) > 1e-5).any()
        ):
            print(
                f"Inconsistency is found in temperatures in {args.efe_file[0]} and "
                "thermal_properties.yaml files."
            )
            print("Temperatures in a thermal_properties.yaml are")
            print(f"{temperatures}.")
            print(f"Temperatures in {args.efe_file[0]} are")
            print(f"{_temperatures.tolist()}.")
            sys.exit(1)

    ########################################################
    # Treatment of thermal properties with imaginary modes #
    ########################################################
    if args.exclude_imaginary and num_modes:
        indices = []
        num_imag_modes = np.array(num_modes) - np.array(num_integrated_modes)
        for i, nim in enumerate(num_imag_modes):
            if nim < 4:
                indices.append(i)
    else:
        indices = range(len(volumes))

    if args.efe_file:
        electronic_energies = electronic_energies[:, indices]
    else:
        electronic_energies = electronic_energies[indices]

    ##########################
    # Analyzing and plotting #
    ##########################
    phonopy_qha = PhonopyQHA(
        volumes=volumes[indices],
        electronic_energies=electronic_energies,
        eos=args.eos,
        temperatures=temperatures,
        pressure=args.pressure,
        free_energy=fe_phonon[:, indices],
        cv=cv[:, indices],
        entropy=entropy[:, indices],
        t_max=args.tmax,
        verbose=True,
    )

    if num_modes:
        num_imag_modes = np.array(num_modes) - np.array(num_integrated_modes)
        for filename, nim in zip(
            args.filenames[1 : (len(volumes) + 1)], num_imag_modes
        ):
            if nim > 3:
                if args.exclude_imaginary:
                    print("# %s has been excluded." % filename)
                else:
                    print("# Warning: %s has imaginary modes." % filename)

    if args.is_graph_plot and not args.is_graph_save:
        # Plot on display
        # - Volume vs Helmholtz free energy
        # - Volume vs Temperature
        # - Thermal expansion coefficient
        phonopy_qha.plot_qha(thin_number=args.thin_number).show()

    if args.is_graph_save:
        # Volume vs Helmholts free energy
        phonopy_qha.plot_pdf_helmholtz_volume(thin_number=args.thin_number)

        # Volume vs Temperature
        phonopy_qha.plot_pdf_volume_temperature()

        # Thermal expansion coefficient
        phonopy_qha.plot_pdf_thermal_expansion()

        # G vs Temperature
        phonopy_qha.plot_pdf_gibbs_temperature()

        # Bulk modulus vs Temperature
        phonopy_qha.plot_pdf_bulk_modulus_temperature()

        # C_P vs Temperature
        phonopy_qha.plot_pdf_heat_capacity_P_numerical()

        # C_P vs Temperature (poly fit)
        phonopy_qha.plot_pdf_heat_capacity_P_polyfit()

        # Gruneisen parameter vs Temperature
        phonopy_qha.plot_pdf_gruneisen_temperature()

    phonopy_qha.write_helmholtz_volume()
    phonopy_qha.write_helmholtz_volume_fitted(thin_number=args.thin_number)
    phonopy_qha.write_volume_temperature()
    phonopy_qha.write_thermal_expansion()
    phonopy_qha.write_gibbs_temperature()
    phonopy_qha.write_bulk_modulus_temperature()
    phonopy_qha.write_heat_capacity_P_numerical()
    phonopy_qha.write_heat_capacity_P_polyfit()
    phonopy_qha.write_gruneisen_temperature()
