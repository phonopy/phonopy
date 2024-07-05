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

import phonopy
from phonopy import PhonopyGruneisen
from phonopy.interface.calculator import (
    get_default_cell_filename,
    get_default_physical_units,
    get_interface_mode,
)


def fracval(frac):
    """Convert fractional value string to float."""
    if frac.find("/") == -1:
        return float(frac)
    else:
        x = frac.split("/")
        return float(x[0]) / float(x[1])


def get_options():
    """Parse command-line options."""
    import argparse

    from phonopy.interface.calculator import (
        add_arguments_of_calculators,
        calculator_info,
    )

    parser = argparse.ArgumentParser(description="Phonopy gruneisen command-line-tool")
    add_arguments_of_calculators(parser, calculator_info)

    parser.set_defaults(
        band_paths=None,
        band_points=51,
        cell_filename=None,
        color_scheme=None,
        cutoff_frequency=None,
        cutoff_wave_vector=1e-4,
        factor=None,
        is_gamma_center=False,
        is_hdf5=False,
        is_nac=False,
        is_mesh_symmetry=True,
        marker="o",
        markersize=None,
        masses=None,
        plot_graph=False,
        primitive_axes=None,
        reads_force_constants=False,
        sampling_mesh=None,
        save_graph=False,
        delta_strain=None,
        supercell_dimension=None,
        symprec=1e-5,
        title=None,
        tmax=2004,
        tmin=0,
        tstep=2,
        volumes_filename=None,
    )
    parser.add_argument(
        "--band", dest="band_paths", help="Band paths in reduced coordinates"
    )
    parser.add_argument(
        "--band_points",
        dest="band_points",
        type=int,
        help="Number of sampling points in a segment of band path",
    )
    parser.add_argument(
        "-c", "--cell", dest="cell_filename", help="Read unit cell", metavar="FILE"
    )
    parser.add_argument("--color", dest="color_scheme", help="Color scheme")
    parser.add_argument(
        "--cutoff",
        dest="cutoff_frequency",
        type=float,
        help="Plot above this cutoff frequency for mesh sampling mode.",
    )
    parser.add_argument(
        "--dim", dest="supercell_dimension", help="Same behavior as DIM tag"
    )
    parser.add_argument(
        "--factor",
        dest="factor",
        type=float,
        help="Conversion factor to favorite frequency unit",
    )
    parser.add_argument(
        "--hdf5",
        dest="is_hdf5",
        action="store_true",
        help="Use hdf5 to read force constants and store results",
    )
    parser.add_argument(
        "--gc",
        "--gamma_center",
        dest="is_gamma_center",
        action="store_true",
        help="Set mesh as Gamma center",
    )
    parser.add_argument("--marker", dest="marker", help="Marker for plot (matplotlib)")
    parser.add_argument(
        "--markersize",
        dest="markersize",
        type=float,
        help="Markersize for plot in points (matplotlib)",
    )
    parser.add_argument("--mass", dest="masses", help="Same as MASS tag")
    parser.add_argument("--mp", "--mesh", dest="sampling_mesh", help="Sampling mesh")
    parser.add_argument(
        "--nac",
        dest="is_nac",
        action="store_true",
        help="Non-analytical term correction",
    )
    parser.add_argument(
        "--nomeshsym",
        dest="is_mesh_symmetry",
        action="store_false",
        help="Symmetry is not imposed for mesh sampling.",
    )
    parser.add_argument(
        "-o", "--output", dest="output_filename", help="Output filename of PDF plot"
    )
    parser.add_argument(
        "-p", "--plot", dest="plot_graph", action="store_true", help="Plot data"
    )
    parser.add_argument(
        "--pa",
        "--primitive-axis",
        "--primitive-axes",
        dest="primitive_axes",
        help="Same as PRIMITIVE_AXES tags",
    )
    parser.add_argument(
        "--q_cutoff",
        dest="cutoff_wave_vector",
        type=float,
        help="Acoustic modes inside cutoff wave vector is treated.",
    )
    parser.add_argument(
        "--readfc",
        dest="reads_force_constants",
        action="store_true",
        help="Read FORCE_CONSTANTS",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save_graph",
        action="store_true",
        help="Save plot data in pdf",
    )
    parser.add_argument(
        "--delta-strain",
        dest="delta_strain",
        type=float,
        help="Delta strain instead of using delta-V/V",
    )
    parser.add_argument("-t", "--title", dest="title", help="Title of plot")
    parser.add_argument(
        "--tmax", dest="tmax", type=float, help="Maximum calculated temperature"
    )
    parser.add_argument(
        "--tmin", dest="tmin", type=float, help="Minimum calculated temperature"
    )
    parser.add_argument(
        "--tolerance", dest="symprec", type=float, help="Symmetry tolerance to search"
    )
    parser.add_argument(
        "--tstep", dest="tstep", type=float, help="Calculated temperature step"
    )
    parser.add_argument(
        "--vf",
        "--volumes_filename",
        dest="volumes_filename",
        help="Filename of volume is contained.",
    )
    parser.add_argument(
        "dirnames",
        nargs="*",
        help=("Directory names of phonons with three different volumes " "(0, +, -)"),
    )
    args = parser.parse_args()
    return args


def run():
    """Run phonopy-gruneisen."""
    args = get_options()

    if len(args.dirnames) != 3:
        sys.stderr.write(
            "Three directory names (original, plus, minus) " "have to be spefied.\n"
        )
        sys.exit(1)

    primitive_matrix = None
    if args.primitive_axes:
        if args.primitive_axes == "auto":
            primitive_matrix = "auto"
        else:
            vals = [fracval(x) for x in args.primitive_axes.split()]
            primitive_matrix = np.array(vals).reshape(3, 3)

    if args.supercell_dimension:
        supercell_matrix = [int(x) for x in args.supercell_dimension.split()]
        if len(supercell_matrix) == 9:
            supercell_matrix = np.array(supercell_matrix).reshape(3, 3)
        elif len(supercell_matrix) == 3:
            supercell_matrix = np.diag(supercell_matrix)
        else:
            print("Number of elements of --dim option has to be 3 or 9.")
            sys.exit(1)
        if np.linalg.det(supercell_matrix) < 1:
            print("Determinant of supercell matrix has to be positive.")
            sys.exit(1)
    else:
        print("--dim option has to be specified.")
        sys.exit(1)

    #
    # Phonopy gruneisen interface mode
    #
    interface_mode = get_interface_mode(vars(args))
    physical_units = get_default_physical_units(interface_mode)

    if args.factor is not None:
        physical_units["factor"] = args.factor

    if args.cell_filename:
        cell_filename = args.cell_filename
    else:
        cell_filename = get_default_cell_filename(interface_mode)

    phonons = []
    for i in range(3):
        directory = args.dirnames[i]

        unitcell_filename = "%s/%s" % (directory, cell_filename)
        print('Unit cell was read from "%s".' % unitcell_filename)

        if args.is_nac:
            born_filename = "%s/BORN" % directory
            print('NAC parameters were read from "%s".' % born_filename)
        else:
            born_filename = None

        if args.reads_force_constants:
            if args.is_hdf5:
                fc_filename = "%s/force_constants.hdf5" % directory
            else:
                fc_filename = "%s/FORCE_CONSTANTS" % directory
            print('Force constants were read from "%s".' % fc_filename)
            phonon = phonopy.load(
                supercell_matrix=supercell_matrix,
                primitive_matrix=primitive_matrix,
                calculator=interface_mode,
                unitcell_filename=unitcell_filename,
                born_filename=born_filename,
                force_constants_filename=fc_filename,
                factor=physical_units["factor"],
                symprec=args.symprec,
            )
            phonons.append(phonon)
        else:
            force_filename = "%s/FORCE_SETS" % directory
            print('Force sets were read from "%s".' % force_filename)
            phonon = phonopy.load(
                supercell_matrix=supercell_matrix,
                primitive_matrix=primitive_matrix,
                calculator=interface_mode,
                unitcell_filename=unitcell_filename,
                born_filename=born_filename,
                force_sets_filename=force_filename,
                factor=physical_units["factor"],
                symprec=args.symprec,
            )
            phonons.append(phonon)

        print("")

    if args.masses:
        masses = [float(v) for v in args.masses.split()]
        for ph in phonons:
            ph.set_masses(masses)

    gruneisen = PhonopyGruneisen(
        phonons[0],  # equilibrium
        phonons[1],  # plus
        phonons[2],  # minus
        delta_strain=args.delta_strain,
    )

    if args.plot_graph:
        if args.save_graph:
            import matplotlib as mpl

            mpl.use("Agg")

    if args.band_paths:
        from phonopy.phonon.band_structure import get_band_qpoints

        band_paths = []
        for path_str in args.band_paths.split(","):
            paths = np.array([fracval(x) for x in path_str.split()])
            if len(paths) % 3 != 0 or len(paths) < 6:
                print("Band path is incorrectly set.")
                sys.exit(1)
            band_paths.append(paths.reshape(-1, 3))
        bands = get_band_qpoints(band_paths, npoints=args.band_points)
        gruneisen.set_band_structure(bands)
        gruneisen.write_yaml_band_structure()
        if args.plot_graph:
            plt = gruneisen.plot_band_structure(
                epsilon=args.cutoff_wave_vector, color_scheme=args.color_scheme
            )
            if args.title is not None:
                plt.suptitle(args.title)

    elif args.sampling_mesh:
        mesh_numbers = np.array([int(x) for x in args.sampling_mesh.split()])
        gruneisen.set_mesh(
            mesh_numbers,
            is_gamma_center=args.is_gamma_center,
            is_mesh_symmetry=args.is_mesh_symmetry,
        )

        if args.is_hdf5:
            gruneisen.write_hdf5_mesh()
        else:
            gruneisen.write_yaml_mesh()

        if args.plot_graph:
            plt = gruneisen.plot_mesh(
                cutoff_frequency=args.cutoff_frequency,
                color_scheme=args.color_scheme,
                marker=args.marker,
                markersize=args.markersize,
            )
            if args.title is not None:
                plt.suptitle(args.title)

        if args.volumes_filename is not None:
            volumes = []
            with open(args.volumes_filename) as f:
                for line in f:
                    if line.strip()[0] == "#":
                        continue
                    volumes.append(float(line.split()[0]))
            if volumes:
                gruneisen.set_thermal_properties(
                    volumes,
                    t_step=args.tstep,
                    t_max=args.tmax,
                    t_min=args.tmin,
                    cutoff_frequency=args.cutoff_frequency,
                )
                gruneisen.write_yaml_thermal_properties()
    else:
        pass

    if args.plot_graph:
        if args.save_graph:
            if args.output_filename:
                plt.savefig(args.output_filename)
            else:
                plt.savefig("gruneisen.pdf")
        else:
            plt.show()
