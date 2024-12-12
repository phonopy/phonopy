"""Phonopy command line argument parser."""

# Copyright (C) 2016 Atsushi Togo
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


def fix_deprecated_option_names(argv):
    """Replace underscore in command option name by hyphen."""
    deprecated = []
    for i, v in enumerate(argv[1:]):
        if v[0] == "-":
            tag = v.split("=")[0]
            if "_" in tag:
                correct_tag = tag.replace("_", "-")
                deprecated.append(tag)
                argv[i + 1] = v.replace(tag, correct_tag)

    return deprecated


def show_deprecated_option_warnings(deprecated):
    """Show warning when underscore is included in command option name."""
    lines = [
        "Option names with underscores are deprecated, by which",
        "the underscores are replaced by dashes. Therefore",
    ]
    for tag in deprecated:
        lines.append("'%s' has to be written as '%s'." % (tag, tag.replace("_", "-")))
    maxlen = max([len(line) for line in lines])
    print("*" * maxlen)
    print("\n".join(lines))
    print("*" * maxlen)
    print("")


def get_parser(fc_symmetry=False, is_nac=False, load_phonopy_yaml=False):
    """Return ArgumentParser instance."""
    deprecated = fix_deprecated_option_names(sys.argv)
    import argparse

    from phonopy.interface.calculator import (
        add_arguments_of_calculators,
        calculator_info,
    )

    try:
        parser = argparse.ArgumentParser(
            description="Phonopy command-line-tool", allow_abbrev=False
        )  # allow_abbrev requires >= python 3.5
    except TypeError:
        parser = argparse.ArgumentParser(description="Phonopy command-line-tool")

    add_arguments_of_calculators(parser, calculator_info)

    parser.add_argument(
        "--alm",
        dest="use_alm",
        action="store_true",
        default=None,
        help="Use ALM for generating force constants",
    )
    parser.add_argument(
        "--amplitude",
        "--amin",
        dest="displacement_distance",
        type=float,
        default=None,
        help=(
            "Distance of displacements and also minimum distance of displacements "
            "in random displacements"
        ),
    )
    parser.add_argument(
        "--amax",
        dest="displacement_distance_max",
        type=float,
        default=None,
        help="Minimum distance of displacements in random displacements",
    )
    parser.add_argument(
        "--anime", nargs="+", dest="anime", default=None, help="Same as ANIME tag"
    )
    parser.add_argument(
        "--band",
        nargs="+",
        dest="band_paths",
        default=None,
        help="Same behavior as BAND tag",
    )
    parser.add_argument(
        "--band-connection",
        dest="is_band_connection",
        action="store_true",
        default=None,
        help="Treat band crossings",
    )
    parser.add_argument(
        "--band-const-interval",
        dest="is_band_const_interval",
        action="store_true",
        default=None,
        help="Band paths are sampled with similar interval.",
    )
    parser.add_argument(
        "--band-labels",
        nargs="+",
        dest="band_labels",
        default=None,
        help="Show labels at band segments",
    )
    parser.add_argument(
        "--band-format",
        dest="band_format",
        default=None,
        help="Band structure output file-format",
    )
    parser.add_argument(
        "--band-points",
        dest="band_points",
        type=int,
        default=None,
        help=(
            "Number of points calculated on a band segment in "
            "the band structure mode"
        ),
    )
    parser.add_argument(
        "--bi",
        "--band-indices",
        nargs="+",
        dest="band_indices",
        default=None,
        help=("Band indices to be included to calcualte thermal " "properties"),
    )
    if not load_phonopy_yaml:
        parser.add_argument(
            "-c",
            "--cell",
            dest="cell_filename",
            metavar="FILE",
            default=None,
            help="Read unit cell",
        )
    if load_phonopy_yaml:
        parser.add_argument(
            "--config",
            dest="conf_filename",
            metavar="FILE",
            default=None,
            help="Phonopy configuration file",
        )
    parser.add_argument(
        "--classical",
        dest="classical",
        action="store_true",
        default=False,
        help=("Compute thermodynamic properties using classical statistics."),
    )
    parser.add_argument(
        "--cutoff-freq",
        "--cutoff-frequency",
        dest="cutoff_frequency",
        type=float,
        default=None,
        help=("Thermal properties are not calculated below this " "cutoff frequency."),
    )
    parser.add_argument(
        "--cutoff-radius",
        dest="cutoff_radius",
        type=float,
        default=None,
        help="Out of cutoff radius, force constants are set zero.",
    )
    if not load_phonopy_yaml:
        parser.add_argument(
            "-d",
            "--displacement",
            dest="is_displacement",
            action="store_true",
            default=None,
            help="Create supercells with displacements",
        )
    if not load_phonopy_yaml:
        parser.add_argument(
            "--dim",
            nargs="+",
            dest="supercell_dimension",
            default=None,
            help="Same behavior as DIM tag",
        )
    parser.add_argument(
        "--dm-decimals",
        dest="dynamical_matrix_decimals",
        default=None,
        type=int,
        help="Decimals of values of decimals",
    )
    parser.add_argument(
        "--dos",
        dest="is_dos_mode",
        action="store_true",
        default=None,
        help="Calculate (P)DOS",
    )
    parser.add_argument(
        "--eigvecs",
        "--eigenvectors",
        dest="is_eigenvectors",
        action="store_true",
        default=None,
        help="Output eigenvectors",
    )
    if load_phonopy_yaml:
        parser.add_argument(
            "--exclude-born",
            "--exclude-nac-params",
            dest="include_nac_params",
            action="store_false",
            default=None,
            help=(
                "Exclude born effective charge and dielectric tensor in " "phonopy.yaml"
            ),
        )
    parser.add_argument(
        "-f",
        "--force-sets",
        nargs="+",
        dest="create_force_sets",
        default=None,
        help="Create FORCE_SETS",
    )
    parser.add_argument(
        "--factor",
        dest="frequency_conversion_factor",
        type=float,
        default=None,
        help="Frequency unit conversion factor",
    )
    parser.add_argument(
        "--fc",
        "--force-constants",
        metavar="FILE",
        dest="create_force_constants",
        default=None,
        help=(
            "Create FORCE_CONSTANTS from vaspurn.xml. "
            "vasprun.xml has to be passed as argument."
        ),
    )
    parser.add_argument(
        "--fc-calc",
        "--fc-calculator",
        dest="fc_calculator",
        default=None,
        help=("Force constants calculator"),
    )
    parser.add_argument(
        "--fc-calc-opt",
        "--fc-calculator-options",
        dest="fc_calculator_options",
        default=None,
        help=(
            "Options for force constants calculator as comma separated "
            "string with the style of key = values"
        ),
    )
    parser.add_argument(
        "--fc-decimals",
        dest="force_constants_decimals",
        type=int,
        default=None,
        help="Decimals of values of force constants",
    )
    parser.add_argument(
        "--fc-format",
        dest="fc_format",
        default=None,
        help="Force constants input/output file-format",
    )
    parser.add_argument(
        "--fc-spg-symmetry",
        dest="fc_spg_symmetry",
        action="store_true",
        default=None,
        help="Enforce space group symmetry to force constants",
    )
    if not fc_symmetry:
        parser.add_argument(
            "--fc-symmetry",
            "--sym-fc",
            dest="fc_symmetry",
            action="store_true",
            default=None,
            help="Symmetrize force constants",
        )
    parser.add_argument(
        "--fits-debye-model",
        dest="fits_debye_model",
        action="store_true",
        default=None,
        help="Fits total DOS to a Debye model",
    )
    # parser.add_argument(
    #     "--freq-scale",
    #     dest="frequency_scale_factor",
    #     type=float,
    #     default=None,
    #     help=(
    #         "Squared scale factor multiplied as fc2 * factor^2. Therefore "
    #         "frequency is changed but the contribution from NAC is not "
    #         "changed."
    #     ),
    # )
    parser.add_argument(
        "--full-fc",
        dest="is_full_fc",
        action="store_true",
        default=None,
        help="Calculate full supercell force constants matrix",
    )
    parser.add_argument(
        "--fz",
        "--force-sets-zero",
        nargs="+",
        dest="create_force_sets_zero",
        default=None,
        help=(
            "Create FORCE_SETS. disp.yaml in the current directory and "
            "vapsrun.xml's for VASP or case.scf(m) for Wien2k as arguments "
            "are required. The first argument is that of the perfect "
            "supercell to subtract residual forces"
        ),
    )
    parser.add_argument(
        "--fmax",
        dest="fmax",
        type=float,
        default=None,
        help="Maximum frequency used for DOS or moment calculation",
    )
    parser.add_argument(
        "--fmin",
        dest="fmin",
        type=float,
        default=None,
        help="Minimum frequency used for DOS or moment calculation",
    )
    parser.add_argument(
        "--fpitch",
        dest="fpitch",
        type=float,
        help="Frequency pitch used for DOS or moment calculation",
    )
    parser.add_argument(
        "--gc",
        "--gamma-center",
        dest="is_gamma_center",
        action="store_true",
        default=None,
        help="Set mesh as Gamma center",
    )
    parser.add_argument(
        "--gv",
        "--group-velocity",
        dest="is_group_velocity",
        action="store_true",
        default=None,
        help="Calculate group velocities at q-points",
    )
    parser.add_argument(
        "--gv-delta-q",
        dest="gv_delta_q",
        type=float,
        default=None,
        help="Delta-q distance used for group velocity calculation",
    )
    parser.add_argument(
        "--hdf5",
        dest="is_hdf5",
        action="store_true",
        default=None,
        help="Use hdf5 for force constants",
    )
    parser.add_argument(
        "--hdf5-compression",
        dest="hdf5_compression",
        default=None,
        help="hdf5 compression filter (default: gzip)",
    )
    parser.add_argument(
        "--irreps",
        "--irreps-qpoint",
        nargs="+",
        dest="irreps_qpoint",
        default=None,
        help="A q-point where characters of irreps are calculated",
    )
    parser.add_argument(
        "--include-fc",
        dest="include_fc",
        action="store_true",
        default=None,
        help="Include force constants in phonopy.yaml",
    )
    parser.add_argument(
        "--include-fs",
        dest="include_fs",
        action="store_true",
        default=None,
        help="Include force sets in phonopy.yaml",
    )
    parser.add_argument(
        "--include-disp",
        dest="include_disp",
        action="store_true",
        default=None,
        help="Include displacements in phonopy.yaml",
    )
    parser.add_argument(
        "--include-all",
        dest="include_all",
        action="store_true",
        default=None,
        help="Include all output file data in phonopy.yaml",
    )
    parser.add_argument(
        "--legend",
        dest="is_legend",
        action="store_true",
        default=None,
        help="Legend of plots is shown in thermal displacements",
    )
    parser.add_argument(
        "--legacy-plot",
        dest="is_legacy_plot",
        action="store_true",
        default=None,
        help="Legacy style band structure pl",
    )
    parser.add_argument(
        "--lcg",
        "--little-cogroup",
        dest="is_little_cogroup",
        action="store_true",
        default=None,
        help=(
            "Show irreps of little co-group (or point-group of "
            "wave vector q) instead of little group"
        ),
    )
    parser.add_argument(
        "--loglevel", dest="loglevel", type=int, default=None, help="Log level"
    )
    parser.add_argument(
        "--mass", nargs="+", dest="masses", default=None, help="Same as MASS tag"
    )
    parser.add_argument(
        "--magmom", nargs="+", dest="magmoms", default=None, help="Same as MAGMOM tag"
    )
    parser.add_argument(
        "--mesh-format",
        dest="mesh_format",
        default=None,
        help="Mesh output file-format",
    )
    parser.add_argument(
        "--modulation",
        nargs="+",
        dest="modulation",
        default=None,
        help="Same as MODULATION tag",
    )
    parser.add_argument(
        "--mp",
        "--mesh",
        nargs="+",
        dest="mesh_numbers",
        default=None,
        help="Same behavior as MP tag",
    )
    parser.add_argument(
        "--mlp-params",
        dest="mlp_params",
        default=None,
        help=(
            "Parameters for machine learning potentials as comma separated "
            "string with the style of key = values"
        ),
    )
    parser.add_argument(
        "--moment",
        dest="is_moment",
        action="store_true",
        default=None,
        help="Calculate moment of phonon states distribution",
    )
    parser.add_argument(
        "--moment-order",
        dest="moment_order",
        default=None,
        type=int,
        help="Order of moment of phonon states distribution",
    )
    if not is_nac:
        parser.add_argument(
            "--nac",
            dest="is_nac",
            action="store_true",
            default=None,
            help="Non-analytical term correction",
        )
    parser.add_argument(
        "--nac-method",
        dest="nac_method",
        default=None,
        help="Non-analytical term correction method: Gonze (default) or Wang",
    )
    if fc_symmetry:
        parser.add_argument(
            "--no-fc-symmetry",
            "--no-sym-fc",
            dest="fc_symmetry",
            action="store_false",
            default=None,
            help="Do not symmetrize force constants",
        )
    parser.add_argument(
        "--nodiag",
        dest="is_nodiag",
        action="store_true",
        default=None,
        help="Set displacements parallel to axes",
    )
    parser.add_argument(
        "--nomeshsym",
        dest="is_nomeshsym",
        action="store_true",
        default=None,
        help="Symmetry is not imposed for mesh sampling.",
    )
    if is_nac:
        parser.add_argument(
            "--nonac",
            dest="is_nac",
            action="store_false",
            default=None,
            help="Non-analytical term correction",
        )
    parser.add_argument(
        "--nosym",
        dest="is_nosym",
        action="store_true",
        default=None,
        help="Symmetry is not imposed.",
    )
    parser.add_argument(
        "--nowritemesh",
        dest="write_mesh",
        action="store_false",
        default=None,
        help="Do not write mesh.yaml or mesh.hdf5",
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="is_graph_plot",
        action="store_true",
        default=None,
        help="Plot data",
    )
    parser.add_argument(
        "--pa",
        "--primitive-axis",
        "--primitive-axes",
        nargs="+",
        dest="primitive_axes",
        default=None,
        help="Same as PRIMITIVE_AXES tag",
    )
    parser.add_argument(
        "--pd",
        "--projection-direction",
        nargs="+",
        dest="projection_direction",
        default=None,
        help="Same as PROJECTION_DIRECTION tag",
    )
    parser.add_argument(
        "--pdos", nargs="+", dest="pdos", default=None, help="Same as PDOS tag"
    )
    parser.add_argument(
        "--pm",
        dest="is_plusminus_displacements",
        action="store_true",
        default=None,
        help="Set plus minus displacements",
    )
    parser.add_argument(
        "--pr",
        "--pretend-real",
        dest="pretend_real",
        action="store_true",
        default=None,
        help=(
            "Use imaginary frequency as real for thermal property "
            "calculation. For a testing purpose only, when a small "
            "amount of imaginary branches obtained."
        ),
    )
    parser.add_argument(
        "--pt",
        "--projected-thermal-property",
        dest="is_projected_thermal_properties",
        action="store_true",
        default=None,
        help="Output projected thermal properties",
    )
    parser.add_argument(
        "--pypolymlp",
        dest="use_pypolymlp",
        action="store_true",
        default=None,
        help="Use pypolymlp for generating force constants",
    )
    parser.add_argument(
        "--qpoints",
        nargs="+",
        dest="qpoints",
        default=None,
        help="Calculate at specified q-points",
    )
    parser.add_argument(
        "--qpoints-format",
        dest="qpoints_format",
        default=None,
        help="Q-points output file-format",
    )
    parser.add_argument(
        "--q-direction",
        nargs="+",
        dest="nac_q_direction",
        default=None,
        help=(
            "Direction of q-vector perturbation used for NAC at "
            "q->0, and group velocity for degenerate phonon "
            "mode in q-points mode"
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=None,
        help="Print out smallest information",
    )
    parser.add_argument(
        "--random-seed",
        dest="random_seed",
        type=int,
        default=None,
        help="Random seed by a 32 bit unsigned integer",
    )
    parser.add_argument(
        "--rd",
        "--random-displacements",
        dest="random_displacements",
        type=int,
        default=None,
        help="Number of supercells with random displacements",
    )
    parser.add_argument(
        "--rd-temperature",
        dest="rd_temperature",
        type=float,
        default=None,
        metavar="TEMPERATURE",
        help="A temperature used to generate random displacements.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        metavar="TEMPERATURE",
        help="(Deprecated) A temperature used to generate random displacements.",
    )
    parser.add_argument(
        "--readfc",
        dest="read_force_constants",
        action="store_true",
        default=None,
        help="Read FORCE_CONSTANTS",
    )
    parser.add_argument(
        "--readfc-format",
        dest="readfc_format",
        default=None,
        help="Force constants input file-format",
    )
    parser.add_argument(
        "--read-qpoints",
        dest="read_qpoints",
        action="store_true",
        default=None,
        help="Read QPOITNS",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="is_graph_save",
        action="store_true",
        default=None,
        help="Save plot data in pdf",
    )
    parser.add_argument(
        "--sp",
        "--save-params",
        dest="save_params",
        action="store_true",
        default=None,
        help="Save parameters that can run phonopy in phonopy_params.yaml.",
    )
    parser.add_argument(
        "--show-irreps",
        dest="show_irreps",
        action="store_true",
        default=None,
        help="Show IR-Reps along with characters",
    )
    parser.add_argument(
        "--sigma", dest="sigma", default=None, help="Smearing width for DOS"
    )
    parser.add_argument(
        "--sscha",
        dest="sscha_iterations",
        type=int,
        default=None,
        help="Number of iterations in SSCHA calculation",
    )
    if not fc_symmetry:
        parser.add_argument(
            "--symfc",
            dest="use_symfc",
            action="store_true",
            default=None,
            help="Use symfc for generating force constants",
        )
    parser.add_argument(
        "--symmetry",
        dest="is_check_symmetry",
        action="store_true",
        default=None,
        help="Check crystal symmetry",
    )
    parser.add_argument(
        "-t",
        "--thermal-property",
        dest="is_thermal_properties",
        action="store_true",
        default=None,
        help="Output thermal properties",
    )
    parser.add_argument(
        "--td",
        "--thermal-displacements",
        dest="is_thermal_displacements",
        action="store_true",
        default=None,
        help="Output thermal displacements",
    )
    parser.add_argument(
        "--tdm",
        "--thermal-displacement-matrix",
        dest="is_thermal_displacement_matrices",
        action="store_true",
        default=None,
        help="Output thermal displacement matrices",
    )
    parser.add_argument(
        "--tdm-cif",
        "--thermal-displacement-matrix-cif",
        metavar="TEMPERATURE",
        dest="thermal_displacement_matrices_cif",
        type=float,
        default=None,
        help="Write cif with aniso_U for which temperature is specified",
    )
    parser.add_argument(
        "--tmax",
        dest="tmax",
        type=float,
        default=None,
        help="Maximum calculated temperature",
    )
    parser.add_argument(
        "--tmin",
        dest="tmin",
        type=float,
        default=None,
        help="Minimum calculated temperature",
    )
    parser.add_argument(
        "--tolerance",
        dest="symmetry_tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance to search",
    )
    parser.add_argument(
        "--trigonal",
        dest="is_trigonal_displacements",
        action="store_true",
        default=None,
        help="Set displacements of all trigonal axes ",
    )
    parser.add_argument(
        "--tstep",
        dest="tstep",
        type=float,
        default=None,
        help="Calculated temperature step",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=None,
        help="Detailed information is shown.",
    )
    parser.add_argument(
        "--wien2k-p1",
        dest="is_wien2k_p1",
        action="store_true",
        default=None,
        help="Assume Wien2k structs with displacements are P1",
    )
    parser.add_argument(
        "--writefc",
        dest="write_force_constants",
        action="store_true",
        default=None,
        help="Write FORCE_CONSTANTS",
    )
    parser.add_argument(
        "--writefc-format",
        dest="writefc_format",
        default=None,
        help="Force constants output file-format",
    )
    parser.add_argument(
        "--writedm",
        dest="write_dynamical_matrices",
        action="store_true",
        default=None,
        help=(
            "Write dynamical matrices. This has to be used "
            "with QPOINTS setting (or --qpoints)"
        ),
    )
    parser.add_argument(
        "--xyz-projection",
        dest="xyz_projection",
        action="store_true",
        default=None,
        help="Project PDOS x, y, z directions in Cartesian coordinates",
    )
    if load_phonopy_yaml:
        parser.add_argument("filename", nargs="*", help="phonopy.yaml like file")
    else:
        parser.add_argument(
            "filename",
            nargs="*",
            help=(
                "Phonopy configure file. However if the file is recognized as "
                "phonopy.yaml like file, this file is read as phonopy.yaml like file."
            ),
        )

    return parser, deprecated
