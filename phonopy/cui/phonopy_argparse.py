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
    deprecated = []
    for i, v in enumerate(argv[1:]):
        if v[0] == '-':
            tag = v.split('=')[0]
            if '_' in tag:
                correct_tag = tag.replace('_', '-')
                deprecated.append(tag)
                argv[i + 1] = v.replace(tag, correct_tag)

    return deprecated

def show_deprecated_option_warnings(deprecated):
    lines = ["Option names with underscores are deprecated, by which",
             "the underscores are replaced by dashes. Therefore"]
    for tag in deprecated:
        lines.append("'%s' has to be written as '%s'." %
                     (tag, tag.replace('_', '-')))
    maxlen = max([len(line) for line in lines])
    print("*" * maxlen)
    print('\n'.join(lines))
    print("*" * maxlen)
    print("")

def get_parser():
    deprecated = fix_deprecated_option_names(sys.argv)
    import argparse
    parser = argparse.ArgumentParser(
        description="Phonopy command-line-tool")
    parser.set_defaults(
        abinit_mode=False,
        anime=None,
        band_format=None,
        band_indices=None,
        band_labels=None,
        band_paths=None,
        band_points=None,
        cell_filename=None,
        crystal_mode=False,
        cutoff_frequency=None,
        cutoff_radius=None,
        displacement_distance=None,
        dynamical_matrix_decimals=None,
        elk_mode=False,
        siesta_mode=False,
        cp2k_mode=False,
        fc_symmetry=None,
        fc_computation_algorithm=None,
        fc_format=None,
        fc_spg_symmetry=False,
        fits_debye_model=False,
        force_constants_decimals=None,
        force_constants=None,
        force_sets=None,
        force_sets_zero=None,
        fmax=None,
        fmin=None,
        frequency_conversion_factor=None,
        fpitch=None,
        gv_delta_q=None,
        is_band_connection=False,
        is_check_symmetry=False,
        is_displacement=False,
        is_dos_mode=False,
        is_eigenvectors=False,
        is_gamma_center=False,
        is_graph_plot=False,
        is_graph_save=False,
        is_group_velocity=False,
        is_hdf5=False,
        is_legend=False,
        is_little_cogroup=False,
        is_moment=False,
        is_nac=False,
        is_nodiag=False,
        is_nomeshsym=False,
        is_nosym=False,
        is_plusminus_displacements=False,
        is_tetrahedron_method=False,
        is_thermal_displacements=False,
        is_thermal_displacement_matrices=False,
        is_thermal_displacement_matrices_cif=None,
        is_thermal_properties=False,
        is_projected_thermal_properties=False,
        is_trigonal_displacements=False,
        is_wien2k_p1=False,
        irreps_qpoint=None,
        lapack_solver=False,
        loglevel=None,
        masses=None,
        magmom=None,
        mesh_format=None,
        mesh_numbers=None,
        modulation=None,
        moment_order=None,
        pretend_real=False,
        primitive_axis=None,
        projection_direction=None,
        pwscf_mode=False,
        qpoints=None,
        qpoints_format=None,
        quiet=False,
        q_direction=None,
        read_fc_format=None,
        read_force_constants=False,
        read_qpoints=False,
        show_irreps=False,
        sigma=None,
        supercell_dimension=None,
        symprec=1e-5,
        tmax=None,
        tmin=None,
        tstep=None,
        vasp_mode=False,
        verbose=False,
        wien2k_mode=False,
        write_dynamical_matrices=False,
        write_force_constants=False,
        write_fc_format=None,
        write_mesh=True,
        yaml_mode=False)

    parser.add_argument(
        "--abinit", dest="abinit_mode", action="store_true",
        help="Invoke Abinit mode")
    parser.add_argument(
        "--amplitude", dest="displacement_distance", type=float,
        help="Distance of displacements")
    parser.add_argument(
        "--anime", dest="anime",
        help="Same as ANIME tag")
    parser.add_argument(
        "--band", dest="band_paths",
        help="Same behavior as BAND tag")
    parser.add_argument(
        "--band-connection", dest="is_band_connection", action="store_true",
        help="Treat band crossings")
    parser.add_argument(
        "--band-labels", dest="band_labels",
        help="Show labels at band segments")
    parser.add_argument(
        "--band-format", dest="band_format",
        help="Band structure output file-format")
    parser.add_argument(
        "--band-points", dest="band_points", type=int,
        help=("Number of points calculated on a band segment in "
              "the band structure mode"))
    parser.add_argument(
        "--bi", "--band-indices", dest="band_indices",
        help=("Band indices to be included to calcualte thermal "
              "properties"))
    parser.add_argument(
        "-c", "--cell", dest="cell_filename", metavar="FILE",
        help="Read unit cell")
    parser.add_argument(
        "--cp2k", dest="cp2k_mode", action="store_true",
        help="Invoke CP2K mode")
    parser.add_argument(
        "--crystal", dest="crystal_mode", action="store_true",
        help="Invoke CRYSTAL mode")
    parser.add_argument(
        "--cutoff-freq", "--cutoff-frequency", dest="cutoff_frequency",
        type=float,
        help=("Thermal properties are not calculated below this "
              "cutoff frequency."))
    parser.add_argument(
        "--cutoff-radius", dest="cutoff_radius", type=float,
        help="Out of cutoff radius, force constants are set zero.")
    parser.add_argument(
        "-d", "--displacement", dest="is_displacement", action="store_true",
        help="Create supercells with displacements")
    parser.add_argument(
        "--dim", dest="supercell_dimension",
        help="Same behavior as DIM tag")
    parser.add_argument(
        "--dm-decimals", dest="dynamical_matrix_decimals",
        type=int, help="Decimals of values of decimals")
    parser.add_argument(
        "--dos", dest="is_dos_mode", action="store_true",
        help="Calculate (P)DOS")
    parser.add_argument(
        "--eigvecs", "--eigenvectors", dest="is_eigenvectors",
        action="store_true",
        help="Output eigenvectors")
    parser.add_argument(
        "--elk", dest="elk_mode", action="store_true",
        help="Invoke elk mode")
    parser.add_argument(
        "-f", "--force-sets", nargs='+', dest="force_sets",
        help="Create FORCE_SETS")
    parser.add_argument(
        "--factor", dest="frequency_conversion_factor", type=float,
        help="Conversion factor to favorite frequency unit")
    parser.add_argument(
        "--fc", "--force-constants", nargs=1, dest="force_constants",
        help=("Create FORCE_CONSTANTS from vaspurn.xml. "
              "vasprun.xml has to be passed as argument."))
    parser.add_argument(
        "--fc-computation-algorithm", dest="fc_computation_algorithm",
        help="Switch computation algorithm of force constants")
    parser.add_argument(
        "--fc-decimals", dest="force_constants_decimals", type=int,
        help="Decimals of values of force constants")
    parser.add_argument(
        "--fc-format", dest="fc_format",
        help="Force constants input/output file-format")
    parser.add_argument(
        "--fc-spg-symmetry", dest="fc_spg_symmetry", action="store_true",
        help="Enforce space group symmetry to force constants")
    parser.add_argument(
        "--fc-symmetry", dest="fc_symmetry", type=int,
        help="Symmetrize force constants")
    parser.add_argument(
        "--fits-debye-model", dest="fits_debye_model", action="store_true",
        help="Fits total DOS to a Debye model")
    parser.add_argument(
        "--fz", "--force-sets-zero", nargs='+', dest="force_sets_zero",
        help=("Create FORCE_SETS. disp.yaml in the current directory and "
              "vapsrun.xml's for VASP or case.scf(m) for Wien2k as arguments "
              "are required. The first argument is that of the perfect "
              "supercell to subtract residual forces"))
    parser.add_argument(
        "--fmax", dest="fmax", type=float,
        help="Maximum frequency used for DOS or moment calculation")
    parser.add_argument(
        "--fmin", dest="fmin", type=float,
        help="Minimum frequency used for DOS or moment calculation")
    parser.add_argument(
        "--fpitch", dest="fpitch", type=float,
        help="Frequency pitch used for DOS or moment calculation")
    parser.add_argument(
        "--gc", "--gamma-center", dest="is_gamma_center", action="store_true",
        help="Set mesh as Gamma center")
    parser.add_argument(
        "--gv", "--group-velocity", dest="is_group_velocity",
        action="store_true",
        help="Calculate group velocities at q-points")
    parser.add_argument(
        "--gv-delta-q", dest="gv_delta_q", type=float,
        help="Delta-q distance used for group velocity calculation")
    parser.add_argument(
        "--hdf5", dest="is_hdf5", action="store_true",
        help="Use hdf5 for force constants")
    parser.add_argument(
        "--irreps", "--irreps-qpoint", dest="irreps_qpoint",
        help="A q-point where characters of irreps are calculated")
    # parser.add_argument(
    #     "--lapack-solver", dest="lapack_solver", action="store_true",
    #     help=("Use Lapack via Lapacke for solving phonons. This "
    #           "option can be used only when phonopy is compiled "
    #           "specially."))
    parser.add_argument(
        "--legend", dest="is_legend", action="store_true",
        help="Legend of plots is shown in thermal displacements")
    parser.add_argument(
        "--lcg", "--little-cogroup", dest="is_little_cogroup",
        action="store_true",
        help=("Show irreps of little co-group (or point-group of "
              "wave vector q) instead of little group"))
    parser.add_argument(
        "--loglevel", dest="loglevel", type=int,
        help="Log level")
    parser.add_argument(
        "--mass", dest="masses",
        help="Same as MASS tag")
    parser.add_argument(
        "--magmom", dest="magmoms",
        help="Same as MAGMOM tag")
    parser.add_argument(
        "--mesh-format", dest="mesh_format",
        help="Mesh output file-format")
    parser.add_argument(
        "--modulation", dest="modulation",
        help="Same as MODULATION tag")
    parser.add_argument(
        "--mp", "--mesh", dest="mesh_numbers",
        help="Same behavior as MP tag")
    parser.add_argument(
        "--moment", dest="is_moment", action="store_true",
        help="Calculate moment of phonon states distribution")
    parser.add_argument(
        "--moment-order", dest="moment_order",
        type=int, help="Order of moment of phonon states distribution")
    parser.add_argument(
        "--nac", dest="is_nac", action="store_true",
        help="Non-analytical term correction")
    parser.add_argument(
        "--nodiag", dest="is_nodiag", action="store_true",
        help="Set displacements parallel to axes")
    parser.add_argument(
        "--nomeshsym", dest="is_nomeshsym", action="store_true",
        help="Symmetry is not imposed for mesh sampling.")
    parser.add_argument(
        "--nowritemesh", dest="write_mesh", action="store_false",
        help="Do not write mesh.yaml or mesh.hdf5")
    parser.add_argument(
        "--nosym", dest="is_nosym", action="store_true",
        help="Symmetry is not imposed.")
    parser.add_argument(
        "-p", "--plot", dest="is_graph_plot", action="store_true",
        help="Plot data")
    parser.add_argument(
        "--pa", "--primitive-axis", dest="primitive_axis",
        help="Same as PRIMITIVE_AXIS tag")
    parser.add_argument(
        "--pd", "--projection-direction", dest="projection_direction",
        help="Same as PROJECTION_DIRECTION tag")
    parser.add_argument(
        "--pdos", dest="pdos",
        help="Same as PDOS tag")
    parser.add_argument(
        "--pm", dest="is_plusminus_displacements", action="store_true",
        help="Set plus minus displacements")
    parser.add_argument(
        "--pr", "--pretend-real", dest="pretend_real", action="store_true",
        help=("Use imaginary frequency as real for thermal property "
              "calculation. For a testing purpose only, when a small "
              "amount of imaginary branches obtained."))
    parser.add_argument(
        "--pt", "--projected-thermal-property",
        dest="is_projected_thermal_properties", action="store_true",
        help="Output projected thermal properties")
    parser.add_argument(
        "--pwscf", dest="pwscf_mode",
        action="store_true", help="Invoke Pwscf mode")
    parser.add_argument(
        "--qpoints", dest="qpoints",
        help="Calculate at specified q-points")
    parser.add_argument(
        "--qpoints-format", dest="qpoints_format",
        help="Q-points output file-format")
    parser.add_argument(
        "--q-direction", dest="q_direction",
        help=("Direction of q-vector perturbation used for NAC at "
              "q->0, and group velocity for degenerate phonon "
              "mode in q-points mode"))
    parser.add_argument(
        "-q", "--quiet", dest="quiet", action="store_true",
        help="Print out smallest information")
    parser.add_argument(
        "--readfc", dest="read_force_constants", action="store_true",
        help="Read FORCE_CONSTANTS")
    parser.add_argument(
        "--readfc-format", dest="readfc_format",
        help="Force constants input file-format")
    parser.add_argument(
        "--read-qpointsfc", dest="read_qpoints", action="store_true",
        help="Read QPOITNS")
    parser.add_argument(
        "-s", "--save", dest="is_graph_save", action="store_true",
        help="Save plot data in pdf")
    parser.add_argument(
        "--show-irreps", dest="show_irreps", action="store_true",
        help="Show IR-Reps along with characters")
    parser.add_argument(
        "--siesta", dest="siesta_mode", action="store_true",
        help="Invoke Siesta mode")
    parser.add_argument(
        "--sigma", dest="sigma",
        help="Smearing width for DOS")
    parser.add_argument(
        "--symmetry", dest="is_check_symmetry", action="store_true",
        help="Check crystal symmetry")
    parser.add_argument(
        "-t", "--thermal-property", dest="is_thermal_properties",
        action="store_true",
        help="Output thermal properties")
    parser.add_argument(
        "--td", "--thermal-displacements", dest="is_thermal_displacements",
        action="store_true",
        help="Output thermal displacements")
    parser.add_argument(
        "--tdm", "--thermal-displacement-matrix",
        dest="is_thermal_displacement_matrices", action="store_true",
        help="Output thermal displacement matrices")
    parser.add_argument(
        "--tdm-cif", "--thermal-displacement-matrix-cif",
        dest="thermal_displacement_matrices_cif", type=float,
        help="Write cif with aniso_U for which temperature is specified")
    parser.add_argument(
        "--thm", "--tetrahedron-method", dest="is_tetrahedron_method",
        action="store_true",
        help="Use tetrahedron method for DOS/PDOS")
    parser.add_argument(
        "--tmax", dest="tmax", type=float,
        help="Maximum calculated temperature")
    parser.add_argument(
        "--tmin", dest="tmin", type=float,
        help="Minimum calculated temperature")
    parser.add_argument(
        "--trigonal", dest="is_trigonal_displacements", action="store_true",
        help="Set displacements of all trigonal axes ")
    parser.add_argument(
        "--tstep", dest="tstep", type=float,
        help="Calculated temperature step")
    parser.add_argument(
        "--tolerance", dest="symprec", type=float,
        help="Symmetry tolerance to search")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true",
        help="Detailed information is shown.")
    parser.add_argument(
        "--vasp", dest="vasp_mode", action="store_true",
        help="Invoke Vasp mode")
    parser.add_argument(
        "--wien2k", dest="wien2k_mode", action="store_true",
        help="Invoke Wien2k mode")
    parser.add_argument(
        "--wien2k_p1", dest="is_wien2k_p1", action="store_true",
        help="Assume Wien2k structs with displacements are P1")
    parser.add_argument(
        "--writefc", dest="write_force_constants", action="store_true",
        help="Write FORCE_CONSTANTS")
    parser.add_argument(
        "--writefc-format", dest="writefc_format",
        help="Force constants output file-format")
    parser.add_argument(
        "--writedm", dest="write_dynamical_matrices", action="store_true",
        help=("Write dynamical matrices. This has to be used "
              "with QPOINTS setting (or --qpoints)"))
    parser.add_argument(
        "--xyz-projection", dest="xyz_projection", action="store_true",
        help="Project PDOS x, y, z directions in Cartesian coordinates")
    parser.add_argument(
        "--yaml", dest="yaml_mode", action="store_true",
        help="Activate phonopy YAML mode")
    parser.add_argument(
        "conf_file", nargs='*',
        help="Phonopy configure file")

    return parser, deprecated
