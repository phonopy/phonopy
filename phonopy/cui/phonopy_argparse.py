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

from optparse import OptionParser

def get_parser():
    parser = OptionParser()
    parser.set_defaults(
        abinit_mode=False,
        anime=None,
        band_indices=None,
        band_labels=None,
        band_paths=None,
        band_points=None,
        cell_filename=None,
        cutoff_frequency=None,
        cutoff_radius=None,
        displacement_distance=None,
        dynamical_matrix_decimals=None,
        elk_mode=False,
        siesta_mode=False,
        fc_symmetry=None,
        fc_computation_algorithm=None,
        fc_spg_symmetry=False,
        fits_debye_model=False,
        force_constants_decimals=None,
        force_constants_mode=False,
        force_sets_mode=False,
        force_sets_zero_mode=False,
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
        is_read_force_constants=False,
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
        mesh_numbers=None,
        modulation=None,
        moment_order=None,
        primitive_axis=None,
        projection_direction=None,
        pwscf_mode=False,
        qpoints=None,
        quiet=False,
        q_direction=None,
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
        write_mesh=True,
        yaml_mode=False)
    
    parser.add_option(
        "--abinit", dest="abinit_mode",
        action="store_true", help="Invoke Abinit mode")
    parser.add_option(
        "--amplitude", dest="displacement_distance", type="float",
        help="Distance of displacements")
    parser.add_option(
        "--anime", dest="anime", action="store", type="string",
        help="Same as ANIME tag")
    parser.add_option(
        "--band", dest="band_paths", action="store", type="string",
        help="Same behavior as BAND tag")
    parser.add_option(
        "--band_connection", dest="is_band_connection",
        action="store_true", help="Treat band crossings")
    parser.add_option(
        "--band_labels", dest="band_labels", action="store", type="string",
        help="Show labels at band segments")
    parser.add_option(
        "--band_points", dest="band_points", type="int",
        help=("Number of points calculated on a band segment in "
              "the band structure mode"))
    parser.add_option(
        "--bi", "--band_indices", dest="band_indices", type="string",
        help=("Band indices to be included to calcualte thermal "
              "properties"))
    parser.add_option(
        "-c", "--cell", dest="cell_filename", action="store", type="string",
        help="Read unit cell", metavar="FILE")
    parser.add_option(
        "--cutoff_freq", "--cutoff_frequency",
        dest="cutoff_frequency", type="float",
        help=("Thermal properties are not calculated below this "
              "cutoff frequency."))
    parser.add_option(
        "--cutoff_radius", dest="cutoff_radius", type="float",
        help="Out of cutoff radius, force constants are set zero.")
    parser.add_option(
        "-d", "--displacement", dest="is_displacement", action="store_true",
        help="Create supercells with displacements")
    parser.add_option(
        "--dim", dest="supercell_dimension", action="store", type="string",
        help="Same behavior as DIM tag")
    parser.add_option(
        "--dm_decimals", dest="dynamical_matrix_decimals",
        type="int", help="Decimals of values of decimals")
    parser.add_option(
        "--dos", dest="is_dos_mode", action="store_true",
        help="Calculate (P)DOS")
    parser.add_option(
        "--eigvecs", "--eigenvectors", dest="is_eigenvectors",
        action="store_true",
        help="Output eigenvectors")
    parser.add_option(
        "--elk", dest="elk_mode", action="store_true",
        help="Invoke elk mode")
    parser.add_option(
        "-f", "--force_sets", dest="force_sets_mode", action="store_true",
        help="Create FORCE_SETS")
    parser.add_option(
        "--factor", dest="frequency_conversion_factor", type="float",
        help="Conversion factor to favorite frequency unit")
    parser.add_option(
        "--fc", "--force_constants", dest="force_constants_mode",
        action="store_true",
        help=("Create FORCE_CONSTANTS from vaspurn.xml. "
              "vasprun.xml has to be passed as argument."))
    parser.add_option(
        "--fc_decimals", dest="force_constants_decimals",
        type="int", help="Decimals of values of force constants")
    parser.add_option(
        "--fc_computation_algorithm", dest="fc_computation_algorithm",
        action="store", type="string",
        help="Switch computation algorithm of force constants")
    parser.add_option(
        "--fc_spg_symmetry", dest="fc_spg_symmetry", action="store_true",
        help="Enforce space group symmetry to force constants")
    parser.add_option(
        "--fc_symmetry", dest="fc_symmetry", type="int",
        help="Symmetrize force constants")
    parser.add_option(
        "--fits_debye_model", dest="fits_debye_model", action="store_true",
        help="Fits total DOS to a Debye model")
    parser.add_option(
        "--fz", "--force_sets_zero", dest="force_sets_zero_mode",
        action="store_true",
        help=("Create FORCE_SETS. disp.yaml in the current directory and "
              "vapsrun.xml's for VASP or case.scf(m) for Wien2k as arguments "
              "are required. The first argument is that of the perfect "
              "supercell to subtract residual forces"))
    parser.add_option(
        "--fmax", dest="fmax", type="float",
        help="Maximum frequency used for DOS or moment calculation")
    parser.add_option(
        "--fmin", dest="fmin", type="float",
        help="Minimum frequency used for DOS or moment calculation")
    parser.add_option(
        "--fpitch", dest="fpitch", type="float",
        help="Frequency pitch used for DOS or moment calculation")
    parser.add_option(
        "--gc", "--gamma_center", dest="is_gamma_center", action="store_true",
        help="Set mesh as Gamma center")
    parser.add_option(
        "--gv", "--group_velocity", dest="is_group_velocity",
        action="store_true",
        help="Calculate group velocities at q-points")
    parser.add_option(
        "--gv_delta_q", dest="gv_delta_q", type="float",
        help="Delta-q distance used for group velocity calculation")
    parser.add_option(
        "--hdf5", dest="is_hdf5", action="store_true",
        help="Use hdf5 for force constants")
    parser.add_option(
        "--irreps", "--irreps_qpoint", dest="irreps_qpoint",
        action="store", type="string",
        help="A q-point where characters of irreps are calculated")
    parser.add_option(
        "--lapack_solver", dest="lapack_solver", action="store_true",
        help=("Use Lapack via Lapacke for solving phonons. This "
              "option can be used only when phonopy is compiled "
              "specially."))
    parser.add_option(
        "--legend", dest="is_legend", action="store_true",
        help="Legend of plots is shown in thermal displacements")
    parser.add_option(
        "--lcg", "--little_cogroup", dest="is_little_cogroup",
        action="store_true",
        help=("Show irreps of little co-group (or point-group of "
              "wave vector q) instead of little group"))
    parser.add_option(
        "--loglevel", dest="loglevel", type="int",
        help="Log level")
    parser.add_option(
        "--mass", dest="masses", action="store", type="string",
        help="Same as MASS tag")
    parser.add_option(
        "--modulation", dest="modulation", action="store", type="string",
        help="Same as MODULATION tag")
    parser.add_option(
        "--mp", "--mesh", dest="mesh_numbers", action="store", type="string",
        help="Same behavior as MP tag")
    parser.add_option(
        "--moment", dest="is_moment", action="store_true",
        help="Calculate moment of phonon states distribution")
    parser.add_option(
        "--moment_order", dest="moment_order",
        type="int", help="Order of moment of phonon states distribution")
    parser.add_option(
        "--nac", dest="is_nac", action="store_true",
        help="Non-analytical term correction")
    parser.add_option(
        "--nodiag", dest="is_nodiag", action="store_true",
        help="Set displacements parallel to axes")
    parser.add_option(
        "--nomeshsym", dest="is_nomeshsym", action="store_true",
        help="Symmetry is not imposed for mesh sampling.")
    parser.add_option(
        "--nowritemesh", dest="write_mesh", action="store_false",
        help="Do not write mesh.yaml or mesh.hdf5")
    parser.add_option(
        "--nosym", dest="is_nosym", action="store_true",
        help="Symmetry is not imposed.")
    parser.add_option(
        "-p", "--plot", dest="is_graph_plot", action="store_true",
        help="Plot data")
    parser.add_option(
        "--pa", "--primitive_axis", dest="primitive_axis",
        action="store", type="string",
        help="Same as PRIMITIVE_AXIS tag")
    parser.add_option(
        "--pd", "--projection_direction", dest="projection_direction",
        action="store", type="string",
        help="Same as PROJECTION_DIRECTION tag")
    parser.add_option(
        "--pdos", dest="pdos", action="store", type="string",
        help="Same as PDOS tag")
    parser.add_option(
        "--pm", dest="is_plusminus_displacements", action="store_true",
        help="Set plus minus displacements")
    parser.add_option(
        "--pt", "--projected_thermal_property",
        dest="is_projected_thermal_properties", action="store_true",
        help="Output projected thermal properties")
    parser.add_option(
        "--pwscf", dest="pwscf_mode",
        action="store_true", help="Invoke Pwscf mode")
    parser.add_option(
        "--qpoints", dest="qpoints", type="string",
        help="Calculate at specified q-points")
    parser.add_option(
        "--q_direction", dest="q_direction", type="string",
        help=("Direction of q-vector perturbation used for NAC at "
              "q->0, and group velocity for degenerate phonon "
              "mode in q-points mode"))
    parser.add_option(
        "--siesta", dest="siesta_mode",
        action="store_true", help="Invoke Siesta mode")
    parser.add_option(
        "-q", "--quiet", dest="quiet", action="store_true",
        help="Print out smallest information")
    parser.add_option(
        "--readfc", dest="is_read_force_constants", action="store_true",
        help="Read FORCE_CONSTANTS")
    parser.add_option(
        "-s", "--save", dest="is_graph_save", action="store_true",
        help="Save plot data in pdf")
    parser.add_option(
        "--show_irreps", dest="show_irreps", action="store_true",
        help="Show IR-Reps along with characters")
    parser.add_option(
        "--sigma", dest="sigma", type="string",
        help="Smearing width for DOS")
    parser.add_option(
        "--symmetry", dest="is_check_symmetry", action="store_true",
        help="Check crystal symmetry")
    parser.add_option(
        "-t", "--thermal_property", dest="is_thermal_properties",
        action="store_true",
        help="Output thermal properties")
    parser.add_option(
        "--td", "--thermal_displacements",
        dest="is_thermal_displacements", action="store_true",
        help="Output thermal displacements")
    parser.add_option(
        "--tdm", "--thermal_displacement_matrix",
        dest="is_thermal_displacement_matrices", action="store_true",
        help="Output thermal displacement matrices")
    parser.add_option(
        "--tdm_cif", "--thermal_displacement_matrix_cif",
        dest="thermal_displacement_matrices_cif", type="float",
        help="Write cif with aniso_U for which temperature is specified")
    parser.add_option(
        "--thm", "--tetrahedron_method",
        dest="is_tetrahedron_method", action="store_true",
        help="Use tetrahedron method for DOS/PDOS")
    parser.add_option(
        "--tmax", dest="tmax", type="float",
        help="Maximum calculated temperature")
    parser.add_option(
        "--tmin", dest="tmin", type="float",
        help="Minimum calculated temperature")
    parser.add_option(
        "--trigonal", dest="is_trigonal_displacements", action="store_true",
        help="Set displacements of all trigonal axes ")
    parser.add_option(
        "--tstep", dest="tstep", type="float",
        help="Calculated temperature step")
    parser.add_option(
        "--tolerance", dest="symprec", type="float",
        help="Symmetry tolerance to search")
    parser.add_option(
        "-v", "--verbose", dest="verbose", action="store_true",
        help="Detailed information is shown.")
    parser.add_option(
        "--vasp", dest="vasp_mode",
        action="store_true", help="Invoke Vasp mode")
    parser.add_option(
        "--wien2k", dest="wien2k_mode",
        action="store_true", help="Invoke Wien2k mode")
    parser.add_option(
        "--wien2k_p1", dest="is_wien2k_p1", action="store_true",
        help="Assume Wien2k structs with displacements are P1")
    parser.add_option(
        "--writefc", dest="write_force_constants", action="store_true",
        help="Write FORCE_CONSTANTS")
    parser.add_option(
        "--writedm", dest="write_dynamical_matrices", action="store_true",
        help=("Write dynamical matrices. This has to be used "
              "with QPOINTS setting (or --qpoints)"))
    parser.add_option(
        "--xyz_projection", dest="xyz_projection", action="store_true",
        help="Project PDOS x, y, z directions in Cartesian coordinates")
    parser.add_option(
        "--yaml", dest="yaml_mode", action="store_true",
        help="Activate phonopy YAML mode")

    return parser
