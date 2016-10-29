# Copyright (C) 2015 Atsushi Togo
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
    parser.set_defaults(band_indices=None,
                        band_paths=None,
                        band_points=None,
                        cell_filename=None,
                        constant_averaged_pp_interaction=None,
                        cutoff_fc3_distance=None,
                        cutoff_frequency=None,
                        boundary_mfp=None,
                        cutoff_pair_distance=None,
                        delta_fc2=False,
                        delta_fc2_sets_mode=False,
                        displacement_distance=None,
                        force_sets_to_forces_fc2_mode=None,
                        forces_fc3_mode=False,
                        forces_fc3_file_mode=False,
                        forces_fc2_mode=False,
                        force_sets_mode=False,
                        frequency_conversion_factor=None,
                        fpitch=None,
                        frequency_scale_factor=None,
                        num_frequency_points=None,
                        freq_scale=None,
                        gamma_unit_conversion=None,
                        grid_addresses=None,
                        grid_points=None,
                        gv_delta_q=None,
                        input_filename=None,
                        input_output_filename=None,
                        ion_clamped=False,
                        is_bterta=False,
                        is_decay_channel=False,
                        is_nodiag=False,
                        is_displacement=False,
                        is_nomeshsym=False,
                        is_gruneisen=False,
                        is_isotope=False,
                        is_joint_dos=False,
                        is_linewidth=False,
                        is_lbte=False,
                        is_frequency_shift=False,
                        is_full_pp=False,
                        is_nac=False,
                        is_plusminus_displacements=False,
                        is_reducible_collision_matrix=False,
                        is_translational_symmetry=False,
                        is_symmetrize_fc2=False,
                        is_symmetrize_fc3_r=False,
                        is_symmetrize_fc3_q=False,                     
                        is_tetrahedron_method=False,
                        log_level=None,
                        max_freepath=None,
                        masses=None,
                        mass_variances=None,
                        mesh_numbers=None,
                        mesh_divisors=None,
                        no_kappa_stars=False,
                        output_filename=None,
                        phonon_supercell_dimension=None,
                        pinv_cutoff=1.0e-8,
                        pp_unit_conversion=None,
                        primitive_axis=None,
                        qpoints=None,
                        quiet=False,
                        q_direction=None,
                        read_amplitude=False,
                        read_collision=None,
                        read_fc2=False,
                        read_fc3=False,
                        read_gamma=False,
                        run_with_g=True,
                        scattering_event_class=None,
                        show_num_triplets=False,
                        sigma=None,
                        supercell_dimension=None,
                        symprec=1e-5,
                        temperatures=None,
                        tmax=None,
                        tmin=None,
                        tstep=None,
                        tsym_type=None,
                        uplo='L',
                        use_ave_pp=False,
                        verbose=False,
                        write_amplitude=False,
                        write_collision=False,
                        write_gamma_detail=False,
                        write_gamma=False,
                        write_phonon=False,
                        write_grid_points=False)
    parser.add_option(
        "--amplitude", dest="displacement_distance", type="float",
        help="Distance of displacements")
    parser.add_option(
        "--band", dest="band_paths", action="store", type="string",
        help="Band structure paths calculated for Gruneisen parameter")
    parser.add_option(
        "--band_points", dest="band_points", type="int",
        help=("Number of points calculated on a band segment in the band "
              "structure Gruneisen parameter calculation"))
    parser.add_option(
        "--bi", "--band_indices", dest="band_indices", type="string",
        help="Band indices where life time is calculated")
    parser.add_option(
        "--boundary_mfp", "--bmfp", dest="boundary_mfp", type="float",
        help=("Boundary mean free path in micrometre for thermal conductivity "
              "calculation"))
    parser.add_option(
        "--br", "--bterta", dest="is_bterta", action="store_true",
        help="Calculate thermal conductivity in BTE-RTA")
    parser.add_option(
        "-c", "--cell", dest="cell_filename", action="store", type="string",
        help="Read unit cell", metavar="FILE")
    parser.add_option(
        "--cf2", "--create_f2", dest="forces_fc2_mode",
        action="store_true", help="Create FORCES_FC2")
    parser.add_option(
        "--cf3", "--create_f3", dest="forces_fc3_mode",
        action="store_true", help="Create FORCES_FC3")
    parser.add_option(
        "--cf3_file", "--create_f3_from_file",
        dest="forces_fc3_file_mode",
        action="store_true",
        help="Create FORCES_FC3 from file name list")
    parser.add_option(
        "--cfs", "--create_force_sets", dest="force_sets_mode",
        action="store_true",
        help="Create phonopy FORCE_SETS from FORCES_FC2")
    parser.add_option(
        "--const_ave_pp",
        dest="constant_averaged_pp_interaction",
        type="float",
        help="Set constant averaged ph-ph interaction (Pqj)")
    parser.add_option(
        "--cutoff_fc3", "--cutoff_fc3_distance",
        dest="cutoff_fc3_distance", type="float",
        help=("Cutoff distance of third-order force constants. Elements where "
              "any pair of atoms has larger distance than cut-off distance are "
              "set zero."))
    parser.add_option(
        "--cutoff_freq", "--cutoff_frequency", dest="cutoff_frequency",
        type="float",
        help="Phonon modes below this frequency are ignored.")
    parser.add_option(
        "--cutoff_pair", "--cutoff_pair_distance",
        dest="cutoff_pair_distance", type="float",
        help=("Cutoff distance between pairs of displaced atoms used for "
              "supercell creation with displacements and making third-order "
              "force constants"))
    parser.add_option(
        "-d", "--disp", dest="is_displacement", action="store_true",
        help="As first stage, get least displacements")
    parser.add_option(
        "--dim", dest="supercell_dimension", type="string",
        help="Supercell dimension")
    parser.add_option(
        "--dim_fc2", dest="phonon_supercell_dimension", type="string",
        help="Supercell dimension for extra fc2")
    parser.add_option(
        "--factor", dest="frequency_conversion_factor", type="float",
        help="Conversion factor to favorite frequency unit")
    parser.add_option(
        "--fc2", dest="read_fc2", action="store_true",
        help="Read second order force constants")
    parser.add_option(
        "--fc3", dest="read_fc3", action="store_true",
        help="Read third order force constants")
    parser.add_option(
        "--fs2f2", "--force_sets_to_forces_fc2",
        dest="force_sets_to_forces_fc2_mode",
        action="store_true", help="Create FORCES_FC2 from FORCE_SETS")
    parser.add_option(
        "--freq_scale", dest="frequency_scale_factor", type="float",
        help=("Squared scale factor multiplied with fc2. Therefore frequency "
              "is changed but the contribution from NAC is not changed."))
    parser.add_option(
        "--freq_pitch", dest="fpitch", type="float",
        help="Pitch in frequency for spectrum")
    parser.add_option(
        "--full_pp", dest="is_full_pp",
        action="store_true",
        help=("Calculate full ph-ph interaction for RTA conductivity."
              "This may be activated when full elements of ph-ph interaction "
              "strength are needed, i.e., to calculate average ph-ph "
              "interaction strength."))
    parser.add_option(
        "--gamma_unit_conversion", dest="gamma_unit_conversion", type="float",
        help="Conversion factor for gamma")
    parser.add_option(
        "--gp", "--grid_points", dest="grid_points", type="string",
        help="Fixed grid points where anharmonic properties are calculated")
    parser.add_option(
        "--ga", "--grid_addresses", dest="grid_addresses", type="string",
        help="Fixed grid addresses where anharmonic properties are calculated")
    parser.add_option(
        "--gruneisen", dest="is_gruneisen", action="store_true",
        help="Calculate phonon Gruneisen parameter")
    parser.add_option(
        "--gv_delta_q", dest="gv_delta_q", type="float",
        help="Delta-q distance used for group velocity calculation")
    parser.add_option(
        "-i", dest="input_filename", type="string",
        help="Input filename extension")
    parser.add_option(
        "--io", dest="input_output_filename", type="string",
        help="Input and output filename extension")
    parser.add_option(
        "--ion_clamped", dest="ion_clamped", action="store_true",
        help=("Atoms are clamped under applied strain in Gruneisen parameter "
              "calculation"))
    parser.add_option(
        "--ise", dest="is_imag_self_energy", action="store_true",
        help="Calculate imaginary part of self energy")
    parser.add_option(
        "--isotope", dest="is_isotope", action="store_true",
        help="Isotope scattering lifetime")
    parser.add_option(
        "--jdos", dest="is_joint_dos", action="store_true",
        help="Calculate joint density of states")
    parser.add_option(
        "--lbte", dest="is_lbte", action="store_true",
        help="Calculate thermal conductivity LBTE with Chaput's method")
    parser.add_option(
        "--loglevel", dest="log_level", type="int", help="Log level")
    parser.add_option(
        "--lw", "--linewidth", dest="is_linewidth",
        action="store_true", help="Calculate linewidths")
    parser.add_option(
        "--fst", "--frequency_shift", dest="is_frequency_shift",
        action="store_true", help="Calculate frequency shifts")
    parser.add_option(
        "--mass", dest="masses", action="store", type="string",
        help="Same as MASS tag")
    parser.add_option(
        "--md", "--mesh_divisors", dest="mesh_divisors", type="string",
        help="Divisors for mesh numbers")
    parser.add_option(
        "--mesh", dest="mesh_numbers", type="string",
        help="Mesh numbers")
    parser.add_option(
        "--mv", "--mass_variances", dest="mass_variances",
        type="string",
        help="Mass variance parameters for isotope scattering")
    parser.add_option(
        "--nac", dest="is_nac", action="store_true",
        help="Non-analytical term correction")
    parser.add_option(
        "--nodiag", dest="is_nodiag", action="store_true",
        help="Set displacements parallel to axes")
    parser.add_option(
        "--noks", "--no_kappa_stars", dest="no_kappa_stars", action="store_true",
        help="Deactivate summation of partial kappa at q-stars"),
    parser.add_option(
        "--nomeshsym", dest="is_nomeshsym", action="store_true",
        help="No symmetrization of triplets is made.")
    parser.add_option(
        "--num_freq_points", dest="num_frequency_points", type="int",
        help="Number of sampling points for spectrum")
    parser.add_option(
        "-o", dest="output_filename", type="string",
        help="Output filename extension")
    parser.add_option(
        "--pa", "--primitive_axis", dest="primitive_axis", action="store",
        type="string", help="Same as PRIMITIVE_AXIS tags")
    parser.add_option(
        "--pinv_cutoff", dest="pinv_cutoff", type="float",
        help="Cutoff frequency (THz) for pseudo inversion of collision matrix")
    parser.add_option(
        "--pm", dest="is_plusminus_displacements", action="store_true",
        help="Set plus minus displacements")
    parser.add_option(
        "--pp_unit_conversion", dest="pp_unit_conversion", type="float",
        help="Conversion factor for ph-ph interaction")
    parser.add_option(
        "--pwscf", dest="pwscf_mode",
        action="store_true", help="Invoke Pwscf mode")
    parser.add_option(
        "--qpoints", dest="qpoints", type="string",
        help="Calculate at specified q-points")
    parser.add_option(
        "--q_direction", dest="q_direction", type="string",
        help="q-vector direction at q->0 for non-analytical term correction")
    parser.add_option(
        "-q", "--quiet", dest="quiet", action="store_true",
        help="Print out smallest information")
    # parser.add_option(
    #     "--read_amplitude", dest="read_amplitude", action="store_true",
    #     help="Read phonon-phonon interaction amplitudes")
    parser.add_option(
        "--use_ave_pp", dest="use_ave_pp", action="store_true",
        help="Use averaged ph-ph interaction")
    parser.add_option(
        "--read_collision", dest="read_collision", type="string",
        help="Read collision matrix and Gammas from files")
    parser.add_option(
        "--read_gamma", dest="read_gamma", action="store_true",
        help="Read Gammas from files")
    parser.add_option(
        "--read_phonon", dest="read_phonon", action="store_true",
        help="Read phonons from files")
    parser.add_option(
        "--reducible_colmat", dest="is_reducible_collision_matrix",
        action="store_true", help="Solve reducible collision matrix")
    parser.add_option(
        "--run_without_g", dest="run_with_g", action="store_false",
        help=("Calculate imag-part self energy without using "
              "integration weights from gaussian smearing function"))
    parser.add_option(
        "--scattering_event_class", dest="scattering_event_class", type="int",
        help=("Scattering event class 1 or 2 to draw imaginary part of self "
              "energy"))
    parser.add_option(
        "--stp", "--show_num_triplets", dest="show_num_triplets",
        action="store_true",
        help=("Show reduced number of triplets to be calculated at "
              "specified grid points"))
    parser.add_option(
        "--sigma", dest="sigma", type="string",
        help=("A sigma value or multiple sigma values (separated by space) for "
              "smearing width used for limited functions"))
    parser.add_option(
        "--sym_fc2", dest="is_symmetrize_fc2", action="store_true",
        help="Symmetrize fc2 by index exchange")
    parser.add_option(
        "--sym_fc3r", dest="is_symmetrize_fc3_r", action="store_true",
        help="Symmetrize fc3 in real space by index exchange")
    parser.add_option(
        "--sym_fc3q", dest="is_symmetrize_fc3_q", action="store_true",
        help="Symmetrize fc3 in reciprocal space by index exchange")
    parser.add_option(
        "--thm", "--tetrahedron_method", dest="is_tetrahedron_method",
        action="store_true", help="Use tetrahedron method")
    parser.add_option(
        "--tmax", dest="tmax", type="string",
        help="Maximum calculated temperature")
    parser.add_option(
        "--tmin", dest="tmin", type="string",
        help="Minimum calculated temperature")
    parser.add_option(
        "--ts", dest="temperatures", type="string",
        help="Temperatures for damping functions")
    parser.add_option(
        "--tstep", dest="tstep", type="string",
        help="Calculated temperature step")
    parser.add_option(
        "--tsym", dest="is_translational_symmetry", action="store_true",
        help="Impose translational invariance condition")
    parser.add_option(
        "--tsym_type", dest="tsym_type", type="int",
        help="Imposing type of translational invariance")
    parser.add_option(
        "--tolerance", dest="symprec", type="float",
        help="Symmetry tolerance to search")
    parser.add_option(
        "--uplo", dest="uplo", type="string", help="Lapack zheev UPLO")
    parser.add_option(
        "-v", "--verbose", dest="verbose", action="store_true",
        help="Detailed run-time information is displayed")
    parser.add_option(
        "--wgp", "--write_grid_points", dest="write_grid_points",
        action="store_true",
        help=("Write grid address of irreducible grid points for specified "
              "mesh numbers to ir_grid_address.yaml"))
    # parser.add_option("--write_amplitude", dest="write_amplitude",
    #                   action="store_true",
    #                   help="Write phonon-phonon interaction amplitudes")
    parser.add_option(
        "--write_collision", dest="write_collision", action="store_true",
        help="Write collision matrix and Gammas to files")
    parser.add_option(
        "--write_gamma_detail", "--write_detailed_gamma",
        dest="write_gamma_detail",
        action="store_true", help="Write out detailed imag-part of self energy")
    parser.add_option(
        "--write_gamma", dest="write_gamma", action="store_true",
        help="Write imag-part of self energy to files")
    parser.add_option(
        "--write_phonon", dest="write_phonon", action="store_true",
        help="Write all phonons on grid points to files")

    return parser
