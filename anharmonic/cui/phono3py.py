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

import os
import sys
import numpy as np

from phonopy.interface.vasp import read_vasp, write_vasp
from phonopy.structure.cells import print_cell
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.file_IO import parse_BORN, write_FORCE_SETS, parse_FORCE_SETS
from phonopy.structure.spglib import get_grid_point_from_address
from phonopy.units import VaspToTHz
from anharmonic.phonon3.fc3 import show_drift_fc3
from anharmonic.file_IO import (parse_disp_fc2_yaml, parse_disp_fc3_yaml,
                                parse_FORCES_FC2, parse_FORCES_FC3,
                                write_FORCES_FC3_vasp, write_FORCES_FC2_vasp,
                                write_FORCES_FC2, write_fc3_to_hdf5,
                                write_fc2_to_hdf5, read_fc3_from_hdf5,
                                read_fc2_from_hdf5, write_ir_grid_points,
                                write_grid_address, write_grid_address_to_hdf5,
                                write_disp_fc3_yaml, write_disp_fc2_yaml)
from anharmonic.phonon3.triplets import (get_coarse_ir_grid_points,
                                         get_number_of_triplets)
from anharmonic.cui.settings import Phono3pyConfParser
from anharmonic.phonon3 import (Phono3py, Phono3pyJointDos, Phono3pyIsotope,
                                get_gruneisen_parameters)
from anharmonic.cui.phono3py_argparse import get_parser

phono3py_version = "0.9.14"

# AA is created at http://www.network-science.de/ascii/.
def print_phono3py():
    print("""        _                      _____             
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _ 
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/ """)

def print_version(version):
    print(" " * 42 + "%s" % version)
    print('')

def print_end():
    print("""                 _ 
   ___ _ __   __| |
  / _ \ '_ \ / _` |
 |  __/ | | | (_| |
  \___|_| |_|\__,_|
""")

def print_error():
    print("""  ___ _ __ _ __ ___  _ __ 
 / _ \ '__| '__/ _ \| '__|
|  __/ |  | | | (_) | |   
 \___|_|  |_|  \___/|_|
""")

def print_error_message(message):
    print(message)

def file_exists(filename, log_level):
    if os.path.exists(filename):
        return True
    else:
        error_text = "%s not found." % filename
        print_error_message(error_text)
        if log_level > 0:
            print_error()
        sys.exit(1)

# Parse arguments
parser = get_parser()
(options, args) = parser.parse_args()
option_list = parser.option_list

# Log level
log_level = 1
if options.verbose:
    log_level = 2
if options.quiet:
    log_level = 0
if options.log_level is not None:
    log_level=options.log_level

# Input and output filename extension
input_filename = options.input_filename
output_filename = options.output_filename
if options.input_output_filename is not None:
    input_filename = options.input_output_filename
    output_filename = options.input_output_filename

# Title
if log_level:
    print_phono3py()
    print_version(phono3py_version)

#####################
# Create FORCES_FC3 #
#####################
if options.forces_fc3_mode or options.forces_fc3_file_mode:
    if input_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + input_filename + '.yaml'
    file_exists(filename, log_level)
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    disp_dataset = parse_disp_fc3_yaml()

    if options.forces_fc3_file_mode:
        file_exists(args[0], log_level)
        filenames = [x.strip() for x in open(args[0])]
    else:
        filenames = args
    write_FORCES_FC3_vasp(filenames, disp_dataset)
    
    if log_level:
        print("FORCES_FC3 has been created.")
        print_end()
    exit(0)

#####################
# Create FORCES_FC2 #
#####################
if options.forces_fc2_mode:
    if input_filename is None:
        filename = 'disp_fc2.yaml'
    else:
        filename = 'disp_fc2.' + input_filename + '.yaml'
    file_exists(filename, log_level)
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    disp_dataset = parse_disp_fc2_yaml()
    write_FORCES_FC2_vasp(args, disp_dataset)

    if log_level:
        print("FORCES_FC2 has been created.")
        print_end()
    exit(0)

if options.force_sets_to_forces_fc2_mode:
    filename = 'FORCE_SETS'
    file_exists(filename, log_level)
    disp_dataset = parse_FORCE_SETS(filename=filename)
    write_FORCES_FC2(disp_dataset)

    if log_level:
        print("FORCES_FC2 has been created from FORCE_SETS.")
        print_end()
    exit(0)
    
#####################################
# Create FORCE_SETS from FORCES_FC* #
#####################################
if options.force_sets_mode:
    if options.phonon_supercell_dimension is not None:
        if input_filename is None:
            filename = 'disp_fc2.yaml'
        else:
            filename = 'disp_fc2.' + input_filename + '.yaml'
        file_exists(filename, log_level)
        disp_dataset = parse_disp_fc2_yaml()
        forces = parse_FORCES_FC2(disp_dataset)
    else:
        if input_filename is None:
            filename = 'disp_fc3.yaml'
        else:
            filename = 'disp_fc3.' + input_filename + '.yaml'
        file_exists(filename, log_level)
        disp_dataset = parse_disp_fc3_yaml()
        forces = parse_FORCES_FC3(disp_dataset)
        
    if log_level:
        print("Displacement dataset is read from %s." % filename)
        
    for force_set, disp1 in zip(forces, disp_dataset['first_atoms']):
        disp1['forces'] = force_set
    write_FORCE_SETS(disp_dataset)
    
    if log_level:
        print("FORCE_SETS has been created.")
        print_end()
    exit(0)
    
##################
# Parse settings #
##################
if len(args) > 0:
    phono3py_conf = Phono3pyConfParser(filename=args[0],
                                       options=options,
                                       option_list=option_list)
    settings = phono3py_conf.get_settings()

else:
    phono3py_conf = Phono3pyConfParser(options=options,
                                       option_list=option_list)
    settings = phono3py_conf.get_settings()
    
###################################
# Read crystal structure (POSCAR) #
###################################
if options.cell_poscar is None:
    file_exists('POSCAR', log_level)
    unitcell_filename = 'POSCAR'
else:
    file_exists(options.cell_poscar, log_level)
    unitcell_filename = options.cell_poscar
unitcell = read_vasp(unitcell_filename, settings.get_chemical_symbols())

#################################################
# Create supercells with displacements and exit #
#################################################
if settings.get_create_displacements():
    if settings.get_displacement_distance() is None:
        displacement_distance = 0.03
    else:
        displacement_distance = settings.get_displacement_distance()
    cutoff_pair_distance = settings.get_cutoff_pair_distance()
    phono3py = Phono3py(
        unitcell,
        settings.get_supercell_matrix(),
        phonon_supercell_matrix=settings.get_phonon_supercell_matrix(),
        symprec=options.symprec)
    supercell = phono3py.get_supercell()
    phono3py.generate_displacements(
        distance=displacement_distance,
        cutoff_pair_distance=cutoff_pair_distance,
        is_plusminus=settings.get_is_plusminus_displacement(),
        is_diagonal=settings.get_is_diagonal_displacement())
    dds = phono3py.get_displacement_dataset()
    
    if log_level:
        print('')
        print("Displacement distance: %s" % displacement_distance)

    if output_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + output_filename + '.yaml'
        
    num_disps, num_disp_files = write_disp_fc3_yaml(dds,
                                                    supercell,
                                                    filename=filename)
    for i, dcell in enumerate(phono3py.get_supercells_with_displacements()):
        if dcell is not None:
            write_vasp('POSCAR-%05d' % (i + 1), dcell, direct=True)

    if log_level:
        print("Number of displacements: %d" % num_disps)
        if cutoff_pair_distance is not None:
            print("Cutoff distance for displacements: %s" %
                  cutoff_pair_distance)
            print("Number of displacement supercell files created: %d" %
                  num_disp_files)
            
    if settings.get_phonon_supercell_matrix() is not None:
        phonon_dds = phono3py.get_phonon_displacement_dataset()
        phonon_supercell = phono3py.get_phonon_supercell()
        if output_filename is None:
            filename = 'disp_fc2.yaml'
        else:
            filename = 'disp_fc2.' + output_filename + '.yaml'
            
        num_disps = write_disp_fc2_yaml(phonon_dds,
                                        phonon_supercell,
                                        filename=filename)
        for i, dcell in enumerate(
                phono3py.get_phonon_supercells_with_displacements()):
            write_vasp('POSCAR_FC2-%05d' % (i + 1), dcell, direct=True)

        if log_level:
            print("Number of displacements for special fc2: %d" % num_disps)
        
    if log_level:
        print_end()
    sys.exit(0)

##############
# Initialize #
##############
mesh = settings.get_mesh_numbers()
mesh_divs = settings.get_mesh_divisors()
grid_points = settings.get_grid_points()
grid_addresses = settings.get_grid_addresses()
if grid_addresses is not None:
    grid_points = [get_grid_point_from_address(ga, mesh)
                   for ga in grid_addresses]
band_indices = settings.get_band_indices()

# Brillouin zone integration: Tetrahedron (default) or smearing method
sigma = settings.get_sigma()
if sigma is None:
    sigmas = []
elif isinstance(sigma, float):
    sigmas = [sigma]
else:
    sigmas = sigma
if settings.get_is_tetrahedron_method():
    sigmas = [None] + sigmas
if len(sigmas) == 0:
    sigmas = [None]

if settings.get_temperatures() is None:
    if options.is_joint_dos:
        temperature_points = None
    else:
        t_max=settings.get_max_temperature()
        t_min=settings.get_min_temperature()
        t_step=settings.get_temperature_step()
        temperature_points = [0.0, 300.0] # For spectra
        temperatures = np.arange(t_min, t_max + float(t_step) / 10, t_step)
else:
    temperature_points = settings.get_temperatures() # For spectra
    temperatures = settings.get_temperatures() # For others
if options.factor is None:
    frequency_factor_to_THz = VaspToTHz
else:
    frequency_factor_to_THz = options.factor
if settings.get_num_frequency_points() is None:
    if settings.get_frequency_pitch() is None:
        num_frequency_points = 201
        frequency_step = None
    else:
        num_frequency_points = None
        frequency_step = settings.get_frequency_pitch()
else:
    num_frequency_points = settings.get_num_frequency_points()
    frequency_step = None
if options.freq_scale is None:
    frequency_scale_factor = 1.0
else:
    frequency_scale_factor = options.freq_scale
if settings.get_cutoff_frequency() is None:
    cutoff_frequency = 1e-2
else:
    cutoff_frequency = settings.get_cutoff_frequency()
if settings.get_is_translational_symmetry():
    tsym_type = 1
elif settings.get_tsym_type() > 0:
    tsym_type = settings.get_tsym_type()
else:
    tsym_type = 0
    
phono3py = Phono3py(
    unitcell,
    settings.get_supercell_matrix(),
    primitive_matrix=settings.get_primitive_matrix(),
    phonon_supercell_matrix=settings.get_phonon_supercell_matrix(),
    masses=settings.get_masses(),
    mesh=mesh,
    band_indices=band_indices,
    sigmas=sigmas,
    cutoff_frequency=cutoff_frequency,
    frequency_factor_to_THz=frequency_factor_to_THz,
    is_symmetry=True,
    is_nosym=options.is_nosym,
    symmetrize_fc3_q=options.is_symmetrize_fc3_q,
    symprec=options.symprec,
    log_level=log_level,
    lapack_zheev_uplo=options.uplo)

supercell = phono3py.get_supercell()
primitive = phono3py.get_primitive()
phonon_supercell = phono3py.get_phonon_supercell()
phonon_primitive = phono3py.get_phonon_primitive()
symmetry = phono3py.get_symmetry()

#################
# Show settings #
#################
if log_level:
    print("Spacegroup: %s" % symmetry.get_international_table())
    print("-" * 30 + " primitive cell " + "-" * 30)
    print_cell(primitive)
    print("-" * 32 + " super cell " + "-" * 32)
    print_cell(supercell, mapping=primitive.get_supercell_to_primitive_map())
    print("-" * 19 + " ratio (supercell for fc)/(primitive) " + "-" * 19)
    for vec in np.dot(supercell.get_cell(),
                      np.linalg.inv(primitive.get_cell())):
        print(("%5.2f" * 3) % tuple(vec))
    if settings.get_phonon_supercell_matrix() is not None:
        print("-" * 19 + " primitive cell for harmonic phonon " + "-" * 20)
        print_cell(phonon_primitive)
        print("-" * 21 + " supercell for harmonic phonon " + "-" * 22)
        print_cell(phonon_supercell,
                   mapping=phonon_primitive.get_supercell_to_primitive_map())
        print("-" * 15 + " ratio (phonon supercell)/(phonon primitive) " +
              "-" * 15)
        for vec in np.dot(phonon_supercell.get_cell(),
                          np.linalg.inv(phonon_primitive.get_cell())):
            print(("%5.2f" * 3) % tuple(vec))

#####################################################  
# Write ir-grid points and grid addresses, and exit #
#####################################################
if options.write_grid_points:
    print("-" * 76)
    if mesh is None:
        print("To write grid points, mesh numbers have to be specified.")
    else:
        (ir_grid_points,
         coarse_grid_weights,
         grid_address) = get_coarse_ir_grid_points(
             primitive,
             mesh,
             mesh_divs,
             settings.get_coarse_mesh_shifts(),
             is_nosym=options.no_kappa_stars,
             symprec=options.symprec)
        write_ir_grid_points(mesh,
                             mesh_divs,
                             ir_grid_points,
                             coarse_grid_weights,
                             grid_address,
                             np.linalg.inv(primitive.get_cell()))
        gadrs_hdf5_fname = write_grid_address_to_hdf5(grid_address, mesh)

        print("Ir-grid points are written into \"ir_grid_points.yaml\".")
        print("Grid addresses are written into \"%s\"." % gadrs_hdf5_fname)

    if log_level:
        print_end()
    sys.exit(0)

if options.show_num_triplets:
    print("-" * 76)

    ir_grid_points, _, grid_address = get_coarse_ir_grid_points(
        primitive,
        mesh,
        mesh_divs,
        settings.get_coarse_mesh_shifts(),
        is_nosym=options.no_kappa_stars,
        symprec=options.symprec)

    if grid_points:
        _grid_points = grid_points
    else:
        _grid_points = ir_grid_points

    print("Grid point        q-point        No. of triplets")
    for gp in _grid_points:
        num_triplets =  get_number_of_triplets(primitive,
                                               mesh,
                                               gp,
                                               symprec=options.symprec)
        q = grid_address[gp] / np.array(mesh, dtype='double')
        print("  %5d     (%5.2f %5.2f %5.2f)  %8d" %
              (gp, q[0], q[1], q[2], num_triplets))
    
    if log_level:
        print_end()
    sys.exit(0)


run_mode = None
if (settings.get_is_isotope() and
    not (settings.get_is_bterta() or settings.get_is_lbte())):
    run_mode = "isotope"
if settings.get_is_bterta() or settings.get_is_lbte():
    run_mode = "conductivity"

###################
# Force constants #
###################
if log_level:
    print("-" * 29 + " Force constants " + "-" * 30)
    if not options.read_fc2:
        print("Imposing translational symmetry to fc2: %s" % 
              (tsym_type > 0))
        print("Imposing symmetry of index exchange to fc2: %s" %
              options.is_symmetrize_fc2)
        
    if not (options.read_fc3 or
            settings.get_is_isotope() or
            options.is_joint_dos):
        print("Imposing translational symmetry to fc3: %s" %
              (tsym_type > 0))
        print("Imposing symmetry of index exchange to fc3 in real space: %s" %
              options.is_symmetrize_fc3_r)
        print(("Imposing symmetry of index exchange to fc3 in reciprocal space: "
              "%s") % options.is_symmetrize_fc3_q)
        
    if settings.get_cutoff_fc3_distance() is not None:
        print("FC3 cutoff distance: %s" % settings.get_cutoff_fc3_distance())
        
#######
# fc3 #
#######
if (options.is_joint_dos or
    (settings.get_is_isotope() and
     not (settings.get_is_bterta() or settings.get_is_lbte())) or
    settings.get_read_gamma() or
    settings.get_read_amplitude() or
    settings.get_constant_averaged_pp_interaction() is not None):
    pass
else:
    if options.read_fc3: # Read fc3.hdf5
        if input_filename is None:
            filename = 'fc3.hdf5'
        else:
            filename = 'fc3.' + input_filename + '.hdf5'
        file_exists(filename, log_level)
        if log_level:
            print("Reading fc3 from %s" % filename)
        fc3 = read_fc3_from_hdf5(filename=filename)
        phono3py.set_fc3(fc3)
    else: # fc3 from FORCES_THIRD and FORCES_SECOND
        if input_filename is None:
            filename = 'disp_fc3.yaml'
        else:
            filename = 'disp_fc3.' + input_filename + '.yaml'
        file_exists(filename, log_level)
        if log_level:
            print("Displacement dataset is read from %s." % filename)
        disp_dataset = parse_disp_fc3_yaml(filename=filename)
        file_exists("FORCES_FC3", log_level)
        if log_level:
            print("Sets of supercell forces are read from %s." % "FORCES_FC3")
        forces_fc3 = parse_FORCES_FC3(disp_dataset)
        phono3py.produce_fc3(
            forces_fc3,
            displacement_dataset=disp_dataset,
            cutoff_distance=settings.get_cutoff_fc3_distance(),
            translational_symmetry_type=tsym_type,
            is_permutation_symmetry=options.is_symmetrize_fc3_r,
            is_permutation_symmetry_fc2=options.is_symmetrize_fc2)
        if output_filename is None:
            filename = 'fc3.hdf5'
        else:
            filename = 'fc3.' + output_filename + '.hdf5'
        if log_level:
            print("Writing fc3 to %s" % filename)
        write_fc3_to_hdf5(phono3py.get_fc3(), filename=filename)

    if log_level:
        show_drift_fc3(phono3py.get_fc3())

##############
# phonon fc2 #
##############
if options.read_fc2:
    if input_filename is None:
        filename = 'fc2.hdf5'
    else:
        filename = 'fc2.' + input_filename + '.hdf5'
    file_exists(filename, log_level)
    if log_level:
        print("Reading fc2 from %s" % filename)
    phonon_fc2 = read_fc2_from_hdf5(filename=filename)
    if phonon_fc2.shape[0] != phonon_supercell.get_number_of_atoms():
        print_error_message("Matrix shape of fc2 doesn't agree with supercell.")
        if log_level:
            print_error()
        sys.exit(1)
    
    phono3py.set_fc2(phonon_fc2)
else:
    if log_level:
        print("Solving fc2")
        
    if settings.get_phonon_supercell_matrix() is None:
        if phono3py.get_fc2() is None:
            if input_filename is None:
                filename = 'disp_fc3.yaml'
            else:
                filename = 'disp_fc3.' + input_filename + '.yaml'
            if log_level:
                print("Displacement dataset is read from %s." % filename)
            file_exists(filename, log_level)
            disp_dataset = parse_disp_fc3_yaml(filename=filename)
            if log_level:
                print("Sets of supercell forces are read from %s." %
                      "FORCES_FC3")
            file_exists("FORCES_FC3", log_level)
            forces_fc3 = parse_FORCES_FC3(disp_dataset)
            phono3py.produce_fc2(
                forces_fc3,
                displacement_dataset=disp_dataset,
                translational_symmetry_type=tsym_type,
                is_permutation_symmetry=options.is_symmetrize_fc2)
    else:
        if input_filename is None:
            filename = 'disp_fc2.yaml'
        else:
            filename = 'disp_fc2.' + input_filename + '.yaml'
        if log_level:
            print("Displacement dataset is read from %s." % filename)
        file_exists(filename, log_level)
        disp_dataset = parse_disp_fc2_yaml(filename=filename)
        if log_level:
            print("Sets of supercell forces are read from %s." %
                  "FORCES_FC2")
        file_exists("FORCES_FC2", log_level)
        forces_fc2 = parse_FORCES_FC2(disp_dataset)
        phono3py.produce_fc2(
            forces_fc2,
            displacement_dataset=disp_dataset,
            translational_symmetry_type=tsym_type,
            is_permutation_symmetry=options.is_symmetrize_fc2)

    if output_filename is None:
        filename = 'fc2.hdf5'
    else:
        filename = 'fc2.' + output_filename + '.hdf5'
    if log_level:
        print("Writing fc2 to %s" % filename)
    write_fc2_to_hdf5(phono3py.get_fc2(), filename=filename)

if log_level:    
    show_drift_force_constants(phono3py.get_fc2(), name='fc2')

if settings.get_is_nac():
    file_exists('BORN', log_level)
    nac_params = parse_BORN(phonon_primitive)
    nac_q_direction = settings.get_nac_q_direction()
else:
    nac_params = None
    nac_q_direction = None

if mesh is None:
    if log_level:
        print_end()
    sys.exit(0)
    
##############################
# Phonon Gruneisen parameter #
##############################
if options.is_gruneisen:
    fc2 = phono3py.get_fc2()
    fc3 = phono3py.get_fc3()
    if len(fc2) != len(fc3):
        print_error_message("Supercells used for fc2 and fc3 have to be same.")
        if log_level:
            print_error()
        sys.exit(1)
    
    band_paths = settings.get_bands()
    qpoints = settings.get_qpoints()
    ion_clamped = settings.get_ion_clamped()

    if (mesh is None and
        band_paths is None and
        qpoints is None):

        print_error_message("An option of --mesh, --band, or --qpoints "
                            "has to be specified.")
        if log_level:
            print_error()
        sys.exit(1)

    if log_level:
        print("------ Phonon Gruneisen parameter ------")
        if mesh is not None:
            print("Mesh sampling: [ %d %d %d ]" % tuple(mesh))
        elif band_paths is not None:
            print("Paths in reciprocal reduced coordinates:")
            for path in band_paths:
                print("[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]" %
                      (tuple(path[0]) + tuple(path[-1])))
        if ion_clamped:
            print("To be calculated with ion clamped.")
            
        sys.stdout.flush()

    gr = get_gruneisen_parameters(fc2,
                                  fc3,
                                  supercell,
                                  primitive,
                                  nac_params=nac_params,
                                  nac_q_direction=nac_q_direction,
                                  ion_clamped=ion_clamped,
                                  factor=VaspToTHz,
                                  symprec=options.symprec)
    if mesh is not None:
        gr.set_sampling_mesh(mesh, is_gamma_center=True)
    elif band_paths is not None:
        gr.set_band_structure(band_paths)
    elif qpoints is not None:
        gr.set_qpoints(qpoints)
    gr.run()

    if output_filename is None:
        filename = 'gruneisen3.yaml'
    else:
        filename = 'gruneisen3.' + output_filename + '.yaml'
    gr.write_yaml(filename=filename)

    if log_level:
        print_end()
    sys.exit(0)

#################
# Show settings #
#################
if log_level:
    print("-" * 33 + " Settings " + "-" * 33)
    if settings.get_is_nac():
        print("Non-analytical term correction: %s" % settings.get_is_nac())
    if mesh is not None:
        print("Mesh sampling: [ %d %d %d ]" % tuple(mesh))
    if mesh_divs is not None and settings.get_is_bterta():
        print("Mesh divisors: [ %d %d %d ]" % tuple(mesh_divs))
    if band_indices is not None and not settings.get_is_bterta():
        print(("Band indices: [" + " %s" * len(band_indices) + " ]") %
              tuple([np.array(bi) + 1 for bi in band_indices]))
    if sigmas:
        text = "BZ integration: "
        for i, sigma in enumerate(sigmas):
            if sigma:
                text += "Smearing=%s" % sigma
            else:
                text += "Tetrahedron-method"
            if i < len(sigmas) - 1:
                text += ", "
        print(text)
        
    if (settings.get_is_linewidth() or
        settings.get_is_frequency_shift() or
        settings.get_is_bterta() or
        settings.get_is_lbte()):
        if len(temperatures) > 5:
            text = (" %.1f " * 5 + "...") % tuple(temperatures[:5])
            text += " %.1f" % temperatures[-1]
        else:
            text = (" %.1f " * len(temperatures)) % tuple(temperatures)
        print("Temperature: " + text)
    elif temperature_points is not None:
        print(("Temperatures:" + " %.1f " * len(temperature_points))
              % tuple(temperature_points))
        if settings.get_scattering_event_class() is not None:
            print("Scattering event class: %s" %
                  settings.get_scattering_event_class())
            
    if grid_points is not None:
        text = "Grid point to be calculated: "
        if len(grid_points) > 8:
            for i, gp in enumerate(grid_points):
                if i % 10 == 0:
                    text += "\n"
                    text += " "
                text += "%d " % gp
        else:
            for gp in grid_points:
                text += "%d " % gp
        print(text)
            
    if cutoff_frequency:
        print("Cutoff frequency: %s" % cutoff_frequency)

    if settings.get_average_pp_interaction() and run_mode == "conductivity":
        print("Use averaged ph-ph interaction")

    
        
    if log_level > 1:
        print("Frequency factor to THz: %s" % frequency_factor_to_THz)
        if frequency_step is not None:
            print("Frequency step for spectrum: %s" % frequency_step)
        if num_frequency_points is not None:
            print("Number of frequency sampling points: %d" %
                  num_frequency_points)
    sys.stdout.flush()
    
#############
# Joint DOS #
#############
if options.is_joint_dos:
    joint_dos = Phono3pyJointDos(
        phonon_supercell,
        phonon_primitive,
        mesh,
        phono3py.get_fc2(),
        nac_params=nac_params,
        nac_q_direction=nac_q_direction,
        sigmas=sigmas,
        cutoff_frequency=cutoff_frequency,
        frequency_step=frequency_step,
        num_frequency_points=num_frequency_points,
        temperatures=temperature_points,
        frequency_factor_to_THz=frequency_factor_to_THz,
        frequency_scale_factor=frequency_scale_factor,
        is_nosym=options.is_nosym,
        symprec=options.symprec,
        output_filename=output_filename,
        log_level=log_level)
    joint_dos.run(grid_points)
    if log_level:
        print_end()
    sys.exit(0)
    
######################
# Isotope scattering #
######################
if settings.get_is_isotope() and settings.get_mass_variances() is None:
    from phonopy.structure.atoms import isotope_data
    symbols = phonon_primitive.get_chemical_symbols()
    in_database = True
    for s in set(symbols):
        if not s in isotope_data:
            print("%s is not in the list of isotope databese" % s)
            print("(not implemented).")
            print("Use --mass_variances option.")
            in_database = False
    if not in_database:
        if log_level:
            print_end()
        sys.exit(0)

if run_mode == "isotope":
    mass_variances = settings.get_mass_variances()
    if band_indices is not None:
        band_indices = np.hstack(band_indices).astype('intc')
    iso = Phono3pyIsotope(
        mesh,
        phonon_primitive,
        mass_variances=mass_variances,
        band_indices=band_indices,
        sigmas=sigmas,
        frequency_factor_to_THz=frequency_factor_to_THz,
        symprec=options.symprec,
        cutoff_frequency=settings.get_cutoff_frequency(),
        lapack_zheev_uplo=options.uplo)
    iso.set_dynamical_matrix(phono3py.get_fc2(),
                             phonon_supercell,
                             phonon_primitive,
                             nac_params=nac_params,
                             frequency_scale_factor=frequency_scale_factor)
    iso.run(grid_points)
    if log_level:
        print_end()
    sys.exit(0)
            
#########
# Ph-ph #
#########
ave_pp = settings.get_constant_averaged_pp_interaction()
phono3py.set_phph_interaction(
    nac_params=nac_params,
    nac_q_direction=nac_q_direction,
    constant_averaged_interaction=ave_pp,
    frequency_scale_factor=frequency_scale_factor,
    unit_conversion=options.pp_unit_conversion)

if settings.get_is_linewidth():
    if grid_points is None:
        print_error_message("Grid point(s) has to be specified with --gp or "
                            "--ga option.")
        if log_level:
            print_error()
        sys.exit(1)
    phono3py.run_linewidth(
        grid_points,
        temperatures=temperatures,
        run_with_g=settings.get_run_with_g(),
        write_details=settings.get_write_detailed_gamma())
    phono3py.write_linewidth(filename=output_filename)
elif settings.get_is_imag_self_energy():
    if not settings.get_run_with_g() and settings.get_scattering_event_class():
        print_error_message("--run_without_g and --scattering_event_class can "
                            "not used together.")
        if log_level:
            print_error()
        sys.exit(1)
    if grid_points is None:
        print_error_message("Grid point(s) has to be specified with --gp or "
                            "--ga option.")
        if log_level:
            print_error()
        sys.exit(1)
    phono3py.run_imag_self_energy(
        grid_points,
        frequency_step=frequency_step,
        num_frequency_points=num_frequency_points,
        temperatures=temperature_points,
        scattering_event_class=settings.get_scattering_event_class(),
        run_with_g=settings.get_run_with_g(),
        write_details=settings.get_write_detailed_gamma())
    phono3py.write_imag_self_energy(filename=output_filename)
elif settings.get_is_frequency_shift():
    phono3py.get_frequency_shift(
        grid_points,
        epsilons=sigmas,
        temperatures=temperatures,
        output_filename=output_filename)
elif settings.get_is_bterta() or settings.get_is_lbte():
    phono3py.run_thermal_conductivity(
        is_LBTE=settings.get_is_lbte(),
        temperatures=temperatures,
        sigmas=sigmas,
        is_isotope=settings.get_is_isotope(),
        mass_variances=settings.get_mass_variances(),
        grid_points=grid_points,
        boundary_mfp=settings.get_boundary_mfp(),
        use_averaged_pp_interaction=settings.get_average_pp_interaction(),
        gamma_unit_conversion=options.gamma_unit_conversion,
        mesh_divisors=mesh_divs,
        coarse_mesh_shifts=settings.get_coarse_mesh_shifts(),
        is_reducible_collision_matrix=options.is_reducible_collision_matrix,
        no_kappa_stars=settings.get_no_kappa_stars(),
        gv_delta_q=settings.get_group_velocity_delta_q(),
        run_with_g=settings.get_run_with_g(),
        pinv_cutoff=settings.get_pinv_cutoff(),
        write_gamma=settings.get_write_gamma(),
        read_gamma=settings.get_read_gamma(),
        write_collision=settings.get_write_collision(),
        read_collision=settings.get_read_collision(),
        input_filename=input_filename,
        output_filename=output_filename)
else:
    if log_level:
        print("** None of anharmonic properties were calculated. **")

if log_level:
    print_end()
