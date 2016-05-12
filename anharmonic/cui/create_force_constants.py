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

import sys
from phonopy.harmonic.force_constants import show_drift_force_constants
from anharmonic.phonon3.fc3 import show_drift_fc3
from anharmonic.file_IO import (parse_disp_fc3_yaml,
                                parse_disp_fc2_yaml,
                                parse_FORCES_FC2,
                                parse_FORCES_FC3,
                                read_fc3_from_hdf5,
                                read_fc2_from_hdf5,
                                write_fc3_to_hdf5,
                                write_fc2_to_hdf5)
from anharmonic.cui.show_log import (show_phono3py_force_constants_settings,
                                     print_error, file_exists)

def create_phono3py_force_constants(phono3py,
                                    phonon_supercell_matrix,
                                    settings,
                                    energy_to_eV=None,
                                    distance_to_A=None,
                                    input_filename=None,
                                    output_filename=None,
                                    log_level=1):
    read_fc3 = settings.get_read_fc3()
    read_fc2 = settings.get_read_fc2()
    symmetrize_fc3_r = settings.get_is_symmetrize_fc3_r()
    symmetrize_fc3_q = settings.get_is_symmetrize_fc3_q()
    symmetrize_fc2 = settings.get_is_symmetrize_fc2()
    if settings.get_is_translational_symmetry():
        tsym_type = 1
    elif settings.get_tsym_type() > 0:
        tsym_type = settings.get_tsym_type()
    else:
        tsym_type = 0

    if log_level:
        show_phono3py_force_constants_settings(read_fc3,
                                               read_fc2,
                                               tsym_type,
                                               symmetrize_fc3_r,
                                               symmetrize_fc3_q,
                                               symmetrize_fc2,
                                               settings)

    # fc3
    if (settings.get_is_joint_dos() or
        (settings.get_is_isotope() and
         not (settings.get_is_bterta() or settings.get_is_lbte())) or
        settings.get_read_gamma() or
        settings.get_read_amplitude() or
        settings.get_constant_averaged_pp_interaction() is not None):
        pass
    else:
        if read_fc3: # Read fc3.hdf5
            if input_filename is None:
                filename = 'fc3.hdf5'
            else:
                filename = 'fc3.' + input_filename + '.hdf5'
            file_exists(filename, log_level)
            if log_level:
                print("Reading fc3 from %s" % filename)
            fc3 = read_fc3_from_hdf5(filename=filename)
            num_atom = phono3py.get_supercell().get_number_of_atoms()
            if fc3.shape[0] != num_atom:
                print("Matrix shape of fc3 doesn't agree with supercell size.")
                if log_level:
                    print_error()
                sys.exit(1)
            phono3py.set_fc3(fc3)
        else: # fc3 from FORCES_FC3
            if not _create_phono3py_fc3(phono3py,
                                        energy_to_eV,
                                        distance_to_A,
                                        tsym_type,
                                        symmetrize_fc3_r,
                                        symmetrize_fc2,
                                        settings.get_cutoff_fc3_distance(),
                                        input_filename,
                                        output_filename,
                                        log_level):
                    print("fc3 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)
        if log_level:
            show_drift_fc3(phono3py.get_fc3())
    
    # fc2
    if read_fc2:
        if input_filename is None:
            filename = 'fc2.hdf5'
        else:
            filename = 'fc2.' + input_filename + '.hdf5'
        file_exists(filename, log_level)
        if log_level:
            print("Reading fc2 from %s" % filename)
        num_atom = phono3py.get_phonon_supercell().get_number_of_atoms()
        phonon_fc2 = read_fc2_from_hdf5(filename=filename)
        if phonon_fc2.shape[0] != num_atom:
            print("Matrix shape of fc2 doesn't agree with supercell size.")
            if log_level:
                print_error()
            sys.exit(1)
        
        phono3py.set_fc2(phonon_fc2)
    else:
        if log_level:
            print("Solving fc2")
            
        if phonon_supercell_matrix is None:
            if phono3py.get_fc2() is None:
                if not _create_phono3py_fc2(phono3py,
                                            energy_to_eV,
                                            distance_to_A,
                                            tsym_type,
                                            symmetrize_fc2,
                                            input_filename,
                                            log_level):
                    print("fc2 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)

        else:
            if not _create_phono3py_phonon_fc2(phono3py,
                                               energy_to_eV,
                                               distance_to_A,
                                               tsym_type,
                                               symmetrize_fc2,
                                               input_filename,
                                               log_level):
                    print("fc2 was not created properly.")
                    if log_level:
                        print_error()
                    sys.exit(1)
        if output_filename is None:
            filename = 'fc2.hdf5'
        else:
            filename = 'fc2.' + output_filename + '.hdf5'
        if log_level:
            print("Writing fc2 to %s" % filename)
        write_fc2_to_hdf5(phono3py.get_fc2(), filename=filename)
    
    if log_level:    
        show_drift_force_constants(phono3py.get_fc2(), name='fc2')

def _create_phono3py_fc3(phono3py,
                         energy_to_eV,
                         distance_to_A,
                         tsym_type,
                         symmetrize_fc3_r,
                         symmetrize_fc2,
                         cutoff_distance,
                         input_filename,
                         output_filename,
                         log_level):
    if input_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + input_filename + '.yaml'
    file_exists(filename, log_level)
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    disp_dataset = parse_disp_fc3_yaml(filename=filename)
    num_atom = phono3py.get_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A)

    file_exists("FORCES_FC3", log_level)
    if log_level:
        print("Sets of supercell forces are read from %s." % "FORCES_FC3")
    forces_fc3 = parse_FORCES_FC3(disp_dataset)
    if not forces_fc3:
        return False

    _convert_force_unit(forces_fc3, energy_to_eV, distance_to_A)

    phono3py.produce_fc3(
        forces_fc3,
        displacement_dataset=disp_dataset,
        cutoff_distance=cutoff_distance,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=symmetrize_fc3_r,
        is_permutation_symmetry_fc2=symmetrize_fc2)
    if output_filename is None:
        filename = 'fc3.hdf5'
    else:
        filename = 'fc3.' + output_filename + '.hdf5'
    if log_level:
        print("Writing fc3 to %s" % filename)
    write_fc3_to_hdf5(phono3py.get_fc3(), filename=filename)

    return True

def _create_phono3py_fc2(phono3py,
                         energy_to_eV,
                         distance_to_A,
                         tsym_type,
                         symmetrize_fc2,
                         input_filename,
                         log_level):
    if input_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + input_filename + '.yaml'
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    file_exists(filename, log_level)
    disp_dataset = parse_disp_fc3_yaml(filename=filename)
    num_atom = phono3py.get_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=True)

    if log_level:
        print("Sets of supercell forces are read from %s." % "FORCES_FC3")
    file_exists("FORCES_FC3", log_level)
    forces_fc2 = parse_FORCES_FC2(disp_dataset, filename="FORCES_FC3")
    if not forces_fc2:
        return False

    _convert_force_unit(forces_fc2, energy_to_eV, distance_to_A)

    phono3py.produce_fc2(
        forces_fc2,
        displacement_dataset=disp_dataset,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=symmetrize_fc2)

    return True

def _create_phono3py_phonon_fc2(phono3py,
                                energy_to_eV,
                                distance_to_A,
                                tsym_type,
                                symmetrize_fc2,
                                input_filename,
                                log_level):
    if input_filename is None:
        filename = 'disp_fc2.yaml'
    else:
        filename = 'disp_fc2.' + input_filename + '.yaml'
    if log_level:
        print("Displacement dataset is read from %s." % filename)
    file_exists(filename, log_level)
    disp_dataset = parse_disp_fc2_yaml(filename=filename)
    num_atom = phono3py.get_phonon_supercell().get_number_of_atoms()
    if disp_dataset['natom'] != num_atom:
        print("Number of atoms in supercell is not consistent with %s" %
              filename)
        if log_level:
            print_error()
        sys.exit(1)
    _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=True)

    if log_level:
        print("Sets of supercell forces are read from %s." %
              "FORCES_FC2")
    file_exists("FORCES_FC2", log_level)
    forces_fc2 = parse_FORCES_FC2(disp_dataset)
    if not forces_fc2:
        return False

    _convert_force_unit(forces_fc2, energy_to_eV, distance_to_A)

    phono3py.produce_fc2(
        forces_fc2,
        displacement_dataset=disp_dataset,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=symmetrize_fc2)

    return True

def _convert_force_unit(force_sets, energy_to_eV, distance_to_A):
    if energy_to_eV is not None and distance_to_A is not None:
        if energy_to_eV is None:
            unit_conversion_factor = 1.0 / distance_to_A
        elif distance_to_A is None:
            unit_conversion_factor = energy_to_eV
        else:
            unit_conversion_factor = energy_to_eV / distance_to_A
        for forces in force_sets:
            if forces is not None:
                forces *= unit_conversion_factor

def _convert_displacement_unit(disp_dataset, distance_to_A, is_fc2=False):
    if distance_to_A is not None:
        for first_atom in disp_dataset['first_atoms']:
            for i in range(3):
                first_atom['displacement'][i] *= distance_to_A
            if not is_fc2:
                for second_atom in first_atom['second_atoms']:
                    for i in range(3):
                        second_atom['displacement'][i] *= distance_to_A
