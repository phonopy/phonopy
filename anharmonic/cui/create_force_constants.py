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

from anharmonic.file_IO import (file_exists,
                                parse_disp_fc3_yaml,
                                parse_disp_fc2_yaml,
                                parse_FORCES_FC2,
                                parse_FORCES_FC3,
                                write_fc3_to_hdf5)

def create_phono3py_fc3(phono3py,
                        tsym_type,
                        is_symmetrize_fc3_r,
                        is_symmetrize_fc2,
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
    file_exists("FORCES_FC3", log_level)
    if log_level:
        print("Sets of supercell forces are read from %s." % "FORCES_FC3")
    forces_fc3 = parse_FORCES_FC3(disp_dataset)
    phono3py.produce_fc3(
        forces_fc3,
        displacement_dataset=disp_dataset,
        cutoff_distance=cutoff_distance,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=is_symmetrize_fc3_r,
        is_permutation_symmetry_fc2=is_symmetrize_fc2)
    if output_filename is None:
        filename = 'fc3.hdf5'
    else:
        filename = 'fc3.' + output_filename + '.hdf5'
    if log_level:
        print("Writing fc3 to %s" % filename)
    write_fc3_to_hdf5(phono3py.get_fc3(), filename=filename)

def create_phono3py_fc2(phono3py,
                        tsym_type,
                        is_symmetrize_fc2,
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
    if log_level:
        print("Sets of supercell forces are read from %s." %
              "FORCES_FC3")
    file_exists("FORCES_FC3", log_level)
    forces_fc3 = parse_FORCES_FC3(disp_dataset)
    phono3py.produce_fc2(
        forces_fc3,
        displacement_dataset=disp_dataset,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=is_symmetrize_fc2)

def create_phono3py_phonon_fc2(phono3py,
                               tsym_type,
                               is_symmetrize_fc2,
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
    if log_level:
        print("Sets of supercell forces are read from %s." %
              "FORCES_FC2")
    file_exists("FORCES_FC2", log_level)
    forces_fc2 = parse_FORCES_FC2(disp_dataset)
    phono3py.produce_fc2(
        forces_fc2,
        displacement_dataset=disp_dataset,
        translational_symmetry_type=tsym_type,
        is_permutation_symmetry=is_symmetrize_fc2)
