# Copyright (C) 2020 Atsushi Togo
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


import numpy as np
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.calculator import get_force_sets, get_force_sets_wien2k
from phonopy.file_IO import parse_disp_yaml, write_FORCE_SETS


def create_FORCE_SETS(interface_mode,
                      force_filenames,
                      symmetry_tolerance=None,
                      wien2k_P1_mode=False,
                      force_sets_zero_mode=False,
                      disp_filename='disp.yaml',
                      force_sets_filename='FORCE_SETS',
                      write_forcesets_yaml=False,
                      log_level=0):
    """Create FORCE_SETS from phonopy_disp.yaml and calculator output files.

    Reading disp.yaml instead of phonopy_disp.yaml is deprecated.

    """

    if log_level > 0:
        if interface_mode:
            print("Calculator interface: %s" % interface_mode)
        print("Displacements were read from \"%s\"." % disp_filename)
        if disp_filename == 'disp.yaml':
            print('')
            print("NOTE:")
            print("  From phonopy v2.0, displacements are written into "
                  "\"phonopy_disp.yaml\".")
            print("  \"disp.yaml\" is still supported for reading except for "
                  "Wien2k interface, ")
            print("  but is deprecated.")
            print('')
        if force_sets_zero_mode:
            print("Forces in %s are subtracted from forces in all "
                  "other files." % force_filenames[0])

    if disp_filename == 'disp.yaml':
        if interface_mode == 'wien2k':
            disp_dataset, supercell = parse_disp_yaml(filename=disp_filename,
                                                      return_cell=True)
        else:
            disp_dataset = parse_disp_yaml(filename=disp_filename)
    else:
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(disp_filename)
        supercell = phpy_yaml.supercell
        disp_dataset = phpy_yaml.dataset

    if 'natom' in disp_dataset:  # type-1 dataset
        num_atoms = disp_dataset['natom']
        num_displacements = len(disp_dataset['first_atoms'])
        dataset_type = 1
    elif 'displacements' in disp_dataset:  # type-2 dataset
        num_atoms = disp_dataset['displacements'].shape[1]
        num_displacements = disp_dataset['displacements'].shape[0]
        dataset_type = 2
    else:
        raise RuntimeError("Number of atoms could not be retrieved from %s"
                           % disp_filename)
    if force_sets_zero_mode:
        num_displacements += 1

    if not check_number_of_force_files(num_displacements,
                                       force_filenames,
                                       disp_filename):
        force_sets = []
    elif interface_mode == 'wien2k':
        force_sets = get_force_sets_wien2k(
            num_displacements,
            force_filenames,
            disp_filename,
            supercell,
            disp_dataset,
            wien2k_P1_mode=wien2k_P1_mode,
            symmetry_tolerance=symmetry_tolerance,
            verbose=(log_level > 0))
    else:
        force_sets = get_force_sets(interface_mode,
                                    num_atoms,
                                    num_displacements,
                                    force_filenames,
                                    disp_filename=disp_filename,
                                    verbose=(log_level > 0))

    if force_sets:
        if force_sets_zero_mode:
            force_sets = _subtract_residual_forces(force_sets)
        if dataset_type == 1:
            for forces, disp in zip(force_sets, disp_dataset['first_atoms']):
                disp['forces'] = forces
        elif dataset_type == 2:
            disp_dataset['forces'] = np.array(
                force_sets, dtype='double', order='C')
        else:
            raise RuntimeError("FORCE_SETS could not be created.")

        write_FORCE_SETS(disp_dataset, filename=force_sets_filename)

        if log_level > 0:
            print("\"%s\" has been created." % force_sets_filename)

        if disp_filename != 'disp.yaml' and write_forcesets_yaml:
            phpy_yaml.dataset = disp_dataset
            with open("phonopy_force_sets.yaml", 'w') as w:
                w.write(str(phpy_yaml))
            if log_level > 0:
                print("\"%s\" has been created." % "phonopy_force_sets.yaml")

    else:
        if log_level > 0:
            print("%s could not be created." % force_sets_filename)


def check_number_of_force_files(num_displacements,
                                force_filenames,
                                disp_filename):
    if num_displacements != len(force_filenames):
        print('')
        print("Number of files to be read (%d) don't match to" %
              len(force_filenames))
        print("the number of displacements (%d) in %s." %
              (num_displacements, disp_filename))
        return False
    else:
        return True


def _subtract_residual_forces(force_sets):
    for i in range(1, len(force_sets)):
        force_sets[i] -= force_sets[0]
    return force_sets[1:]
