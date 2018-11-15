# Copyright (C) 2018 Atsushi Togo
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
from phonopy.harmonic.force_constants import distribute_force_constants


def get_fc2(supercell,
            primitive,
            disp_dataset,
            atom_list=None,
            log_level=0):
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    numbers = supercell.get_atomic_numbers()
    natom = len(numbers)
    disps, forces = _collect_disps_and_forces(disp_dataset)
    p2s_map = primitive.p2s_map
    p2p_map = primitive.p2p_map

    if log_level:
        print("-------------------------------"
              " ALM FC2 start "
              "------------------------------")
        print("ALM by T. Tadano, https://github.com/ttadano/ALM")
        if log_level == 1:
            print("Use -v option to watch detailed ALM log.")

    try:
        from alm import ALM
    except ImportError:
        raise ModuleNotFoundError("ALM python module was not found.")

    sys.stdout.flush()
    with ALM(lattice, positions, numbers) as alm:
        if log_level > 0:
            log_level_alm = log_level - 1
        else:
            log_level_alm = 0
        alm.set_verbosity(log_level_alm)
        alm.define(1)
        alm.set_displacement_and_force(disps, forces)
        alm.optimize()
        if (atom_list == p2s_map).all():
            fc2 = np.zeros((len(p2s_map), natom, 3, 3),
                           dtype='double', order='C')
            for fc, indices in zip(*alm.get_fc(1, mode='origin')):
                v1, v2 = indices // 3
                c1, c2 = indices % 3
                fc2[p2p_map[v1], v2, c1, c2] = fc
        elif atom_list is None or (atom_list == np.range(natom)):
            fc2 = np.zeros((natom, natom, 3, 3), dtype='double', order='C')
            for fc, indices in zip(*alm.get_fc(1, mode='all')):
                v1, v2 = indices // 3
                c1, c2 = indices % 3
                fc2[v1, v2, c1, c2] = fc
        else:  # This case would not happen.
            fc2 = np.zeros((natom, natom, 3, 3), dtype='double', order='C')
            for fc, indices in zip(*alm.get_fc(1, mode='origin')):
                v1, v2 = indices // 3
                c1, c2 = indices % 3
                fc2[v1, v2, c1, c2] = fc
            N = natom // primitive.get_number_of_atoms()
            rotations = np.array([np.eye(3, dtype='intc'), ] * N,
                                 dtype='intc', order='C')
            distribute_force_constants(fc2,
                                       p2s_map,
                                       lattice,
                                       rotations,
                                       primitive.atomic_permutations,
                                       atom_list=atom_list)
            fc2 = np.array(fc2[atom_list], dtype='double', order='C')

    if log_level:
        print("--------------------------------"
              " ALM FC2 end "
              "-------------------------------")

    return fc2


def _collect_disps_and_forces(disp_dataset):
    if 'first_atoms' in disp_dataset:
        natom = disp_dataset['natom']
        disps = np.zeros((len(disp_dataset['first_atoms']), natom, 3),
                         dtype='double', order='C')
        forces = np.zeros_like(disps)
        for i, disp1 in enumerate(disp_dataset['first_atoms']):
            if 'forces' in disp1:
                disps[i, disp1['number']] = disp1['displacement']
                forces[i] = disp1['forces']
            else:
                return [], []
        return disps, forces
    elif 'forces' in disp_dataset and 'displacements' in disp_dataset:
        return disp_dataset['displacements'], disp_dataset['forces']
