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
from ase.atoms import Atoms
from ase.io import write
from ase.build import bulk
from ase.calculators.emt import EMT
from hiphive.structure_generation import generate_mc_rattled_structures
from hiphive.utilities import get_displacements
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer


def get_fc2(supercell,
            primitive,
            displacements,
            forces,
            atom_list=None,
            options=None,
            log_level=0):
    if options is None:
        option_dict = {}
    else:
        option_dict = decode_options(options)

    structures = []
    for d, f in zip(displacements, forces):
        structure = Atoms(cell=supercell.cell,
                          scaled_positions=supercell.scaled_positions,
                          numbers=supercell.numbers,
                          pbc=True)
        structure.new_array('displacements', d)
        structure.new_array('forces', f)
        structure.calc = None
        structures.append(structure)

    cutoffs = [option_dict['cutoff'], ]
    cs = ClusterSpace(structures[0], cutoffs)
    print(cs)
    cs.print_orbits()

    sc = StructureContainer(cs)
    for structure in structures:
        sc.add_structure(structure)
    print(sc)

    opt = Optimizer(sc.get_fit_data())
    opt.train()
    print(opt)

    fcp = ForceConstantPotential(cs, opt.parameters)
    print(fcp)


def decode_options(options):
    option_dict = {}
    for pair in options.split(','):
        key, value = [x.strip() for x in pair.split('=')]
        if key == 'cutoff':
            option_dict[key] = float(value)
    return option_dict
