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

import numpy as np
from ase.atoms import Atoms
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.fitting import Optimizer
from hiphive.cutoffs import estimate_maximum_cutoff


def get_fc2(supercell,
            primitive,
            displacements,
            forces,
            atom_list=None,
            options=None,
            log_level=0):
    if log_level:
        msg = [
            "-------------------------------"
            " hiPhive start "
            "------------------------------",
            "hiPhive is a non-trivial force constants calculator. "
            "Please cite the paper:",
            "\"The Hiphive Package for the Extraction of High‐Order Force "
            "Constants",
            " by Machine Learning\"",
            "by Fredrik Eriksson, Erik Fransson, and Paul Erhart,",
            "Advanced Theory and Simulations, DOI:10.1002/adts.201800184 "
            "(2019)",
            ""]
        print("\n".join(msg))

    fc2 = run_hiphive(supercell,
                      primitive,
                      displacements,
                      forces,
                      options,
                      log_level)

    p2s_map = primitive.p2s_map
    is_compact_fc = (atom_list is not None and
                     (atom_list == p2s_map).all())
    if is_compact_fc:
        fc2 = np.array(fc2[p2s_map], dtype='double', order='C')
    elif atom_list is not None:
        fc2 = np.array(fc2[atom_list], dtype='double', order='C')

    if log_level:
        print("--------------------------------"
              " hiPhive end "
              "-------------------------------")

    return fc2


def run_hiphive(supercell,
                primitive,
                displacements,
                forces,
                options,
                log_level):
    """Run hiphive

    supercell : Supercell
        Perfect supercell.
    primitive : Primitive
        Primitive cell.
    displacements : ndarray
        Displacements of atoms in supercell.
        shape=(supercells, natom, 3)
    forces : ndarray
        Forces on atoms in supercell.
        shape=(supercells, natom, 3)
    options : str
        Force constants calculation options.
    log_level : int
        Log control. 0: quiet, 1: normal, 2: verbose 3: debug

    """

    if options is None:
        options_dict = {}
    else:
        options_dict = _decode_options(options)

    ase_supercell = Atoms(cell=supercell.cell,
                          scaled_positions=supercell.scaled_positions,
                          numbers=supercell.numbers,
                          pbc=True)
    structures = []
    for d, f in zip(displacements, forces):
        structure = ase_supercell.copy()
        structure.new_array('displacements', d)
        structure.new_array('forces', f)
        structure.calc = None
        structures.append(structure)

    if 'cutoff' in options_dict:
        cutoff = options_dict['cutoff']
    else:
        cutoff = estimate_maximum_cutoff(ase_supercell) - 1e-5
    cutoffs = [cutoff, ]
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

    print('RMSE train : {:.4f}'.format(opt.rmse_train))
    print('RMSE test  : {:.4f}'.format(opt.rmse_test))

    fcp = ForceConstantPotential(cs, opt.parameters)
    print(fcp)

    fcs = fcp.get_force_constants(ase_supercell)
    fc2 = fcs.get_fc_array(order=2)

    return fc2


def _decode_options(options):
    """This is an example to parse options given in str.

    When options = 'cutoff = 4.0', options is converted to {'cutoff': 4.0}.

    In this implementation (can be modified), using phonopy command line
    options, ``options`` is passed by  --fc-calc-opt such as::

       phonopy --hiphiveph --fc-calc-opt "cutoff = 4" ...

    """

    option_dict = {}
    for pair in options.split(','):
        key, value = [x.strip() for x in pair.split('=')]
        if key == 'cutoff':
            option_dict[key] = float(value)
    return option_dict
