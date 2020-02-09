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
from hiphive.input_output.logging_tools import set_config
set_config(level=30)


def phonopy_atoms_to_ase(atoms_phonopy):
    ase_atoms = Atoms(cell=atoms_phonopy.cell,
                      scaled_positions=atoms_phonopy.scaled_positions,
                      numbers=atoms_phonopy.numbers,
                      pbc=True)
    return ase_atoms


def get_fc2(supercell,
            primitive,
            displacements,
            forces,
            symprec,
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

    fc2 = run_hiphive(supercell=supercell,
                      primitive=primitive,
                      displacements=displacements,
                      forces=forces,
                      options=options,
                      symprec=symprec,
                      log_level=log_level)

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
                symprec,
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

    ase_supercell = phonopy_atoms_to_ase(supercell)
    ase_prim = phonopy_atoms_to_ase(primitive)

    # setup training structures
    structures = []
    for d, f in zip(displacements, forces):
        structure = ase_supercell.copy()
        structure.new_array('displacements', d)
        structure.new_array('forces', f)
        structures.append(structure)

    # parse options
    if options is None:
        options_dict = {}
    else:
        options_dict = _decode_options(options)

    # select cutoff
    max_cutoff = estimate_maximum_cutoff(ase_supercell) - 1e-5
    if 'cutoff' in options_dict:
        cutoff = options_dict['cutoff']
        if cutoff > max_cutoff:
            raise ValueError('Cutoff {:.4f} is larger than maximum allowed '
                             'cutoff, {:.4f}, for the given supercell.'
                             '\nDecrease cutoff or provide larger supercells.'
                             .format(cutoff, max_cutoff))
    else:
        cutoff = max_cutoff

    # setup ClusterSpace
    cutoffs = [cutoff]
    cs = ClusterSpace(ase_prim, cutoffs, symprec=symprec)
    cs.print_orbits()

    sc = StructureContainer(cs)
    for structure in structures:
        sc.add_structure(structure)
    n_rows, n_cols = sc.data_shape
    if n_rows < n_cols:
        raise ValueError('Fitting problem is under-determined.'
                         '\nProvide more structures or decrease cutoff.')

    # Estimate error
    opt = Optimizer(sc.get_fit_data(), train_size=0.75)
    opt.train()
    print(opt)
    print('RMSE train : {:.4f}'.format(opt.rmse_train))
    print('RMSE test  : {:.4f}'.format(opt.rmse_test))

    # Final train
    opt = Optimizer(sc.get_fit_data(), train_size=1.0)
    opt.train()

    # get force constants
    fcp = ForceConstantPotential(cs, opt.parameters)
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
