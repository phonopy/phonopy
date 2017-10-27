# vim: set fileencoding=utf-8 :
# Copyright (C) 2017 Tiziano Müller
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

from phonopy.file_IO import collect_forces
from phonopy.interface.vasp import (check_forces, get_drift_forces)
from phonopy.structure.atoms import (PhonopyAtoms, symbol_map)
from phonopy.units import Bohr


def parse_set_of_forces(num_atoms, forces_filenames, verbose=True):
    hook = '# Atom   Kind   Element'

    force_sets = []

    for i, filename in enumerate(forces_filenames):
        if verbose:
            sys.stdout.write("%d. " % (i + 1))

        with open(filename) as fhandle:
            cp2k_forces = collect_forces(fhandle, num_atoms, hook, [3, 4, 5])
            if check_forces(cp2k_forces, num_atoms, filename, verbose=verbose):
                drift_force = get_drift_forces(cp2k_forces,
                                               filename=filename,
                                               verbose=verbose)
                force_sets.append(np.array(cp2k_forces) - drift_force)
            else:
                return []  # if one file is invalid, the whole thing is broken

    return force_sets


def read_cp2k(filename):
    from cp2k_tools.parser import CP2KInputParser

    with open(filename) as fhandle:
        parser = CP2KInputParser()
        cp2k_in = parser.parse(fhandle)

    cp2k_cell = cp2k_in['force_eval']['subsys']['cell']
    cp2k_coord = cp2k_in['force_eval']['subsys']['coord']

    if 'abc' in cp2k_cell:
        unit = '[angstrom]'  # CP2K default unit
        unit_cell = np.eye(3)
        if isinstance(cp2k_cell['abc'][0], str):
            unit = cp2k_cell['abc'][0]
            unit_cell *= cp2k_cell['abc'][1:]
        else:
            unit_cell *= cp2k_cell['abc']

        if unit == '[angstrom]':
            unit_cell /= Bohr  # phonopy expects the lattice to be in Bohr
        else:
            raise NotImplementedError("unit scaling for other units than angstrom not yet implemented")
    else:
        raise NotImplementedError("unit cell can only be specified via ABC")

    # the keyword can be unavailable, true, false or None, with unavailable=false, None=true
    scaled_coords = cp2k_coord.get('scaled', False) is not False

    if not scaled_coords:
        raise NotImplementedError("only scaled coordinates are currently supported")


    numbers = [symbol_map[e[0]] for e in cp2k_coord['*']]
    positions = [e[1:] for e in cp2k_coord['*']]

    return PhonopyAtoms(numbers=numbers, cell=unit_cell, scaled_positions=positions)


def write_cp2k(filename, cell):
    with open(filename, 'w') as fhandle:
        fhandle.write(get_cp2k_structure(cell))


def write_supercells_with_displacements(supercell,
                                        cells_with_displacements,
                                        pre_filename="supercell",
                                        width=3):

    write_cp2k("supercell.inp", supercell)

    for i, cell in enumerate(cells_with_displacements):
        filename = "{pre_filename}-{0:0{width}}.inp".format(
            i + 1,
            pre_filename=pre_filename,
            width=width)
        write_cp2k(filename, cell)


def get_cp2k_structure(atoms):
    """Convert the atoms structure to a CP2K input file skeleton string"""

    from cp2k_tools.generator import dict2cp2k

    # CP2K's default unit is angstrom, convert it, but still declare it explictly:
    cp2k_cell = {sym: ('[angstrom]',) + tuple(coords) for sym, coords in zip(('a', 'b', 'c'), atoms.get_cell()*Bohr)}
    cp2k_cell['periodic'] = 'XYZ'  # anything else does not make much sense
    cp2k_coord = {
        'scaled': True,
        '*': [[sym] + list(coord) for sym, coord in zip(atoms.get_chemical_symbols(), atoms.get_scaled_positions())],
        }

    return dict2cp2k(
        {
            'global': {
                'run_type': 'ENERGY_FORCE',
                },
            'force_eval': {
                'subsys': {
                    'cell': cp2k_cell,
                    'coord': cp2k_coord,
                    },
                'print': {
                    'forces': {
                        'filename': 'forces',
                        },
                    },
                },
            }
        )
