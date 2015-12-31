# Copyright (C) 2014 Atsushi Togo
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

from phonopy.file_IO import iter_collect_forces, get_drift_forces
from phonopy.interface.vasp import get_scaled_positions_lines
from phonopy.units import Bohr
from phonopy.cui.settings import fracval
from phonopy.structure.atoms import Atoms, symbol_map

def parse_set_of_forces(num_atoms, forces_filenames):
    hook = 'Forces acting on atoms'
    force_sets = []
    for filename in forces_filenames:
        pwscf_forces = iter_collect_forces(filename,
                                           num_atoms,
                                           hook,
                                           [6, 7, 8],
                                           word='force')
        if not pwscf_forces:
            return []

        drift_force = get_drift_forces(pwscf_forces)
        force_sets.append(np.array(pwscf_forces) - drift_force)
        
    return force_sets

def read_pwscf(filename):
    pwscf_in = PwscfIn(open(filename).readlines())
    tags = pwscf_in.get_tags()
    lattice = tags['cell_parameters']
    positions = [pos[1] for pos in tags['atomic_positions']]
    species = [pos[0] for pos in tags['atomic_positions']]
    mass_map = {}
    pp_map = {}
    for vals in tags['atomic_species']:
        mass_map[vals[0]] = vals[1]
        pp_map[vals[0]] = vals[2]
    masses = [mass_map[x] for x in species]
    pp_all_filenames = [pp_map[x] for x in species]

    unique_species = []
    for x in species:
        if x not in unique_species:
            unique_species.append(x)
    
    numbers = []
    is_unusual = False
    for x in species:
        if x in symbol_map:
            numbers.append(symbol_map[x])
        else:
            numbers.append(-unique_species.index(x))
            is_unusual = True

    if is_unusual:
        positive_numbers = []
        for n in numbers:
            if n > 0:
                if n not in positive_numbers:
                    positive_numbers.append(n)
    
        available_numbers = range(1, 119)
        for pn in positive_numbers:
            available_numbers.remove(pn)
        
        for i, n in enumerate(numbers):
            if n < 1:
                numbers[i] = available_numbers[-n]

        cell = Atoms(numbers=numbers,
                     masses=masses,
                     cell=lattice,
                     scaled_positions=positions)
    else:
        cell = Atoms(numbers=numbers,
                     cell=lattice,
                     scaled_positions=positions)

    unique_symbols = []
    pp_filenames = {}
    for i, symbol in enumerate(cell.get_chemical_symbols()):
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
            pp_filenames[symbol] = pp_all_filenames[i]

    return cell, pp_filenames

def write_pwscf(filename, cell, pp_filenames):
    f = open(filename, 'w')
    f.write(get_pwscf_structure(cell, pp_filenames=pp_filenames))

def write_supercells_with_displacements(supercell,
                                        cells_with_displacements,
                                        pp_filenames):
    write_pwscf("supercell.in", supercell, pp_filenames)
    for i, cell in enumerate(cells_with_displacements):
        write_pwscf("supercell-%03d.in" % (i + 1), cell, pp_filenames)

def get_pwscf_structure(cell, pp_filenames=None):
    lattice = cell.get_cell()
    positions = cell.get_scaled_positions()
    masses = cell.get_masses()
    chemical_symbols = cell.get_chemical_symbols()
    unique_symbols = []
    atomic_species = []
    for symbol, m in zip(chemical_symbols, masses):
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
            atomic_species.append((symbol, m))
    
    lines = ""
    lines += ("!    ibrav = 0, nat = %d, ntyp = %d\n" %
              (len(positions), len(unique_symbols)))
    lines += "CELL_PARAMETERS bohr\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "ATOMIC_SPECIES\n"
    for symbol, mass in atomic_species:
        if pp_filenames is None:
            lines += " %2s %10.5f   %s_PP_filename\n" % (symbol, mass, symbol)
        else:
            lines += " %2s %10.5f   %s\n" % (symbol, mass, pp_filenames[symbol])
    lines += "ATOMIC_POSITIONS crystal\n"
    for i, (symbol, pos_line) in enumerate(zip(
            chemical_symbols,
            get_scaled_positions_lines(positions).split('\n'))):
        lines += (" %2s " % symbol) + pos_line
        if i < len(chemical_symbols) - 1:
            lines += "\n"

    return lines
    
class PwscfIn:
    def __init__(self, lines):
        self._set_methods = {'ibrav':            self._set_ibrav,
                             'nat':              self._set_nat,
                             'ntyp':             self._set_ntyp,
                             'atomic_species':   self._set_atom_types,
                             'atomic_positions': self._set_positions,
                             'cell_parameters':  self._set_lattice}
        self._tags = {'ibrav':            None,
                      'nat':              None,
                      'ntyp':             None,
                      'atomic_species':   None,
                      'atomic_positions': None,
                      'cell_parameters':  None}

        self._values = None
        self._collect(lines)

    def get_tags(self):
        return self._tags

    def _collect(self, lines):
        elements = {}
        tag = None
        for line_tmp in lines:
            line = line_tmp.split('!')[0]
            if ('atomic_positions' in line.lower() or
                'cell_parameters' in line.lower()):
                if len(line.split()) == 1:
                    words = [line.lower().strip(), 'alat']
                else:
                    words = line.lower().split()[:2]
            elif 'atomic_species' in line.lower():
                words = line.lower().split()
            else:
                line_replaced = line.replace('=', ' ').replace(',', ' ')
                words = line_replaced.split()
            for val in words:
                if val.lower() in self._set_methods:
                    tag = val.lower()
                    elements[tag] = []
                elif tag is not None:
                    elements[tag].append(val)

        for tag in ['ibrav', 'nat', 'ntyp']:
            if tag not in elements:
                print "%s is not found in the input file." % tag
                sys.exit(1)
                    
        for tag, self._values in elements.iteritems():
            if tag == 'ibrav' or tag == 'nat' or tag == 'ntyp':
                self._set_methods[tag]()

        for tag, self._values in elements.iteritems():
            if tag != 'ibrav' and tag != 'nat' and tag != 'ntyp':
                self._set_methods[tag]()

    def _set_ibrav(self):
        ibrav = int(self._values[0])
        if ibrav != 0:
            print "Only ibrav = 0 is supported."
            sys.exit(1)

        self._tags['ibrav'] = ibrav
                
    def _set_nat(self):
        self._tags['nat'] = int(self._values[0])

    def _set_ntyp(self):
        self._tags['ntyp'] = int(self._values[0])

    def _set_lattice(self):
        unit = self._values[0]
        if unit == 'alat':
            print "Only CELL_PARAMETERS format with alat is not supported."
            sys.exit(1)
        if unit == 'angstrom':
            factor = 1.0 / Bohr
        else:
            factor = 1.0

        if len(self._values[1:]) < 9:
            print "CELL_PARAMETERS is wrongly set."
            sys.exit(1)
            
        lattice = np.reshape([float(x) for x in self._values[1:10]], (3, 3))
        self._tags['cell_parameters'] = lattice * factor
        
    def _set_positions(self):
        unit = self._values[0]
        if unit != 'crystal':
            print ("Only ATOMIC_POSITIONS format with "
                   "crystal coordinates is supported.")
            sys.exit(1)
            
        natom = self._tags['nat']
        pos_vals = self._values[1:]
        if len(pos_vals) < natom * 4:
            print "ATOMIC_POSITIONS is wrongly set."
            sys.exit(1)

        positions = []
        for i in range(natom):
            positions.append(
                [pos_vals[i * 4],
                 [float(x) for x in pos_vals[i * 4 + 1:i * 4 + 4]]])
            
        self._tags['atomic_positions'] = positions

    def _set_atom_types(self):
        num_types = self._tags['ntyp']
        if len(self._values) < num_types * 3:
            print "ATOMIC_SPECIES is wrongly set."
            sys.exit(1)

        species = []
        
        for i in range(num_types):
            species.append(
                [self._values[i * 3],
                 float(self._values[i * 3 + 1]),
                 self._values[i * 3 + 2]])
            
        self._tags['atomic_species'] = species
        
if __name__ == '__main__':
    import sys
    from phonopy.structure.symmetry import Symmetry
    # abinit = PwscfIn(open(sys.argv[1]).readlines())
    cell, pp_filenames = read_pwscf(sys.argv[1])
    # symmetry = Symmetry(cell)
    # print "#", symmetry.get_international_table()
    print get_pwscf_structure(cell, pp_filenames)
