# Copyright (C) 2015 Henrique Pereira Coutada Miranda
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
import re

from phonopy.file_IO import iter_collect_forces, get_drift_forces
from phonopy.interface.vasp import get_scaled_positions_lines
from phonopy.units import Bohr
from phonopy.cui.settings import fracval
from phonopy.structure.atoms import Atoms, symbol_map

def parse_set_of_forces(num_atoms, forces_filenames):
    hook = '' # Just for skipping the first line
    force_sets = []
    for filename in forces_filenames:
        siesta_forces = iter_collect_forces(filename,
                                            num_atoms,
                                            hook,
                                            [1, 2, 3],
                                            word='')
        if not siesta_forces:
            return []
        drift_force = get_drift_forces(siesta_forces)
        force_sets.append(np.array(siesta_forces) - drift_force)

    return force_sets

def read_siesta(filename):
    siesta_in = SiestaIn(open(filename).read())
    numbers = siesta_in._tags["atomicnumbers"]
    lattice = siesta_in._tags["latticevectors"]
    positions = siesta_in._tags["atomiccoordinates"]
    atypes = siesta_in._tags["chemicalspecieslabel"]
    cell = Atoms(numbers=numbers,
                 cell=lattice)

    coordformat = siesta_in._tags["atomiccoordinatesformat"]
    if coordformat == "fractional" or coordformat == "scaledbylatticevectors":
        cell.set_scaled_positions(positions)
    elif coordformat == "scaledcartesian":
        if siesta_in._tags['latticeconstant'] == 'ang':
            cell.set_positions(np.array(positions) / Bohr)#convert from angstroem to Bohr 
        else:
            cell.set_positions(np.array(positions))
    elif coordformat == "notscaledcartesianang" or coordformat == "ang":
        cell.set_positions(np.array(positions) / Bohr) #convert from angstroem to Bohr 
    elif coordformat == "notscaledcartesianbohr" or coordformat == "bohr":
        cell.set_positions(np.array(positions))
    else:
        print "The format %s for the AtomicCoordinatesFormat is not implemented"%coordformat
        exit() 
    
    return cell, atypes

def write_siesta(filename, cell, atypes):
    f = open(filename, 'w')
    f.write(get_siesta_structure(cell,atypes))

def write_supercells_with_displacements(supercell,
                                        cells_with_displacements,
                                        atypes):
    write_siesta("supercell.fdf", supercell, atypes)
    for i, cell in enumerate(cells_with_displacements):
        write_siesta("supercell-%03d.fdf" % (i + 1), cell, atypes)

def get_siesta_structure(cell,atypes):
    lattice = cell.get_cell()
    positions = cell.get_scaled_positions()
    masses = cell.get_masses()
    chemical_symbols = cell.get_chemical_symbols()
   
    lines = ""
    
    lines += "NumberOfAtoms %d\n\n"% len(positions)

    lines += "%block LatticeVectors\n"
    lines += ((" %21.16f" * 3 + "\n") * 3) % tuple(lattice.ravel())
    lines += "%endblock LatticeVectors\n\n"

    lines += "AtomicCoordinatesFormat  Fractional\n\n"

    lines += "LatticeConstant 1.0 Bohr\n\n"

    lines += "%block AtomicCoordinatesAndAtomicSpecies\n"
    for pos, i in zip(positions,chemical_symbols): 
        lines += ("%21.16lf"*3+" %d\n") % tuple(pos.tolist()+[atypes[i]])
    lines += "%endblock AtomicCoordinatesAndAtomicSpecies\n"

    return lines

class SiestaIn:
    _num_regex = '([+-]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
    _tags = { "latticeconstant":          1.0,
              "latticeconstantunit":     None,
              "chemicalspecieslabel":    None,
              "atomiccoordinatesformat": None,
              "atomicnumbers":           None,
              "atomicspecies":           None,
              "atomiccoordinates":       None }

    def __init__(self, lines):
        self._collect(lines)
    
    def _collect(self, lines):
        """ This routine reads the following from the Siesta file:
            - atomic positions
            - cell_parameters
            - atomic_species
        """
        #capture tags
        for tag,value,unit in re.findall('([\.A-Za-z]+)\s+?%s(?:[ ]+)?([A-Za-z]+)?'%self._num_regex,lines):
            tag = tag.lower()
            if tag == "latticeconstant":
                self._tags['latticeconstantunit'] = unit.lower()
                if unit == 'Ang':
                    self._tags[tag] = float(value) / Bohr
                else:
                    self._tags[tag] = float(value)

        for tag,value in re.findall('([\.A-Za-z]+)[ \t]+([a-zA-Z]+)',lines):
            tag = tag.replace('_','').lower()
            if tag == "atomiccoordinatesformat":
                self._tags[tag] = value.strip().lower() 

        #check if the necessary tags are present
        self.check_present('atomiccoordinatesformat')
        acell = self._tags['latticeconstant']

        #capture the blocks
        blocks = re.findall('%block\s+([A-Za-z_]+)\s((?:.+\n)+?(?=(?:\s+)?%endblock))',lines,re.MULTILINE)
        for tag,block in blocks:
            tag = tag.replace('_','').lower()
            if   tag == "chemicalspecieslabel":
                lines = block.split('\n')[:-1]
                self._tags["atomicnumbers"] = dict([map(int,species.split()[:2]) for species in lines])
                self._tags[tag] = dict([(lambda x: (x[2],int(x[0])))(species.split()) for species in lines])
            elif tag == "latticevectors":
                self._tags[tag] = [[ float(v)*acell for v in vector.split()] for vector in block.split('\n')[:3]]
            elif tag == "atomiccoordinatesandatomicspecies":
                lines = block.split('\n')[:-1]
                self._tags["atomiccoordinates"] = [ [float(x)  for x in atom.split()[:3]] for atom in lines ]
                self._tags["atomicspecies"] = [ int(atom.split()[3]) for atom in lines]
       
        #check if the block are present
        self.check_present("atomicspecies")
        self.check_present("atomiccoordinates")
        self.check_present("latticevectors")
        self.check_present("chemicalspecieslabel")
            
        #translate the atomicspecies to atomic numbers
        self._tags["atomicnumbers"] = [self._tags["atomicnumbers"][atype] for atype in self._tags["atomicspecies"]]
    
    def check_present(self,tag):
        if not self._tags[tag]:
            print "%s not present"%tag
            exit()
 
    def __str__(self):
        return self._tags

        
if __name__ == '__main__':
    import sys
    from phonopy.structure.symmetry import Symmetry
    cell,atypes = read_siesta(sys.argv[1])
    symmetry = Symmetry(cell)
    print "#", symmetry.get_international_table()
    print get_siesta_structure(cell,atypes)
