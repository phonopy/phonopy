# Copyright (C) 2011 Atsushi Togo
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
import phonopy.structure.spglib as spg
from phonopy.structure.atoms import Atoms

def find_primitive(cell, symprec=1e-5):
    """
    A primitive cell is searched in the input cell. When a primitive
    cell is found, an object of Atoms class of the primitive cell is
    returned. When not, None is returned.
    """
    lattice, positions, numbers = spg.find_primitive(cell, symprec)
    if lattice == None:
        return None
    else:
        return Atoms(numbers=numbers,
                     scaled_positions=positions,
                     cell=lattice,
                     pbc=True)

class Symmetry:
    def __init__(self, cell, symprec=1e-5, is_symmetry=True):
        self.__cell = cell
        self.symprec = symprec

        self.symmetry_operations = None
        self.international_table = None
        self.dataset = None
        self.wyckoff_letters = None
        self.map_atoms = None
        if not is_symmetry:
            self._set_nosym()
        elif cell.get_magnetic_moments() is None:
            self._symmetry_dataset()
        else:
            self._symmetry_operations()

        self.pointgroup_operations = None
        self.pointgroup = None
        self._pointgroup_operations()

        self.independent_atoms = None
        self.map_operations = None
        self._map_operations()

    def get_symmetry_operations(self):
        return self.symmetry_operations

    def get_symmetry_operation(self, operation_number):
        operation = self.symmetry_operations
        return {'rotations': operation['rotations'][operation_number],
                'translations': operation['translations'][operation_number]}

    def get_pointgroup_operations(self):
        return self.pointgroup_operations

    def get_pointgroup(self):
        return self.pointgroup

    def get_international_table(self):
        return self.international_table

    def get_Wyckoff_letters(self):
        return self.wyckoff_letters

    def get_dataset(self):
        """Detail of dataset is found in spglib.get_symmetry_dataset.
        """
        return self.dataset

    def get_independent_atoms(self):
        return self.independent_atoms

    def get_map_atoms(self):
        return self.map_atoms

    def get_map_operations(self):
        return self.map_operations

    def get_site_symmetry(self, atom_number):
        pos = self.__cell.get_scaled_positions()[atom_number]
        symprec = self.symprec
        rot = self.symmetry_operations['rotations']
        trans = self.symmetry_operations['translations']
        site_symmetries = []

        for r, t in zip(rot, trans):
            rot_pos = np.dot(pos, r.T) + t
            diff = pos - rot_pos
            if (abs(diff - np.rint(diff)) < symprec).all():
                site_symmetries.append(r)

        return np.array(site_symmetries, dtype='intc')

    def get_symmetry_tolerance(self):
        return self.symprec

    def _map_atoms(self):
        rotations = self.symmetry_operations['rotations']
        translations = self.symmetry_operations['translations']
        positions = self.__cell.get_scaled_positions()
        lattice = self.__cell.get_cell()
        map_atoms = range(self.__cell.get_number_of_atoms())
        for i, p in enumerate(positions):
            is_found = False
            for j in range(i):
                for r, t in zip(rotations, translations):
                    diff = np.dot(p, r.T) + t - positions[j]
                    diff -= np.rint(diff)
                    dist = np.linalg.norm(np.dot(diff, lattice))
                    if dist < self.symprec:
                        map_atoms[i] = j
                        is_found = True
                        break
                if is_found:
                    break
        self.map_atoms = np.array(map_atoms, dtype=int)

    def _symmetry_dataset(self):
        self.dataset = spg.get_symmetry_dataset(self.__cell, self.symprec)
        self.symmetry_operations = \
            {'rotations': self.dataset['rotations'],
             'translations': self.dataset['translations']}
        self.international_table = "%s (%d)" % (self.dataset['international'],
                                                 self.dataset['number'])
        self.wyckoff_letters = self.dataset['wyckoffs']
        self.map_atoms = self.dataset['equivalent_atoms']

    def _symmetry_operations(self):
        self.symmetry_operations = \
            spg.get_symmetry(self.__cell, self.symprec)
        self._map_atoms()

    def _pointgroup_operations(self):
        rotations = []
        for rot in self.symmetry_operations['rotations']:
            is_same = False
            for tmp_rot in rotations:
                if (tmp_rot==rot).all():
                    is_same = True
                    break
            if not is_same:
                rotations.append(rot)

        self.pointgroup_operations = np.array(rotations, dtype='intc')
        self.pointgroup = get_pointgroup(self.pointgroup_operations)[0]

    def _map_operations(self):
        ops = self.symmetry_operations
        pos = self.__cell.get_scaled_positions()
        map_operations = np.zeros(len(pos), dtype=int)
        independent_atoms = []

        for i, eq_atom in enumerate(self.map_atoms):
            if i == eq_atom:
                independent_atoms.append(i)
            for j, (r, t) in enumerate(
                zip(ops['rotations'], ops['translations'])):
                
                diff = np.dot(pos[i], r.T) + t - pos[eq_atom]
                if (abs(diff - np.rint(diff)) < self.symprec).all():
                    map_operations[i] = j
                    break

        self.independent_atoms = np.array(independent_atoms)
        self.map_operations = map_operations

    def _set_nosym(self):
        translations = []
        rotations = []
        
        if 'get_supercell_to_unitcell_map' in dir(self.__cell):
            s2u_map = self.__cell.get_supercell_to_unitcell_map()
            positions = self.__cell.get_scaled_positions()
    
            for i, j in enumerate(s2u_map):
                if j==0:
                    ipos0 = i
                    break
    
            for i, p in zip(s2u_map, positions):
                if i==0:
                    trans = p - positions[ipos0]
                    trans -= np.floor(trans)
                    translations.append(trans)
                    rotations.append(np.eye(3, dtype='intc'))

            self.map_atoms = s2u_map
        else:
            rotations.append(np.eye(3, dtype=int))
            translations.append(np.zeros(3, dtype='double'))
            self.map_atoms = range(self.__cell.get_number_of_atoms())

        self.symmetry_operations = {'rotations': np.array(rotations,
                                                          dtype='intc'),
                                    'translations': np.array(translations,
                                                             dtype='double')}
        self.international_table = 'P1 (1)'
        self.wyckoff_letters = ['a'] * self.__cell.get_number_of_atoms()

def get_pointgroup(rotations):
    ptg = spg.get_pointgroup(rotations)
    return ptg[0].strip(), ptg[2]

def get_ir_reciprocal_mesh(mesh,
                           cell,
                           is_shift=np.zeros(3, dtype=int),
                           is_time_reversal=False,
                           symprec=1e-5):
    """
    Return k-point map to the irreducible k-points and k-point grid points .
    The symmetry is serched from the input cell.
    is_shift=[ 0, 0, 0 ] gives Gamma center mesh.
    """

    return spg.get_ir_reciprocal_mesh(mesh,
                                      cell,
                                      is_shift,
                                      is_time_reversal,
                                      symprec)

def get_ir_kpoints(kpoints, cell, is_time_reversal=False, symprec=1e-5):

    return spg.get_ir_kpoints(kpoints, cell, is_time_reversal, symprec)

    


if __name__ == '__main__':
    from phonopy.structure.symmetry import Symmetry
    from phonopy.interface.vasp import read_vasp
    
    def get_magmom(text):
        magmom = []
        for numxmag in text.split():
            if '*' in numxmag:
                num, mag = numxmag.split('*')
                magmom += [float(mag)] * int(num)
            else:
                magmom.append(float(numxmag))
        return magmom
    
    def parse_incar(filename):
        for line in open(filename):
            for conf in line.split(';'):
                if 'MAGMOM' in conf:
                    return get_magmom(conf.split('=')[1])
    
    cell = read_vasp("POSCAR")
    symmetry = Symmetry(cell, symprec=1e-3)
    map_nonspin = symmetry.get_map_atoms()
    print "Number of operations w/o spin", len(symmetry.get_symmetry_operations()['rotations'])
    magmoms = parse_incar("INCAR")
    cell.set_magnetic_moments(magmoms)
    symmetry = Symmetry(cell, symprec=1e-3)
    print "Number of operations w spin", len(symmetry.get_symmetry_operations()['rotations'])
    map_withspin = symmetry.get_map_atoms()
    if ((map_nonspin - map_withspin) == 0).all():
        print True
    else:
        print False
        print map_nonspin
        print map_withspin
