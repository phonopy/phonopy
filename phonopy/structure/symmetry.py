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
from phonopy.structure.atoms import PhonopyAtoms as Atoms

class Symmetry(object):
    def __init__(self, cell, symprec=1e-5, is_symmetry=True):
        self._cell = cell
        self._symprec = symprec

        self._symmetry_operations = None
        self._international_table = None
        self._dataset = None
        self._wyckoff_letters = None
        self._map_atoms = None

        magmom = cell.get_magnetic_moments()
        if type(magmom) is np.ndarray:
            if (magmom < symprec).all():
                magmom = None

        if not is_symmetry:
            self._set_nosym()
        elif magmom is None:
            self._set_symmetry_dataset()
        else:
            self._set_symmetry_operations_with_magmoms()

        self._pointgroup_operations = None
        self._pointgroup = None
        self._set_pointgroup_operations()

        self._independent_atoms = None
        self._set_independent_atoms()
        self._map_operations = None

    def get_symmetry_operations(self):
        return self._symmetry_operations

    def get_symmetry_operation(self, operation_number):
        operation = self._symmetry_operations
        return {'rotations': operation['rotations'][operation_number],
                'translations': operation['translations'][operation_number]}

    def get_pointgroup_operations(self):
        return self._pointgroup_operations

    def get_pointgroup(self):
        return self._pointgroup

    def get_international_table(self):
        return self._international_table

    def get_Wyckoff_letters(self):
        return self._wyckoff_letters

    def get_dataset(self):
        """Detail of dataset is found in spglib.get_symmetry_dataset.
        """
        return self._dataset

    def get_independent_atoms(self):
        return self._independent_atoms

    def get_map_atoms(self):
        return self._map_atoms

    def get_map_operations(self):
        if self._map_operations is None:
            self._set_map_operations()
        return self._map_operations

    def get_site_symmetry(self, atom_number):
        positions = self._cell.get_scaled_positions()
        lattice = self._cell.get_cell()
        rotations = self._symmetry_operations['rotations']
        translations = self._symmetry_operations['translations']

        return get_site_symmetry(atom_number,
                                 lattice,
                                 positions,
                                 rotations,
                                 translations,
                                 self._symprec)

    def get_symmetry_tolerance(self):
        return self._symprec

    def get_reciprocal_operations(self):
        """
        Definition of operation:
        q' = Rq

        This is transpose of that shown in ITA (q' = qR).
        """
        return self._reciprocal_operations

    def _set_symmetry_dataset(self):
        self._dataset = spg.get_symmetry_dataset(self._cell, self._symprec)
        self._symmetry_operations = {
            'rotations': self._dataset['rotations'],
            'translations': self._dataset['translations']}
        self._international_table = "%s (%d)" % (self._dataset['international'],
                                                 self._dataset['number'])
        self._wyckoff_letters = self._dataset['wyckoffs']

        self._map_atoms = self._dataset['equivalent_atoms']

    def _set_symmetry_operations_with_magmoms(self):
        cell = (self._cell.get_cell(),
                self._cell.get_scaled_positions(),
                self._cell.get_atomic_numbers(),
                self._cell.get_magnetic_moments())
        self._symmetry_operations = spg.get_symmetry(cell,
                                                     symprec=self._symprec)
        self._map_atoms = self._symmetry_operations['equivalent_atoms']
        self._set_map_atoms()

    def _set_map_atoms(self):
        rotations = self._symmetry_operations['rotations']
        translations = self._symmetry_operations['translations']
        positions = self._cell.get_scaled_positions()
        lattice = self._cell.get_cell()
        map_atoms = np.arange(self._cell.get_number_of_atoms())
        for i, p in enumerate(positions):
            is_found = False
            for j in range(i):
                for r, t in zip(rotations, translations):
                    diff = np.dot(p, r.T) + t - positions[j]
                    diff -= np.rint(diff)
                    dist = np.linalg.norm(np.dot(diff, lattice))
                    if dist < self._symprec:
                        map_atoms[i] = j
                        is_found = True
                        break
                if is_found:
                    break
        self._map_atoms = np.array(map_atoms, dtype='intc')

    def _set_independent_atoms(self):
        indep_atoms = []
        for i, atom_map in enumerate(self._map_atoms):
            if i == atom_map:
                indep_atoms.append(i)
        self._independent_atoms = np.array(indep_atoms, dtype='intc')

    def _set_pointgroup_operations(self):
        rotations = self._symmetry_operations['rotations']
        ptg_ops = get_pointgroup_operations(rotations)
        reciprocal_rotations = [rot.T for rot in ptg_ops]
        exist_r_inv = False
        for rot in ptg_ops:
            if (rot + np.eye(3, dtype='intc') == 0).all():
                exist_r_inv = True
                break
        if not exist_r_inv:
            reciprocal_rotations += [-rot.T for rot in ptg_ops]

        self._pointgroup_operations = np.array(ptg_ops, dtype='intc')
        self._pointgroup = get_pointgroup(self._pointgroup_operations)[0]
        self._reciprocal_operations = np.array(reciprocal_rotations,
                                               dtype='intc')

    def _set_map_operations(self):
        ops = self._symmetry_operations
        pos = self._cell.get_scaled_positions()
        lattice = self._cell.get_cell()
        map_operations = np.zeros(len(pos), dtype='intc')

        for i, eq_atom in enumerate(self._map_atoms):
            for j, (r, t) in enumerate(
                zip(ops['rotations'], ops['translations'])):

                diff = np.dot(pos[i], r.T) + t - pos[eq_atom]
                diff -= np.rint(diff)
                dist = np.linalg.norm(np.dot(diff, lattice))
                if dist < self._symprec:
                    map_operations[i] = j
                    break
        self._map_operations = map_operations

    def _set_nosym(self):
        translations = []
        rotations = []

        if 'get_supercell_to_unitcell_map' in dir(self._cell):
            s2u_map = self._cell.get_supercell_to_unitcell_map()
            positions = self._cell.get_scaled_positions()

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

            self._map_atoms = s2u_map
        else:
            rotations.append(np.eye(3, dtype='intc'))
            translations.append(np.zeros(3, dtype='double'))
            self._map_atoms = range(self._cell.get_number_of_atoms())

        self._symmetry_operations = {
            'rotations': np.array(rotations, dtype='intc'),
            'translations': np.array(translations, dtype='double')}
        self._international_table = 'P1 (1)'
        self._wyckoff_letters = ['a'] * self._cell.get_number_of_atoms()

def find_primitive(cell, symprec=1e-5):
    """
    A primitive cell is searched in the input cell. When a primitive
    cell is found, an object of Atoms class of the primitive cell is
    returned. When not, None is returned.
    """
    lattice, positions, numbers = spg.find_primitive(cell, symprec)
    if lattice is None:
        return None
    else:
        return Atoms(numbers=numbers,
                     scaled_positions=positions,
                     cell=lattice,
                     pbc=True)

def get_pointgroup(rotations):
    ptg = spg.get_pointgroup(rotations)
    return ptg[0].strip(), ptg[2]

def get_lattice_vector_equivalence(point_symmetry):
    """Return (b==c, c==a, a==b)"""
    # primitive_vectors: column vectors

    equivalence = [False, False, False]
    for r in point_symmetry:
        if (np.abs(r[:, 0]) == [0, 1, 0]).all():
            equivalence[2] = True
        if (np.abs(r[:, 0]) == [0, 0, 1]).all():
            equivalence[1] = True
        if (np.abs(r[:, 1]) == [1, 0, 0]).all():
            equivalence[2] = True
        if (np.abs(r[:, 1]) == [0, 0, 1]).all():
            equivalence[0] = True
        if (np.abs(r[:, 2]) == [1, 0, 0]).all():
            equivalence[1] = True
        if (np.abs(r[:, 2]) == [0, 1, 0]).all():
            equivalence[0] = True

    return equivalence

def get_site_symmetry(atom_number,
                      lattice,
                      positions,
                      rotations,
                      translations,
                      symprec):
    pos = positions[atom_number]
    site_symmetries = []

    for r, t in zip(rotations, translations):
        rot_pos = np.dot(pos, r.T) + t
        diff = pos - rot_pos
        diff -= np.rint(diff)
        diff = np.dot(diff, lattice)
        if np.linalg.norm(diff) < symprec:
            site_symmetries.append(r)

    return np.array(site_symmetries, dtype='intc')

def get_pointgroup_operations(rotations):
    ptg_ops = []
    for rot in rotations:
        is_same = False
        for tmp_rot in ptg_ops:
            if (tmp_rot == rot).all():
                is_same = True
                break
        if not is_same:
            ptg_ops.append(rot)

    return ptg_ops
