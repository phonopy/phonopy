import unittest

import numpy as np
import time

from phonopy.structure.symmetry import (Symmetry, symmetrize_borns_and_epsilon,
                                        _get_supercell_and_primitive,
                                        _get_mapping_between_cells)
from phonopy.structure.cells import get_supercell
from phonopy.interface.phonopy_yaml import read_cell_yaml
import os

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestSymmetry(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_map_operations(self):
        symprec = 1e-5
        cell = read_cell_yaml(os.path.join(data_dir, "..", "NaCl.yaml"))
        scell = get_supercell(cell, np.diag([2, 2, 2]), symprec=symprec)
        symmetry = Symmetry(scell, symprec=symprec)
        # start = time.time()
        symmetry._set_map_operations()
        # end = time.time()
        # print(end - start)
        map_ops = symmetry.get_map_operations()
        map_atoms = symmetry.get_map_atoms()
        positions = scell.get_scaled_positions()
        rotations = symmetry.get_symmetry_operations()['rotations']
        translations = symmetry.get_symmetry_operations()['translations']
        for i, (op_i, atom_i) in enumerate(zip(map_ops, map_atoms)):
            r_pos = np.dot(rotations[op_i], positions[i]) + translations[op_i]
            diff = positions[atom_i] - r_pos
            diff -= np.rint(diff)
            self.assertTrue((diff < symprec).all())

    def test_magmom(self):
        symprec = 1e-5
        cell = read_cell_yaml(os.path.join(data_dir, "Cr.yaml"))
        symmetry_nonspin = Symmetry(cell, symprec=symprec)
        atom_map_nonspin = symmetry_nonspin.get_map_atoms()
        len_sym_nonspin = len(
            symmetry_nonspin.get_symmetry_operations()['rotations'])

        spin = [1, -1]
        cell_withspin = cell.copy()
        cell_withspin.set_magnetic_moments(spin)
        symmetry_withspin = Symmetry(cell_withspin, symprec=symprec)
        atom_map_withspin = symmetry_withspin.get_map_atoms()
        len_sym_withspin = len(
            symmetry_withspin.get_symmetry_operations()['rotations'])

        broken_spin = [1, -2]
        cell_brokenspin = cell.copy()
        cell_brokenspin = cell.copy()
        cell_brokenspin.set_magnetic_moments(broken_spin)
        symmetry_brokenspin = Symmetry(cell_brokenspin, symprec=symprec)
        atom_map_brokenspin = symmetry_brokenspin.get_map_atoms()
        len_sym_brokenspin = len(
            symmetry_brokenspin.get_symmetry_operations()['rotations'])

        self.assertTrue((atom_map_nonspin == atom_map_withspin).all())
        self.assertFalse((atom_map_nonspin == atom_map_brokenspin).all())
        self.assertTrue(len_sym_nonspin == len_sym_withspin)
        self.assertFalse(len_sym_nonspin == len_sym_brokenspin)


class TestSymmetrizeBornsAndEpsilon(unittest.TestCase):
    def setUp(self):
        self.pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        self.smat = np.eye(3, dtype='int') * 2
        self.cell = read_cell_yaml(os.path.join(data_dir, "..", "NaCl.yaml"))
        self.scell, self.pcell = _get_supercell_and_primitive(
            self.cell,
            primitive_matrix=self.pmat,
            supercell_matrix=self.smat)
        self.idx = [self.scell.u2u_map[i]
                    for i in self.scell.s2u_map[self.pcell.p2s_map]]
        self.epsilon = np.array([[2.43533967, 0, 0],
                                 [0, 2.43533967, 0],
                                 [0, 0, 2.43533967]])
        self.borns = np.array([[[1.08875538, 0, 0],
                                [0, 1.08875538, 0],
                                [0, 0, 1.08875538]],
                               [[1.08875538, 0, 0],
                                [0, 1.08875538, 0],
                                [0, 0, 1.08875538]],
                               [[1.08875538, 0, 0],
                                [0, 1.08875538, 0],
                                [0, 0, 1.08875538]],
                               [[1.08875538, 0, 0],
                                [0, 1.08875538, 0],
                                [0, 0, 1.08875538]],
                               [[-1.08875538, 0, 0],
                                [0, -1.08875538, 0],
                                [0, 0, -1.08875538]],
                               [[-1.08875538, 0, 0],
                                [0, -1.08875538, 0],
                                [0, 0, -1.08875538]],
                               [[-1.08875538, 0, 0],
                                [0, -1.08875538, 0],
                                [0, 0, -1.08875538]],
                               [[-1.08875538, 0, 0],
                                [0, -1.08875538, 0],
                                [0, 0, -1.08875538]]])

    def tearDown(self):
        pass

    def test_only_symmetrize(self):
        borns, epsilon = symmetrize_borns_and_epsilon(
            self.borns,
            self.epsilon,
            self.cell)
        np.testing.assert_allclose(borns, self.borns)
        np.testing.assert_allclose(epsilon, self.epsilon)
        # primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])

    def test_with_pmat_and_smat(self):
        borns, epsilon = symmetrize_borns_and_epsilon(
            self.borns,
            self.epsilon,
            self.cell,
            primitive_matrix=self.pmat,
            supercell_matrix=self.smat)
        np.testing.assert_allclose(borns, self.borns[self.idx])
        np.testing.assert_allclose(epsilon, self.epsilon)

    def test_with_pcell(self):
        idx2 = _get_mapping_between_cells(self.pcell, self.pcell)

        np.testing.assert_array_equal(
            idx2, np.arange(self.pcell.get_number_of_atoms()))

        borns, epsilon = symmetrize_borns_and_epsilon(
            self.borns,
            self.epsilon,
            self.cell,
            primitive=self.pcell)
        np.testing.assert_allclose(borns, self.borns[self.idx][idx2])
        np.testing.assert_allclose(epsilon, self.epsilon)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSymmetry)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestSymmetrizeBornsAndEpsilon)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
