import unittest

import os
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell, get_primitive, TrimmedCell
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestSupercell(unittest.TestCase):
    def setUp(self):
        self._cells = []
        symbols = ['Si'] * 2 + ['O'] * 4
        lattice = [[4.65, 0, 0],
                   [0, 4.75, 0],
                   [0, 0, 3.25]]
        points = [[0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.5],
                  [0.3, 0.3, 0.0],
                  [0.7, 0.7, 0.0],
                  [0.2, 0.8, 0.5],
                  [0.8, 0.2, 0.5]]

        self._cells.append(PhonopyAtoms(cell=lattice,
                                        scaled_positions=points,
                                        symbols=symbols))

        symbols = ['Si'] * 2
        lattice = [[0, 2.73, 2.73],
                   [2.73, 0, 2.73],
                   [2.73, 2.73, 0]]
        points = [[0.75, 0.75, 0.75],
                  [0.5, 0.5, 0.5]]

        self._cells.append(PhonopyAtoms(cell=lattice,
                                        scaled_positions=points,
                                        symbols=symbols))

        self._smats = []
        self._smats.append(np.diag([1, 2, 3]))
        self._smats.append([[-1, 1, 1],
                            [1, -1, 1],
                            [1, 1, -1]])

        self._fnames = ("SiO2-123.yaml", "Si-conv.yaml")

    def tearDown(self):
        pass

    def test_get_supercell(self):
        for i, (cell, smat, fname) in enumerate(zip(self._cells,
                                                    self._smats,
                                                    self._fnames)):
            scell = get_supercell(cell, smat)
            scell_yaml = read_cell_yaml(os.path.join(data_dir, fname))
            np.testing.assert_allclose(scell.cell, scell_yaml.get_cell(),
                                       atol=1e-5)
            pos = scell.get_scaled_positions()
            pos -= np.rint(pos)
            pos_yaml = scell_yaml.get_scaled_positions()
            pos_yaml -= np.rint(pos_yaml)
            np.testing.assert_allclose(pos, pos_yaml, atol=1e-5)
            np.testing.assert_array_equal(scell.numbers,
                                          scell_yaml.get_atomic_numbers())
            np.testing.assert_allclose(scell.masses,
                                       scell_yaml.get_masses(),
                                       atol=1e-5)


class TestPrimitive(unittest.TestCase):
    def setUp(self):
        cell = read_cell_yaml(os.path.join(data_dir, "..", "NaCl.yaml"))
        self._pcell = get_primitive(cell, [[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])

    def tearDown(self):
        pass

    def test_properties(self):
        np.testing.assert_array_equal(
            self._pcell.p2s_map,
            self._pcell.get_primitive_to_supercell_map())
        np.testing.assert_array_equal(
            self._pcell.s2p_map,
            self._pcell.get_supercell_to_primitive_map())
        np.testing.assert_array_equal(
            self._pcell.atomic_permutations,
            self._pcell.get_atomic_permutations())
        self.assertTrue(id(self._pcell.p2p_map)
                        == id(self._pcell.get_primitive_to_primitive_map()))


class TestTrimmedCell(unittest.TestCase):
    def setUp(self):
        pmat = [[0, 0.5, 0.5],
                [0.5, 0, 0.5],
                [0.5, 0.5, 0]]
        smat2 = np.eye(3, dtype='intc') * 2
        pmat2 = np.dot(np.linalg.inv(smat2), pmat)
        smat3 = np.eye(3, dtype='intc') * 3
        pmat3 = np.dot(np.linalg.inv(smat3), pmat)

        cell = read_cell_yaml(os.path.join(data_dir, "..", "NaCl.yaml"))
        scell2 = get_supercell(cell, smat2)
        scell3 = get_supercell(cell, smat3)
        n = len(scell3) // 2
        # swap first and last half of atomic order
        indices = [i + n for i in range(n)] + list(range(n))
        scell3_swap = PhonopyAtoms(
            cell=scell3.cell,
            scaled_positions=scell3.scaled_positions[indices],
            numbers=scell3.numbers[indices])
        self._tcell2 = TrimmedCell(pmat2, scell2)
        self._tcell3 = TrimmedCell(
            pmat3, scell3_swap,
            positions_to_reorder=self._tcell2.scaled_positions)

    def tearDown(self):
        pass

    def test_reorder(self):
        np.testing.assert_array_equal(self._tcell2.numbers,
                                      self._tcell3.numbers)
        np.testing.assert_allclose(self._tcell2.cell, self._tcell3.cell)
        diff = self._tcell2.scaled_positions - self._tcell3.scaled_positions
        diff -= np.rint(diff)
        dist = (np.dot(diff, self._tcell2.cell) ** 2).sum(axis=1)
        self.assertTrue((dist < 1e-5).all())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSupercell)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPrimitive)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrimmedCell)
    unittest.TextTestRunner(verbosity=2).run(suite)
