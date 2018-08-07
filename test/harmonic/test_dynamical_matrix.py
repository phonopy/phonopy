import unittest
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
import os

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestDynamicalMatrix(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_properties(self):
        cell = read_vasp(os.path.join(data_dir, "..", "POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        dynmat = phonon.dynamical_matrix
        dynmat.set_dynamical_matrix([0, 0, 0])
        self.assertTrue(id(dynmat.primitive)
                        == id(dynmat.get_primitive()))
        self.assertTrue(id(dynmat.supercell)
                        == id(dynmat.get_supercell()))
        np.testing.assert_allclose(dynmat.dynamical_matrix,
                                   dynmat.get_dynamical_matrix())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynamicalMatrix)
    unittest.TextTestRunner(verbosity=2).run(suite)
