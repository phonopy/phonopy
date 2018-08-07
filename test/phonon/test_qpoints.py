import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestQpoints(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_phonon(self):
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
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon

    def testQpoints(self):
        phonon = self._get_phonon()
        phonon.set_qpoints_phonon([0, 0, 0], is_eigenvectors=True)
        qpoints_phonon = phonon.qpoints
        np.testing.assert_allclose(qpoints_phonon.eigenvectors,
                                   qpoints_phonon.get_eigenvectors())
        np.testing.assert_allclose(qpoints_phonon.frequencies,
                                   qpoints_phonon.get_frequencies())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQpoints)
    unittest.TextTestRunner(verbosity=2).run(suite)
