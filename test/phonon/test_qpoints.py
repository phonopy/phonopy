import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.units import VaspToTHz

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
        phonon.dataset = force_sets
        phonon.produce_force_constants()
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        nac_params = parse_BORN(phonon.primitive, filename=filename_born)
        phonon.nac_params = nac_params
        return phonon

    def testQpoints(self):
        phonon = self._get_phonon()
        qpoints = [[0, 0, 0], [0, 0, 0.5]]
        phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
        for i, q in enumerate(qpoints):
            dm = phonon.qpoints.dynamical_matrices[i]
            dm_eigs = np.linalg.eigvalsh(dm).real
            eigs = phonon.qpoints.eigenvalues[i]
            freqs = phonon.qpoints.frequencies[i] / VaspToTHz
            np.testing.assert_allclose(dm_eigs, eigs)
            np.testing.assert_allclose(freqs ** 2 * np.sign(freqs), eigs)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQpoints)
    unittest.TextTestRunner(verbosity=2).run(suite)
