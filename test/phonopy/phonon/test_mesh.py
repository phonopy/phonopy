import unittest
import os
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN

data_dir = os.path.dirname(os.path.abspath(__file__))

class TestIterMesh(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIterMesh(self):
        phonon = self._get_phonon()
        phonon.set_iter_mesh([3, 3, 3], is_eigenvectors=True)
        imesh = phonon.get_iter_mesh()
        freqs = []
        eigvecs = []
        for i, (f, e) in enumerate(imesh):
            freqs.append(f)
            eigvecs.append(e)

        phonon.set_mesh([3, 3, 3], is_eigenvectors=True)
        _, _, mesh_freqs, mesh_eigvecs = phonon.get_mesh()

        np.testing.assert_allclose(mesh_freqs, freqs)
        np.testing.assert_allclose(mesh_eigvecs, eigvecs)

    def _get_phonon(self):
        cell = read_vasp(os.path.join(data_dir, "../POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir,"../FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        filename_born = os.path.join(data_dir, "../BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon

    def _set_mesh(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIterMesh)
    unittest.TextTestRunner(verbosity=2).run(suite)
