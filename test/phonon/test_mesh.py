import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.phonon.mesh import Mesh

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestMesh(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIterMesh(self):
        phonon = self._get_phonon()
        phonon.init_mesh(mesh=[3, 3, 3],
                         with_eigenvectors=True,
                         use_iter_mesh=True)
        freqs = []
        eigvecs = []
        for i, (f, e) in enumerate(phonon.mesh):
            freqs.append(f)
            eigvecs.append(e)

        phonon.run_mesh([3, 3, 3], with_eigenvectors=True)
        mesh_freqs = phonon.mesh.frequencies
        mesh_eigvecs = phonon.mesh.eigenvectors

        np.testing.assert_allclose(mesh_freqs, freqs)
        np.testing.assert_allclose(mesh_eigvecs, eigvecs)

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

    def _set_mesh(self):
        pass

    def testMesh(self):
        phonon = self._get_phonon()
        mesh_obj = Mesh(phonon.dynamical_matrix, [10, 10, 10],
                        with_eigenvectors=True)
        mesh_obj.run()
        for i, x in enumerate(mesh_obj):
            pass
        for j, x in enumerate(mesh_obj):
            pass
        assert i == j
        self.assertTrue(id(mesh_obj.dynamical_matrix)
                        == id(mesh_obj.get_dynamical_matrix()))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMesh)
    unittest.TextTestRunner(verbosity=2).run(suite)
