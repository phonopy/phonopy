import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestPhonopy(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPhonopy(self):
        phonon = self._get_phonon()
        self.assertTrue(phonon.version == phonon.get_version())
        self.assertTrue(id(phonon.primitive) == id(phonon.get_primitive()))
        self.assertTrue(id(phonon.unitcell) == id(phonon.get_unitcell()))
        self.assertTrue(id(phonon.supercell) == id(phonon.get_supercell()))
        self.assertTrue(id(phonon.symmetry) == id(phonon.get_symmetry()))
        self.assertTrue(id(phonon.primitive_symmetry)
                        == id(phonon.get_primitive_symmetry()))
        self.assertTrue(id(phonon.supercell_matrix)
                        == id(phonon.get_supercell_matrix()))
        self.assertTrue(id(phonon.primitive_matrix)
                        == id(phonon.get_primitive_matrix()))
        self.assertTrue(id(phonon.unit_conversion_factor)
                        == id(phonon.get_unit_conversion_factor()))
        self.assertTrue(id(phonon.displacements)
                        == id(phonon.get_displacements()))
        self.assertTrue(id(phonon.force_constants)
                        == id(phonon.get_force_constants()))
        self.assertTrue(id(phonon.nac_params) == id(phonon.get_nac_params()))
        self.assertTrue(id(phonon.supercells_with_displacements)
                        == id(phonon.get_supercells_with_displacements()))
        self.assertTrue(id(phonon.dataset)
                        == id(phonon.get_displacement_dataset()))
        phonon.run_mesh([11, 11, 11], with_eigenvectors=True)
        phonon.get_mesh_dict()

    def _get_phonon(self):
        cell = read_vasp(os.path.join(data_dir, "POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir, "FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        filename_born = os.path.join(data_dir, "BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhonopy)
    unittest.TextTestRunner(verbosity=2).run(suite)
