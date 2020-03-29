import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.phonon.band_structure import get_band_qpoints

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestBandStructure(unittest.TestCase):
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

    def test_band(self):
        self._test_band()

    def test_with_group_velocities(self):
        self._test_band(with_group_velocities=True)

    def test_is_band_connection(self):
        self._test_band(is_band_connection=True)

    def _test_band(self,
                   with_group_velocities=False,
                   is_band_connection=False):
        band_paths = [[[0, 0, 0], [0.5, 0.5, 0.5]],
                      [[0.5, 0.5, 0], [0, 0, 0], [0.5, 0.25, 0.75]]]
        qpoints = get_band_qpoints(band_paths, npoints=11)
        phonon = self._get_phonon()
        phonon.run_band_structure(qpoints,
                                  with_group_velocities=with_group_velocities,
                                  is_band_connection=is_band_connection)
        band_structure = phonon.band_structure
        phonon.get_band_structure_dict()

        self.assertTrue(id(band_structure.distances),
                        id(band_structure.get_distances()))
        self.assertTrue(id(band_structure.qpoints),
                        id(band_structure.get_qpoints()))
        self.assertTrue(id(band_structure.frequencies),
                        id(band_structure.get_frequencies()))
        self.assertTrue(id(band_structure.eigenvectors),
                        id(band_structure.get_eigenvectors()))
        self.assertTrue(id(band_structure.group_velocities),
                        id(band_structure.get_group_velocities()))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBandStructure)
    unittest.TextTestRunner(verbosity=2).run(suite)
