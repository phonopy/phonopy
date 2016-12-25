import unittest

import numpy as np
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
import os
data_dir = os.path.dirname(os.path.abspath(__file__))

class TestPhonopyYaml(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_read_poscar_yaml(self):
        filename = os.path.join(data_dir,"NaCl-vasp.yaml")
        cell = self._get_unitcell(filename)
        self._compare(cell)

    def test_read_phonopy_yaml(self):
        filename = os.path.join(data_dir,"phonopy.yaml")
        cell = self._get_unitcell(filename)
        self._compare(cell)

    def test_write_phonopy_yaml(self):
        phonopy = self._get_phonon()
        phpy_yaml = PhonopyYaml(calculator='vasp')
        phpy_yaml.set_phonon_info(phonopy)
        # print(phpy_yaml)

    def _compare(self, cell):
        cell_ref = read_vasp(os.path.join(data_dir, "../POSCAR_NaCl"))
        self.assertTrue(
            (np.abs(cell.get_cell() - cell_ref.get_cell()) < 1e-5).all())
        diff_pos = cell.get_scaled_positions() - cell_ref.get_scaled_positions()
        diff_pos -= np.rint(diff_pos)
        self.assertTrue((np.abs(diff_pos) < 1e-5).all())
        for s, s_r in zip(cell.get_chemical_symbols(),
                          cell_ref.get_chemical_symbols()):
            self.assertTrue(s == s_r)

    def _get_unitcell(self, filename):
        phpy_yaml = PhonopyYaml()
        phpy_yaml.read(filename)
        unitcell = phpy_yaml.get_unitcell()
        return unitcell

    def _get_phonon(self):
        cell = read_vasp(os.path.join(data_dir, "../POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir, "FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        supercell = phonon.get_supercell()
        born_elems = {'Na': [[1.08703, 0, 0],
                             [0, 1.08703, 0],
                             [0, 0, 1.08703]],
                      'Cl': [[-1.08672, 0, 0],
                             [0, -1.08672, 0],
                             [0, 0, -1.08672]]}
        born = [born_elems[s] for s in ['Na', 'Cl']]
        epsilon = [[2.43533967, 0, 0],
                   [0, 2.43533967, 0],
                   [0, 0, 2.43533967]]
        factors = 14.400
        phonon.set_nac_params({'born': born,
                               'factor': factors,
                               'dielectric': epsilon})
        return phonon

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhonopyYaml)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
