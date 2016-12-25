import unittest

import numpy as np
from phonopy.interface.phonopy_yaml import get_unitcell_from_phonopy_yaml
from phonopy.interface.wien2k import parse_wien2k_struct
from phonopy.file_IO import parse_disp_yaml
import os
data_dir = os.path.dirname(os.path.abspath(__file__))

class TestWien2k(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_parse_wien2k_struct(self):
        filename_BaGa2 = os.path.join(data_dir,"BaGa2.struct")
        cell, npts, r0s, rmts = parse_wien2k_struct(filename_BaGa2)
        filename = os.path.join(data_dir,"BaGa2-wien2k.yaml")
        cell_ref = get_unitcell_from_phonopy_yaml(filename)
        self.assertTrue(
            (np.abs(cell.get_cell() - cell_ref.get_cell()) < 1e-5).all())
        diff_pos = cell.get_scaled_positions() - cell_ref.get_scaled_positions()
        diff_pos -= np.rint(diff_pos)
        self.assertTrue((np.abs(diff_pos) < 1e-5).all())
        for s, s_r in zip(cell.get_chemical_symbols(),
                          cell_ref.get_chemical_symbols()):
            self.assertTrue(s == s_r)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWien2k)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
