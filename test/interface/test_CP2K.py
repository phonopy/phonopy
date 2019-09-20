import unittest

import numpy as np
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.cp2k import read_cp2k
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

CP2K_INPUT_TOOLS_AVAILABLE = True

try:
    import cp2k_input_tools
except ImportError:
    CP2K_INPUT_TOOLS_AVAILABLE = False


class TestCP2K(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipUnless(CP2K_INPUT_TOOLS_AVAILABLE, "the cp2k-input-tools package is not installed")
    def test_read_cp2k(self):
        cell, _ = read_cp2k(os.path.join(data_dir, "Si-CP2K.inp"))
        cell_ref = read_cell_yaml(os.path.join(data_dir, "Si-CP2K.yaml"))
        self.assertTrue(
            (np.abs(cell.get_cell() - cell_ref.get_cell()) < 1e-5).all())
        diff_pos = (cell.get_scaled_positions()
                    - cell_ref.get_scaled_positions())
        diff_pos -= np.rint(diff_pos)
        self.assertTrue((np.abs(diff_pos) < 1e-5).all())
        for s, s_r in zip(cell.get_chemical_symbols(),
                          cell_ref.get_chemical_symbols()):
            self.assertTrue(s == s_r)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCP2K)
    unittest.TextTestRunner(verbosity=2).run(suite)
