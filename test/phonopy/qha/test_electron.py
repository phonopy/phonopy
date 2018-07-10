import unittest
import sys
import numpy as np
from phonopy.qha.electron import ElectronFreeEnergy
from phonopy.interface.vasp import VasprunxmlExpat
import os
data_dir=os.path.dirname(os.path.abspath(__file__))

class TestElectronFreeEnergy(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_entropy(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestElectronFreeEnergy)
    unittest.TextTestRunner(verbosity=2).run(suite)
