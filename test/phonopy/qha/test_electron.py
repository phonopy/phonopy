import unittest
import io
import sys
import numpy as np
import tarfile
from phonopy.qha.electron import ElectronFreeEnergy
from phonopy.interface.vasp import VasprunxmlExpat
import os
data_dir=os.path.dirname(os.path.abspath(__file__))

class TestElectronFreeEnergy(unittest.TestCase):
    def setUp(self):
        # fullpath = os.path.join(data_dir, "..", "interface",
        #                         "vasprun.xml.tar.bz2")
        # tar_files = tarfile.open(fullpath)
        # f = tar_files.extractfile(tar_files.getmembers()[0])
        fullpath = os.path.join(data_dir, "vasprun.xml")
        f = io.open(fullpath, 'rb')
        self.vasprun = VasprunxmlExpat(f)
        if self.vasprun.parse(debug=True):
            pass
        else:
            raise

    def tearDown(self):
        pass

    def test_entropy(self):
        eigvals = self.vasprun.eigenvalues[:, :, :, 0]
        weights = self.vasprun.k_weights_int
        n_electrons = self.vasprun.NELECT
        efe = ElectronFreeEnergy(eigvals, weights, n_electrons)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestElectronFreeEnergy)
    unittest.TextTestRunner(verbosity=2).run(suite)
