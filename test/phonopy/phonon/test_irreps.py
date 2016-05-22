import unittest
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import sys
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.moment import PhononMoment
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN

chars_Amm2 = """1. 0. -1. 0. -1. 0.  1. 0.
1. 0. -1. 0.  1. 0. -1. 0.
1. 0.  1. 0.  1. 0.  1. 0.
1. 0. -1. 0. -1. 0.  1. 0.
1. 0. -1. 0.  1. 0. -1. 0.
1. 0.  1. 0.  1. 0.  1. 0.
1. 0. -1. 0. -1. 0.  1. 0.
1. 0. -1. 0.  1. 0. -1. 0.
1. 0. -1. 0.  1. 0. -1. 0.
1. 0.  1. 0. -1. 0. -1. 0.
1. 0.  1. 0.  1. 0.  1. 0.
1. 0.  1. 0.  1. 0.  1. 0.
1. 0. -1. 0. -1. 0.  1. 0.
1. 0. -1. 0.  1. 0. -1. 0.
1. 0.  1. 0.  1. 0.  1. 0."""

chars_Pbar6m2 = """ 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0.  0.  0.  0.  0. -0.  0.  0.  0.  0.  0. -0.  0.
 1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0.  1.  0.
 1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0. -1.  0. -1.  0. -1.  0. -1.  0. -1.  0. -1.  0.
 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 2.  0.  1.  0. -1.  0. -2.  0. -1.  0.  1.  0. -0.  0. -0.  0. -0.  0.  0.  0.  0.  0.  0.  0.
 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0.  0.  0. -0.  0. -0.  0.  0.  0. -0.  0. -0.  0.
 1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0.  1.  0.
 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0.  0.  0.  0.  0. -0.  0.  0.  0.  0.  0. -0.  0.
 1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0.  1.  0.
 1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0.  1.  0.
 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0. -0.  0. -0.  0.  0.  0. -0.  0. -0.  0.  0.  0.
 1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0. -1.  0.  1.  0. -1.  0.  1.  0. -1.  0.  1.  0.
 1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  1.  0.
 2.  0. -1.  0. -1.  0.  2.  0. -1.  0. -1.  0. -0.  0. -0.  0.  0.  0. -0.  0. -0.  0.  0.  0."""

chars_Pc = """ 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0. -1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0.
 1.  0. -1.  0.
 1.  0.  1.  0.
 1.  0. -1.  0."""

class TestIrreps(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_Amm2(self):
        data = np.loadtxt(StringIO(chars_Amm2)).view(complex)
        phonon = self._get_phonon("Amm2", 
                                  [3, 2, 2],
                                  [[1, 0, 0],
                                   [0, 0.5, -0.5],
                                   [0, 0.5, 0.5]])
        phonon.set_irreps([0, 0, 0])
        chars = phonon.get_irreps().get_characters()
        self.assertTrue(np.abs(chars - data).all() < 1e-5)

    def test_Pbar6m2(self):
        data = np.loadtxt(StringIO(chars_Pbar6m2)).view(complex)
        phonon = self._get_phonon("P-6m2", 
                                  [2, 2, 3],
                                  np.eye(3))
        phonon.set_irreps([0, 0, 0])
        chars = phonon.get_irreps().get_characters()
        self.assertTrue(np.abs(chars - data).all() < 1e-5)

    def test_Pc(self):
        data = np.loadtxt(StringIO(chars_Pc)).view(complex)
        phonon = self._get_phonon("Pc", 
                                  [2, 2, 2],
                                  np.eye(3))
        phonon.set_irreps([0, 0, 0])
        chars = phonon.get_irreps().get_characters()
        self.assertTrue(np.abs(chars - data).all() < 1e-5)

    def _show(self, vals):
        for v in vals:
            print(("%9.6f " * len(v)) % tuple(v))

    def _get_phonon(self, spgtype, dim, pmat):
        cell = read_vasp("POSCAR_%s" % spgtype)
        phonon = Phonopy(cell,
                         np.diag(dim),
                         primitive_matrix=pmat,
                         is_auto_displacements=False)
        force_sets = parse_FORCE_SETS(filename="FORCE_SETS_%s" % spgtype)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        return phonon

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIrreps)
    unittest.TextTestRunner(verbosity=2).run(suite)
