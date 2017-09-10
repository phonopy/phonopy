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
        pass

    def _get_phonon(self):
        cell = read_vasp(os.path.join(data_dir, "../POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag(dim),
                         primitive_matrix=pmat)
        filename = os.path.join(data_dir,"../FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        filename_born = os.path.join(data_dir, "../BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIterMesh)
    unittest.TextTestRunner(verbosity=2).run(suite)
