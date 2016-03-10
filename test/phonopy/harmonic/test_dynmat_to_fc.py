from __future__ import print_function
import unittest
import numpy as np
from phonopy.interface.phonopy_yaml import phonopyYaml
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.cells import get_supercell, get_primitive

class TestDynmatToFc(unittest.TestCase):

    def setUp(self):
        filename = "POSCAR.yaml"
        self._cell = phonopyYaml(filename).get_atoms()
    
    def tearDown(self):
        pass
    
    def test_get_commensurate_points(self):
        smat = np.diag([2, 2, 2])
        pmat = np.dot(np.linalg.inv(smat),
                      [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        supercell = get_supercell(self._cell, smat)
        primitive = get_primitive(supercell, pmat)
        comm_points = get_commensurate_points(np.linalg.inv(primitive.get_primitive_matrix()).T)
        for i, p in enumerate(comm_points):
            print("%d %s" % (i + 1, p))

if __name__ == '__main__':
    unittest.main()
