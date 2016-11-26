import unittest
import numpy as np
from phonopy.interface.phonopy_yaml import get_unitcell_from_phonopy_yaml
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.cells import get_supercell, get_primitive

import os
data_dir=os.path.dirname(os.path.abspath(__file__))

class TestDynmatToFc(unittest.TestCase):

    def setUp(self):
        filename = "../NaCl.yaml"
        self._cell = get_unitcell_from_phonopy_yaml(os.path.join(data_dir,filename))
    
    def tearDown(self):
        pass
    
    def test_get_commensurate_points(self):
        smat = np.diag([2, 2, 2])
        pmat = np.dot(np.linalg.inv(smat),
                      [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        supercell = get_supercell(self._cell, smat)
        primitive = get_primitive(supercell, pmat)
        supercell_matrix = np.linalg.inv(primitive.get_primitive_matrix())
        supercell_matrix = np.rint(supercell_matrix).astype('intc')
        comm_points = get_commensurate_points(supercell_matrix)
        # self._write(comm_points)
        self._compare(comm_points)

    def _compare(self, comm_points, filename="comm_points.dat"):
        with open(os.path.join(data_dir,filename)) as f:
            comm_points_in_file = np.loadtxt(f)
            diff = comm_points_in_file[:,1:] - comm_points
            np.testing.assert_allclose(diff, np.rint(diff), atol=1e-3)

    def _write(self, comm_points, filename="comm_points.dat"):
        with open(os.path.join(data_dir,filename), 'w') as w:
            lines = []
            for i, p in enumerate(comm_points):
                lines.append("%d %5.2f %5.2f %5.2f" % ((i + 1,) + tuple(p)))
            w.write("\n".join(lines))


if __name__ == '__main__':
    unittest.main()
