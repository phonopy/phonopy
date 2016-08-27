import unittest
import numpy as np
from phonopy.interface.phonopy_yaml import get_unitcell_from_phonopy_yaml
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.cells import get_supercell, get_primitive

class TestDynmatToFc(unittest.TestCase):

    def setUp(self):
        filename = "POSCAR.yaml"
        self._cell = get_unitcell_from_phonopy_yaml(filename)
    
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
        with open(filename) as f:
            comm_points_in_file = np.loadtxt(f)
            self.assertTrue(
                (np.abs(comm_points_in_file[:,1:] - comm_points) < 1e-3).all())

    def _write(self, comm_points, filename="comm_points.dat"):
        with open(filename, 'w') as w:
            lines = []
            for i, p in enumerate(comm_points):
                lines.append("%d %5.2f %5.2f %5.2f" % ((i + 1,) + tuple(p)))
            w.write("\n".join(lines))


if __name__ == '__main__':
    unittest.main()
