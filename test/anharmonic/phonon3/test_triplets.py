import unittest
import numpy as np

from phonopy.interface.phonopy_yaml import PhonopyYaml
from anharmonic.phonon3.triplets import (get_grid_point_from_address,
                                         get_grid_point_from_address_py)

class TestTriplets(unittest.TestCase):

    def setUp(self):
        filename = "POSCAR.yaml"
        self._cell = PhonopyYaml(filename).get_atoms()
    
    def tearDown(self):
        pass
    
    def test_get_grid_point_from_address(self):
        self._mesh = (10, 10, 10)
        print("Compare get_grid_point_from_address from spglib and that "
              "written in python")
        print("with mesh numbers [%d %d %d]" % self._mesh)

        for address in list(np.ndindex(self._mesh)):
            gp_spglib = get_grid_point_from_address(address, self._mesh)
            gp_py = get_grid_point_from_address_py(address, self._mesh)
            # print("%s %d %d" % (address, gp_spglib, gp_py))
            self.assertEqual(gp_spglib, gp_py)

if __name__ == '__main__':
    unittest.main()
