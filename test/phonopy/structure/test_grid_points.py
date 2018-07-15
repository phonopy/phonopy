import unittest
import os
import numpy as np
from phonopy.structure.grid_points import GridPoints

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestGridPoints(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testGridPoints(self):
        gp = GridPoints([11, 11, 11],
                        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        self.assertTrue((gp.grid_address == gp.get_grid_address()).all())
        self.assertTrue((gp.ir_grid_points == gp.get_ir_grid_points()).all())
        np.testing.assert_allclose(gp.qpoints, gp.get_ir_qpoints())
        self.assertTrue((gp.weights == gp.get_ir_grid_weights()).all())
        self.assertTrue((gp.grid_mapping_table ==
                         gp.get_grid_mapping_table()).all())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGridPoints)
    unittest.TextTestRunner(verbosity=2).run(suite)
