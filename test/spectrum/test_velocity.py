import unittest

import numpy as np
from phonopy.spectrum.velocity import Velocity
from phonopy.interface.vasp import read_XDATCAR
import os
data_dir=os.path.dirname(os.path.abspath(__file__))

class TestVelocity(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_Velocity(self):
        positions, lattice = read_XDATCAR(os.path.join(data_dir,"XDATCAR"))
        v = Velocity(positions=positions,
                     lattice=lattice,
                     timestep=2)
        v.run()
        velocity = v.get_velocities()
        velocity_cmp = np.loadtxt(os.path.join(data_dir,"velocities.dat"))
        self.assertTrue(
            (np.abs(velocity.ravel() - velocity_cmp.ravel()) < 1e-1).all())

    def _show(self, velocity):
        print(velocity)

    def _write(self, velocity):
        np.savetxt(os.path.join(data_dir,"velocities.dat"), velocity.reshape((-1, 3)))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVelocity)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
