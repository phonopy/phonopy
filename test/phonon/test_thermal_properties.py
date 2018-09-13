import unittest
import os
import numpy as np
import phonopy

data_dir = os.path.dirname(os.path.abspath(__file__))

tp_str = """0.000000 100.000000 200.000000 300.000000 400.000000
500.000000 600.000000 700.000000 800.000000 900.000000
4.856373 3.916036 -0.276031 -6.809284 -14.961974
-24.342086 -34.708562 -45.898943 -57.796507 -70.313405
0.000000 26.328820 55.258118 74.269718 88.156943
99.054231 108.009551 115.606163 122.200204 128.024541
0.000000 36.207859 45.673598 47.838871 48.634251
49.009334 49.214960 49.339592 49.420745 49.476502"""


class TestThermalProperties(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testThermalProperties(self):
        phonon = self._get_phonon()
        phonon.set_mesh([5, 5, 5])
        phonon.set_thermal_properties(t_step=100, t_max=900)
        tp = phonon.thermal_properties
        np.testing.assert_allclose(tp.temperatures, tp.get_temperatures())
        np.testing.assert_allclose(tp.thermal_properties,
                                   tp.get_thermal_properties())
        np.testing.assert_allclose(tp.zero_point_energy,
                                   tp.get_zero_point_energy())
        np.testing.assert_allclose(tp.high_T_entropy,
                                   tp.get_high_T_entropy())
        np.testing.assert_allclose(tp.number_of_integrated_modes,
                                   tp.get_number_of_integrated_modes())
        np.testing.assert_allclose(tp.number_of_modes,
                                   tp.get_number_of_modes())

        tp_ref = np.reshape([float(x) for x in tp_str.split()], (-1, 10))
        np.testing.assert_allclose(tp.thermal_properties, tp_ref, atol=1e-5)

    def _get_phonon(self):
        phonon = phonopy.load(
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            primitive_matrix=[[0, 0.5, 0.5],
                              [0.5, 0, 0.5],
                              [0.5, 0.5, 0]],
            unitcell_filename=os.path.join(data_dir, "..", "POSCAR_NaCl"),
            force_sets_filename=os.path.join(data_dir, "..",
                                             "FORCE_SETS_NaCl"),
            born_filename=os.path.join(data_dir, "..", "BORN_NaCl"))
        return phonon


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThermalProperties)
    unittest.TextTestRunner(verbosity=2).run(suite)
