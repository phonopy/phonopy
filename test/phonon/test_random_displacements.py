import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.phonon.random_displacements import RandomDisplacements

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestRandomDisplacements(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_phonon_NaCl(self):
        cell = read_vasp(os.path.join(data_dir, "..", "POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon

    def test_NaCl(self):
        phonon = self._get_phonon_NaCl()
        np.random.seed(19680801)
        rd = RandomDisplacements(phonon.dynamical_matrix,
                                 cutoff_frequency=0.01)

        # import matplotlib.pyplot as plt
        # fig = plt.figure()

        # ax1 = fig.add_subplot(211)
        # vals_Na = []
        # vals_Cl = []
        # for i in range(1000):
        #     rd.run(500)
        #     vals_Na.append(rd.u[:32])
        #     vals_Cl.append(rd.u[32:])
        # d_Na = np.sqrt((np.reshape(vals_Na, (-1, 3)) ** 2).sum(axis=1))
        # d_Cl = np.sqrt((np.reshape(vals_Cl, (-1, 3)) ** 2).sum(axis=1))
        # ax1.hist(d_Na, 50)
        # ax1.hist(d_Cl, 50)
        # ax1.set_xlim(0, 1)

        # ax2 = fig.add_subplot(212)
        # vals_Na = []
        # vals_Cl = []
        # for i in range(1000):
        #     rd.run(1000)
        #     vals_Na.append(rd.u[:32])
        #     vals_Cl.append(rd.u[32:])
        # d_Na = np.sqrt((np.reshape(vals_Na, (-1, 3)) ** 2).sum(axis=1))
        # d_Cl = np.sqrt((np.reshape(vals_Cl, (-1, 3)) ** 2).sum(axis=1))
        # ax2.hist(d_Na, 50)
        # ax2.hist(d_Cl, 50)
        # ax2.set_xlim(0, 1)

        # plt.show()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRandomDisplacements)
    unittest.TextTestRunner(verbosity=2).run(suite)
