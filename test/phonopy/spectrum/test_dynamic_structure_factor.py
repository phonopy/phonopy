import unittest

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.units import THzToEv
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

# D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
# f(Q) = \sum_i a_i \exp((-b_i Q^2) + c
# Q is in angstron^-1
# a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
f_params = {'Na': [3.148690, 2.594987, 4.073989, 6.046925,
                   0.767888, 0.070139, 0.995612, 14.1226457,
                   0.968249, 0.217037, 0.045300],  # 1+
            'Cl': [1.061802, 0.144727, 7.139886, 1.171795,
                   6.524271, 19.467656, 2.355626, 60.320301,
                   35.829404, 0.000436, -34.916604],  # 1-
            'Si': [5.275329, 2.631338, 3.191038, 33.730728,
                   1.511514, 0.081119, 1.356849, 86.288640,
                   2.519114, 1.170087, 0.145073],  # neutral
            'Pb': [32.505656, 1.047035, 20.014240, 6.670321,
                   14.645661, 0.105279, 5.029499, 16.525040,
                   1.760138, 0.105279, 4.044678],  # 4+
            'Pb0': [16.419567, 0.105499, 32.738592, 1.055049,
                    6.530247, 25.025890, 2.342742, 80.906596,
                    19.916475, 6.664449, 4.049824],
            'Te': [6.660302, 33.031656, 6.940756, 0.025750,
                   19.847015, 5.065547, 1.557175, 84.101613,
                   17.802427, 0.487660, -0.806668]}  # neutral

data_str = """0.205043308 4.407286257 1.956218187 0.000104857 0.000540313 0.000315644
0.158921899 1.003748377 0.523964179 0.000626127 0.000010037 0.000300514
0.021827309 0.504075274 0.250870053 0.000600811 0.000032282 0.000274957
0.023738365 0.278889658 0.152882598 0.000000079 0.000602730 0.000224140
0.196502005 0.002774374 0.106709063 0.000212955 0.000294880 0.000137485
0.032304029 0.111100967 0.081399008 0.000092400 0.000231328 0.000032545
0.006701520 0.103384727 0.000031104 0.000051431 0.066050787 0.000024674
0.001180117 0.087412827 0.000001203 0.000071282 0.055615996 0.000570346
0.000245315 0.071678812 0.001486225 0.000810228 0.046214590 0.003316495
0.000859873 0.045889006 0.017310162 0.000148125 0.033398866 0.011523264"""

class TestDynamicStructureFactor(unittest.TestCase):
    def setUp(self):
        self.phonon = self._get_phonon()
        mesh = [11, 11, 11]
        self.phonon.set_mesh(mesh,
                             is_mesh_symmetry=False,
                             is_eigenvectors=True)

    def tearDown(self):
        pass

    def show(self):
        directions = np.array([[0.5, 0.5, 0.5],
                               [-0.5, 0.5, 0.5],
                               [0.5, -0.5, 0.5],
                               [0.5, 0.5, -0.5],
                               [0.5, -0.5, -0.5],
                               [-0.5, 0.5, -0.5],
                               [-0.5, -0.5, 0.5],
                               [-0.5, -0.5, -0.5]])
        G_points_cubic = ([7, 1, 1], )
        sign = 1
        self._run(G_points_cubic, directions, sign, verbose=True)

    def test_IXS_G_to_L(self):
        directions = np.array([[0.5, 0.5, 0.5], ])
        n_points = 11
        G_points_cubic = ([7, 1, 1], )
        sign = 1
        self._run(G_points_cubic, directions, sign, n_points=n_points)
        Q, S = self.phonon.get_dynamic_structure_factor()
        data_cmp = np.reshape([float(x) for x in data_str.split()], (-1, 6))
        print(S.shape)
        print(data_cmp.shape)
        for i in (([0, 1], [2], [3, 4], [5])):
            np.testing.assert_allclose(
                S[:, i].sum(axis=1), data_cmp[:, i].sum(axis=1), atol=1e-6)

    def _run(self,
             G_points_cubic,
             directions,
             sign,
             n_points=51,
             verbose=False):
        P = [[0, 0.5, 0.5],
             [0.5, 0, 0.5],
             [0.5, 0.5, 0]]
        G_to_L = np.array(
            [directions[0] * x
             for x in np.arange(0, n_points) / float(n_points - 1)])
        self.phonon.set_band_structure([G_to_L])
        _, distances, frequencies, _ = self.phonon.get_band_structure()

        T = 300
        for G_cubic in G_points_cubic:
            G_prim = np.dot(G_cubic, P)
            if verbose:
                print("# G_cubic %s, G_prim %s, sign=%d" %
                      (G_cubic, G_prim, sign))

            for direction in directions:
                direction_prim = np.dot(direction, P)
                G_to_L = np.array(
                    [direction_prim * x
                     for x in np.arange(1, n_points) / float(n_points - 1)])
                self.phonon.set_dynamic_structure_factor(G_to_L,
                                                         G_prim,
                                                         T,
                                                         f_params=f_params,
                                                         sign=sign,
                                                         freq_min=1e-3,
                                                         run_immediately=False)
                dsf = self.phonon.dynamic_structure_factor
                for i, S in enumerate(dsf):
                    Q_prim = dsf.qpoints[i]
                    Q_cubic = np.dot(Q_prim, np.linalg.inv(P))

                    if verbose:
                        print("%f  %f %f %f  %f %f %f %f  %f %f %f %f" %
                              ((distances[0][i + 1], ) + tuple(Q_cubic) +
                               tuple(frequencies[0][i + 1][[0, 2, 3, 5]]
                                     * THzToEv * 1000) +
                               ((S[0] + S[1]) / 2, S[2], (S[3] + S[4]) / 2,
                                S[5])))
                if verbose:
                    print("")

    def plot_f_Q(f_params):
        import matplotlib.pyplot as plt
        x = np.linspace(0.0, 6.0, 101)
        for elem in ('Si', 'Na', 'Cl', 'Pb', 'Pb0', 'Te'):
            y = [atomic_form_factor(Q, f_params[elem]) for Q in x]
            plt.plot(x, y, label=elem)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()

    def _get_phonon(self):
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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDynamicStructureFactor)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
