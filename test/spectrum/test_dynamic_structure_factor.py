import unittest

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
from phonopy import load
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

data_AFF = """5.171588679 1.845620236 3.025894647 0.124171430 0.531444271 0.357372083
0.023856320 1.675511060 0.792733731 0.046252215 0.607606835 0.365938849
0.201611049 0.529561499 0.375299494 0.629569610 0.028195996 0.376744812
0.122785611 0.283082649 0.233015457 0.057604235 0.606819089 0.388606570
0.241073793 0.027598574 0.174838303 0.120170236 0.551166577 0.399921690
0.017676892 0.191317799 0.155253863 0.001382605 0.675194157 0.408290981
0.124073001 0.069254933 0.037429027 0.639561207 0.163254273 0.408934148
0.003757843 0.217553444 0.213773706 0.446843472 0.205635357 0.389828659
0.335273837 0.009258283 0.461748466 0.111918275 0.311894016 0.321054550
0.155257213 0.560104925 0.005767736 0.261429629 0.511077151 0.176362423"""

data_b = """99.367366854 35.461912077 57.989024624 1.109493894 4.748549446 3.191839110
0.483763005 33.976333211 15.902329182 0.422741848 5.553481879 3.338937967
4.324846943 11.359855711 7.850959264 5.886910551 0.263652029 3.509134839
2.773336666 6.393937243 5.030350431 0.550562310 5.799777024 3.688180663
5.647859305 0.646577389 3.821967934 1.171203888 5.371783065 3.853918874
0.418131844 4.525459914 3.344658336 0.013681138 6.681172229 3.970863352
2.850888142 1.591305647 0.373114126 6.375514924 3.354075619 3.970894110
0.079674882 4.612631104 2.112023136 4.414685828 3.888741010 3.698457365
6.120367922 0.169008406 4.303015992 1.042962054 5.252650957 2.818141845
2.305763345 8.318257049 0.039499076 1.790343459 7.590133448 1.207779368"""


def get_func_AFF(f_params):
    def func(symbol, Q):
        return atomic_form_factor_WK1995(Q, f_params[symbol])
    return func


class TestDynamicStructureFactor(unittest.TestCase):
    def setUp(self):
        self.phonon = self._get_phonon()
        mesh = [5, 5, 5]
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
        self._run(G_points_cubic, directions, verbose=True)

    def test_IXS_G_to_L(self, verbose=False):
        directions = np.array([[0.5, 0.5, 0.5], ])
        n_points = 11
        G_points_cubic = ([7, 1, 1], )

        # Atomic form factor
        self._run(G_points_cubic,
                  directions,
                  func_AFF=get_func_AFF(f_params),
                  n_points=n_points)
        Q, S = self.phonon.get_dynamic_structure_factor()
        data_cmp = np.reshape([float(x) for x in data_AFF.split()], (-1, 6))
        for i in (([0, 1], [2], [3, 4], [5])):
            np.testing.assert_allclose(
                S[:6, i].sum(axis=1), data_cmp[:6, i].sum(axis=1), atol=1e-5)
            if verbose:
                print(S[:6, i].sum(axis=1) - data_cmp[:6, i].sum(axis=1))

        for i in (([0, 1], [2, 3], [4], [5])):
            np.testing.assert_allclose(
                S[6:, i].sum(axis=1), data_cmp[6:, i].sum(axis=1), atol=1e-5)
            if verbose:
                print(S[6:, i].sum(axis=1) - data_cmp[6:, i].sum(axis=1))

        # Scattering lengths
        self._run(G_points_cubic,
                  directions,
                  scattering_lengths={'Na': 3.63, 'Cl': 9.5770},
                  n_points=n_points)
        Q, S = self.phonon.get_dynamic_structure_factor()
        data_cmp = np.reshape([float(x) for x in data_b.split()], (-1, 6))
        for i in (([0, 1], [2], [3, 4], [5])):
            np.testing.assert_allclose(
                S[:6, i].sum(axis=1), data_cmp[:6, i].sum(axis=1), atol=1e-5)
            if verbose:
                print(S[:6, i].sum(axis=1) - data_cmp[:6, i].sum(axis=1))
        for i in (([0, 1], [2, 3], [4], [5])):
            np.testing.assert_allclose(
                S[6:, i].sum(axis=1), data_cmp[6:, i].sum(axis=1), atol=1e-5)
            if verbose:
                print(S[6:, i].sum(axis=1) - data_cmp[6:, i].sum(axis=1))

    def plot_f_Q(f_params):
        import matplotlib.pyplot as plt
        x = np.linspace(0.0, 6.0, 101)
        for elem in ('Si', 'Na', 'Cl', 'Pb', 'Pb0', 'Te'):
            y = [atomic_form_factor_WK1995(Q, f_params[elem]) for Q in x]
            plt.plot(x, y, label=elem)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend()
        plt.show()

    def _run(self,
             G_points_cubic,
             directions,
             func_AFF=None,
             scattering_lengths=None,
             n_points=51):
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
            for direction in directions:
                direction_prim = np.dot(direction, P)
                G_to_L = np.array(
                    [direction_prim * x
                     for x in np.arange(1, n_points) / float(n_points - 1)])
                if func_AFF is not None:
                    self.phonon.set_dynamic_structure_factor(
                        G_to_L,
                        G_prim,
                        T,
                        func_atomic_form_factor=func_AFF,
                        freq_min=1e-3,
                        run_immediately=False)
                elif scattering_lengths is not None:
                    self.phonon.set_dynamic_structure_factor(
                        G_to_L,
                        G_prim,
                        T,
                        scattering_lengths=scattering_lengths,
                        freq_min=1e-3,
                        run_immediately=False)
                else:
                    raise SyntaxError
                dsf = self.phonon.dynamic_structure_factor
                for i, S in enumerate(dsf):
                    pass

    def _get_phonon(self):
        filename_cell = os.path.join(data_dir, "..", "POSCAR_NaCl")
        filename_forces = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        phonon = load(np.diag([2, 2, 2]),
                      primitive_matrix=[[0, 0.5, 0.5],
                                        [0.5, 0, 0.5],
                                        [0.5, 0.5, 0]],
                      unitcell_filename=filename_cell,
                      force_sets_filename=filename_forces,
                      born_filename=filename_born)
        phonon.symmetrize_force_constants()
        return phonon


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDynamicStructureFactor)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
