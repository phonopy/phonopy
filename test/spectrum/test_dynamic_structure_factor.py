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

data_AFF = """24.667711 107.241033  56.790650   1.503984   6.511022   4.367984
 11.096544  20.711955  14.740982   1.433353   6.396630   4.378300
  0.278391  13.365829   6.897108   1.011313   6.702354   4.409389
  1.236102   6.291708   4.204622   0.003613   7.620877   4.443622
  4.883593   0.023340   3.061993   0.132558   7.394646   4.458698
  0.617678   3.079363   2.596512   0.277884   7.115873   4.423664
  2.402336   0.834150   0.477915   6.703289   2.558175   4.282036
  3.403626   0.000000   0.000000   6.746252   2.960792   3.903300
  0.843466   3.844651   4.055997   1.451136   4.058122   2.996062
  7.076301   1.312907   0.041285   2.115464   5.993513   1.423553"""

data_b = """419.130446 1822.138324 963.242515  13.761210  59.574804  39.950123
197.165487 368.013923 260.035228  13.428822  59.928869  40.951942
  5.179328 248.664406 126.202841   9.703159  64.306530  42.148073
 23.998851 122.153147  79.243864   0.035467  74.818658  43.331628
 97.966965   0.468213  58.692973   1.328661  74.118529  44.206394
 12.580089  62.716603  49.697403   2.831812  72.515311  44.331894
 48.416674  16.811463   4.914526  68.931708  47.766185  42.920032
 65.462476   0.000000   0.000000  68.889885  52.478615  38.287576
 14.660547  66.825066  39.113987  13.994019  66.322994  27.156663
105.091902  19.498305   0.282731  14.487294  89.011152   9.748889"""


def get_func_AFF(f_params):
    def func(symbol, Q):
        return atomic_form_factor_WK1995(Q, f_params[symbol])
    return func


class TestDynamicStructureFactor(unittest.TestCase):
    def setUp(self):
        self.phonon = self._get_phonon()
        mesh = [5, 5, 5]
        self.phonon.run_mesh(mesh,
                             is_mesh_symmetry=False,
                             with_eigenvectors=True)

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
        n_points = 11  # Gamma point is excluded, so will be len(S) = 10.
        G_points_cubic = ([7, 1, 1], )

        # Atomic form factor
        self._run(G_points_cubic,
                  directions,
                  func_AFF=get_func_AFF(f_params),
                  n_points=n_points)
        Q, S = self.phonon.get_dynamic_structure_factor()
        data_cmp = np.reshape([float(x) for x in data_AFF.split()], (-1, 6))
        if verbose:
            for S_at_Q in S:
                print(("%10.6f " * 6) % tuple(S_at_Q))

        # Treatment of degeneracy
        for i in (([0, 1], [2], [3, 4], [5])):
            np.testing.assert_allclose(
                S[:6, i].sum(axis=1), data_cmp[:6, i].sum(axis=1), atol=1e-5)
        for i in (([0, 1], [2, 3], [4], [5])):
            np.testing.assert_allclose(
                S[6:, i].sum(axis=1), data_cmp[6:, i].sum(axis=1), atol=1e-5)

        # Scattering lengths
        self._run(G_points_cubic,
                  directions,
                  scattering_lengths={'Na': 3.63, 'Cl': 9.5770},
                  n_points=n_points)
        Q, S = self.phonon.get_dynamic_structure_factor()
        data_cmp = np.reshape([float(x) for x in data_b.split()], (-1, 6))
        if verbose:
            for S_at_Q in S:
                print(("%10.6f " * 6) % tuple(S_at_Q))

        # Treatment of degeneracy
        for i in (([0, 1], [2], [3, 4], [5])):
            np.testing.assert_allclose(
                S[:6, i].sum(axis=1), data_cmp[:6, i].sum(axis=1), atol=1e-5)
        for i in (([0, 1], [2, 3], [4], [5])):
            np.testing.assert_allclose(
                S[6:, i].sum(axis=1), data_cmp[6:, i].sum(axis=1), atol=1e-5)

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
        self.phonon.run_band_structure([G_to_L])

        T = 300
        for G_cubic in G_points_cubic:
            G_prim = np.dot(G_cubic, P)
            for direction in directions:
                direction_prim = np.dot(direction, P)
                G_to_L = np.array(
                    [direction_prim * x
                     for x in np.arange(1, n_points) / float(n_points - 1)])
                if func_AFF is not None:
                    self.phonon.init_dynamic_structure_factor(
                        G_to_L + G_prim,
                        T,
                        atomic_form_factor_func=func_AFF,
                        freq_min=1e-3)
                elif scattering_lengths is not None:
                    self.phonon.init_dynamic_structure_factor(
                        G_to_L + G_prim,
                        T,
                        scattering_lengths=scattering_lengths,
                        freq_min=1e-3)
                else:
                    raise SyntaxError
                dsf = self.phonon.dynamic_structure_factor
                for i, S in enumerate(dsf):
                    pass

    def _get_phonon(self):
        filename_cell = os.path.join(data_dir, "..", "POSCAR_NaCl")
        filename_forces = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        phonon = load(supercell_matrix=[2, 2, 2],
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
