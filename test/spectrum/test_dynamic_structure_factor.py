import unittest

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995
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

data_AFF = """51.516246931 18.384955515 29.905577144 0.001623658 0.006949134 0.004561443
0.249097614 17.494978817 8.017501496 0.000608040 0.007987706 0.004364780
2.216994883 5.823267810 3.839425632 0.008391811 0.000375837 0.004024696
1.400799651 3.229548424 2.339811994 0.000742831 0.007825185 0.003307951
2.736529702 0.313282986 1.633024386 0.001321441 0.006060851 0.002041480
0.185626350 2.009042403 1.245577852 0.000009774 0.004773226 0.000481168
1.081122495 0.603459780 0.000067729 0.001157309 1.010662594 0.000384217
0.023018236 1.332598541 0.000362585 0.000757898 0.850998533 0.008738735
1.071005293 0.029574839 0.028298892 0.006859066 0.707168385 0.050763178
0.155257197 0.560104869 0.005767737 0.261429685 0.511077106 0.176362456"""

data_b = """489.334703965 174.632223816 284.168234941 0.078315115 0.335182733 0.223112376
2.418189439 169.837728889 77.950373216 0.030738690 0.403808512 0.233652859
21.989681681 57.759179530 38.214017010 0.442288550 0.019808399 0.242243866
14.204860288 32.749354353 23.875131165 0.041950927 0.441922770 0.241778572
28.425115222 3.254159814 17.131550944 0.086412362 0.396334462 0.222270111
1.982276035 21.454263455 13.499046170 0.000895709 0.437418806 0.172855557
11.944583112 6.667214428 0.018114464 0.309527374 11.405981860 0.088933474
0.266372000 15.421118189 0.043361198 0.090636349 10.144451724 0.004431416
13.395060379 0.369892437 0.017196325 0.004168033 9.161005619 0.145581653
2.305763188 8.318256484 0.039499088 1.790344015 7.590132995 1.207779697"""


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

    def test_IXS_G_to_L(self):
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
                S[:, i].sum(axis=1), data_cmp[:, i].sum(axis=1), atol=1e-1)

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
                print("# G_cubic %s, G_prim %s" % (G_cubic, G_prim))

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
                    Q_prim = dsf.qpoints[i]
                    Q_cubic = np.dot(Q_prim, np.linalg.inv(P))

                    # print(("%11.9f " * len(S)) % tuple(S))

                    if verbose:
                        print("%f  %f %f %f  %f %f %f %f  %f %f %f %f" %
                              ((distances[0][i + 1], ) + tuple(Q_cubic) +
                               tuple(frequencies[0][i + 1][[0, 2, 3, 5]]
                                     * THzToEv * 1000) +
                               ((S[0] + S[1]) / 2, S[2], (S[3] + S[4]) / 2,
                                S[5])))
                if verbose:
                    print("")

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
        phonon.symmetrize_force_constants()
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDynamicStructureFactor)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
