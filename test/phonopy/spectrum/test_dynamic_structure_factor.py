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

data_AFF = """51.376255296 18.334995742 29.824311212 0.001619351 0.006930700 0.004549347
0.248408682 17.446592747 7.995327709 0.000606398 0.007966140 0.004353010
2.210754520 5.806876564 3.828618814 0.008368748 0.000374804 0.004013668
1.396786889 3.220296987 2.333109686 0.000740754 0.007803308 0.003298757
2.728552194 0.312369707 1.628264217 0.001317689 0.006043644 0.002035759
0.185075698 2.003082681 1.241883374 0.000009746 0.004759531 0.000479857
1.077859312 0.601638340 0.000067538 0.001154045 1.007612628 0.000382949
0.022947555 1.328506566 0.000361393 0.000755408 0.848386042 0.008711364
1.067660562 0.029482478 0.028209365 0.006837366 0.704960564 0.050603265
0.154764494 0.558327397 0.005749330 0.260595344 0.509455223 0.175799603"""

data_b = """488.005476119 174.157853180 283.396323316 0.078104375 0.334280783 0.222512030
2.411504032 169.368189835 77.734871696 0.030654493 0.402702435 0.233012989
21.927810313 57.596665152 38.106498944 0.441055393 0.019753171 0.241568763
14.164185514 32.655578521 23.806769465 0.041831884 0.440668744 0.241093041
28.342285489 3.244677313 17.081633995 0.086162866 0.395190136 0.221629241
1.976398221 21.390647607 13.459023369 0.000893079 0.436134441 0.172349296
11.908545725 6.647099119 0.018060438 0.308604222 11.371574693 0.088669862
0.265554389 15.373784116 0.043230528 0.090363214 10.113320387 0.004418905
13.353241625 0.368737650 0.017140131 0.004154413 9.132411968 0.145120620
2.298445938 8.291858819 0.039373029 1.784630217 7.566046002 1.203925125"""

def get_func_AFF(f_params):
    def func(symbol, Q):
        return atomic_form_factor_WK1995(Q, f_params[symbol])
    return func


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
