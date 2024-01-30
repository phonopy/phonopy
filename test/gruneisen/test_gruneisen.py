"""Tests for mode Grueneisen parameter calculations."""

import numpy as np

from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.phonon.band_structure import get_band_qpoints

g_mesh = """0.12500 0.12500 0.12500
1.04385 1.04385 1.68241 4.56751 4.56751 7.32195
2.38570 2.38570 1.07782 2.71789 2.71789 1.14708
0.37500 0.12500 0.12500
2.29266 2.63503 3.82479 4.06171 4.68276 6.60720
0.75436 2.92943 1.47937 2.55250 2.77701 1.20036
-0.37500 0.12500 0.12500
3.07344 3.48896 3.91785 4.14173 4.98868 6.08005
1.89395 1.99378 2.33219 2.73322 1.49340 1.29484
-0.12500 0.12500 0.12500
1.69982 1.88158 3.16537 4.49215 4.65310 6.74445
0.95084 1.42584 2.02329 2.68580 2.77269 1.14648
0.37500 0.37500 0.12500
2.28442 2.55009 4.45582 4.52108 4.75914 5.67818
0.56672 0.74916 2.10550 2.58308 2.80330 1.29811
0.62500 0.37500 0.12500
3.10169 3.72884 3.84990 4.62905 5.05864 5.16538
0.75942 0.78216 1.15448 2.30108 2.49176 2.49031
-0.12500 0.37500 0.12500
2.82346 3.32426 3.91327 4.62218 4.76938 5.82155
0.76246 1.98425 2.02484 2.04431 2.69811 1.34916
-0.37500 -0.37500 0.12500
2.59000 2.90443 4.17309 4.82480 5.01936 5.18587
0.46441 0.51933 0.41135 2.79428 2.46040 2.73791
0.37500 0.37500 0.37500
2.81962 2.81962 4.10816 4.10816 4.50450 6.61282
2.41042 2.41042 2.62666 2.62666 1.22028 1.22182
0.62500 0.37500 0.37500
3.24912 3.44175 4.14300 4.54509 4.95687 5.44715
1.63622 0.53249 2.81233 1.63424 2.32545 1.68339
"""
g_mesh_weights = [2, 6, 6, 6, 6, 12, 12, 6, 2, 6]

band_distances = [
    0.00000,
    0.01524,
    0.03048,
    0.04573,
    0.06097,
    0.07621,
    0.09145,
    0.10670,
    0.12194,
    0.13718,
]

band_qpoints = [
    [0.05, 0.05, 0.05],
    [0.1, 0.1, 0.1],
    [0.15, 0.15, 0.15],
    [0.2, 0.2, 0.2],
    [0.25, 0.25, 0.25],
    [0.3, 0.3, 0.3],
    [0.35, 0.35, 0.35],
    [0.4, 0.4, 0.4],
    [0.45, 0.45, 0.45],
    [0.5, 0.5, 0.5],
]

band_freqs = [
    [0.42308, 0.42308, 0.68031, 4.64043, 4.64043, 7.42914],
    [0.83972, 0.83972, 1.35220, 4.59739, 4.59739, 7.36643],
    [1.24443, 1.24443, 2.00744, 4.53339, 4.53339, 7.27021],
    [1.63317, 1.63317, 2.63788, 4.45593, 4.45593, 7.14938],
    [2.00294, 2.00294, 3.23475, 4.36962, 4.36962, 7.01118],
    [2.35072, 2.35072, 3.78773, 4.27444, 4.27444, 6.85963],
    [2.67175, 2.67175, 4.28363, 4.16713, 4.16713, 6.69691],
    [2.95657, 2.95657, 4.70278, 4.04627, 4.04627, 6.52914],
    [3.18151, 3.18151, 5.00801, 3.92482, 3.92482, 6.38074],
    [3.27963, 3.27963, 5.12773, 3.86212, 3.86212, 6.31436],
]

band_gammas = [
    [2.36459, 2.36459, 1.05823, 2.68689, 2.68689, 1.11547],
    [2.37765, 2.37765, 1.06969, 2.70671, 2.70671, 1.13436],
    [2.39379, 2.39379, 1.08730, 2.72775, 2.72775, 1.16105],
    [2.40734, 2.40734, 1.10963, 2.73745, 2.73745, 1.18954],
    [2.41461, 2.41461, 1.13585, 2.72714, 2.72714, 1.21346],
    [2.41536, 2.41536, 1.16600, 2.69593, 2.69593, 1.22738],
    [2.41229, 2.41229, 1.20083, 2.65080, 2.65080, 1.22756],
    [2.40875, 2.40875, 1.24105, 2.60380, 2.60380, 1.21218],
    [2.40561, 2.40561, 1.28357, 2.56938, 2.56938, 1.18435],
    [2.40258, 2.40258, 1.30649, 2.55861, 2.55861, 1.16628],
]


def test_gruneisen_mesh(ph_nacl_gruneisen):
    """Test of mode Grueneisen parameter calculation on sampling mesh."""
    ph0, ph_minus, ph_plus = ph_nacl_gruneisen
    phg = PhonopyGruneisen(ph0, ph_minus, ph_plus)
    phg.set_mesh([4, 4, 4])
    # qpoints, weights, freqs, eigvecs, gamma = phg.get_mesh()
    weights = []
    g_mesh_vals = np.reshape(np.asarray(g_mesh.split(), dtype="double"), (-1, 15))
    for i, (qpt, w, freqs, _, gammas) in enumerate(zip(*phg.get_mesh())):
        weights.append(w)
        # print(" ".join(["%.5f" % v for v in qpt]))
        # print(" ".join(["%.5f" % v for v in freqs]))
        # print(" ".join(["%.5f" % v for v in gammas]))
        np.testing.assert_allclose(np.r_[qpt, freqs, gammas], g_mesh_vals[i], atol=1e-5)
    np.testing.assert_array_equal(weights, g_mesh_weights)


def test_gruneisen_band(ph_nacl_gruneisen):
    """Test of mode Grueneisen parameter calculation along band paths."""
    paths = get_band_qpoints([[[0.05, 0.05, 0.05], [0.5, 0.5, 0.5]]], npoints=10)
    ph0, ph_minus, ph_plus = ph_nacl_gruneisen
    phg = PhonopyGruneisen(ph0, ph_minus, ph_plus)
    phg.set_band_structure(paths)
    qpoints, distances, freqs, _, gammas = phg.get_band_structure()
    # print(", ".join(["%.5f" % v for v in distances[0]]))
    np.testing.assert_allclose(distances[0], band_distances, atol=1e-5)
    np.testing.assert_allclose(qpoints[0], band_qpoints, atol=1e-5)
    # for line in freqs[0].reshape(-1, 6):
    #     print(", ".join(["%.5f" % v for v in line]))
    for freqs_q, band_freqs_q in zip(freqs[0], band_freqs):
        np.testing.assert_allclose(np.sort(freqs_q), np.sort(band_freqs_q), atol=1e-5)
    # for line in gammas[0].reshape(-1, 6):
    #     print(", ".join(["%.5f" % v for v in line]))
    for gammas_q, band_gammas_q in zip(gammas[0], band_gammas):
        np.testing.assert_allclose(np.sort(gammas_q), np.sort(band_gammas_q), atol=1e-5)
