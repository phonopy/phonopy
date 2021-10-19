"""Tests for mode Grueneisen parameter calculations."""
import numpy as np
from phonopy.api_gruneisen import PhonopyGruneisen

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
