"""Tests for dynamic structure factor."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from phonopy.api_phonopy import Phonopy
from phonopy.spectrum.dynamic_structure_factor import atomic_form_factor_WK1995

# D. Waasmaier and A. Kirfel, Acta Cryst. A51, 416 (1995)
# f(Q) = \sum_i a_i \exp((-b_i Q^2) + c
# Q is in angstron^-1
# a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, c
f_params = {
    "Na": [
        3.148690,
        2.594987,
        4.073989,
        6.046925,
        0.767888,
        0.070139,
        0.995612,
        14.1226457,
        0.968249,
        0.217037,
        0.045300,
    ],  # 1+
    "Cl": [
        1.061802,
        0.144727,
        7.139886,
        1.171795,
        6.524271,
        19.467656,
        2.355626,
        60.320301,
        35.829404,
        0.000436,
        -34.916604,
    ],  # 1-
    "Si": [
        5.275329,
        2.631338,
        3.191038,
        33.730728,
        1.511514,
        0.081119,
        1.356849,
        86.288640,
        2.519114,
        1.170087,
        0.145073,
    ],  # neutral
    "Pb": [
        32.505656,
        1.047035,
        20.014240,
        6.670321,
        14.645661,
        0.105279,
        5.029499,
        16.525040,
        1.760138,
        0.105279,
        4.044678,
    ],  # 4+
    "Pb0": [
        16.419567,
        0.105499,
        32.738592,
        1.055049,
        6.530247,
        25.025890,
        2.342742,
        80.906596,
        19.916475,
        6.664449,
        4.049824,
    ],
    "Te": [
        6.660302,
        33.031656,
        6.940756,
        0.025750,
        19.847015,
        5.065547,
        1.557175,
        84.101613,
        17.802427,
        0.487660,
        -0.806668,
    ],
}  # neutral

data_AFF_Si = """1617.345756 1873.066252 503.856264  11.001319   8.079923   0.055787
932.668536   4.059507 124.346263  12.115047   0.020839   7.462799
442.713078   8.428824  53.809376  13.270402   2.033486   4.808897
277.504172   0.742826  29.008863  14.488787   3.570989   2.622653
  7.700914 190.329529  17.434621  15.811378   0.956863   4.567044
128.242061  27.189284  11.021400  17.307531   3.613591   1.213426
 42.672187  88.737326   6.948860  19.094662   4.045577   0.061621
  5.078139 112.962641   3.919888  21.394182   3.381237   0.000000
 17.263664  94.427377   1.063912  24.636357   0.316365   2.361231
109.690680   1.039754   4.324087  22.750438   1.942733   0.087975"""

data_b_Si = """1207.369580 1398.268247 376.135235   8.212628   6.031767   0.041645
710.488237   3.092451  94.724496   9.229001   0.015875   5.685011
344.213037   6.553479  41.837230  10.317846   1.581052   3.738956
220.257270   0.589587  23.024565  11.499866   2.834322   2.081621
  6.240743 154.241134  14.128841  12.813381   0.775432   3.701087
106.128611  22.500894   9.120923  14.323103   2.990480   1.004189
 36.068376  75.004623   5.873477  16.139634   3.419496   0.052085
  4.384648  97.536024   3.384573  18.472509   2.919482   0.000000
 15.229216  83.299522   0.938535  21.733070   0.279083   2.082970
 98.876152   0.937243   3.897770  20.507447   1.751197   0.079302 """

data_AFF_NaCl = """13.224708 710.969308 311.677433   2.873317  36.549683  21.483221
 13.145117 168.331089  83.948759  11.537634  27.187230  21.646819
  4.999420  75.939632  40.711985  22.059842  16.300749  21.908902
  1.389114  44.997343  25.647179   9.988515  28.129482  22.174846
  4.475042  26.807966  19.185649   0.607678  37.194983  22.317920
  5.456779  18.725760  16.551901   5.143921  32.097030  22.156059
  9.026793  12.399434   0.143074  36.016896  16.372369  21.360950
  0.021718  22.320823   0.478572  33.239975  18.712015  19.205739
  0.466439  29.112946   7.255176  19.464002  24.860848  14.174948
 25.956886  23.258523   0.301924   8.709088  35.161032   5.947678"""

data_b_NaCl = """40.928422 2200.339853 963.242578   5.345042  67.990960  39.950117
 40.938414 524.240900 260.035243  21.856090  51.501594  40.951936
 15.679344 238.164364 126.202848  42.560396  31.449289  42.148068
  4.376745 141.775245  79.243867  19.614922  55.239203  43.331623
 14.081177  84.353999  58.692973   1.212815  74.234377  44.206389
 16.990664  58.306028  49.697402  10.407351  64.939771  44.331889
 27.480380  37.747754   0.292186  73.554041  47.766181  42.920026
  0.063631  65.398838   0.977764  67.912112  52.478607  38.287570
  1.284951  80.200648  14.420648  38.687349  66.322982  27.156660
 65.710583  58.879603   0.494887  14.275135  89.011137   9.748886 """


def _get_func_AFF(f_params: dict):
    """Return atomic_form_factor function."""

    def func(symbol: str, Q: float) -> float:
        return atomic_form_factor_WK1995(Q, f_params[symbol])

    return func


def test_IXS_G_to_L_Si(ph_si: Phonopy):
    """Test of dynamic structure factor calculation."""
    degeneracies = [[[0, 1], [2], [3], [4, 5]]] * 10
    _test_IXS_G_to_L(
        ph_si, data_b_Si, data_AFF_Si, {"Si": 4.1491}, degeneracies, verbose=True
    )


def test_IXS_G_to_L_NaCl(ph_nacl: Phonopy):
    """Test of dynamic structure factor calculation with NaCl."""
    degeneracies = [[[0, 1], [2], [3, 4], [5]]] * 6 + [[[0, 1], [2, 3], [4], [5]]] * 4
    _test_IXS_G_to_L(
        ph_nacl,
        data_b_NaCl,
        data_AFF_NaCl,
        {"Na": 3.63, "Cl": 9.5770},
        degeneracies,
        verbose=False,
    )


def _test_IXS_G_to_L(
    phonon: Phonopy,
    data_b: str,
    data_AFF: str,
    scattering_lengths: dict,
    degeneracies: list,
    verbose: bool = True,
):
    mesh = [5, 5, 5]
    phonon.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True)

    directions = np.array(
        [
            [0.5, 0.5, 0.5],
        ]
    )
    n_points = 11  # Gamma point is excluded, so will be len(S) = 10.
    G_points_cubic = ([7, 1, 1],)

    # Atomic form factor
    _run(
        phonon,
        G_points_cubic,
        directions,
        func_AFF=_get_func_AFF(f_params),
        n_points=n_points,
    )
    Q, S = phonon.get_dynamic_structure_factor()
    data_cmp = np.reshape([float(x) for x in data_AFF.split()], (-1, 6))
    if verbose:
        for S_at_Q in S:
            print(("%10.6f " * 6) % tuple(S_at_Q))
        for qpt in Q:
            print(qpt)

    # Treatment of degeneracy
    for j, deg_set in enumerate(degeneracies):
        for deg in deg_set:
            np.testing.assert_allclose(
                S[j, deg].sum(), data_cmp[j, deg].sum(), atol=1e-5
            )

    # Scattering lengths
    _run(
        phonon,
        G_points_cubic,
        directions,
        scattering_lengths=scattering_lengths,
        n_points=n_points,
    )
    Q, S = phonon.get_dynamic_structure_factor()
    data_cmp = np.reshape([float(x) for x in data_b.split()], (-1, 6))
    if verbose:
        for S_at_Q in S:
            print(("%10.6f " * 6) % tuple(S_at_Q))
        for qpt in Q:
            print(qpt)

    # Treatment of degeneracy
    for j, deg_set in enumerate(degeneracies):
        for deg in deg_set:
            np.testing.assert_allclose(
                S[j, deg].sum(), data_cmp[j, deg].sum(), atol=1e-5
            )


def plot_f_Q(f_params: dict):
    """Plot f_Q."""
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 6.0, 101)
    for elem in ("Si", "Na", "Cl", "Pb", "Pb0", "Te"):
        y = [atomic_form_factor_WK1995(Q, f_params[elem]) for Q in x]
        plt.plot(x, y, label=elem)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()


def show(phonon):
    """Show the calculation result along many directions."""
    directions = np.array(
        [
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ]
    )
    G_points_cubic = ([7, 1, 1],)
    _run(phonon, G_points_cubic, directions, verbose=True)


def _run(
    phonon: Phonopy,
    G_points_cubic: tuple[list],
    directions: ArrayLike,
    func_AFF: Callable | None = None,
    scattering_lengths: dict | None = None,
    n_points: int = 51,
):
    P = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    G_to_L = np.array(
        [directions[0] * x for x in np.arange(0, n_points) / float(n_points - 1)]
    )

    phonon.run_band_structure([G_to_L])

    T = 300
    for G_cubic in G_points_cubic:
        G_prim = np.dot(G_cubic, P)
        for direction in directions:
            direction_prim = np.dot(direction, P)
            G_to_L = np.array(
                [
                    direction_prim * x
                    for x in np.arange(1, n_points) / float(n_points - 1)
                ]
            )
            if func_AFF is not None:
                phonon.init_dynamic_structure_factor(
                    G_to_L + G_prim,
                    T,
                    atomic_form_factor_func=func_AFF,
                    freq_min=1e-3,
                )
            elif scattering_lengths is not None:
                phonon.init_dynamic_structure_factor(
                    G_to_L + G_prim,
                    T,
                    scattering_lengths=scattering_lengths,
                    freq_min=1e-3,
                )
            else:
                raise SyntaxError
            dsf = phonon.dynamic_structure_factor
            assert dsf is not None
            for _, _ in enumerate(dsf):
                pass
