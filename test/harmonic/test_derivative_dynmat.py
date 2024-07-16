"""Tests for routines in derivative_dynmat.py."""

import numpy as np

from phonopy import Phonopy
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

ddm_ph_nacl = [
    [
        -0.2568036,
        0.6947168,
        0.0337825,
        0.0342511,
        0.0337825,
        0.0342511,
        0.6947168,
        0.0561427,
        0.0342511,
        0.1430624,
        0.0342511,
        0.1430624,
        2.0750808,
        -1.5152406,
        2.0750808,
        -1.5152406,
        -1.5152406,
        1.3859454,
        -1.5152406,
        1.3859454,
        2.0750808,
        -1.5152406,
        2.0750808,
        -1.5152406,
        -1.5152406,
        1.3859454,
        -1.5152406,
        1.3859454,
    ],
    [
        -0.4582279,
        1.1733186,
        0.0783508,
        0.0543731,
        0.0783508,
        0.0543731,
        1.1733186,
        0.0724132,
        0.0543731,
        0.2215595,
        0.0543731,
        0.2215595,
        0.5943896,
        -0.2136538,
        0.5943896,
        -0.2136538,
        -0.2136538,
        0.4540728,
        -0.2136538,
        0.4540728,
        0.5943896,
        -0.2136538,
        0.5943896,
        -0.2136538,
        -0.2136538,
        0.4540728,
        -0.2136538,
        0.4540728,
    ],
]

ddm_ph_nacl_nonac = [
    [
        0.8832905,
        -0.2233653,
        0.0337825,
        0.0342511,
        0.0337825,
        0.0342511,
        -0.2233653,
        0.7954454,
        0.0342511,
        0.1430624,
        0.0342511,
        0.1430624,
        0.3493735,
        -0.1255825,
        0.3493735,
        -0.1255825,
        -0.1255825,
        0.2668973,
        -0.1255825,
        0.2668973,
        0.3493735,
        -0.1255825,
        0.3493735,
        -0.1255825,
        -0.1255825,
        0.2668973,
        -0.1255825,
        0.2668973,
    ],
    [
        0.2826501,
        0.5767127,
        0.0783508,
        0.0543731,
        0.0783508,
        0.0543731,
        0.5767127,
        0.5528413,
        0.0543731,
        0.2215595,
        0.0543731,
        0.2215595,
        0.5943896,
        -0.2136538,
        0.5943896,
        -0.2136538,
        -0.2136538,
        0.4540728,
        -0.2136538,
        0.4540728,
        0.5943896,
        -0.2136538,
        0.5943896,
        -0.2136538,
        -0.2136538,
        0.4540728,
        -0.2136538,
        0.4540728,
    ],
]


def test_ddm_nac(ph_nacl: Phonopy):
    """Test by NaCl."""
    _assert(ph_nacl, ddm_ph_nacl)


def test_ddm_nac_wang(ph_nacl_wang: Phonopy):
    """Test by NaCl with Wang's NAC."""
    _assert(ph_nacl_wang, ddm_ph_nacl, show=True)


def test_ddm_nac_compact(ph_nacl_compact_fcsym: Phonopy):
    """Test by NaCl with compact fc."""
    _assert(ph_nacl_compact_fcsym, ddm_ph_nacl)


def test_ddm_nonac(ph_nacl_nonac: Phonopy):
    """Test by NaCl without NAC."""
    _assert(ph_nacl_nonac, ddm_ph_nacl_nonac)


def test_ddm_nonac_compact(ph_nacl_nonac_compact_fc: Phonopy):
    """Test by NaCl without NAC and with compact fc."""
    _assert(ph_nacl_nonac_compact_fc, ddm_ph_nacl_nonac)


def _assert(ph: Phonopy, ref_vals, show=False):
    dynmat = ph.dynamical_matrix
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    for i, q in enumerate(([0, 0.1, 0.1], [0, 0.25, 0.25])):
        ddynmat.run(q)
        ddm = ddynmat.d_dynamical_matrix
        condition = np.abs(ddm) > 1e-8
        vals = np.extract(condition, ddm).real
        if show:
            _show(vals)
        np.testing.assert_allclose(vals, ref_vals[i], rtol=0, atol=1e-7)


def _show(vals):
    for i, v in enumerate(vals):
        print("%.7f, " % v, end="")
        if (i + 1) % 5 == 0:
            print("")
    print("")
