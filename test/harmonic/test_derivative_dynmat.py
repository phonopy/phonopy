"""Tests for routines in derivative_dynmat.py."""

import numpy as np

from phonopy import Phonopy
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL

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
    """Test by NaCl with Wang's NAC (C and Python paths)."""
    _assert(ph_nacl_wang, ddm_ph_nacl, show=True)
    _assert(ph_nacl_wang, ddm_ph_nacl, show=True, force_python=True)


def test_ddm_nac_compact(ph_nacl_compact_fcsym: Phonopy):
    """Test by NaCl with compact fc."""
    _assert(ph_nacl_compact_fcsym, ddm_ph_nacl)


def test_ddm_nonac(ph_nacl_nonac: Phonopy):
    """Test by NaCl without NAC."""
    _assert(ph_nacl_nonac, ddm_ph_nacl_nonac)


def test_ddm_nonac_compact(ph_nacl_nonac_compact_fc: Phonopy):
    """Test by NaCl without NAC and with compact fc."""
    _assert(ph_nacl_nonac_compact_fc, ddm_ph_nacl_nonac)


def _assert(
    ph: Phonopy, ref_vals: list, show: bool = False, force_python: bool = False
):
    dynmat = ph.dynamical_matrix
    assert dynmat is not None
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    for i, q in enumerate(([0, 0.1, 0.1], [0, 0.25, 0.25])):
        ddynmat.run(q, force_python=force_python)
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


def test_ddm_gonze_lee_matches_fd(ph_nacl: Phonopy):
    """Analytic Python Gonze-Lee derivative matches central FD.

    The Python derivative path computes d D / d q_cart of the full
    Gonze-Lee dynamical matrix (short-range + dipole-dipole).  Validate
    against a central finite difference in the same q_cart convention:
    a step of ``h * e_mu`` in Cartesian q corresponds to a reduced-q
    step of ``h * (e_mu @ cell.T)``.

    """
    dm = ph_nacl.dynamical_matrix
    assert isinstance(dm, DynamicalMatrixGL)

    q = np.array([0.1, 0.2, 0.3])
    h = 1e-4

    ddm_obj = DerivativeOfDynamicalMatrix(dm)
    ddm_obj.run(q, force_python=True)
    analytic = ddm_obj.d_dynamical_matrix
    assert analytic is not None

    cell = ph_nacl.primitive.cell
    fd = np.zeros_like(analytic)
    for mu in range(3):
        step_red = h * (np.eye(3)[mu] @ cell.T)  # h * e_mu in q_cart
        dm.run(q + step_red)
        assert dm.dynamical_matrix is not None
        d_plus = dm.dynamical_matrix.copy()
        dm.run(q - step_red)
        assert dm.dynamical_matrix is not None
        d_minus = dm.dynamical_matrix.copy()
        fd[mu] = (d_plus - d_minus) / (2 * h)
        # FD is not exactly Hermitian; symmetrize for fair comparison.
        fd[mu] = (fd[mu] + fd[mu].conj().T) / 2

    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-6)
