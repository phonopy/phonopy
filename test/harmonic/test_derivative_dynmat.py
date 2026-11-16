# SPDX-License-Identifier: BSD-3-Clause
"""Tests for routines in derivative_dynmat.py."""

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL

# Wang NAC reference values (the C kernel computes Wang correctly).
ddm_ph_nacl_wang = [
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

# Gonze-Lee NAC reference values (Rust kernel +
# _gonze_lee_d_recip_dipole_dipole, validated against central FD in
# test_ddm_gonze_lee_matches_fd).
ddm_ph_nacl_gonze = [
    [
        -0.2568989,
        0.6946391,
        0.0338302,
        0.0342899,
        0.0338302,
        0.0342899,
        0.6946391,
        0.0560808,
        0.0342899,
        0.1430933,
        0.0342899,
        0.1430933,
        2.3669647,
        -1.7615471,
        2.3669647,
        -1.7615471,
        -1.7615471,
        1.5752198,
        -1.7615471,
        1.5752198,
        2.3669647,
        -1.7615471,
        2.3669647,
        -1.7615471,
        -1.7615471,
        1.5752198,
        -1.7615471,
        1.5752198,
    ],
    [
        -0.4580575,
        1.1734573,
        0.0782656,
        0.0543037,
        0.0782656,
        0.0543037,
        1.1734573,
        0.0725238,
        0.0543037,
        0.2215042,
        0.0543037,
        0.2215042,
        0.8931162,
        -0.4662398,
        0.8931162,
        -0.4662398,
        -0.4662398,
        0.6477843,
        -0.4662398,
        0.6477843,
        0.8931162,
        -0.4662398,
        0.8931162,
        -0.4662398,
        -0.4662398,
        0.6477843,
        -0.4662398,
        0.6477843,
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
    """Test by NaCl with Gonze-Lee NAC."""
    _assert(ph_nacl, ddm_ph_nacl_gonze)


def test_ddm_nac_wang(ph_nacl_wang: Phonopy):
    """Test by NaCl with Wang's NAC (C and Python paths)."""
    _assert(ph_nacl_wang, ddm_ph_nacl_wang, show=True)
    _assert(ph_nacl_wang, ddm_ph_nacl_wang, show=True, force_python=True)


def test_ddm_nac_compact(ph_nacl_compact_fcsym: Phonopy):
    """Test by NaCl with Gonze-Lee NAC and compact fc."""
    _assert(ph_nacl_compact_fcsym, ddm_ph_nacl_gonze)


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


_QPOINTS_FOR_BATCH: list[list[float]] = [
    [0, 0, 0],
    [0.5, 0, 0],
    [0.5, 0.25, 0.75],
    [0.1, 0.2, 0.3],
    [1.0, 0, 0],
    [0, 0, 0],
]
_NACL_FIXTURES = [
    "ph_nacl_nonac",
    "ph_nacl_nonac_compact_fc",
    "ph_nacl_wang",
    "ph_nacl",
    "ph_nacl_compact_fcsym",
]


def _run_each(
    ddynmat: DerivativeOfDynamicalMatrix,
    q_direction: list[float] | None,
    force_python: bool = False,
) -> np.ndarray:
    """Return derivatives by calling run at each q-point."""
    ddms = []
    for q in _QPOINTS_FOR_BATCH:
        ddynmat.run(q, q_direction=q_direction, force_python=force_python)
        ddm = ddynmat.d_dynamical_matrix
        assert ddm is not None
        ddms.append(ddm.copy())
    return np.array(ddms)


@pytest.mark.parametrize("q_direction", [None, [1, 0, 0], [0.3, -0.2, 0.9]])
@pytest.mark.parametrize("fixture_name", _NACL_FIXTURES)
def test_run_at_qpoints(
    request: pytest.FixtureRequest, fixture_name: str, q_direction: list[float] | None
):
    """Test that run_at_qpoints agrees with run at each q-point.

    The kernel for many q-points makes all elements Hermitian and that for
    one q-point does not, so the agreement is not bitwise.

    """
    ph: Phonopy = request.getfixturevalue(fixture_name)
    dynmat = ph.dynamical_matrix
    assert dynmat is not None
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    assert ddynmat._lang == "Rust"
    num_band = len(ph.primitive) * 3

    ddynmat.run([0.1, 0.1, 0.1])
    ddm_before = ddynmat.d_dynamical_matrix
    ddms = ddynmat.run_at_qpoints(_QPOINTS_FOR_BATCH, q_direction=q_direction)
    assert ddynmat.d_dynamical_matrix is ddm_before

    assert ddms.shape == (len(_QPOINTS_FOR_BATCH), 3, num_band, num_band)
    assert ddms.dtype == np.cdouble
    assert np.abs(ddms).max() > 0.1
    np.testing.assert_allclose(
        ddms, _run_each(ddynmat, q_direction), rtol=0, atol=1e-12
    )

    ddms_py = ddynmat.run_at_qpoints(
        _QPOINTS_FOR_BATCH, q_direction=q_direction, force_python=True
    )
    assert ddynmat.d_dynamical_matrix is not None
    np.testing.assert_allclose(ddms_py, ddms, rtol=0, atol=1e-12)


def test_run_at_qpoints_q_direction_is_used(ph_nacl_wang: Phonopy):
    """Test that q_direction changes the results of run_at_qpoints.

    This confirms that test_run_at_qpoints can detect q_direction that is
    not passed to the kernel.  With Wang's NAC, q_direction replaces q in
    the NAC term at every q-point.  The Gamma points and the general
    q-point are examined.  At the other q-points, the results do not depend
    on q_direction: the NAC term vanishes at q-points commensurate with the
    supercell, and [1, 0, 0] is parallel to one of the q-directions.

    """
    dynmat = ph_nacl_wang.dynamical_matrix
    assert dynmat is not None
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    ddms = ddynmat.run_at_qpoints(_QPOINTS_FOR_BATCH)
    ddms_x = ddynmat.run_at_qpoints(_QPOINTS_FOR_BATCH, q_direction=[1, 0, 0])
    ddms_y = ddynmat.run_at_qpoints(_QPOINTS_FOR_BATCH, q_direction=[0, 1, 0])
    for i in (0, 3, 5):
        assert np.abs(ddms_x[i] - ddms[i]).max() > 1e-3
        assert np.abs(ddms_x[i] - ddms_y[i]).max() > 1e-3


@pytest.mark.parametrize("q_direction", [None, [1, 0, 0]])
@pytest.mark.parametrize(
    "fixture_name", ["ph_nacl_nonac", "ph_nacl_nonac_compact_fc", "ph_nacl_wang"]
)
def test_run_at_qpoints_c(
    request: pytest.FixtureRequest, fixture_name: str, q_direction: list[float] | None
):
    """Test run_at_qpoints with the C backend against the Rust backend."""
    pytest.importorskip("phonopy._phonopy")
    ph: Phonopy = request.getfixturevalue(fixture_name)
    dynmat = ph.dynamical_matrix
    assert dynmat is not None
    ddms_c = DerivativeOfDynamicalMatrix(dynmat, lang="C").run_at_qpoints(
        _QPOINTS_FOR_BATCH, q_direction=q_direction
    )
    ddms_rust = DerivativeOfDynamicalMatrix(dynmat, lang="Rust").run_at_qpoints(
        _QPOINTS_FOR_BATCH, q_direction=q_direction
    )
    np.testing.assert_allclose(ddms_c, ddms_rust, rtol=0, atol=1e-12)


def test_run_at_qpoints_c_gonze_lee_raises(ph_nacl: Phonopy):
    """Test that the C backend refuses Gonze-Lee NAC in run_at_qpoints."""
    pytest.importorskip("phonopy._phonopy")
    dynmat = ph_nacl.dynamical_matrix
    assert dynmat is not None
    ddynmat = DerivativeOfDynamicalMatrix(dynmat, lang="C")
    with pytest.raises(NotImplementedError):
        ddynmat.run_at_qpoints(_QPOINTS_FOR_BATCH)


def test_run_at_qpoints_wrong_shape(ph_nacl_nonac: Phonopy):
    """Test that run_at_qpoints refuses one q-point given as a 1D array."""
    dynmat = ph_nacl_nonac.dynamical_matrix
    assert dynmat is not None
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    with pytest.raises(ValueError):
        ddynmat.run_at_qpoints(np.array([0.1, 0.2, 0.3]))


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
