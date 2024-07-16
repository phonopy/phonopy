"""Tests for dynamical matrix classes."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixGL

dynmat_ref_000 = [
    0.052897,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.042597,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.052897,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.042597,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.052897,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.042597,
    0.000000,
    -0.042597,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.034302,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.042597,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.034302,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    -0.042597,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.034302,
    0.000000,
]


dynmat_ref_252525 = [
    0.075295,
    0.000000,
    0.016777,
    0.000000,
    0.016777,
    0.000000,
    -0.040182,
    0.000000,
    -0.004226,
    0.000000,
    -0.004226,
    0.000000,
    0.016777,
    0.000000,
    0.075295,
    0.000000,
    0.016777,
    0.000000,
    -0.004226,
    0.000000,
    -0.040182,
    0.000000,
    -0.004226,
    0.000000,
    0.016777,
    0.000000,
    0.016777,
    0.000000,
    0.075295,
    0.000000,
    -0.004226,
    0.000000,
    -0.004226,
    0.000000,
    -0.040182,
    0.000000,
    -0.040182,
    0.000000,
    -0.004226,
    0.000000,
    -0.004226,
    0.000000,
    0.055704,
    0.000000,
    0.011621,
    0.000000,
    0.011621,
    0.000000,
    -0.004226,
    0.000000,
    -0.040182,
    0.000000,
    -0.004226,
    0.000000,
    0.011621,
    0.000000,
    0.055704,
    0.000000,
    0.011621,
    0.000000,
    -0.004226,
    0.000000,
    -0.004226,
    0.000000,
    -0.040182,
    0.000000,
    0.011621,
    0.000000,
    0.011621,
    0.000000,
    0.055704,
    0.000000,
]

dynmat_gonze_lee_ref_252525 = [
    0.081339,
    0.000000,
    0.029509,
    0.000000,
    0.029509,
    0.000000,
    -0.045098,
    0.000000,
    -0.015204,
    0.000000,
    -0.015204,
    0.000000,
    0.029509,
    0.000000,
    0.081339,
    0.000000,
    0.029509,
    0.000000,
    -0.015204,
    0.000000,
    -0.045098,
    0.000000,
    -0.015204,
    0.000000,
    0.029509,
    0.000000,
    0.029509,
    0.000000,
    0.081339,
    0.000000,
    -0.015204,
    0.000000,
    -0.015204,
    0.000000,
    -0.045098,
    0.000000,
    -0.045098,
    0.000000,
    -0.015204,
    0.000000,
    -0.015204,
    0.000000,
    0.059623,
    0.000000,
    0.019878,
    0.000000,
    0.019878,
    0.000000,
    -0.015204,
    0.000000,
    -0.045098,
    0.000000,
    -0.015204,
    0.000000,
    0.019878,
    0.000000,
    0.059623,
    0.000000,
    0.019878,
    0.000000,
    -0.015204,
    0.000000,
    -0.015204,
    0.000000,
    -0.045098,
    0.000000,
    0.019878,
    0.000000,
    0.019878,
    0.000000,
    0.059623,
    0.000000,
]

dynmat_gonze_lee_full_ref_252525 = [
    0.076944,
    0.000000,
    0.020251,
    -0.000000,
    0.020251,
    0.000000,
    -0.041523,
    -0.000000,
    -0.007221,
    0.000000,
    -0.007221,
    0.000000,
    0.020251,
    0.000000,
    0.076944,
    0.000000,
    0.020251,
    0.000000,
    -0.007221,
    0.000000,
    -0.041523,
    -0.000000,
    -0.007221,
    0.000000,
    0.020251,
    -0.000000,
    0.020251,
    -0.000000,
    0.076944,
    0.000000,
    -0.007221,
    0.000000,
    -0.007221,
    0.000000,
    -0.041523,
    -0.000000,
    -0.041523,
    0.000000,
    -0.007221,
    -0.000000,
    -0.007221,
    -0.000000,
    0.056774,
    0.000000,
    0.013874,
    -0.000000,
    0.013874,
    0.000000,
    -0.007221,
    -0.000000,
    -0.041523,
    0.000000,
    -0.007221,
    -0.000000,
    0.013874,
    0.000000,
    0.056774,
    0.000000,
    0.013874,
    -0.000000,
    -0.007221,
    -0.000000,
    -0.007221,
    -0.000000,
    -0.041523,
    0.000000,
    0.013874,
    -0.000000,
    0.013874,
    0.000000,
    0.056774,
    0.000000,
]

dynmat_wang_ref_252525 = [
    0.081339,
    -0.000000,
    0.022821,
    0.000000,
    0.022821,
    0.000000,
    -0.045098,
    -0.000000,
    -0.009142,
    -0.000000,
    -0.009142,
    0.000000,
    0.022821,
    0.000000,
    0.081339,
    0.000000,
    0.022821,
    0.000000,
    -0.009142,
    0.000000,
    -0.045098,
    0.000000,
    -0.009142,
    0.000000,
    0.022821,
    0.000000,
    0.022821,
    0.000000,
    0.081339,
    0.000000,
    -0.009142,
    0.000000,
    -0.009142,
    0.000000,
    -0.045098,
    0.000000,
    -0.045098,
    0.000000,
    -0.009142,
    0.000000,
    -0.009142,
    0.000000,
    0.059623,
    0.000000,
    0.015541,
    0.000000,
    0.015541,
    0.000000,
    -0.009142,
    0.000000,
    -0.045098,
    0.000000,
    -0.009142,
    0.000000,
    0.015541,
    0.000000,
    0.059623,
    0.000000,
    0.015541,
    0.000000,
    -0.009142,
    0.000000,
    -0.009142,
    0.000000,
    -0.045098,
    0.000000,
    0.015541,
    0.000000,
    0.015541,
    -0.000000,
    0.059623,
    0.000000,
]

dynmat_ref_555 = [
    0.091690,
    0.000000,
    0.033857,
    0.000000,
    0.033857,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.033857,
    0.000000,
    0.091690,
    0.000000,
    0.033857,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.033857,
    0.000000,
    0.033857,
    0.000000,
    0.091690,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.064909,
    0.000000,
    0.021086,
    0.000000,
    0.021086,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.021086,
    0.000000,
    0.064909,
    0.000000,
    0.021086,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.021086,
    0.000000,
    0.021086,
    0.000000,
    0.064909,
    0.000000,
]


@pytest.mark.parametrize(
    "is_compact_fc,lang", [(True, "C"), (False, "C"), (True, "Py"), (False, "Py")]
)
def test_dynmat(
    ph_nacl_nonac: Phonopy,
    ph_nacl_nonac_compact_fc: Phonopy,
    is_compact_fc: bool,
    lang: str,
):
    """Test dynamical matrix of NaCl in C and python implementations.

    1. Without NAC.
    2. Without NAC with comapact fc2.

    """
    if is_compact_fc:
        ph = ph_nacl_nonac_compact_fc
    else:
        ph = ph_nacl_nonac
    dynmat = ph.dynamical_matrix
    _test_dynmat(dynmat, lang=lang)
    _test_dynmat_252525(dynmat, dynmat_ref_252525, lang=lang)


@pytest.mark.parametrize("lang", ["C", "Py"])
def test_dynmat_dense_svecs(ph_nacl_nonac_dense_svecs: Phonopy, lang: str):
    """Test with dense svecs."""
    ph = ph_nacl_nonac_dense_svecs
    dynmat = ph.dynamical_matrix
    _test_dynmat(dynmat, lang=lang)
    _test_dynmat_252525(dynmat, dynmat_ref_252525, lang=lang)


def test_dynmat_gonze_lee(ph_nacl: Phonopy):
    """Test with NAC by Gonze and Lee."""
    dynmat = ph_nacl.dynamical_matrix
    _test_dynmat_252525(dynmat, dynmat_gonze_lee_ref_252525)


def test_dynmat_gonze_lee_short_range_fc(ph_nacl: Phonopy):
    """Test force constants in dynamical matrix with NAC by Gonze and Lee."""
    # Test getter
    ph_nacl.dynamical_matrix.make_Gonze_nac_dataset()

    assert ph_nacl.dynamical_matrix._G_cutoff == pytest.approx(1.1584988384375283)
    assert ph_nacl.dynamical_matrix._G_list.shape == (307, 3)
    np.testing.assert_allclose(
        ph_nacl.dynamical_matrix._dd_q0.view("double").ravel(),
        [
            0.5509692730441111,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5509692730441109,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5509692730441113,
            0.0,
            0.5509692730441111,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5509692730441109,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5509692730441113,
            0.0,
        ],
        atol=1e-5,
    )
    fc = ph_nacl.dynamical_matrix.force_constants
    sr_fc = ph_nacl.dynamical_matrix.short_range_force_constants
    np.testing.assert_allclose(
        np.diag(fc[0, 1]), [-0.3017767, 0.0049673, 0.0049673], atol=1e-5
    )
    np.testing.assert_allclose(
        np.diag(sr_fc[0, 1]), [-0.13937495, -0.04645899, -0.04645899], atol=1e-5
    )

    # Test setter.
    ph_nacl.dynamical_matrix.short_range_force_constants = fc
    sr_fc = ph_nacl.dynamical_matrix.short_range_force_constants
    np.testing.assert_allclose(
        np.diag(sr_fc[0, 1]), [-0.3017767, 0.0049673, 0.0049673], atol=1e-5
    )
    ph_nacl.dynamical_matrix.make_Gonze_nac_dataset()
    sr_fc = ph_nacl.dynamical_matrix.short_range_force_constants
    np.testing.assert_allclose(
        np.diag(sr_fc[0, 1]), [-0.13937495, -0.04645899, -0.04645899], atol=1e-5
    )


def test_dynmat_gonze_lee_full_term(ph_nacl: Phonopy):
    """Test with NAC by Gonze and Lee."""
    dynmat = ph_nacl.dynamical_matrix
    _dynmat = DynamicalMatrixGL(
        dynmat.supercell,
        dynmat.primitive,
        dynmat.force_constants,
        nac_params=dynmat.nac_params,
        with_full_terms=True,
    )
    _test_dynmat_252525(_dynmat, dynmat_gonze_lee_full_ref_252525)


def test_dynmat_wang(ph_nacl_wang: Phonopy):
    """Test with NAC by Wang et al."""
    dynmat = ph_nacl_wang.dynamical_matrix
    _test_dynmat_252525(dynmat, dynmat_wang_ref_252525)


def _test_dynmat(dynmat: DynamicalMatrix, lang=None):
    dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)
    if lang:
        dynmat.run([0, 0, 0], lang=lang)
    else:
        dynmat.run([0, 0, 0])
    dynmat_ref = (
        np.array(dynmat_ref_000, dtype="double").view(dtype=dtype_complex).reshape(6, 6)
    )
    np.testing.assert_allclose(dynmat.dynamical_matrix, dynmat_ref, atol=1e-5)

    if lang:
        dynmat.run([0.5, 0.5, 0.5], lang=lang)
    else:
        dynmat.run([0.5, 0.5, 0.5])
    dynmat_ref = (
        np.array(dynmat_ref_555, dtype="double").view(dtype=dtype_complex).reshape(6, 6)
    )
    np.testing.assert_allclose(dynmat.dynamical_matrix, dynmat_ref, atol=1e-5)


def _test_dynmat_252525(dynmat: DynamicalMatrix, dynmat_ref: list, lang=None):
    dtype_complex = "c%d" % (np.dtype("double").itemsize * 2)
    if lang:
        dynmat.run([0.25, 0.25, 0.25], lang=lang)
    else:
        dynmat.run([0.25, 0.25, 0.25])
    # for row in dynmat.dynamical_matrix:
    #     print("".join(["%f, %f, " % (c.real, c.imag) for c in row]))

    np.testing.assert_allclose(
        dynmat.dynamical_matrix,
        np.array(dynmat_ref, dtype="double").view(dtype=dtype_complex).reshape(6, 6),
        atol=1e-5,
    )
