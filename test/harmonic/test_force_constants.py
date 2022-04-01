"""Tests for routines in force_constants.py."""
import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.force_constants import cutoff_force_constants
from phonopy.structure.cells import get_primitive

fc_1_10_ref = [
    -0.037549,
    0.000000,
    0.000000,
    0.000000,
    0.002415,
    -0.001746,
    0.000000,
    -0.001746,
    0.002415,
]

fc_1_10_nofcsym_ref = [
    -0.005051,
    0.000000,
    0.000000,
    0.000000,
    0.094457,
    0.000000,
    0.000000,
    0.000000,
    -0.020424,
]

fc_1_10_compact_fcsym_ref = [
    -0.004481,
    0.000000,
    0.000000,
    0.000000,
    0.095230,
    0.000000,
    0.000000,
    0.000000,
    -0.019893,
]


def test_fc(ph_nacl: Phonopy):
    """Test of force constants calculation with fcsym by NaCl."""
    fc_1_10 = ph_nacl.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(fc_1_10.ravel(), fc_1_10_ref, atol=1e-5)


def test_fc_nofcsym(ph_nacl_nofcsym: Phonopy):
    """Test of force constants calculation without fcsym by NaCl."""
    fc_1_10 = ph_nacl_nofcsym.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(fc_1_10.ravel(), fc_1_10_nofcsym_ref, atol=1e-5)


def test_fc_compact_fcsym(ph_nacl_compact_fcsym: Phonopy):
    """Test of force constants calculation in compact format with fcsym by NaCl."""
    fc_1_10 = ph_nacl_compact_fcsym.force_constants[1, 10]
    # print("".join(["%f, " % v for v in fc_1_10.ravel()]))
    np.testing.assert_allclose(fc_1_10.ravel(), fc_1_10_compact_fcsym_ref, atol=1e-5)


@pytest.mark.parametrize("is_compact", [False, True])
def test_fc_cutoff_radius(ph_nacl: Phonopy, ph_nacl_compact_fcsym: Phonopy, is_compact):
    """Test of cutoff radius of force constants calculation by NaCl."""
    if is_compact:
        ph = ph_nacl_compact_fcsym
    else:
        ph = ph_nacl

    # Need restore fc because fc are overwritten.
    fc_orig = ph.force_constants.copy()
    ph.set_force_constants_zero_with_radius(4.0)
    changed = np.abs(fc_orig - ph.force_constants) > 1e-8
    ph.force_constants = fc_orig

    if is_compact:
        assert np.sum(changed) == 534
    else:
        assert np.sum(changed) == 17088


@pytest.mark.parametrize(
    "is_compact,store_dense_svecs",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_fc_cutoff_radius_svecs(
    ph_nacl: Phonopy, ph_nacl_compact_fcsym: Phonopy, is_compact, store_dense_svecs
):
    """Test of cutoff radius with dense-svecs format by NaCl."""
    if is_compact:
        ph = ph_nacl_compact_fcsym
    else:
        ph = ph_nacl

    fc = ph.force_constants.copy()
    primitive_matrix = np.dot(np.linalg.inv(ph.supercell_matrix), ph.primitive_matrix)
    primitive = get_primitive(
        ph.supercell, primitive_matrix, store_dense_svecs=store_dense_svecs
    )

    cutoff_force_constants(fc, ph.supercell, primitive, 4.0)
    changed = np.abs(ph.force_constants - fc) > 1e-8

    if is_compact:
        assert np.sum(changed) == 534
    else:
        assert np.sum(changed) == 17088
