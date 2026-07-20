# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the pure-Python backend of get_dynamical_matrices_at_qpoints.

The Python (NumPy) backend needs neither the C extension nor ``phonors``.
These tests compare it against whichever backend ``dm.lang`` selects
(the default build backend, always available) on real NaCl data, for
both full and compact force constants, and check the non-NAC-only
contract.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy.harmonic.dynamical_matrix import (
    get_dynamical_matrices_at_qpoints,
    get_dynamical_matrices_at_qpoints_py,
)

cwd = pathlib.Path(__file__).parent.parent

_QPOINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [0.5, 0.0, 0.5],
        [-0.25, 0.25, 0.0],
    ],
    dtype="double",
    order="C",
)


def _load_nacl(*, with_nac=False, is_compact_fc=False):
    return phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl" if with_nac else None,
        is_nac=with_nac,
        is_compact_fc=is_compact_fc,
        log_level=0,
        produce_fc=True,
    )


@pytest.mark.parametrize("is_compact_fc", [False, True])
def test_python_matches_default_backend(is_compact_fc: bool):
    """Python backend matches the default backend at machine precision."""
    ph = _load_nacl(is_compact_fc=is_compact_fc)
    dm = ph.dynamical_matrix
    ref = get_dynamical_matrices_at_qpoints(dm, _QPOINTS, lang=dm.lang)
    py = get_dynamical_matrices_at_qpoints_py(dm, _QPOINTS)
    assert py.shape == (len(_QPOINTS), 6, 6)
    np.testing.assert_allclose(py, ref, atol=1e-10)


def test_python_single_qpoint_is_squeezed():
    """A 1-D q-point input squeezes the leading axis from the result."""
    ph = _load_nacl()
    dm = ph.dynamical_matrix
    ref = get_dynamical_matrices_at_qpoints(dm, _QPOINTS[1], lang=dm.lang)
    py = get_dynamical_matrices_at_qpoints(dm, _QPOINTS[1], lang="Python")
    assert py.shape == (6, 6)
    np.testing.assert_allclose(py, ref, atol=1e-10)


def test_python_result_is_hermitian():
    """Each q-point block is Hermitian when hermitianize is True."""
    ph = _load_nacl()
    py = get_dynamical_matrices_at_qpoints_py(ph.dynamical_matrix, _QPOINTS)
    for block in py:
        np.testing.assert_allclose(block, block.conj().T, atol=1e-12)


def test_python_backend_rejects_nac():
    """The Python backend supports the non-NAC case only."""
    ph = _load_nacl(with_nac=True)
    with pytest.raises(NotImplementedError):
        get_dynamical_matrices_at_qpoints_py(ph.dynamical_matrix, _QPOINTS)
    # The same contract holds through the dispatcher.
    with pytest.raises(NotImplementedError):
        get_dynamical_matrices_at_qpoints(ph.dynamical_matrix, _QPOINTS, lang="Python")
