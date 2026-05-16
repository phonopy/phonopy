"""Parity tests for DerivativeOfDynamicalMatrix.

Confirms machine-precision parity between the C and Rust backends of
``DerivativeOfDynamicalMatrix`` (``_run_c`` and ``_run_rust``,
dispatched via ``_run_compiled`` on the value of ``self._lang``).
The lang is inherited from the parent ``DynamicalMatrix`` instance by
default, and can be overridden through the ``lang`` keyword on the
``DerivativeOfDynamicalMatrix`` constructor.

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

pytest.importorskip("phonors")
pytest.importorskip("phonopy._phonopy")

cwd = pathlib.Path(__file__).parent.parent

_QPOINTS = [
    (0.1, 0.2, 0.3),
    (0.5, 0.5, 0.0),
    (0.25, 0.25, 0.25),
    (0.0, 0.0, 0.0),
]


def _load(lang, *, with_nac=True, is_compact_fc=False):
    return phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl" if with_nac else None,
        is_nac=with_nac,
        is_compact_fc=is_compact_fc,
        log_level=0,
        produce_fc=True,
        lang=lang,
    )


def _ddm(ph: Phonopy, q, q_dir=None) -> np.ndarray:
    dd = DerivativeOfDynamicalMatrix(ph.dynamical_matrix)
    dd.run(q, q_direction=q_dir)
    assert dd.d_dynamical_matrix is not None
    return dd.d_dynamical_matrix


@pytest.mark.parametrize("with_nac", [False, True])
@pytest.mark.parametrize("is_compact_fc", [False, True])
@pytest.mark.parametrize("q", _QPOINTS)
def test_derivative_dynmat_matches(with_nac: bool, is_compact_fc: bool, q):
    """C and Rust paths produce machine-precision-equal derivatives."""
    ph_c = _load("C", with_nac=with_nac, is_compact_fc=is_compact_fc)
    ph_r = _load("Rust", with_nac=with_nac, is_compact_fc=is_compact_fc)
    np.testing.assert_allclose(_ddm(ph_c, q), _ddm(ph_r, q), atol=1e-15)


def test_derivative_dynmat_with_q_direction_matches():
    """Gamma point with explicit q-direction routes through the q_dir branch."""
    ph_c = _load("C")
    ph_r = _load("Rust")
    q_dir = [1.0, 0.0, 0.0]
    np.testing.assert_allclose(
        _ddm(ph_c, [0.0, 0.0, 0.0], q_dir=q_dir),
        _ddm(ph_r, [0.0, 0.0, 0.0], q_dir=q_dir),
        atol=1e-15,
    )


def test_explicit_lang_overrides_inherited():
    """The constructor's ``lang`` keyword wins over the parent's lang."""
    ph_c = _load("C")
    dd_inherit = DerivativeOfDynamicalMatrix(ph_c.dynamical_matrix)
    dd_override = DerivativeOfDynamicalMatrix(ph_c.dynamical_matrix, lang="Rust")
    assert dd_inherit._lang == "C"
    assert dd_override._lang == "Rust"
    dd_inherit.run([0.1, 0.2, 0.3])
    dd_override.run([0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        dd_inherit.d_dynamical_matrix,
        dd_override.d_dynamical_matrix,
        atol=1e-15,
    )
