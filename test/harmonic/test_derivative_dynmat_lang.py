# SPDX-License-Identifier: BSD-3-Clause
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


def _load(lang, *, with_nac=True, is_compact_fc=False, wang=False):
    ph = phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl" if with_nac else None,
        is_nac=with_nac,
        is_compact_fc=is_compact_fc,
        log_level=0,
        produce_fc=True,
        lang=lang,
    )
    if wang:
        nac_params = ph.nac_params
        nac_params["method"] = "wang"
        ph.nac_params = nac_params
    return ph


def _ddm(ph: Phonopy, q, q_dir=None, force_python=False) -> np.ndarray:
    dd = DerivativeOfDynamicalMatrix(ph.dynamical_matrix)
    dd.run(q, q_direction=q_dir, force_python=force_python)
    assert dd.d_dynamical_matrix is not None
    return dd.d_dynamical_matrix


@pytest.mark.parametrize("is_compact_fc", [False, True])
@pytest.mark.parametrize("q", _QPOINTS)
def test_derivative_dynmat_matches_nonac(is_compact_fc: bool, q):
    """C and Rust paths agree without NAC."""
    ph_c = _load("C", with_nac=False, is_compact_fc=is_compact_fc)
    ph_r = _load("Rust", with_nac=False, is_compact_fc=is_compact_fc)
    np.testing.assert_allclose(_ddm(ph_c, q), _ddm(ph_r, q), atol=1e-15)


@pytest.mark.parametrize("is_compact_fc", [False, True])
@pytest.mark.parametrize("q", _QPOINTS)
def test_derivative_dynmat_matches_wang(is_compact_fc: bool, q):
    """C and Rust paths agree with Wang NAC."""
    ph_c = _load("C", is_compact_fc=is_compact_fc, wang=True)
    ph_r = _load("Rust", is_compact_fc=is_compact_fc, wang=True)
    np.testing.assert_allclose(_ddm(ph_c, q), _ddm(ph_r, q), atol=1e-15)


@pytest.mark.parametrize("is_compact_fc", [False, True])
@pytest.mark.parametrize("q", _QPOINTS)
def test_derivative_dynmat_matches_gonze_lee(is_compact_fc: bool, q):
    """Rust and Python paths agree for Gonze-Lee NAC.

    The C path raises ``NotImplementedError`` for Gonze-Lee, so Python
    (``force_python=True``) is used as the parity reference.

    """
    ph_r = _load("Rust", is_compact_fc=is_compact_fc)
    np.testing.assert_allclose(
        _ddm(ph_r, q),
        _ddm(ph_r, q, force_python=True),
        atol=1e-13,
    )


def test_derivative_dynmat_gonze_lee_c_raises():
    """The C path explicitly refuses Gonze-Lee dynmat to avoid silent wrong values."""
    ph_c = _load("C")
    dd = DerivativeOfDynamicalMatrix(ph_c.dynamical_matrix)
    with pytest.raises(NotImplementedError):
        dd.run([0.1, 0.2, 0.3])


def test_derivative_dynmat_with_q_direction_matches():
    """Gamma point with explicit q-direction (Wang NAC) -- C and Rust agree."""
    ph_c = _load("C", wang=True)
    ph_r = _load("Rust", wang=True)
    q_dir = [1.0, 0.0, 0.0]
    np.testing.assert_allclose(
        _ddm(ph_c, [0.0, 0.0, 0.0], q_dir=q_dir),
        _ddm(ph_r, [0.0, 0.0, 0.0], q_dir=q_dir),
        atol=1e-15,
    )


def test_explicit_lang_overrides_inherited():
    """The constructor's ``lang`` keyword wins over the parent's lang."""
    ph_c = _load("C", with_nac=False)
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
