"""Parity tests for get_dynamical_matrices_at_qpoints.

Confirms bit-level (machine-epsilon) parity between
``get_dynamical_matrices_at_qpoints_c`` (which routes to the C kernel
``phonoc.dynamical_matrices_with_dd_openmp_over_qpoints``) and
``get_dynamical_matrices_at_qpoints_rust`` (which routes to the Rust
pair ``phonors.dynamical_matrices_at_qpoints[_gonze]``) on real NaCl
data, across the three NAC modes (no NAC, Wang, Gonze).

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy.harmonic.dynamical_matrix import (
    get_dynamical_matrices_at_qpoints_c,
    get_dynamical_matrices_at_qpoints_rust,
)

pytest.importorskip("phonors")

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


def _load_nacl(*, with_nac=True):
    return phonopy.load(
        cwd / "phonopy_disp_NaCl.yaml",
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl" if with_nac else None,
        is_nac=with_nac,
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
    )


@pytest.mark.parametrize("with_nac", [False, True])
def test_gonze_or_nonac_matches(with_nac: bool):
    """No-NAC and Gonze NAC paths produce machine-precision-equal DMs."""
    ph = _load_nacl(with_nac=with_nac)
    dm_c = get_dynamical_matrices_at_qpoints_c(ph.dynamical_matrix, _QPOINTS)
    dm_r = get_dynamical_matrices_at_qpoints_rust(ph.dynamical_matrix, _QPOINTS)
    np.testing.assert_allclose(dm_c, dm_r, atol=1e-15)


def test_wang_matches():
    """Wang NAC path produces machine-precision-equal DMs."""
    ph = _load_nacl()
    nac = ph.nac_params
    nac["method"] = "wang"
    ph.nac_params = nac
    dm_c = get_dynamical_matrices_at_qpoints_c(ph.dynamical_matrix, _QPOINTS)
    dm_r = get_dynamical_matrices_at_qpoints_rust(ph.dynamical_matrix, _QPOINTS)
    np.testing.assert_allclose(dm_c, dm_r, atol=1e-15)


def test_with_q_direction_matches():
    """The Gamma point with explicit q-direction routes through the q_dir branch."""
    ph = _load_nacl()
    q_dir = np.array([1.0, 0.0, 0.0], dtype="double")
    qpoints = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype="double")
    dm_c = get_dynamical_matrices_at_qpoints_c(
        ph.dynamical_matrix, qpoints, nac_q_direction=q_dir
    )
    dm_r = get_dynamical_matrices_at_qpoints_rust(
        ph.dynamical_matrix, qpoints, nac_q_direction=q_dir
    )
    np.testing.assert_allclose(dm_c, dm_r, atol=1e-15)


def test_phonopy_run_qpoints_lang_rust_dynamical_matrices_match():
    """End-to-end: stored DMs match through Phonopy.run_qpoints().

    Compared on the dynamical matrices rather than frequencies because
    acoustic modes at Gamma have numerical-noise-level frequencies whose
    sign is unstable across backends.  The DMs themselves agree to
    machine precision.

    """
    common_kwargs = dict(
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
    )
    ph_c = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="C", **common_kwargs)
    ph_r = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="Rust", **common_kwargs)
    ph_c.run_qpoints(_QPOINTS, with_eigenvectors=False, with_dynamical_matrices=True)
    ph_r.run_qpoints(_QPOINTS, with_eigenvectors=False, with_dynamical_matrices=True)
    np.testing.assert_allclose(
        np.asarray(ph_c.qpoints.dynamical_matrices),
        np.asarray(ph_r.qpoints.dynamical_matrices),
        atol=1e-15,
    )
