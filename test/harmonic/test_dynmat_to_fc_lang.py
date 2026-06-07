"""Parity tests for DynmatToForceConstants.

Verifies machine-precision parity between the C and Rust backends of
``DynmatToForceConstants.run(lang=...)``, which inverts dynamical
matrices at commensurate q-points back to real-space fc.

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants

pytest.importorskip("phonors")
pytest.importorskip("phonopy._phonopy")

cwd = pathlib.Path(__file__).parent.parent


def _run_d2f(ph: Phonopy, lang: str, *, full_fc: bool) -> np.ndarray:
    d2f = DynmatToForceConstants(
        ph.primitive, ph.supercell, is_full_fc=full_fc, lang=lang
    )
    ph.run_qpoints(d2f.commensurate_points, with_dynamical_matrices=True)
    assert ph.qpoints is not None
    d2f.dynamical_matrices = ph.qpoints.dynamical_matrices
    d2f.run()
    assert d2f.force_constants is not None
    return d2f.force_constants


@pytest.mark.parametrize("full_fc", [False, True])
def test_transform_dynmat_to_fc_matches(ph_nacl: Phonopy, full_fc: bool):
    """Direct leaf check: C and Rust produce machine-precision-equal fc.

    Both backends iterate over the same (i, j) pairs in the same q-point
    order, so the floating-point reductions occur in identical sequence.

    """
    fc_c = _run_d2f(ph_nacl, lang="C", full_fc=full_fc)
    fc_r = _run_d2f(ph_nacl, lang="Rust", full_fc=full_fc)
    np.testing.assert_allclose(fc_c, fc_r, atol=1e-15)


def test_make_gonze_nac_dataset_lang_rust_matches_c():
    """End-to-end: GL NAC short-range fc agrees between C and Rust.

    `DynamicalMatrixGL.make_Gonze_nac_dataset` calls
    `DynmatToForceConstants.run` internally; this test exercises that
    path under `Phonopy(lang=...)` propagation.

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
    ph_c.dynamical_matrix.make_Gonze_nac_dataset()
    ph_r.dynamical_matrix.make_Gonze_nac_dataset()
    fc_c = ph_c.dynamical_matrix._Gonze_force_constants
    fc_r = ph_r.dynamical_matrix._Gonze_force_constants
    np.testing.assert_allclose(fc_c, fc_r, atol=1e-15)
