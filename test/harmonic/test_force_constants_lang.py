# SPDX-License-Identifier: BSD-3-Clause
"""Parity tests comparing Rust and C dispatch paths for fc helpers.

Confirms bit-for-bit parity between ``phonopy._phonopy`` and
``phonors`` at the leaf for ``perm_trans_symmetrize_fc``,
``transpose_compact_fc``, and ``perm_trans_symmetrize_compact_fc``,
plus end-to-end equivalence of ``Phonopy.symmetrize_force_constants``
under ``lang="C"`` vs ``lang="Rust"``.

Leaf parity reuses the session-scoped ``ph_nacl`` fixture from
``test/conftest.py`` since the kernel takes a numpy array and a
``lang`` keyword -- the Phonopy instance just provides representative
fc data.  The end-to-end tests intentionally build two ``Phonopy``
instances (with ``lang="C"`` and ``lang="Rust"``) to confirm
``Phonopy.symmetrize_force_constants`` plumbs ``self._lang`` to the
module-level helper.

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.harmonic.force_constants import (
    get_drift_force_constants,
    symmetrize_compact_force_constants,
    symmetrize_force_constants,
)

pytest.importorskip("phonors")
pytest.importorskip("phonopy._phonopy")

cwd = pathlib.Path(__file__).parent.parent


@pytest.mark.parametrize("level", [1, 2, 3])
def test_perm_trans_symmetrize_fc_matches(ph_nacl: Phonopy, level: int):
    """Direct leaf check: same result from C and Rust on real fc data."""
    fc_c = ph_nacl.force_constants.copy()
    fc_rust = ph_nacl.force_constants.copy()
    symmetrize_force_constants(fc_c, level=level, lang="C")
    symmetrize_force_constants(fc_rust, level=level, lang="Rust")
    np.testing.assert_array_equal(fc_c, fc_rust)


def test_phonopy_symmetrize_force_constants_lang_rust_matches_c():
    """Phonopy.symmetrize_force_constants(lang="Rust") matches lang="C" end-to-end."""
    common_kwargs = dict(
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
    )
    ph_c = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="C", **common_kwargs)
    ph_rust = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="Rust", **common_kwargs)
    ph_c.symmetrize_force_constants()
    ph_rust.symmetrize_force_constants()
    np.testing.assert_array_equal(ph_c.force_constants, ph_rust.force_constants)


def test_transpose_compact_fc_matches(ph_nacl_compact_fcsym: Phonopy):
    """Direct leaf check: transpose_compact_fc identical between C and Rust.

    The drift query path applies the transpose twice, so this also covers
    the involution behaviour on real-world symmetry data.

    """
    fc_c = ph_nacl_compact_fcsym.force_constants.copy()
    fc_rust = ph_nacl_compact_fcsym.force_constants.copy()
    primitive = ph_nacl_compact_fcsym.primitive
    res_c = get_drift_force_constants(fc_c, primitive=primitive, lang="C")
    res_rust = get_drift_force_constants(fc_rust, primitive=primitive, lang="Rust")
    np.testing.assert_array_equal(fc_c, fc_rust)
    assert res_c == res_rust


@pytest.mark.parametrize("level", [1, 2, 3])
def test_perm_trans_symmetrize_compact_fc_matches(
    ph_nacl_compact_fcsym: Phonopy, level: int
):
    """Direct leaf check: compact-fc symmetrization identical between C and Rust.

    Both backends iterate in the same loop order over the same atomic-
    permutation tables, so the floating-point operations occur in
    identical sequence and the result is bit-identical.

    """
    fc_c = ph_nacl_compact_fcsym.force_constants.copy()
    fc_rust = ph_nacl_compact_fcsym.force_constants.copy()
    primitive = ph_nacl_compact_fcsym.primitive
    symmetrize_compact_force_constants(fc_c, primitive, level=level, lang="C")
    symmetrize_compact_force_constants(fc_rust, primitive, level=level, lang="Rust")
    np.testing.assert_array_equal(fc_c, fc_rust)


def test_phonopy_symmetrize_force_constants_compact_lang_rust_matches_c():
    """End-to-end: Phonopy(lang="Rust") on compact fc matches lang="C"."""
    common_kwargs = dict(
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=True,
        log_level=0,
        produce_fc=True,
    )
    ph_c = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="C", **common_kwargs)
    ph_rust = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="Rust", **common_kwargs)
    ph_c.symmetrize_force_constants()
    ph_rust.symmetrize_force_constants()
    np.testing.assert_array_equal(ph_c.force_constants, ph_rust.force_constants)


@pytest.mark.parametrize("is_compact_fc", [False, True])
def test_phonopy_produce_force_constants_lang_rust_matches_c(is_compact_fc: bool):
    """phonopy.load(lang="Rust") produces fc identical to lang="C".

    Hits ``distribute_force_constants`` (and, in the full-fc branch,
    also ``distribute_force_constants_by_translations``).  Bit-identical
    parity is expected: both backends iterate over identical symmetry
    tables in the same order, so the rotation-by-block multiply-add
    proceeds in the same floating-point sequence.

    """
    common_kwargs = dict(
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=is_compact_fc,
        log_level=0,
        produce_fc=True,
    )
    ph_c = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="C", **common_kwargs)
    ph_rust = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="Rust", **common_kwargs)
    np.testing.assert_array_equal(ph_c.force_constants, ph_rust.force_constants)
