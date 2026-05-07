"""Parity tests comparing Rust and C dispatch paths for cells helpers.

Confirms bit-for-bit parity between ``phonopy._phonopy.compute_permutation``
and ``phonors.compute_permutation`` at the leaf, and end-to-end
equivalence of the higher-level helpers
(``compute_permutation_for_rotation``, ``compute_all_sg_permutations``)
and the data they fill on ``Symmetry`` / ``Primitive`` when
``Phonopy(lang="Rust")`` is used.

Most tests reuse the session-scoped ``ph_nacl`` fixture from
``test/conftest.py`` since the Phonopy instance itself does not need
to be lang-aware -- we only consume its supercell and symmetry
operations, and dispatch to C/Rust at the leaf via the ``lang``
argument.  The end-to-end load test
``test_phonopy_load_lang_rust_end_to_end`` is the one case that
intentionally builds two ``Phonopy`` instances with different ``lang``
values to confirm load-time plumbing.

The whole module is skipped when phonors is not importable.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.structure.cells import (
    Primitive,
    _compute_permutation,
    compute_all_sg_permutations,
    compute_permutation_for_rotation,
)
from phonopy.structure.symmetry import Symmetry

pytest.importorskip("phonors")

cwd = pathlib.Path(__file__).parent.parent


def _supercell_positions(ph: Phonopy):
    sc = ph.supercell
    lattice = np.array(sc.cell.T, dtype="double", order="C")
    positions = np.array(sc.scaled_positions, dtype="double", order="C")
    return lattice, positions


def test_compute_permutation_leaf_matches(ph_nacl: Phonopy):
    """Direct leaf check: same identity-rotation permutation from C and Rust."""
    lattice, positions = _supercell_positions(ph_nacl)
    perm_c = _compute_permutation(positions, positions, lattice, 1e-5, lang="C")
    perm_rust = _compute_permutation(positions, positions, lattice, 1e-5, lang="Rust")
    np.testing.assert_array_equal(perm_c, perm_rust)


def test_compute_permutation_for_rotation_matches(ph_nacl: Phonopy):
    """End-to-end through the sort-and-match wrapper."""
    lattice, positions = _supercell_positions(ph_nacl)
    rotations = ph_nacl.symmetry.symmetry_operations["rotations"]
    translations = ph_nacl.symmetry.symmetry_operations["translations"]
    # Pick a non-identity operation.
    sym = rotations[1]
    t = translations[1]
    rotated = positions @ sym.T + t
    perm_c = compute_permutation_for_rotation(
        positions, rotated, lattice, 1e-5, lang="C"
    )
    perm_rust = compute_permutation_for_rotation(
        positions, rotated, lattice, 1e-5, lang="Rust"
    )
    np.testing.assert_array_equal(perm_c, perm_rust)


def test_compute_all_sg_permutations_matches(ph_nacl: Phonopy):
    """All space-group operations produce identical permutation tables."""
    lattice, positions = _supercell_positions(ph_nacl)
    rotations = ph_nacl.symmetry.symmetry_operations["rotations"]
    translations = ph_nacl.symmetry.symmetry_operations["translations"]
    perms_c = compute_all_sg_permutations(
        positions, rotations, translations, lattice, 1e-5, lang="C"
    )
    perms_rust = compute_all_sg_permutations(
        positions, rotations, translations, lattice, 1e-5, lang="Rust"
    )
    np.testing.assert_array_equal(perms_c, perms_rust)


def test_symmetry_atomic_permutations_match(ph_nacl: Phonopy):
    """Symmetry.atomic_permutations is identical under C and Rust backends."""
    sym_c = Symmetry(ph_nacl.supercell, symprec=1e-5, lang="C")
    sym_rust = Symmetry(ph_nacl.supercell, symprec=1e-5, lang="Rust")
    np.testing.assert_array_equal(
        sym_c.atomic_permutations, sym_rust.atomic_permutations
    )


def test_primitive_atomic_permutations_match(ph_nacl: Phonopy):
    """Primitive cell construction produces identical atomic_permutations."""
    pmat = ph_nacl.primitive.primitive_matrix
    prim_c = Primitive(ph_nacl.supercell, pmat, symprec=1e-5, lang="C")
    prim_rust = Primitive(ph_nacl.supercell, pmat, symprec=1e-5, lang="Rust")
    np.testing.assert_array_equal(
        prim_c.atomic_permutations, prim_rust.atomic_permutations
    )


def test_phonopy_load_lang_rust_end_to_end():
    """phonopy.load(lang="Rust") matches default lang="C" on Symmetry/Primitive."""
    common_kwargs = dict(
        force_sets_filename=cwd / "FORCE_SETS_NaCl",
        born_filename=cwd / "BORN_NaCl",
        is_compact_fc=False,
        log_level=0,
        produce_fc=True,
    )
    ph_c = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="C", **common_kwargs)
    ph_rust = phonopy.load(cwd / "phonopy_disp_NaCl.yaml", lang="Rust", **common_kwargs)
    np.testing.assert_array_equal(
        ph_c.symmetry.atomic_permutations,
        ph_rust.symmetry.atomic_permutations,
    )
    np.testing.assert_array_equal(
        ph_c.primitive.atomic_permutations,
        ph_rust.primitive.atomic_permutations,
    )
