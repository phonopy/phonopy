# SPDX-License-Identifier: BSD-3-Clause
"""Tests for symmetry-adapted phonon modes."""

from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest

import phonopy
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.phonon.symmetry_adapted_modes import (
    SymmetryAdaptedModes,
    _get_symmetrized_lattice,
    _get_symmetrized_positions,
    get_little_group_operations,
)

data_dir = pathlib.Path(__file__).parent

# (space-group type, supercell, primitive matrix, q-points)
CASES = [
    ("P2", [3, 2, 2], np.eye(3), [[0, 0, 0], [0, 0, 0.5]]),
    ("Pc", [2, 2, 2], np.eye(3), [[0, 0, 0], [0, 0.5, 0]]),
    (
        "P222_1",
        [2, 2, 1],
        np.eye(3),
        [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0.5]],
    ),
    (
        "Amm2",
        [3, 2, 2],
        [[1, 0, 0], [0, 0.5, -0.5], [0, 0.5, 0.5]],
        [[0, 0, 0], [0, 0, -0.5], [0.5, 0, -0.5], [0.5, 0.25, -0.25]],
    ),
    ("P4_1", [2, 2, 1], np.eye(3), [[0, 0, 0], [0, 0, 0.5]]),
    ("P-3m1", [3, 3, 2], np.eye(3), [[0, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0.5]]),
    ("P6_222", [2, 2, 2], np.eye(3), [[0, 0, 0], [1 / 3, 1 / 3, 0.5]]),
    ("Pa-3", [2, 2, 2], np.eye(3), [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0.5]]),
    ("P4_332", [1, 1, 1], np.eye(3), [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0]]),
    ("P-43m", [2, 2, 2], np.eye(3), [[0, 0, 0], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]),
]


def _get_phonon(spgtype, dim, pmat) -> Phonopy:
    cell = read_vasp(data_dir / f"POSCAR_{spgtype}")
    return phonopy.load(
        unitcell=cell,
        supercell_matrix=np.diag(dim),
        primitive_matrix=pmat,
        force_sets_filename=data_dir / f"FORCE_SETS_{spgtype}",
        symmetrize_fc=True,
    )


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def case(request):
    """Return (phonon, q-points)."""
    spgtype, dim, pmat, qpoints = request.param
    return _get_phonon(spgtype, dim, pmat), qpoints


def _get_modes(phonon: Phonopy, q) -> SymmetryAdaptedModes:
    assert phonon.dynamical_matrix is not None
    return SymmetryAdaptedModes(phonon.dynamical_matrix, q, phonon.primitive_symmetry)


def test_operations_leave_dynamical_matrix_invariant(case):
    """Test the phase convention of the operations."""
    phonon, qpoints = case
    for q in qpoints:
        modes = _get_modes(phonon, q)
        dm = modes._dynamical_matrix
        scale = np.abs(dm).max()
        for op in modes.operations:
            np.testing.assert_allclose(
                op.transform_matrix(dm), dm, atol=1e-6 * scale, err_msg=f"q={q}"
            )


def test_operations_rotate_complex_displacements(case):
    """Test T(S) against the rotation of a displacement field in real space.

    A complex displacement field u(j', n) = e_j' exp(2 pi i q . (x_j' + n)),
    i.e., a plane wave with wave vector q modulated by e over the atoms, is
    built from a random vector e over lattice vectors n.  The operation
    sends the displacement of atom j' in cell n to the image
    R (x_j' + n) + t as R_cart u.  For S Theta the field is complex
    conjugated first, which gives the same form with -q.  The rotated field
    must equal the field of the same form with q built from T e (or
    T conj(e)).  Unlike T D T^dagger, this comparison is sensitive to the
    overall phase exp(-i Rq . tau) of T.

    """
    phonon, qpoints = case
    primitive = phonon.primitive
    symmetry = phonon.primitive_symmetry
    rotations = symmetry.symmetry_operations["rotations"]
    translations = symmetry.symmetry_operations["translations"]
    lattice = _get_symmetrized_lattice(primitive.cell, rotations)
    positions = _get_symmetrized_positions(
        primitive.scaled_positions, lattice, rotations, translations, symmetry.tolerance
    )
    num_atom = len(positions)
    rng = np.random.default_rng(0)
    e = rng.standard_normal((3 * num_atom, 1)) + 1j * rng.standard_normal(
        (3 * num_atom, 1)
    )
    cells = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype="double")
    for q in qpoints:
        q = np.array(q, dtype="double")
        for op in get_little_group_operations(q, primitive, symmetry):
            eta = -1 if op.is_antiunitary else 1
            e_src = (e.conj() if op.is_antiunitary else e).reshape(num_atom, 3)
            te = op.transform_vectors(e).reshape(num_atom, 3)
            # x[j', n]: positions of atom j' in cell n, shape (num_atom, 27, 3)
            x = positions[:, None, :] + cells[None, :, :]
            u = e_src[:, None, :] * np.exp(2j * np.pi * eta * (x @ q))[:, :, None]
            u_rot = u @ op.rotation_cartesian.T
            images = x @ op.rotation.T + op.translation
            x_dst = positions[op.permutation][:, None, :]
            shifts = images - x_dst
            np.testing.assert_allclose(shifts, np.rint(shifts), atol=1e-8)
            expected = (
                te[op.permutation][:, None, :]
                * np.exp(2j * np.pi * ((x_dst + np.rint(shifts)) @ q))[:, :, None]
            )
            np.testing.assert_allclose(
                u_rot, expected, atol=1e-10, err_msg=f"q={q}, rot={op.rotation}"
            )


def test_frequencies_and_eigenvectors(case):
    """Test frequencies against direct diagonalization and eigen equation."""
    phonon, qpoints = case
    for q in qpoints:
        modes = _get_modes(phonon, q)
        dm = modes._dynamical_matrix
        scale = np.abs(dm).max()
        eigvals = np.linalg.eigvalsh(dm)
        freqs = modes.frequencies / modes._factor
        vals = np.sign(freqs) * freqs**2
        np.testing.assert_allclose(vals, eigvals, atol=1e-6 * scale, err_msg=f"q={q}")
        vecs = modes.eigenvectors
        np.testing.assert_allclose(vecs.conj().T @ vecs, np.eye(len(vecs)), atol=1e-10)
        np.testing.assert_allclose(dm @ vecs, vecs * vals, atol=1e-6 * scale)


def test_sets_are_irreducible(case):
    """Test sum of |chi|^2 over the unitary operations.

    It is the order for an irreducible representation, and 2 or 4 times the
    order for a set made degenerate by time reversal.

    """
    phonon, qpoints = case
    for q in qpoints:
        modes = _get_modes(phonon, q)
        order = modes.characters.shape[1]
        for chars in modes.characters:
            ratio = np.vdot(chars, chars).real / order
            assert int(np.rint(ratio)) in (1, 2, 4)
            assert abs(ratio - np.rint(ratio)) < 1e-6


def test_characters_match_irreps_at_gamma(case):
    """Test characters against IrReps where both give the same sets.

    IrReps splits the acoustic modes at Gamma when their frequencies differ
    by more than its tolerance, which gives non-integer characters.  Such
    sets are excluded by requiring the sum of |chi|^2 to be a multiple of
    the order.

    """
    phonon, _ = case
    phonon.run_irreps([0, 0, 0])
    irreps = phonon.irreps
    assert irreps is not None
    modes = _get_modes(phonon, [0, 0, 0])
    order = modes.characters.shape[1]
    assert len(irreps.characters[0]) == order
    irreps_sets = {}
    for bands, chars in zip(irreps.band_indices, irreps.characters, strict=True):
        ratio = np.vdot(chars, chars).real / order
        if abs(ratio - np.rint(ratio)) < 1e-6:
            irreps_sets[tuple(bands)] = chars
    num_compared = 0
    for bands, chars in zip(modes.degenerate_sets, modes.characters, strict=True):
        if tuple(bands) in irreps_sets:
            np.testing.assert_allclose(chars, irreps_sets[tuple(bands)], atol=1e-5)
            num_compared += 1
    assert num_compared > 0


def test_result_is_reproducible(case):
    """Test two runs give identical results."""
    phonon, qpoints = case
    for q in qpoints:
        m1 = _get_modes(phonon, q)
        m2 = _get_modes(phonon, q)
        np.testing.assert_array_equal(m1.frequencies, m2.frequencies)
        np.testing.assert_array_equal(m1.eigenvectors, m2.eigenvectors)
        assert m1.degenerate_sets == m2.degenerate_sets
