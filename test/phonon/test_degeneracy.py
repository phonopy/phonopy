"""Tests of routines in degeneracy.py."""

import numpy as np

from phonopy.phonon.degeneracy import (
    _degenerate_sets_in_any_order,
    degenerate_sets,
    get_degenerate_ids,
    lift_degeneracy,
)


def test_degenerate_sets_ascending():
    """Test that ascending frequencies give the sets of the general algorithm.

    Gaps are drawn from values below, at and above the cutoff, including zero.
    With cutoff=0.5 and multiples of 0.25, gaps equal to the cutoff are exact.

    """
    rng = np.random.default_rng(0)
    settings = [
        (1e-4, np.array([0, 0.3, 0.999, 1.0, 1.001, 2.0, 50.0]) * 1e-4, 1.0),
        (0.5, np.array([0, 0.25, 0.5, 0.75, 1.0]), 0.0),
    ]
    for cutoff, gap_choices, scale in settings:
        num_degenerate = 0
        for i in range(1000):
            n = rng.integers(1, 30)
            gaps = rng.choice(gap_choices, size=n - 1)
            start = scale * rng.normal() * 10 ** (i % 4)
            freqs = np.concatenate([[start], gaps]).cumsum()
            sets = degenerate_sets(freqs, cutoff=cutoff)
            assert sets == _degenerate_sets_in_any_order(freqs, cutoff)
            num_degenerate += len(sets) < n
        assert num_degenerate > 500
    assert degenerate_sets(np.array([])) == []
    assert degenerate_sets(np.array([3.0])) == [[0]]


def test_degenerate_sets_unsorted():
    """Test degenerate_sets for frequencies not in ascending order."""
    freqs = np.array([2.0, 1.0, 2.00005, 1.00005, 3.0])
    assert degenerate_sets(freqs) == [[0, 2], [1, 3], [4]]


def test_lift_degeneracy():
    """Test lift_degeneracy against diagonalization of each projected block."""
    rng = np.random.default_rng(0)
    num_band = 12
    for _ in range(50):
        # Degenerate sets of 1 to 3 bands.
        freqs = np.repeat(np.arange(num_band, dtype=float), rng.integers(1, 4, 12))
        freqs = freqs[:num_band]
        mat = rng.normal(size=(2, num_band, num_band))
        eigvecs, _ = np.linalg.qr(mat[0] + 1j * mat[1])
        mat = rng.normal(size=(2, num_band, num_band))
        dDdq = mat[0] + 1j * mat[1]
        dDdq += dDdq.T.conj()

        eigvals_pert, rot_eigvecs = lift_degeneracy(freqs, eigvecs, dDdq)

        for deg in degenerate_sets(freqs):
            block = eigvecs[:, deg].T.conj() @ dDdq @ eigvecs[:, deg]
            vals_ref, u = np.linalg.eigh(block)
            np.testing.assert_allclose(eigvals_pert[deg], vals_ref, atol=1e-12)
            # Each column is unique up to a phase factor.
            overlap = (rot_eigvecs[:, deg].conj() * (eigvecs[:, deg] @ u)).sum(axis=0)
            np.testing.assert_allclose(np.abs(overlap), 1, atol=1e-12)


def test_get_degenerate_ids():
    """Test that get_degenerate_ids gives the sets of degenerate_sets.

    Frequencies in ascending order are drawn with ties, with pairs just below
    and just above the cutoff, and with a chain of bands whose neighbours are
    closer than the cutoff.

    """
    rng = np.random.default_rng(0)
    freqs = np.sort(rng.integers(0, 5, size=(200, 9)).astype(float), axis=1)
    freqs += np.sort(rng.choice([0, 5e-5, 1.5e-4], size=(200, 9)), axis=1)
    freqs = np.vstack([freqs, [[1.0, 1.00005, 1.0001, 1.00015, 2, 3, 3, 3, 4]]])
    freqs = np.sort(freqs, axis=1)
    ids = get_degenerate_ids(freqs)
    assert ids.dtype == np.int64
    num_degenerate = 0
    for f, ids_q in zip(freqs, ids, strict=True):
        sets = [np.flatnonzero(ids_q == i).tolist() for i in np.unique(ids_q)]
        assert sets == degenerate_sets(f)
        num_degenerate += len(sets) < len(f)
    assert num_degenerate > 100
