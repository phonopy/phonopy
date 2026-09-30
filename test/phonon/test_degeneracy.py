"""Tests of routines in degeneracy.py."""

import numpy as np

from phonopy.phonon.degeneracy import degenerate_sets, get_degenerate_ids


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
