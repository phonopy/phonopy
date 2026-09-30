# SPDX-License-Identifier: BSD-3-Clause
"""Utility routines to handle degeneracy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix

DEFAULT_Q_LENGTH = 1e-5
DEFAULT_CUTOFF = 1e-4


def degenerate_sets(
    freqs: NDArray[np.double], cutoff: float = DEFAULT_CUTOFF
) -> list[list[int]]:
    """Find degenerate bands from frequencies.

    Parameters
    ----------
    freqs : ndarray
        A list of values.
        shape=(values,)
    cutoff : float, optional
        Equivalent of values is defined by this value, i.e.,
            abs(val1 - val2) < cutoff
        Default is 1e-4.

    Returns
    -------
    indices : list of list
        Indices of equivalent values are grouped as a list and those groups are
        stored in a list.

    Example
    -------
    In : degenerate_sets(np.array([1.5, 2.1, 2.1, 3.4, 8]))
    Out: [[0], [1, 2], [3], [4]]

    """
    indices = []
    done = []
    for i in range(len(freqs)):
        if i in done:
            continue
        else:
            f_set = [i]
            done.append(i)
        for j in range(i + 1, len(freqs)):
            if (np.abs(freqs[f_set] - freqs[j]) < cutoff).any():
                f_set.append(j)
                done.append(j)
        indices.append(f_set[:])

    return indices


def get_degenerate_ids(
    freqs: NDArray[np.double], cutoff: float = DEFAULT_CUTOFF
) -> NDArray[np.int64]:
    """Return degenerate sets of bands as the smallest band index of each set.

    The frequencies at each q-point have to be in ascending order, as returned
    by the eigenvalue solver. A new set starts at a band whose frequency is
    higher than that of the band below by cutoff or more. For frequencies in
    ascending order, the sets are those of degenerate_sets.

    Parameters
    ----------
    freqs : ndarray
        Phonon frequencies. shape=(qpoints, num_band), dtype='double'
    cutoff : float, optional
        Frequency difference below which two bands are degenerate. Default is
        1e-4.

    Returns
    -------
    ndarray
        Smallest band index in the degenerate set of each band.
        shape=(qpoints, num_band), dtype='int64'

    Example
    -------
    In : get_degenerate_ids(np.array([[1.5, 2.1, 2.1, 3.4, 8]]))
    Out: array([[0, 1, 1, 3, 4]])

    """
    num_band = freqs.shape[1]
    is_first = np.ones(freqs.shape, dtype=bool)
    is_first[:, 1:] = np.diff(freqs, axis=1) >= cutoff
    first_bands = np.where(is_first, np.arange(num_band, dtype="int64"), 0)
    return np.maximum.accumulate(first_bands, axis=1)


def lift_degeneracy(
    freqs: NDArray[np.double],
    eigvecs: NDArray[np.cdouble],
    dDdq: NDArray[np.cdouble],
) -> tuple[NDArray[np.double], NDArray[np.cdouble]]:
    """Lift degeneracy by diagonalizing a perturbation within degenerate subspaces.

    For each degenerate subspace of ``eigvals``, the ``dDdq`` operator
    is projected onto that subspace and diagonalized. The returned
    eigenvectors remain eigenvectors of the original operator (with the same
    ``eigvals``), but are additionally the "good" zeroth-order basis of
    degenerate perturbation theory for ``dDdq``. Non-degenerate
    eigenvectors are returned unchanged.

    A typical use is to pick a unique basis among degenerate phonon bands by
    taking ``dDdq`` as the q-derivative of the dynamical matrix along
    some chosen direction. The resulting eigenvectors then vary analytically
    along that direction in the limit of small perturbation.

    Parameters
    ----------
    freqs :
        Eigenvalues of the unperturbed operator (e.g. dynamical matrix).
        shape=(num_band,)
    eigvecs :
        Eigenvectors of the unperturbed operator, column-wise.
        shape=(num_band, num_band)
    dDdq :
        Perturbation operator in the same basis as ``eigvecs``. For phonon
        group velocities or Grueneisen parameters, this is the q-derivative
        of the dynamical matrix along a chosen direction.
        shape=(num_band, num_band)

    Returns
    -------
    eigvals_pert :
        Eigenvalues of ``dDdq`` projected onto each degenerate
        subspace of ``freqs``. For non-degenerate bands, this is simply
        ``<e|dDdq|e>``.
        shape=(num_band,)
    rot_eigvecs :
        Rotated eigenvectors that diagonalize ``dDdq`` within each
        degenerate subspace of ``freqs``.
        shape=(num_band, num_band)

    """
    rot_eigvecs = np.zeros_like(eigvecs)
    eigvals_pert = np.zeros_like(freqs)
    for deg in degenerate_sets(freqs):
        block = eigvecs[:, deg].T.conj() @ dDdq @ eigvecs[:, deg]
        eigvals_pert[deg], u = np.linalg.eigh(block)
        rot_eigvecs[:, deg] = eigvecs[:, deg] @ u
    return eigvals_pert, rot_eigvecs


def delta_dynamical_matrix(
    q: NDArray[np.double],
    delta_q: NDArray[np.double],
    dynmat: DynamicalMatrix,
) -> NDArray[np.cdouble]:
    """Compute the difference of dynamical matrices at q+delta_q and q-delta_q."""
    dynmat.run(q - delta_q)
    dm1 = dynmat.dynamical_matrix
    dynmat.run(q + delta_q)
    dm2 = dynmat.dynamical_matrix
    assert dm1 is not None and dm2 is not None
    return dm2 - dm1
