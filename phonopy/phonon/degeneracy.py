"""Utility routines to handle degeneracy."""

# Copyright (C) 2014 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
