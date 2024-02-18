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

from typing import Union

import numpy as np

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC


def degenerate_sets(freqs, cutoff=1e-4):
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


def get_eigenvectors(
    q,
    dm: Union[DynamicalMatrix, DynamicalMatrixNAC],
    ddm: DerivativeOfDynamicalMatrix,
    perturbation=None,
    derivative_order=None,
    nac_q_direction=None,
):
    """Return degenerated eigenvalues and rotated eigenvalues."""
    if nac_q_direction is not None and (np.abs(q) < 1e-5).all():
        dm.run(q, q_direction=nac_q_direction)
    else:
        dm.run(q)
    eigvals, eigvecs = np.linalg.eigh(dm.dynamical_matrix)
    eigvals = eigvals.real
    if perturbation is None:
        return eigvals, eigvecs

    if derivative_order is not None:
        ddm.set_derivative_order(derivative_order)
    dD = _get_dD(q, ddm, perturbation)
    rot_eigvecs, _ = rotate_eigenvectors(eigvals, eigvecs, dD)

    return eigvals, rot_eigvecs


def rotate_eigenvectors(eigvals, eigvecs, dD):
    """Rotate eigenvectors among degenerated band.

    Parameters
    ----------
    eigvals :
        Eigenvalues.
        shape=(num_band, )
    eigvecs :
        Eigenvectors.
        shape=(num_atom * 3, num_band)
    dD :
        q-point derivative of dynamical matrix.

    """
    rot_eigvecs = np.zeros_like(eigvecs)
    eigvals_dD = np.zeros_like(eigvals)
    for deg in degenerate_sets(eigvals):
        dD_part = np.dot(eigvecs[:, deg].T.conj(), np.dot(dD, eigvecs[:, deg]))
        eigvals_dD[deg], eigvecs_dD = np.linalg.eigh(dD_part)
        rot_eigvecs[:, deg] = np.dot(eigvecs[:, deg], eigvecs_dD)
    return rot_eigvecs, eigvals_dD


def _get_dD(q, ddm: DerivativeOfDynamicalMatrix, perturbation):
    """Return q-vector derivative of dynamical matrix.

    Returns
    -------
    shape=(3, num_band, num_band).

    """
    ddm.run(q)
    ddm_vals = ddm.get_derivative_of_dynamical_matrix()
    dD = np.zeros(ddm_vals.shape[1:], dtype=ddm_vals.dtype, order="C")
    if len(ddm_vals) == 3:
        for i in range(3):
            dD += perturbation[i] * ddm_vals[i]
        return dD / np.linalg.norm(perturbation)
    else:
        dD += perturbation[0] * perturbation[0] * ddm_vals[0]
        dD += perturbation[1] * perturbation[1] * ddm_vals[1]
        dD += perturbation[2] * perturbation[2] * ddm_vals[2]
        dD += 2 * perturbation[0] * perturbation[1] * ddm_vals[5]
        dD += 2 * perturbation[0] * perturbation[2] * ddm_vals[4]
        dD += 2 * perturbation[1] * perturbation[2] * ddm_vals[3]
        return dD / np.linalg.norm(perturbation) ** 2
