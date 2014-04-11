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

import numpy as np

def degenerate_sets(freqs, cutoff=1e-4):
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

def get_eigenvectors(q,
                     dm,
                     ddm,
                     perturbation=None,
                     derivative_order=None,
                     nac_q_direction=None):
    if nac_q_direction is not None and (np.abs(q) < 1e-5).all():
        dm.set_dynamical_matrix(q, q_direction=nac_q_direction)
    else:        
        dm.set_dynamical_matrix(q)
    eigvals, eigvecs = np.linalg.eigh(dm.get_dynamical_matrix())
    eigvals = eigvals.real
    if perturbation is None:
        return eigvals, eigvecs

    eigvecs_new = np.zeros_like(eigvecs)
    deg_sets = degenerate_sets(eigvals)

    if derivative_order is not None:
        ddm.set_derivative_order(derivative_order)
    dD = _get_dD(q, ddm, perturbation)

    for deg in deg_sets:
        if len(deg) == 1:
            continue
        
        eigvecs_arranged = _rearrange_eigenvectors(dD, eigvecs[:, deg])

        if eigvecs_arranged is not None:
            eigvecs[:, deg] = eigvecs_arranged

    return eigvals, eigvecs

def _get_dD(q, ddm, perturbation):
    ddm.run(q)
    ddm_vals = ddm.get_derivative_of_dynamical_matrix()
    dD = np.zeros(ddm_vals.shape[1:], dtype='complex128')
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

def _rearrange_eigenvectors(dD, eigvecs_deg):
    dD_part = np.dot(eigvecs_deg.T.conj(), np.dot(dD, eigvecs_deg))
    p_eigvals, p_eigvecs = np.linalg.eigh(dD_part)
    return np.dot(eigvecs_deg, p_eigvecs)
