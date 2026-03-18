"""Mode Grueneisen parameter calculation."""

# Copyright (C) 2012 Atsushi Togo
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

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import estimate_band_connection
from phonopy.phonon.degeneracy import rotate_eigenvectors


class GruneisenBase:
    """Base class of mode Grueneisen parameter calculation classes."""

    def __init__(
        self,
        dynmat: DynamicalMatrix | DynamicalMatrixNAC,
        dynmat_plus: DynamicalMatrix | DynamicalMatrixNAC,
        dynmat_minus: DynamicalMatrix | DynamicalMatrixNAC,
        delta_strain: float | None = None,
        qpoints: NDArray[np.double] | None = None,
        is_band_connection: bool = False,
    ) -> None:
        """Init method."""
        self._dynmat = dynmat
        self._dynmat_plus = dynmat_plus
        self._dynmat_minus = dynmat_minus
        if delta_strain is None:
            volume = dynmat.primitive.volume
            volume_plus = dynmat_plus.primitive.volume
            volume_minus = dynmat_minus.primitive.volume
            dV = volume_plus - volume_minus
            self._delta_strain = dV / volume
        else:
            self._delta_strain = delta_strain
        self._is_band_connection = is_band_connection
        self._qpoints = qpoints

        self._gruneisen: NDArray[np.double] | None = None
        self._eigenvalues: NDArray[np.double] | None = None
        self._eigenvectors: NDArray[np.cdouble] | None = None
        self._q_direction: NDArray[np.double] | None = None
        if qpoints is not None:
            self._set_gruneisen()

    def set_qpoints(self, qpoints: NDArray[np.double]) -> None:
        """Set q-points."""
        self._qpoints = qpoints
        self._set_gruneisen()

    def get_gruneisen(self) -> NDArray[np.double] | None:
        """Return mode Grueneisen parameters."""
        return self._gruneisen

    def get_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues."""
        return self._eigenvalues

    def get_eigenvectors(self) -> NDArray[np.cdouble] | None:
        """Return eigenvectors."""
        return self._eigenvectors

    def _set_gruneisen(self) -> None:
        assert self._qpoints is not None
        if self._is_band_connection:
            self._q_direction = self._qpoints[0] - self._qpoints[-1]
            band_order: Sequence[int] = list(range(len(self._dynmat.primitive) * 3))
            prev_eigvecs: NDArray[np.cdouble] | None = None

        edDe: list[NDArray[np.double]] = []  # <e|dD|e>
        eigvals: list[NDArray[np.double]] = []
        eigvecs: list[NDArray[np.cdouble]] = []
        for _, q in enumerate(self._qpoints):
            if self._is_band_connection and isinstance(
                self._dynmat, DynamicalMatrixNAC
            ):
                self._dynmat.run(q, q_direction=self._q_direction)
            else:
                self._dynmat.run(q)

            dm = self._dynmat.dynamical_matrix
            assert dm is not None
            evals, evecs = np.linalg.eigh(dm)
            evals_at_q = np.array(evals, dtype="double")
            dD = self._get_dD(q, self._dynmat_minus, self._dynmat_plus)
            evecs_at_q, edDe_at_q = rotate_eigenvectors(evals_at_q, evecs, dD)

            if self._is_band_connection:
                if prev_eigvecs is not None:
                    band_order = estimate_band_connection(
                        prev_eigvecs, evecs_at_q, band_order
                    )
                eigvals.append(evals_at_q[band_order])
                eigvecs.append(evecs_at_q[:, band_order])
                edDe.append(edDe_at_q[band_order])
                prev_eigvecs = evecs_at_q
            else:
                eigvals.append(evals_at_q)
                eigvecs.append(evecs_at_q)
                edDe.append(edDe_at_q)

        edDe_arr = np.array(edDe, dtype="double", order="C")
        self._eigenvalues = np.array(eigvals, dtype="double", order="C")
        self._eigenvectors = np.array(eigvecs, dtype="cdouble", order="C")
        self._gruneisen = -edDe_arr / self._delta_strain / self._eigenvalues / 2

    def _get_dD(
        self,
        q: NDArray[np.double],
        d_a: DynamicalMatrix | DynamicalMatrixNAC,
        d_b: DynamicalMatrix | DynamicalMatrixNAC,
    ) -> NDArray[np.cdouble]:
        if (
            self._is_band_connection
            and isinstance(d_a, DynamicalMatrixNAC)
            and isinstance(d_b, DynamicalMatrixNAC)
        ):
            d_a.run(q, q_direction=self._q_direction)
            d_b.run(q, q_direction=self._q_direction)
        else:
            d_a.run(q)
            d_b.run(q)
        dm_a = d_a.dynamical_matrix
        dm_b = d_b.dynamical_matrix
        assert dm_a is not None
        assert dm_b is not None
        return dm_b - dm_a
