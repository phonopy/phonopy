# SPDX-License-Identifier: BSD-3-Clause
"""Mode Grueneisen parameter calculation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import estimate_band_connection
from phonopy.phonon.degeneracy import lift_degeneracy


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
            edDe_at_q, evecs_at_q = lift_degeneracy(evals_at_q, evecs, dD)

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
