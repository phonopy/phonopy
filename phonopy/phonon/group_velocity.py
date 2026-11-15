# SPDX-License-Identifier: BSD-3-Clause
"""Group velocity calculation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    diagonalize_dynamical_matrices,
    get_dynamical_matrices_at_qpoints,
)
from phonopy.phonon.degeneracy import (
    delta_dynamical_matrix,
    derivative_along_direction,
    lift_degeneracy,
)
from phonopy.physical_units import get_physical_units
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation


class GroupVelocity:
    r"""Class to calculate group velocities of phonons.

    d omega   ----
    ------- = \  / omega
    d q        \/q

    Gradient of omega in reciprocal space, which is calculated here by

       1             d D(q)
    ------- <e(q,nu)|------|e(q,nu)>
    2 omega           d q

    Attributes
    ----------
    group_velocity : ndarray
        Group velocities at q-points.
        shape=(q-points, num_band, 3), dtype='double', order='C'
    q_length : float
        Distance in reciprocal space used to calculate finite difference of
        dynamcial matrix.

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        q_length: float | None = None,
        symmetry: Symmetry | None = None,
        frequency_factor_to_THz: float | None = None,
        cutoff_frequency: float = 1e-4,
    ) -> None:
        """Init method.

        dynamical_matrix : DynamicalMatrix
            Dynamical matrix class instance.
        q_length : float, optional
            When set, the dynamical-matrix derivative is approximated by a
            central finite difference ``D(q + q_length) - D(q - q_length)``.
            Default is None, which selects the analytical derivative via
            ``DerivativeOfDynamicalMatrix`` (including Gonze-Lee NAC, since
            the dipole-dipole derivative kernel is now available).
        symmetry : Symmetry
            This is used to symmetrize group velocity at each q-points.
            Default is None, which means no symmetrization.
        frequency_factor_to_THz : float
            Unit conversion factor to convert to THz. Default is VaspToTHz.
        cutoff_frequency : float
            Group velocity is set zero if phonon frequency is below this value.

        """
        self._dynmat = dynamical_matrix
        primitive = dynamical_matrix.primitive
        self._reciprocal_lattice_inv = primitive.cell
        self._reciprocal_lattice: NDArray[np.double] = np.linalg.inv(
            self._reciprocal_lattice_inv
        )  # type: ignore
        self._q_length = q_length

        self._ddm: DerivativeOfDynamicalMatrix | None
        if self._q_length is None:
            self._ddm = DerivativeOfDynamicalMatrix(
                dynamical_matrix, lang=dynamical_matrix.lang
            )
        else:
            self._ddm = None

        self._symmetry = symmetry
        if symmetry is not None:
            # Rotations of q in reduced coordinates and of v in Cartesian.
            self._reciprocal_ops = symmetry.reciprocal_operations
            self._reciprocal_ops_cart = np.array(
                [
                    similarity_transformation(self._reciprocal_lattice, r)
                    for r in self._reciprocal_ops
                ],
                dtype="double",
            )
        if frequency_factor_to_THz is None:
            self._factor = get_physical_units().DefaultToTHz
        else:
            self._factor = frequency_factor_to_THz
        self._cutoff_frequency = cutoff_frequency

        self._directions = np.array(
            [[1, 2, 3], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="double", order="C"
        )
        self._directions[0] /= np.linalg.norm(self._directions[0])

        self._group_velocities: NDArray[np.double] | None = None

    def run(
        self,
        q_points: Sequence[Sequence[float]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
        perturbation: Sequence[float] | NDArray[np.double] | None = None,
    ) -> None:
        """Group velocities are computed at q-points.

        Calculated group velocities are stored in self._group_velocities.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like
            Direction in fractional coordinates of reciprocal space.

        """
        if perturbation is None:
            # Give an random direction to break symmetry
            self._directions[0] = np.array([1, 2, 3], dtype="double")
        else:
            self._directions[0] = np.dot(self._reciprocal_lattice, perturbation)
        self._directions[0] /= np.linalg.norm(self._directions[0])

        qpoints = np.array(q_points, dtype="double")
        if len(qpoints) == 0:
            self._group_velocities = np.array([], dtype="double")
            return

        dynmats = get_dynamical_matrices_at_qpoints(self._dynmat, qpoints)
        eigvals, eigvecs = diagonalize_dynamical_matrices(
            dynmats, lang=self._dynmat.lang
        )
        assert eigvecs is not None
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor
        gv = np.array(
            [
                self._calculate_group_velocity_at_q(q, freqs_at_q, eigvecs_at_q)
                for q, freqs_at_q, eigvecs_at_q in zip(
                    qpoints, freqs, eigvecs, strict=True
                )
            ],
            dtype="double",
            order="C",
        )
        if self._symmetry is not None:
            gv = self._symmetrize_group_velocities(gv, qpoints)
        self._group_velocities = gv

    @property
    def q_length(self) -> float | None:
        """Setter an getter of q_length."""
        return self._q_length

    @q_length.setter
    def q_length(self, q_length: float | None) -> None:
        self._q_length = q_length

    @property
    def group_velocities(self) -> NDArray[np.double] | None:
        """Return group velocities."""
        return self._group_velocities

    def _calculate_group_velocity_at_q(
        self,
        q: NDArray[np.double],
        freqs: NDArray[np.double],
        eigvecs: NDArray[np.cdouble],
    ) -> NDArray[np.double]:
        gv = np.zeros((len(freqs), 3), dtype="double", order="C")

        # Degenerate eigenvectors are rotated by the derivative along
        # self._directions[0] so that their group velocities along the
        # other directions are well defined.
        ddms = self._get_dD(np.array(q))
        _, rot_eigvecs = lift_degeneracy(freqs, eigvecs, ddms[0])
        for i, ddm in enumerate(ddms[1:]):
            gv[:, i] = (rot_eigvecs.conj() * (ddm @ rot_eigvecs)).sum(axis=0).real

        for i, f in enumerate(freqs):
            if f > self._cutoff_frequency:
                gv[i, :] *= self._factor**2 / f / 2
            else:
                gv[i, :] = 0

        return gv

    def _symmetrize_group_velocities(
        self, gv: NDArray[np.double], qpoints: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Symmetrize group velocities by the little group of each q-point.

        This is done after the loop over q-points, which calls rayon-parallel
        kernels of phonors, so that BLAS is not called in that loop.

        gv : shape=(qpoints, bands, 3)
        qpoints : shape=(qpoints, 3)

        """
        q_in_BZ = qpoints - np.rint(qpoints)
        r_cart_mean = np.array([self._get_r_cart_mean(q) for q in q_in_BZ])
        return gv @ r_cart_mean.transpose(0, 2, 1)

    def _get_r_cart_mean(self, q: NDArray[np.double]) -> NDArray[np.double]:
        """Return the mean of the Cartesian rotations that satisfy R q = q."""
        assert self._symmetry is not None
        diff = q - self._reciprocal_ops @ q
        is_little = (np.abs(diff) < self._symmetry.tolerance).all(axis=1)
        return self._reciprocal_ops_cart[is_little].mean(axis=0)

    def _get_dD(self, q: NDArray[np.double]) -> list[NDArray[np.cdouble]]:
        """Compute derivative or finite difference of dynamcial matrices."""
        if self._q_length is None:
            return self._get_dD_analytical(q)
        else:
            return [self._get_dD_FD(q, direction) for direction in self._directions]

    def _get_dD_analytical(self, q: NDArray[np.double]) -> list[NDArray[np.cdouble]]:
        """Compute derivative of dynamcial matrices."""
        assert self._ddm is not None
        self._ddm.run(q)
        ddm = self._ddm.d_dynamical_matrix
        assert ddm is not None
        return [
            derivative_along_direction(ddm, direction) for direction in self._directions
        ]

    def _get_dD_FD(
        self,
        q: NDArray[np.double],
        direction: NDArray[np.double],
    ) -> NDArray[np.cdouble]:
        """Compute finite difference of dynamcial matrices.

        dq :
            q-shift in reduced coordinates in reciprocal space.
            dq = L*^-1 dq_cart

        """
        assert self._q_length is not None
        dq_cart = direction / np.linalg.norm(direction) * self._q_length
        dq = self._dynmat.primitive.cell @ dq_cart
        ddm = delta_dynamical_matrix(q, dq, self._dynmat) / self._q_length / 2
        return ddm
