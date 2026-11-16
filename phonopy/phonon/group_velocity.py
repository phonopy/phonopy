# SPDX-License-Identifier: BSD-3-Clause
"""Group velocity calculation."""

from __future__ import annotations

import warnings
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

        self._lifting_direction = self._get_lifting_direction(None)

        self._group_velocities: NDArray[np.double] | None = None

    def run(
        self,
        q_points: Sequence[Sequence[float]]
        | Sequence[NDArray[np.double]]
        | NDArray[np.double],
        perturbation: Sequence[float] | NDArray[np.double] | None = None,
        degeneracy_lifting_direction: Sequence[float]
        | NDArray[np.double]
        | None = None,
    ) -> None:
        """Group velocities are computed at q-points.

        Calculated group velocities are stored in self._group_velocities.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like, optional
            Deprecated.  Use ``degeneracy_lifting_direction``.
        degeneracy_lifting_direction : array-like, optional
            Direction in reduced coordinates of the reciprocal lattice used to
            choose eigenvectors of degenerate bands.  In each degenerate
            subspace, the eigenvectors are rotated to diagonalize the
            derivative of the dynamical matrix along this direction.  Only the
            direction is used.  When None, the direction (1, 2, 3) in
            Cartesian coordinates is used.  ``shape=(3,)``.

        """
        degeneracy_lifting_direction = self._handle_deprecated_perturbation(
            perturbation, degeneracy_lifting_direction
        )
        self._lifting_direction = self._get_lifting_direction(
            degeneracy_lifting_direction
        )

        qpoints = np.array(q_points, dtype="double")
        if len(qpoints) == 0:
            self._group_velocities = np.array([], dtype="double")
            return

        freqs, eigvecs = self._get_frequencies_and_eigenvectors(qpoints)
        dD_lifting, dD_cart = self._get_dD(qpoints)
        gv = np.array(
            [
                self._calculate_group_velocity_at_q(
                    freqs_at_q, eigvecs_at_q, dD_lifting_at_q, dD_cart_at_q
                )
                for freqs_at_q, eigvecs_at_q, dD_lifting_at_q, dD_cart_at_q in zip(
                    freqs, eigvecs, dD_lifting, dD_cart, strict=True
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

    def _get_frequencies_and_eigenvectors(
        self, qpoints: NDArray[np.double]
    ) -> tuple[NDArray[np.double], NDArray[np.cdouble]]:
        """Return frequencies and eigenvectors at q-points.

        Dynamical matrices at all q-points are built and diagonalized at
        once.

        Returns
        -------
        freqs : ndarray
            Frequencies in THz. Imaginary frequencies are given as negative
            values.
            shape=(num_qpoints, num_band), dtype='double'
        eigvecs : ndarray
            Eigenvectors in columns.
            shape=(num_qpoints, num_band, num_band), dtype='cdouble'

        """
        dynmats = get_dynamical_matrices_at_qpoints(self._dynmat, qpoints)
        eigvals, eigvecs = diagonalize_dynamical_matrices(
            dynmats, lang=self._dynmat.lang
        )
        assert eigvecs is not None
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * self._factor
        return freqs, eigvecs

    def _calculate_group_velocity_at_q(
        self,
        freqs: NDArray[np.double],
        eigvecs: NDArray[np.cdouble],
        dD_lifting: NDArray[np.cdouble],
        dD_cart: NDArray[np.cdouble],
    ) -> NDArray[np.double]:
        """Return group velocities at a q-point.

        dD_lifting and dD_cart are those of ``_get_dD`` at this q-point.

        """
        gv = np.zeros((len(freqs), 3), dtype="double", order="C")

        # Degenerate eigenvectors are rotated by the derivative along
        # self._lifting_direction so that their group velocities along the
        # Cartesian directions are well defined.
        _, rot_eigvecs = lift_degeneracy(freqs, eigvecs, dD_lifting)
        for i, dD in enumerate(dD_cart):
            gv[:, i] = (rot_eigvecs.conj() * (dD @ rot_eigvecs)).sum(axis=0).real

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

    def _handle_deprecated_perturbation(
        self,
        perturbation: Sequence[float] | NDArray[np.double] | None,
        degeneracy_lifting_direction: Sequence[float] | NDArray[np.double] | None,
    ) -> Sequence[float] | NDArray[np.double] | None:
        """Return degeneracy_lifting_direction given by either argument of run.

        REMOVE-AT-V5: perturbation is removed, and so is this method.

        """
        if perturbation is None:
            return degeneracy_lifting_direction
        if degeneracy_lifting_direction is not None:
            raise ValueError(
                "perturbation and degeneracy_lifting_direction cannot be given "
                "together."
            )
        warnings.warn(
            "perturbation is deprecated. Use degeneracy_lifting_direction.",
            DeprecationWarning,
            stacklevel=3,
        )
        return perturbation

    def _get_lifting_direction(
        self,
        degeneracy_lifting_direction: Sequence[float] | NDArray[np.double] | None,
    ) -> NDArray[np.double]:
        """Return the Cartesian unit vector along which degeneracy is lifted.

        degeneracy_lifting_direction :
            Direction in reduced coordinates of the reciprocal lattice. With
            None, an arbitrary direction that breaks the symmetry is used.

        """
        if degeneracy_lifting_direction is None:
            direction = np.array([1, 2, 3], dtype="double")
        else:
            direction = np.dot(self._reciprocal_lattice, degeneracy_lifting_direction)
        return direction / np.linalg.norm(direction)

    def _get_dD(
        self, qpoints: NDArray[np.double]
    ) -> tuple[NDArray[np.cdouble], NDArray[np.cdouble]]:
        """Compute derivatives or finite differences of dynamcial matrices.

        The arrays of all q-points are allocated at once.  This is done
        before the loop over q-points that calls BLAS, so that kernels of
        phonors are not called in that loop.

        Parameters
        ----------
        qpoints : ndarray
            q-points in reduced coordinates.
            shape=(num_qpoints, 3)

        Returns
        -------
        dD_lifting : ndarray
            Derivatives along self._lifting_direction.
            shape=(num_qpoints, num_band, num_band)
        dD_cart : ndarray
            Derivatives with respect to Cartesian components of q.
            shape=(num_qpoints, 3, num_band, num_band)

        """
        if self._q_length is None:
            return self._get_dD_analytical(qpoints)
        return self._get_dD_FD_at_qpoints(qpoints)

    def _get_dD_FD_at_qpoints(
        self, qpoints: NDArray[np.double]
    ) -> tuple[NDArray[np.cdouble], NDArray[np.cdouble]]:
        """Compute finite differences of dynamcial matrices at q-points."""
        dD_lifting = np.array(
            [self._get_dD_FD(q, self._lifting_direction) for q in qpoints],
            dtype="cdouble",
            order="C",
        )
        dD_cart = np.array(
            [
                [self._get_dD_FD(q, direction) for direction in np.eye(3)]
                for q in qpoints
            ],
            dtype="cdouble",
            order="C",
        )
        return dD_lifting, dD_cart

    def _get_dD_analytical(
        self, qpoints: NDArray[np.double]
    ) -> tuple[NDArray[np.cdouble], NDArray[np.cdouble]]:
        """Compute derivatives of dynamcial matrices at q-points."""
        assert self._ddm is not None
        dD_cart = self._ddm.run_at_qpoints(qpoints)
        # derivative_along_direction takes the Cartesian axis first.
        dD_lifting = derivative_along_direction(
            dD_cart.transpose(1, 0, 2, 3), self._lifting_direction
        )
        return dD_lifting, dD_cart

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
