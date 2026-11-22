# SPDX-License-Identifier: BSD-3-Clause
"""Calculate group velocity matrix."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.phonon.degeneracy import lift_degeneracy
from phonopy.phonon.group_velocity import GroupVelocity

LITTLE_GROUP_TOLERANCE = 1e-5


class VelocityMatrix(GroupVelocity):
    """Class to calculate group velocities matrices of phonons.

     v_qjj' = 1/(2*sqrt(omega_qj*omega_qj')) * <e(q,j)|dD/dq|e(q,j')>

    Attributes
    ----------
    velocity_matrices : ndarray

    """

    def __init__(
        self,
        dynamical_matrix: DynamicalMatrix,
        q_length: float | None = None,
        rotations_cartesian: NDArray[np.double] | None = None,
        reciprocal_operations: NDArray[np.int64] | None = None,
        frequency_factor_to_THz: float | None = None,
        cutoff_frequency: float = 1e-4,
    ) -> None:
        """Init method.

        See details of parameters at phonopy `GroupVelocity` class.

        ``rotations_cartesian`` is generated from ``reciprocal_operations`` and
        their orders are corresponding.

        """
        GroupVelocity.__init__(
            self,
            dynamical_matrix,
            q_length=q_length,
            frequency_factor_to_THz=frequency_factor_to_THz,
            cutoff_frequency=cutoff_frequency,
        )

        self._reciprocal_operations = reciprocal_operations
        self._rotations_cartesian = rotations_cartesian

        self._velocity_matrices: NDArray[np.cdouble] | None = None

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
        """Run group velocity matrix calculate at q-points.

        Calculated group velocities are stored in
        self._velocity_matrices.

        Parameters
        ----------
        q_points : array-like
            List of q-points such as [[0, 0, 0], [0.1, 0.2, 0.3], ...].
        perturbation : array-like, optional
            Deprecated.  Use ``degeneracy_lifting_direction``.
        degeneracy_lifting_direction : array-like, optional
            Direction in reduced coordinates of the reciprocal lattice used to
            choose eigenvectors of degenerate bands.  See ``GroupVelocity.run``.

        """
        degeneracy_lifting_direction = self._handle_deprecated_perturbation(
            perturbation, degeneracy_lifting_direction
        )
        self._lifting_direction = self._get_lifting_direction(
            degeneracy_lifting_direction
        )
        _q_points = np.asarray(q_points, dtype="double")
        if len(_q_points) == 0:
            self._velocity_matrices = np.array([], dtype="cdouble")
            self._group_velocities = np.array([], dtype="double")
            return

        freqs, eigvecs = self._get_frequencies_and_eigenvectors(_q_points)
        dD_lifting, dD_cart = self._get_dD(_q_points)
        vm = [
            self._calculate_velocity_matrix_at_q(
                freqs_at_q, eigvecs_at_q, dD_lifting_at_q, dD_cart_at_q
            )
            for freqs_at_q, eigvecs_at_q, dD_lifting_at_q, dD_cart_at_q in zip(
                freqs, eigvecs, dD_lifting, dD_cart, strict=True
            )
        ]
        self._velocity_matrices = np.array(vm, dtype="cdouble", order="C")
        self._group_velocities = np.array(
            [
                np.einsum("aii->ia", vm_at_q).real @ self._get_projector(q).T
                for vm_at_q, q in zip(vm, _q_points, strict=True)
            ],
            dtype="double",
            order="C",
        )

    @property
    def velocity_matrices(self) -> NDArray[np.cdouble] | None:
        """Return group velocity matrices.

        Returns
        -------
        velocity_matrices : ndarray
            shape=(q-points, 3, num_band, num_band), order='C'
            dtype=complex that is "c%d" % (np.dtype('double').itemsize * 2)

        """
        return self._velocity_matrices

    @property
    def group_velocities(
        self,
    ) -> NDArray[np.double] | None:
        """Return group velocities."""
        return self._group_velocities

    def _calculate_velocity_matrix_at_q(
        self,
        freqs: NDArray[np.double],
        eigvecs: NDArray[np.cdouble],
        dD_lifting: NDArray[np.cdouble],
        dD_cart: NDArray[np.cdouble],
    ) -> NDArray[np.cdouble]:
        """Return velocity matrices at a q-point.

        dD_lifting and dD_cart are those of ``_get_dD`` at this q-point.

        """
        # Degenerate eigenvectors are rotated by the derivative along
        # self._lifting_direction.
        _, rot_eigvecs = lift_degeneracy(freqs, eigvecs, dD_lifting)
        mask = freqs > self._cutoff_frequency
        scale = np.zeros_like(freqs)
        scale[mask] = 1 / np.sqrt(2 * freqs[mask])
        rot_eigvecs *= scale

        vm = np.zeros((3,) + eigvecs.shape, dtype="cdouble", order="C")
        for i, dD in enumerate(dD_cart):
            vm[i] = rot_eigvecs.T.conj() @ dD @ rot_eigvecs

        return self._hermitian_velocity_matrix(vm) * self._factor**2

    def _hermitian_velocity_matrix(
        self, vm: NDArray[np.cdouble]
    ) -> NDArray[np.cdouble]:
        vm = (vm + vm.transpose(0, 2, 1).conj()) / 2
        return vm

    def _get_projector(self, q: NDArray[np.double]) -> NDArray[np.double]:
        """Return little group rotations at q."""
        assert (
            self._reciprocal_operations is not None
            and self._rotations_cartesian is not None
        )
        projector = np.zeros((3, 3), dtype="double", order="C")
        order = 0
        for r, r_cart in zip(
            self._reciprocal_operations, self._rotations_cartesian, strict=True
        ):
            q_in_BZ = q - np.rint(q)
            diff = q_in_BZ - r @ q_in_BZ
            if (np.abs(diff) < LITTLE_GROUP_TOLERANCE).all():
                projector += r_cart
                order += 1
        return projector / order
