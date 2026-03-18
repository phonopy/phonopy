"""Use first Brillouin zone (Wigner–Seitz cell) to locate q-points."""

# Copyright (C) 2013 Atsushi Togo
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

from phonopy.structure.cells import get_reduced_bases

search_space: NDArray[np.int64] = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
        [-1, -1, -1],
        [-1, -1, 0],
        [-1, -1, 1],
        [-1, 0, -1],
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, -1],
        [-1, 1, 0],
        [-1, 1, 1],
        [0, -1, -1],
        [0, -1, 0],
        [0, -1, 1],
        [0, 0, -1],
    ],
    dtype="int64",
)


def get_qpoints_in_Brillouin_zone(
    reciprocal_lattice: Sequence[Sequence[float]] | NDArray[np.double],
    qpoints: Sequence[Sequence[float]] | NDArray[np.double],
    only_unique: bool = False,
    tolerance: float = 0.01,
) -> NDArray[np.double] | list[NDArray[np.double]]:
    """Move qpoints to first Brillouin zone by lattice translation.

    Parameters
    ----------
    reciprocal_lattice : array_like
        Reciprocal primitive cell basis vectors given in column vectors.
        shape=(3,3), dtype=float
    qpoints : array_like
        q-points in reduced coordinates.
        shape=(n_qpoints,3), dtype=float
    only_unique : bool, optional
        With True, only unique q-points are returned. Default is False.
    tolerance : float, optional
        Tolerance parameter to distinguish equivalent points. Default is 0.01.

    """
    bz = BrillouinZone(reciprocal_lattice, tolerance=tolerance)
    bz.run(qpoints)
    assert bz.shortest_qpoints is not None
    if only_unique:
        return np.array(
            [pts[0] for pts in bz.shortest_qpoints], dtype="double", order="C"
        )
    else:
        return bz.shortest_qpoints


class BrillouinZone:
    """Move qpoints to first Brillouin zone by lattice translation.

    Attributes
    ----------
    shortest_qpoints : list[NDArray[np.double]]
        This must be list.
        Each element of the list contains a set of q-points that are in first
        Brillouin zone (BZ). When inside BZ, there is only one q-point for
        each element, but on the surface, multiple q-points that are
        distinguished by non-zero lattice translation are stored.

    """

    def __init__(
        self,
        reciprocal_lattice: Sequence[Sequence[float]] | NDArray[np.double],
        tolerance: float = 0.01,
    ) -> None:
        """Init method.

        Parameters
        ----------
        reciprocal_lattice : array_like
            Reciprocal primitive cell basis vectors given in column vectors.
            shape=(3,3), dtype=float
        tolerance : float, optional
            Tolerance parameter to distinguish equivalent points. Default is
            0.01.

        """
        _rec_lat = np.array(reciprocal_lattice, dtype="double")
        self._reciprocal_lattice: NDArray[np.double] = _rec_lat
        self._tolerance: float = float(min(np.sum(_rec_lat**2, axis=0)) * tolerance)
        self._reduced_bases: NDArray[np.double] = get_reduced_bases(_rec_lat.T)
        self._tmat: NDArray[np.double] = np.dot(
            np.linalg.inv(_rec_lat), self._reduced_bases.T
        )
        self._tmat_inv: NDArray[np.double] = np.array(
            np.linalg.inv(self._tmat), dtype="double"
        )
        self._shortest_qpoints: list[NDArray[np.double]] | None = None

    def run(
        self,
        qpoints: Sequence[Sequence[float]] | NDArray[np.double],
    ) -> None:
        """Find q-points inside Wigner–Seitz cell.

        Parameters
        ----------
        qpoints : Sequence[Sequence[float]] | NDArray[np.double]
            q-points in reduced coordinates.

        Returns
        -------
        list[NDArray[np.double]]
            Each element of the list contains a set of q-points whose size can
            be different.

        """
        reduced_qpoints = np.dot(qpoints, self._tmat_inv.T)
        reduced_qpoints -= np.rint(reduced_qpoints)
        self._shortest_qpoints = []
        for q in reduced_qpoints:
            distances = (np.dot(q + search_space, self._reduced_bases) ** 2).sum(axis=1)
            min_dist = min(distances)
            shortest_indices = np.where(distances < min_dist + self._tolerance)[0]
            self._shortest_qpoints.append(
                np.dot(search_space[shortest_indices] + q, self._tmat.T)
            )

    @property
    def shortest_qpoints(self) -> list[NDArray[np.double]]:
        """Return shortest qpoints including equivalents."""
        if self._shortest_qpoints is None:
            raise RuntimeError("run method has not been called yet.")
        return self._shortest_qpoints
