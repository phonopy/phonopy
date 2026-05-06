"""Calculated accumulated property with respect to other property."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy.other.tetrahedron_method import get_integration_weights
from phonopy.phonon.dos import NormalDistribution
from phonopy.phonon.grid import BZGrid

epsilon = 1.0e-8


class KappaDOSTHM:
    """Class to calculate DOS like spectram with tetrahedron method.

    To compute usual DOS on all GR grid points:

    ```
    freqs, _, _ = ph3.get_phonon_data()
    freqs_grg = freqs[bzgrid.grg2bzg]
    kappados = KappaDOSTHM(
        np.ones(freqs_grg.shape, dtype=float)[None, :, :, None],
        freqs_grg,
        bzgrid,
        num_sampling_points=201
    )
    ```

    To compute DOS on ir-grid points:

    ```
    freqs, _, _ = ph3.get_phonon_data()
    ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bzgrid)
    freqs_ir = freqs[bzgrid.grg2bzg[ir_grid_points]]
    kappados = KappaDOSTHM(
        np.ones(freqs_ir.shape, dtype=float)[None, :, :, None],
        freqs_ir,
        bzgrid,
        ir_grid_points=ir_grid_points,
        ir_grid_weights=ir_grid_weights,
        ir_grid_map=ir_grid_map,
        num_sampling_points=201,
    )
    ```

    """

    def __init__(
        self,
        mode_kappa: NDArray[np.double],
        frequencies: NDArray[np.double],
        bz_grid: BZGrid,
        ir_grid_points: NDArray[np.int64] | None = None,
        ir_grid_weights: NDArray[np.int64] | None = None,
        ir_grid_map: NDArray[np.int64] | None = None,
        frequency_points: NDArray[np.double] | Sequence[float] | None = None,
        num_sampling_points: int = 100,
    ):
        """Init method.

        Parameters
        ----------
        mode_kappa : ndarray
            Target value.
            shape=(temperatures, ir_grid_points, num_band, num_elem),
            dtype='double'
        frequencies : ndarray
            Frequencies at ir-grid points.
            shape=(ir_grid_points, num_band), dtype='double'
        bz_grid : BZGrid
            BZGrid instance.
        ir_grid_points : ndarray
            Irreducible grid point indices in GR-grid (as obtained by
            get_ir_grid_points).
            shape=(num_ir_grid_points, ), dtype='int64'
        ir_grid_weights : ndarray
            Weights of irreducible grid points. Its sum is the number of grid
            points in GR-grid (prod(D_diag)) (as obtained by
            get_ir_grid_points).
            shape=(num_ir_grid_points, ), dtype='int64'
        ir_grid_map : ndarray
            Index mapping table to irreducible grid points from all grid points
            in GR-grid such as, [0, 0, 2, 3, 3, ...]. (as obtained by
            get_ir_grid_points).
            shape=(prod(D_diag), ), dtype='int64'
        frequency_points : array_like, optional, default=None
            This is used as the frequency points. When None, frequency points
            are created from `num_sampling_points`.
        num_sampling_points : int, optional, default=100
            Number of uniform sampling points.

        """
        min_freq = min(frequencies.ravel())
        max_freq = max(frequencies.ravel()) + epsilon
        if frequency_points is None:
            self._frequency_points = np.linspace(
                min_freq, max_freq, num_sampling_points, dtype="double"
            )
        else:
            self._frequency_points = np.array(frequency_points, dtype="double")

        n_temp, _, _, n_elem = mode_kappa.shape
        self._kdos = np.zeros(
            (n_temp, len(self._frequency_points), 2, n_elem), dtype="double"
        )
        _ir_grid_points: NDArray[np.int64]
        if ir_grid_points is None:
            _ir_grid_points = np.arange(len(frequencies), dtype="int64")
        else:
            _ir_grid_points = ir_grid_points
        grid_points = bz_grid.grg2bzg[_ir_grid_points]
        _ir_grid_map: NDArray[np.int64]
        if ir_grid_map is None:
            _ir_grid_map = np.arange(len(frequencies), dtype="int64")
        else:
            _ir_grid_map = ir_grid_map
        bzgp2irgp_map = self._get_bzgp2irgp_map(
            bz_grid.bzg2grg, _ir_grid_map, _ir_grid_points
        )
        grid_weights: NDArray[np.int64]
        if ir_grid_weights is None:
            grid_weights = np.ones(mode_kappa.shape[1], dtype="int64")
        else:
            grid_weights = ir_grid_weights
        func: Literal["J", "I"]
        for j, func in enumerate(("J", "I")):  # type: ignore[assignment]
            iweights = get_integration_weights(
                self._frequency_points,
                frequencies,
                bz_grid,
                grid_points=grid_points,
                bzgp2irgp_map=bzgp2irgp_map,
                function=func,
            )
            for i, iw in enumerate(iweights):
                self._kdos[:, :, j] += np.transpose(
                    np.dot(iw, mode_kappa[:, i] * grid_weights[i]), axes=(1, 0, 2)
                )
        self._kdos /= np.prod(bz_grid.D_diag)

    def get_kdos(self) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Return thermal conductivity spectram.

        Returns
        -------
        tuple
            frequency_points : ndarray
                shape=(sampling_points, ), dtype='double'
            kdos : ndarray
                shape=(temperatures, sampling_points, 2 (J, I), num_elem),
                dtype='double', order='C'

        """
        return self._frequency_points, self._kdos

    def _get_bzgp2irgp_map(
        self,
        bzg2grg: NDArray[np.int64],
        ir_grid_map: NDArray[np.int64],
        ir_grid_points: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Return mapping table from BZ-grid indices to ir-grid point indices.

        More precisely, return mapping table from grid points in BZ-grid to
        indices of ir-grid points. The length of the set of the indices is the
        number of the ir-grid points.

        Parameters
        ----------
        bzg2grg : ndarray
            Mapping table from BZ-grid to GR-grid.
            shape=(len(all-BZ-grid-points), ), dtype='int64'
        ir_grid_map : ndarray
            Mapping table from all grid points to ir-grid points in GR-grid.
            shape=(np.prod(D_diag), ), dtype='int64'
        ir_grid_points : ndarray
            Irreducible grid points in GR-grid. shape=(num_ir_grid_points, ),
            dtype='int64'

        Returns
        -------
        np.ndarray
            Mapping table from BZ-grid to indices of ir-grid points.
            shape=(len(ir-grid-points), ), dtype='

        """
        unique_gps = np.unique(ir_grid_map)
        assert np.array_equal(unique_gps, ir_grid_points)
        # ir-grid points in GR-grid to the index of unique grid points.
        gp_map = {j: i for i, j in enumerate(unique_gps)}
        bzgp2irgp_map = np.array(
            [gp_map[ir_grid_map[grgp]] for grgp in bzg2grg], dtype="int64"
        )
        return bzgp2irgp_map


class GammaDOSsmearing:
    """Class to calculate Gamma spectram by smearing method."""

    def __init__(
        self,
        gamma: NDArray[np.double],
        frequencies: NDArray[np.double],
        ir_grid_weights: NDArray[np.int64],
        sigma: float | None = None,
        num_sampling_points: int = 200,
    ):
        """Init method.

        gamma : ndarray
            Target value.
            shape=(temperatures, ir_grid_points, num_band)
            dtype='double'
        frequencies : ndarray
            shape=(ir_grid_points, num_band), dtype='double'
        ir_grid_weights : ndarray
            Grid point weights at ir-grid points.
            shape=(ir_grid_points, ), dtype='int64'
        sigma : float
            Smearing width.
        num_sampling_points : int, optional, default=100
            Number of uniform sampling points.

        """
        self._gamma = gamma
        self._frequencies = frequencies
        self._ir_grid_weights = ir_grid_weights
        self._num_sampling_points = num_sampling_points
        self._set_frequency_points()
        self._gdos = np.zeros(
            (len(gamma), len(self._frequency_points), 2), dtype="double"
        )
        if sigma is None:
            self._sigma = (
                max(self._frequency_points) - min(self._frequency_points)
            ) / 100
        else:
            self._sigma = sigma
        self._smearing_function = NormalDistribution(self._sigma)
        self._run_smearing_method()

    def get_gdos(self) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Return Gamma spectram.

        gdos[:, :, 0] is not used but eixts to be similar shape to kdos.

        """
        return self._frequency_points, self._gdos

    def _set_frequency_points(self) -> None:
        min_freq = np.min(self._frequencies)
        max_freq = np.max(self._frequencies) + epsilon
        self._frequency_points = np.linspace(
            min_freq, max_freq, self._num_sampling_points
        )

    def _run_smearing_method(self) -> None:
        num_gp = np.sum(self._ir_grid_weights)
        for i, f in enumerate(self._frequency_points):
            dos = self._smearing_function.calc(self._frequencies - f)
            for j, g_t in enumerate(self._gamma):
                self._gdos[j, i, 1] = (
                    np.sum(np.dot(self._ir_grid_weights, dos * g_t)) / num_gp
                )


def run_prop_dos(
    frequencies: NDArray[np.double],
    mode_prop: NDArray[np.double],
    ir_grid_map: NDArray[np.int64] | None,
    ir_grid_points: NDArray[np.int64] | None,
    num_sampling_points: int,
    bz_grid: BZGrid,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Run DOS-like calculation.

    This is a simple wrapper of KappsDOSTHM.

    Parameters
    ----------
    frequencies:
        Frequencies at ir-grid points.
    mode_prop:
        Properties at  ir-grid points.
    ir_grid_map:
        Obtained by get_ir_grid_points(bz_grid)[2].
    ir_grid_points:
        Obtained by get_ir_grid_points(bz_grid)[0].
    num_sampling_points:
        Number of sampling points in horizontal axis.
    bz_grid:
        BZ grid.

    """
    kappa_dos = KappaDOSTHM(
        mode_prop,
        frequencies,
        bz_grid,
        ir_grid_points=ir_grid_points,
        ir_grid_map=ir_grid_map,
        num_sampling_points=num_sampling_points,
    )
    freq_points, kdos = kappa_dos.get_kdos()
    sampling_points = np.tile(freq_points, (len(kdos), 1))
    return kdos, sampling_points


def run_mfp_dos(
    mean_freepath: NDArray[np.double],
    mode_prop: NDArray[np.double],
    ir_grid_map: NDArray[np.int64] | None,
    ir_grid_points: NDArray[np.int64] | None,
    num_sampling_points: int,
    bz_grid: BZGrid,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Run DOS-like calculation for mean free path.

    mean_freepath : shape=(temperatures, ir_grid_points, 6)
    mode_prop : shape=(temperatures, ir_grid_points, 6, 6)

    """
    kdos: list[NDArray[np.double]] = []
    sampling_points: list[NDArray[np.double]] = []
    for i, _ in enumerate(mean_freepath):
        kappa_dos = KappaDOSTHM(
            mode_prop[i : i + 1, :, :],
            mean_freepath[i],
            bz_grid,
            ir_grid_points=ir_grid_points,
            ir_grid_map=ir_grid_map,
            num_sampling_points=num_sampling_points,
        )
        sampling_points_at_T, kdos_at_T = kappa_dos.get_kdos()
        kdos.append(kdos_at_T[0])
        sampling_points.append(sampling_points_at_T)
    kdos_array: NDArray[np.double] = np.array(kdos)
    sampling_points_array: NDArray[np.double] = np.array(sampling_points)

    return kdos_array, sampling_points_array


def get_mfp(
    g: NDArray[np.double],
    gv: NDArray[np.double],
) -> NDArray[np.double]:
    """Calculate mean free path from inverse lifetime and group velocity."""
    g = np.where(g > 0, g, -1)
    gv_norm = np.sqrt((gv**2).sum(axis=2))
    mean_freepath = np.where(g > 0, gv_norm / (2 * 2 * np.pi * g), 0)
    return mean_freepath
