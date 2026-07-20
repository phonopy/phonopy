# SPDX-License-Identifier: BSD-3-Clause
"""Spectrum-like quantities computed from per-mode values on an irreducible grid.

Two complementary backends are provided:

- ``TetrahedronDOSAccumulator``: linear tetrahedron method.  Returns both the
  differential spectrum (density) and its cumulative integral.
- ``SmearingDOSAccumulator``: Gaussian-smeared accumulation.  Returns only the
  differential spectrum.

Both reduce to the plain phonon DOS when ``mode_property`` is left at its
default of unit weights.  Typical non-DOS uses include kappa(omega),
kappa(MFP), and gamma(omega) spectra.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from phonopy._lang import resolve_lang
from phonopy.phonon.grid import BZGrid
from phonopy.phonon.tetrahedron_method import get_integration_weights

_BIN_EPS = 1.0e-8


@dataclass(frozen=True)
class TetrahedronDOSResult:
    """Output of TetrahedronDOSAccumulator.

    Attributes
    ----------
    sampling_points : NDArray[np.double]
        Values along the accumulation axis.  shape=(n_sampling,) when
        ``bin_values`` was 2D (axis shared across batches), or
        shape=(n_batch, n_sampling) when ``bin_values`` was 3D and the
        sampling axis was auto-generated per batch.
    cumulative : NDArray[np.double]
        Running integral of the property from -inf to each sampling point.
        For DOS this is the integrated density of states.
        shape=(n_batch, n_sampling, n_elem)
    density : NDArray[np.double]
        Differential value at each sampling point.  For unit weights this is
        the plain DOS.
        shape=(n_batch, n_sampling, n_elem)

    """

    sampling_points: NDArray[np.double]
    cumulative: NDArray[np.double]
    density: NDArray[np.double]


class TetrahedronDOSAccumulator:
    """Tetrahedron-method spectrum accumulator with optional per-mode weights.

    Computes both the cumulative integral and the differential spectrum of a
    per-mode property along an accumulation axis (typically frequency, but any
    per-mode scalar works -- e.g. mean free path).

    With ``mode_property=None`` the accumulator reduces to the plain phonon DOS:
    ``cumulative`` is the integrated DOS up to each sampling point, ``density``
    is the DOS itself.

    Examples
    --------
    Plain DOS on all GR-grid points::

        freqs, _, _ = ph3.get_phonon_data()
        freqs_grg = freqs[bzgrid.grg2bzg]
        acc = TetrahedronDOSAccumulator(
            freqs_grg, bzgrid, num_sampling_points=201
        )
        result = acc.result

    DOS on ir-grid points::

        ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bzgrid)
        freqs_ir = freqs[bzgrid.grg2bzg[ir_grid_points]]
        acc = TetrahedronDOSAccumulator(
            freqs_ir,
            bzgrid,
            ir_grid_points=ir_grid_points,
            ir_grid_weights=ir_grid_weights,
            ir_grid_map=ir_grid_map,
            num_sampling_points=201,
        )

    """

    def __init__(
        self,
        bin_values: NDArray[np.double],
        bz_grid: BZGrid,
        mode_property: NDArray[np.double] | None = None,
        ir_grid_points: NDArray[np.int64] | None = None,
        ir_grid_weights: NDArray[np.int64] | None = None,
        ir_grid_map: NDArray[np.int64] | None = None,
        sampling_points: NDArray[np.double] | Sequence[float] | None = None,
        num_sampling_points: int = 100,
        lang: Literal["C", "Rust"] = "Rust",
    ) -> None:
        """Init method.

        Parameters
        ----------
        bin_values : NDArray[np.double]
            Per-mode values defining the accumulation axis.  Either:

            - shape=(n_ir, n_band) -- shared across all batches (e.g. phonon
              frequencies),
            - shape=(n_batch, n_ir, n_band) -- one set per batch (e.g. mean
              free path at each temperature).

        bz_grid : BZGrid
            Reciprocal-space grid information.
        mode_property : NDArray[np.double] | None, optional
            Per-mode property to accumulate.
            shape=(n_batch, n_ir, n_band, n_elem).  ``None`` (default) is
            treated as unit weights with n_batch=1, n_elem=1, giving the plain
            DOS.
        ir_grid_points : NDArray[np.int64] | None, optional
            Irreducible grid point indices in GR-grid (as obtained from
            ``get_ir_grid_points``).  ``None`` defaults to ``arange(n_ir)``.
        ir_grid_weights : NDArray[np.int64] | None, optional
            Multiplicities of irreducible grid points.  ``None`` defaults to
            ones (correct only when the grid is full or trivially symmetric).
        ir_grid_map : NDArray[np.int64] | None, optional
            Mapping from all GR-grid points to their irreducible image
            (as obtained from ``get_ir_grid_points``).  ``None`` defaults to
            ``arange(n_ir)``.
        sampling_points : array_like, optional
            Explicit sampling-point grid (1D, shared across batches).  If
            ``None`` and ``bin_values`` is 2D, a single uniform grid spanning
            the bin_values range is generated.  If ``None`` and ``bin_values``
            is 3D, a per-batch uniform grid spanning each batch's range is
            generated and returned as a 2D array.
        num_sampling_points : int, optional
            Number of uniform sampling points (used only when
            ``sampling_points`` is None).
        lang : {"C", "Rust"}, optional
            Backend for the tetrahedron-weight kernel.

        """
        lang = resolve_lang(lang)
        bin_values_arr = np.asarray(bin_values, dtype="double")
        if bin_values_arr.ndim == 2:
            shared_bins = True
            n_ir, n_band = bin_values_arr.shape
        elif bin_values_arr.ndim == 3:
            shared_bins = False
            _, n_ir, n_band = bin_values_arr.shape
        else:
            raise ValueError(
                f"bin_values must be 2D or 3D, got shape {bin_values_arr.shape}"
            )

        if mode_property is None:
            n_batch = 1 if shared_bins else bin_values_arr.shape[0]
            n_elem = 1
            mode_property_arr = np.ones((n_batch, n_ir, n_band, n_elem), dtype="double")
        else:
            mode_property_arr = np.asarray(mode_property, dtype="double")
            if mode_property_arr.ndim != 4:
                raise ValueError(
                    "mode_property must be 4D (n_batch, n_ir, n_band, n_elem)"
                    f", got shape {mode_property_arr.shape}"
                )
            n_batch = mode_property_arr.shape[0]
            n_elem = mode_property_arr.shape[3]
            if not shared_bins and bin_values_arr.shape[0] != n_batch:
                raise ValueError(
                    f"bin_values batch ({bin_values_arr.shape[0]}) does not "
                    f"match mode_property batch ({n_batch})"
                )

        self._sampling_points = self._build_sampling_points(
            bin_values_arr, sampling_points, num_sampling_points, shared_bins
        )

        if ir_grid_points is None:
            _ir_grid_points = np.arange(n_ir, dtype="int64")
        else:
            _ir_grid_points = np.asarray(ir_grid_points, dtype="int64")
        if ir_grid_map is None:
            _ir_grid_map = np.arange(n_ir, dtype="int64")
        else:
            _ir_grid_map = np.asarray(ir_grid_map, dtype="int64")
        if ir_grid_weights is None:
            _ir_grid_weights = np.ones(n_ir, dtype="int64")
        else:
            _ir_grid_weights = np.asarray(ir_grid_weights, dtype="int64")

        grid_points = bz_grid.grg2bzg[_ir_grid_points]
        bzgp2irgp_map = self._build_bzgp2irgp_map(
            bz_grid.bzg2grg, _ir_grid_map, _ir_grid_points
        )

        if shared_bins:
            self._cumulative, self._density = self._accumulate_shared(
                bin_values_arr,
                mode_property_arr,
                _ir_grid_weights,
                bz_grid,
                grid_points,
                bzgp2irgp_map,
                lang,
            )
        else:
            self._cumulative, self._density = self._accumulate_per_batch(
                bin_values_arr,
                mode_property_arr,
                _ir_grid_weights,
                bz_grid,
                grid_points,
                bzgp2irgp_map,
                lang,
            )
        norm = float(np.prod(bz_grid.D_diag))
        self._cumulative /= norm
        self._density /= norm

    @property
    def result(self) -> TetrahedronDOSResult:
        """Return the accumulated spectrum."""
        return TetrahedronDOSResult(
            sampling_points=self._sampling_points,
            cumulative=self._cumulative,
            density=self._density,
        )

    def _build_sampling_points(
        self,
        bin_values_arr: NDArray[np.double],
        sampling_points: NDArray[np.double] | Sequence[float] | None,
        num_sampling_points: int,
        shared_bins: bool,
    ) -> NDArray[np.double]:
        if sampling_points is not None:
            return np.array(sampling_points, dtype="double")
        if shared_bins:
            bin_min = float(bin_values_arr.min())
            bin_max = float(bin_values_arr.max()) + _BIN_EPS
            return np.linspace(bin_min, bin_max, num_sampling_points, dtype="double")
        n_batch = bin_values_arr.shape[0]
        out = np.empty((n_batch, num_sampling_points), dtype="double")
        for b in range(n_batch):
            bin_min = float(bin_values_arr[b].min())
            bin_max = float(bin_values_arr[b].max()) + _BIN_EPS
            out[b] = np.linspace(bin_min, bin_max, num_sampling_points)
        return out

    def _accumulate_shared(
        self,
        bin_values_2d: NDArray[np.double],
        mode_property_arr: NDArray[np.double],
        ir_grid_weights: NDArray[np.int64],
        bz_grid: BZGrid,
        grid_points: NDArray[np.int64],
        bzgp2irgp_map: NDArray[np.int64],
        lang: Literal["C", "Rust"],
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        n_batch, _, _, n_elem = mode_property_arr.shape
        n_sampling = self._sampling_points.shape[0]
        cumulative = np.zeros((n_batch, n_sampling, n_elem), dtype="double")
        density = np.zeros((n_batch, n_sampling, n_elem), dtype="double")
        targets: tuple[tuple[Literal["J", "I"], NDArray[np.double]], ...] = (
            ("J", cumulative),
            ("I", density),
        )
        for func, target in targets:
            iweights = get_integration_weights(
                self._sampling_points,
                bin_values_2d,
                bz_grid,
                grid_points=grid_points,
                bzgp2irgp_map=bzgp2irgp_map,
                function=func,
                lang=lang,
            )
            for i, iw in enumerate(iweights):
                target += np.transpose(
                    np.dot(iw, mode_property_arr[:, i] * ir_grid_weights[i]),
                    axes=(1, 0, 2),
                )
        return cumulative, density

    def _accumulate_per_batch(
        self,
        bin_values_3d: NDArray[np.double],
        mode_property_arr: NDArray[np.double],
        ir_grid_weights: NDArray[np.int64],
        bz_grid: BZGrid,
        grid_points: NDArray[np.int64],
        bzgp2irgp_map: NDArray[np.int64],
        lang: Literal["C", "Rust"],
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        n_batch, _, _, n_elem = mode_property_arr.shape
        # sampling_points may be 1D (explicit user override) or 2D (per-batch
        # auto-generated); use the last axis length.
        n_sampling = self._sampling_points.shape[-1]
        per_batch_sampling = self._sampling_points.ndim == 2
        cumulative = np.zeros((n_batch, n_sampling, n_elem), dtype="double")
        density = np.zeros((n_batch, n_sampling, n_elem), dtype="double")
        for b in range(n_batch):
            if per_batch_sampling:
                sp = self._sampling_points[b]
            else:
                sp = self._sampling_points
            targets: tuple[tuple[Literal["J", "I"], NDArray[np.double]], ...] = (
                ("J", cumulative[b]),
                ("I", density[b]),
            )
            for func, target in targets:
                iweights = get_integration_weights(
                    sp,
                    bin_values_3d[b],
                    bz_grid,
                    grid_points=grid_points,
                    bzgp2irgp_map=bzgp2irgp_map,
                    function=func,
                    lang=lang,
                )
                for i, iw in enumerate(iweights):
                    target += np.dot(iw, mode_property_arr[b, i] * ir_grid_weights[i])
        return cumulative, density

    @staticmethod
    def _build_bzgp2irgp_map(
        bzg2grg: NDArray[np.int64],
        ir_grid_map: NDArray[np.int64],
        ir_grid_points: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        unique_gps = np.unique(ir_grid_map)
        assert np.array_equal(unique_gps, ir_grid_points)
        gp_map = {j: i for i, j in enumerate(unique_gps)}
        return np.array([gp_map[ir_grid_map[grgp]] for grgp in bzg2grg], dtype="int64")


@dataclass(frozen=True)
class SmearingDOSResult:
    """Output of SmearingDOSAccumulator.

    Attributes
    ----------
    sampling_points : NDArray[np.double]
        Frequency points.  shape=(n_sampling,)
    density : NDArray[np.double]
        Smeared differential spectrum.  shape=(n_batch, n_sampling)

    """

    sampling_points: NDArray[np.double]
    density: NDArray[np.double]


class SmearingDOSAccumulator:
    """Spectrum accumulator using Gaussian smearing.

    Faster than the tetrahedron method but with smearing-width-dependent
    broadening.  Returns only the differential spectrum (no cumulative).

    """

    def __init__(
        self,
        frequencies: NDArray[np.double],
        ir_grid_weights: NDArray[np.int64],
        mode_property: NDArray[np.double] | None = None,
        sigma: float | None = None,
        sampling_points: NDArray[np.double] | Sequence[float] | None = None,
        num_sampling_points: int = 200,
    ) -> None:
        """Init method.

        Parameters
        ----------
        frequencies : NDArray[np.double]
            Phonon frequencies on irreducible grid points.
            shape=(n_ir, n_band)
        ir_grid_weights : NDArray[np.int64]
            Multiplicities of irreducible grid points.  shape=(n_ir,)
        mode_property : NDArray[np.double] | None, optional
            Per-mode property to accumulate.  shape=(n_batch, n_ir, n_band).
            ``None`` (default) gives the plain DOS with n_batch=1.
        sigma : float | None, optional
            Gaussian smearing width.  ``None`` defaults to 1/100 of the
            sampling-points span.
        sampling_points : array_like, optional
            Explicit sampling-point grid.  If ``None``, uniform points spanning
            the frequency range are generated.
        num_sampling_points : int, optional
            Number of uniform sampling points (used only when
            ``sampling_points`` is None).

        """
        self._frequencies = np.asarray(frequencies, dtype="double")
        self._ir_grid_weights = np.asarray(ir_grid_weights, dtype="int64")
        if mode_property is None:
            self._mode_property = np.ones(
                (1,) + self._frequencies.shape, dtype="double"
            )
        else:
            self._mode_property = np.asarray(mode_property, dtype="double")
            if self._mode_property.ndim != 3:
                raise ValueError(
                    "mode_property must be 3D (n_batch, n_ir, n_band)"
                    f", got shape {self._mode_property.shape}"
                )

        if sampling_points is not None:
            self._sampling_points = np.array(sampling_points, dtype="double")
        else:
            f_min = float(self._frequencies.min())
            f_max = float(self._frequencies.max()) + _BIN_EPS
            self._sampling_points = np.linspace(
                f_min, f_max, num_sampling_points, dtype="double"
            )

        if sigma is None:
            self._sigma = (
                float(self._sampling_points.max() - self._sampling_points.min()) / 100
            )
        else:
            self._sigma = sigma
        # Local import to avoid a circular dependency with phonopy.phonon.dos
        # (dos.py now imports TetrahedronDOSAccumulator from this module).
        from phonopy.phonon.dos import NormalDistribution

        self._smearing = NormalDistribution(self._sigma)
        self._density = self._compute_density()

    @property
    def result(self) -> SmearingDOSResult:
        """Return the smeared spectrum."""
        return SmearingDOSResult(
            sampling_points=self._sampling_points,
            density=self._density,
        )

    def _compute_density(self) -> NDArray[np.double]:
        n_batch = self._mode_property.shape[0]
        density = np.zeros((n_batch, self._sampling_points.shape[0]), dtype="double")
        num_gp = float(np.sum(self._ir_grid_weights))
        for i, f in enumerate(self._sampling_points):
            kernel = self._smearing.calc(self._frequencies - f)
            for b in range(n_batch):
                w = np.dot(self._ir_grid_weights, kernel * self._mode_property[b])
                density[b, i] = np.sum(w) / num_gp
        return density
