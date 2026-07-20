# SPDX-License-Identifier: BSD-3-Clause
"""Equation of states and fitting routine."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

EosFunc = Callable[[NDArray[np.double], NDArray[np.double]], NDArray[np.double]]


def get_eos(eos: str) -> EosFunc:
    """Return equation of states."""

    def birch_murnaghan(
        v: NDArray[np.double], p: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return Third-order Birch-Murnaghan EOS.

        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0

        """
        return p[0] + 9.0 / 16 * p[3] * p[1] * (
            ((p[3] / v) ** (2.0 / 3) - 1) ** 3 * p[2]
            + ((p[3] / v) ** (2.0 / 3) - 1) ** 2 * (6 - 4 * (p[3] / v) ** (2.0 / 3))
        )

    def murnaghan(v: NDArray[np.double], p: NDArray[np.double]) -> NDArray[np.double]:
        """Return Murnaghan EOS.

        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0

        """
        return (
            p[0]
            + p[1] * v / p[2] * ((p[3] / v) ** p[2] / (p[2] - 1) + 1)
            - p[1] * p[3] / (p[2] - 1)
        )

    def vinet(v: NDArray[np.double], p: NDArray[np.double]) -> NDArray[np.double]:
        """Return Vinet EOS.

        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0

        """
        x = np.cbrt(v / p[3])
        xi = 3.0 / 2 * (p[2] - 1)
        return p[0] + (
            9 * p[1] * p[3] / (xi**2) * (1 + (xi * (1 - x) - 1) * np.exp(xi * (1 - x)))
        )

    if eos == "murnaghan":
        return murnaghan
    elif eos == "birch_murnaghan":
        return birch_murnaghan
    else:
        return vinet


def fit_to_eos(
    volumes: NDArray[np.double],
    fe: NDArray[np.double],
    eos: EosFunc,
) -> NDArray[np.double]:
    """Fit volume-energy data to EOS."""
    fit = EOSFit(volumes, fe, eos)
    fit.fit([fe[len(fe) // 2], 1.0, 4.0, volumes[len(volumes) // 2]])
    assert fit.parameters is not None
    return fit.parameters


class EOSFit:
    """Class to fit volume-energy data to EOS.

    Attributes
    ----------
    parameters: ndarray
        Fitting parameters to EOS corresponding to [energy, B, B', V].
        dtype=float
        shape=(4,)

    """

    def __init__(
        self,
        volume: Sequence[float] | NDArray[np.double],
        energy: Sequence[float] | NDArray[np.double],
        eos: EosFunc,
    ) -> None:
        """Init method."""
        self._energy = np.array(energy)
        self._volume = np.array(volume)
        self._eos = eos

        self.parameters = None

    def fit(self, initial_parameters: Sequence[float] | NDArray[np.double]) -> None:
        """Fit."""
        try:
            import scipy
            from scipy.optimize import leastsq
        except ImportError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            def residuals(
                p: NDArray[np.double],
                eos: EosFunc,
                v: NDArray[np.double],
                e: NDArray[np.double],
            ) -> NDArray[np.double]:
                """Return residuals."""
                return eos(v, p) - e

            try:
                result = leastsq(
                    residuals,
                    initial_parameters,
                    args=(self._eos, self._volume, self._energy),
                    full_output=True,
                )
            except RuntimeError as exc:
                raise RuntimeError("Fitting to EOS failed.") from exc
            except (RuntimeWarning, scipy.optimize.OptimizeWarning) as exc:
                raise RuntimeError("Met difficulty in fitting to EOS.") from exc
            else:
                self.parameters = result[0]
