"""Equation of states and fitting routine."""

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

import warnings

import numpy as np


def get_eos(eos):
    """Return equation of states."""

    def birch_murnaghan(v, *p):
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

    def murnaghan(v, *p):
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

    def vinet(v, *p):
        """Return Vinet EOS.

        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0

        """
        x = (v / p[3]) ** (1.0 / 3)
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


def fit_to_eos(volumes, fe, eos):
    """Fit volume-energy data to EOS."""
    fit = EOSFit(volumes, fe, eos)
    fit.fit([fe[len(fe) // 2], 1.0, 4.0, volumes[len(volumes) // 2]])

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

    def __init__(self, volume, energy, eos):
        """Init method."""
        self._energy = np.array(energy)
        self._volume = np.array(volume)
        self._eos = eos

        self.parameters = None

    def fit(self, initial_parameters):
        """Fit."""
        try:
            import scipy
            from scipy.optimize import leastsq
        except ImportError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            def residuals(p, eos, v, e):
                """Return residuals."""
                return eos(v, *p) - e

            try:
                result = leastsq(
                    residuals,
                    initial_parameters,
                    args=(self._eos, self._volume, self._energy),
                    full_output=1,
                )
            except RuntimeError as exc:
                raise RuntimeError("Fitting to EOS failed.") from exc
            except (RuntimeWarning, scipy.optimize.OptimizeWarning) as exc:
                raise RuntimeError("Met difficulty in fitting to EOS.") from exc
            else:
                self.parameters = result[0]
