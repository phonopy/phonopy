"""Calculate phonon state moments."""

# Copyright (C) 2016 Atsushi Togo
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


class PhononMoment:
    """Calculate phonon state moments.

    Attributes
    ----------
    moment : float or ndarray
        Phonon state moment of specified order (float) or
        projected phonon state moment of specified order (ndarray).

    """

    def __init__(self, frequencies, weights, eigenvectors=None):
        """Init method."""
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._weights = weights
        self._fmin = None
        self._fmax = None
        self.set_frequency_range()
        self._moment = None

    @property
    def moment(self):
        """Return phonon state moment."""
        return self._moment

    def get_moment(self):
        """Return phonon state moment."""
        warnings.warn(
            "PhononMoment.get_moment() is deprecated. "
            "Use PhononMoment.moment attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.moment

    def set_frequency_range(self, freq_min=None, freq_max=None, tolerance=1e-8):
        """Set frequeny range where moment is computed."""
        if freq_min is None:
            self._fmin = tolerance
        else:
            self._fmin = freq_min - tolerance

        if freq_max is None:
            self._fmax = np.max(self._frequencies) + tolerance
        else:
            self._fmax = freq_max + tolerance

    def run(self, order=1):
        """Calculate phonon state moment of specified order."""
        if self._eigenvectors is None:
            self._get_moment(order)
        else:
            self._get_projected_moment(order)

    def _get_moment(self, order):
        moment = 0
        norm0 = 0
        for i, w in enumerate(self._weights):
            for freq in self._frequencies[i]:
                if self._fmin < freq and freq < self._fmax:
                    norm0 += w
                    moment += freq**order * w
        self._moment = moment / norm0

    def _get_projected_moment(self, order):
        moment = np.zeros(self._frequencies.shape[1], dtype="double")
        norm0 = np.zeros_like(moment)
        for i, w in enumerate(self._weights):
            for freq, eigvec in zip(self._frequencies[i], self._eigenvectors[i].T):
                if self._fmin < freq and freq < self._fmax:
                    projection = np.abs(eigvec) ** 2
                    norm0 += w * projection
                    moment += freq**order * w * projection
        self._moment = (
            np.array(
                [
                    np.sum((moment / norm0)[i * 3 : (i + 1) * 3])
                    for i in range(len(moment) // 3)
                ]
            )
            / 3
        )
