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

import numpy as np

class PhononMoment:
    def __init__(self,
                 frequencies,
                 eigenvectors,
                 weights):
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._weights = weights
        self._fmin = None
        self._fmax = None
        self.set_frequency_range()
        self._moment = None

    def get_moment(self):
        return self._moment

    def set_frequency_range(self, freq_min=None, freq_max=None):
        if freq_min is not None:
            self._fmin = 0
        else:
            self._fmin = freq_min

        if freq_max is None:
            self._fmax = np.max(self._frequencies) + 1e-5
        else:
            self._fmax = freq_max

    def run(self, order=1):
        moment = 0
        norm0 = 0
        for i, w in enumerate(self._weights):
            for freq, eigvec in zip(self._frequencies[i],
                                    self._eigenvectors[i].T):
                norm0 += w
                moment += freq ** order * w
        self._moment = moment / norm0
