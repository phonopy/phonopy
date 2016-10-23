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

import sys
import numpy as np

def get_eos(eos):

    # Third-order Birch-Murnaghan EOS
    def birch_murnaghan(v, *p):
        """
        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0
        """
        return p[0] + 9.0 / 16 * p[3] * p[1] * (
            ((p[3] / v)**(2.0 / 3) - 1)**3 * p[2] +
            ((p[3] / v)**(2.0 / 3) - 1)**2 * (6 - 4 * (p[3] / v)**(2.0 / 3)))

    # Murnaghan EOS
    def murnaghan(v, *p):
        """
        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0
        """
        return (p[0]
                + p[1] * v / p[2] *((p[3] / v)**p[2] / (p[2] - 1) + 1)
                - p[1] * p[3] / (p[2] - 1))

    # Vinet EOS
    def vinet(v, *p):
        """
        p[0] = E_0
        p[1] = B_0
        p[2] = B'_0
        p[3] = V_0
        """

        x = (v / p[3]) ** (1.0 / 3)
        xi = 3.0 / 2 * (p[2] - 1)
        return p[0] + (9 * p[1] * p[3] / (xi**2)
                       * (1 + (xi * (1 - x) - 1) * np.exp(xi * (1 - x))))

    if eos=='murnaghan':
        return murnaghan
    elif eos=='birch_murnaghan':
        return birch_murnaghan
    else:
        return vinet


def fit_to_eos(volumes, fe, eos):
    fit = EOSFit(volumes, fe, eos)
    fit.run([fe[len(fe) // 2], 1.0, 4.0, volumes[len(volumes) // 2]])
    ev = fit.get_volume()
    ee = fit.get_energy()
    eb = fit.get_bulk_modulus()
    ebp = fit.get_b_prime()
    return ee, eb, ebp, ev

class EOSFit(object):
    def __init__(self, volume, energy, eos):
        self._energy = np.array(energy)
        self._volume = np.array(volume)
        self._eos = eos

    def run(self, initial_parameter):
        import sys
        import logging
        import warnings

        try:
            from scipy.optimize import leastsq
            import scipy
        except ImportError:
            print("You need to install python-scipy.")
            sys.exit(1)

        warnings.filterwarnings('error')

        residuals = lambda p, eos, v, e: eos(v, *p) - e

        try:
            result = leastsq(residuals,
                             initial_parameter,
                             args=(self._eos, self._volume, self._energy),
                             full_output=1)
            #
            # leastsq is more stable than curve_fit.
            # The reason is unclear, maybe the default parameters used for
            # leastsq are different though curve_fit seems to call curve_fit.
            #
            # result = curve_fit(self._eos, self._volume, self._energy,
            #                    p0=initial_parameter)

        except (RuntimeError,
                RuntimeWarning,
                scipy.optimize.optimize.OptimizeWarning):
            logging.exception('')
            self._parameters = None
            return sys.exc_info()
        else:
            self._parameters = result[0]
            return True

    def get_energy(self):
        if self._parameters is None:
            return None
        else:
            return self._parameters[0]

    def get_volume(self):
        if self._parameters is None:
            return None
        else:
            return self._parameters[3]

    def get_bulk_modulus(self):
        if self._parameters is None:
            return None
        else:
            return self._parameters[1]

    def get_b_prime(self):
        if self._parameters is None:
            return None
        else:
            return self._parameters[2]

    def get_parameters(self):
        return self._parameters
