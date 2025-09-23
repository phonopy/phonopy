import os

use_rocm = os.getenv("ROCM_PATH") is not None

import numba

if use_rocm:
    from numba import hip

    hip.pose_as_cuda()

from numba import cuda

if use_rocm:
    from math import copysign, cos, exp, pi, sin, sqrt

    @cuda.jit(device=True)
    def rsqrt(x):
        """Reciprocal square root."""
        return 1 / sqrt(x)

    @cuda.jit(device=True)
    def sincospi(x):
        """(sin(pi*x), cos(pi*x))."""
        return (sin(pi * x), cos(pi * x))

else:
    from numba.cuda.libdevice import copysign, cos, exp, rsqrt, sin, sincospi, sqrt

def use_acc():
    """Check if device is available."""
    return numba.cuda.is_available()
