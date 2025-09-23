import numpy as np
from math import ceil
import cupy

import phonopy
from phonopy.acc.numba_imports import copysign, sqrt
from phonopy.acc.numba_imports import cuda
from phonopy.acc.harmonic.dynamical_matrix import DynMatWorkspace

@cuda.jit
def _freq(frequencies, eigenvalues, factor):
    """Kernel function to compute frequencies from eigenvalues."""
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= eigenvalues.shape[0]:
        return
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if j >= eigenvalues.shape[1]:
        return

    frequencies[i, j] = copysign(
        sqrt(abs(eigenvalues[i, j])), eigenvalues[i, j]) * factor
    return

def frequencies(eigenvalues, factor):
    """Compute frequencies from eigenvalues."""
    tpb_y = min(128, eigenvalues.shape[1])
    tpb_x = max(1, 128 // tpb_y)
    threadsperblock = (tpb_x, tpb_y)
    freq_path = cuda.device_array(eigenvalues.shape, dtype=np.float64)
    blockspergrid_x = ceil(eigenvalues.shape[0] / threadsperblock[0])
    blockspergrid_y = ceil(eigenvalues.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    _freq[blockspergrid, threadsperblock](freq_path, eigenvalues, factor)
    return freq_path

def _run_qpoints_phonon(qpoints, dynamical_matrix, nac_q_direction=None,
        with_eigenvectors=False, factor=None, dm_ws=None):
    """Accelerated version of QpointsPhonon.run()."""
    if factor is None:
        factor = phonopy.physical_units.get_physical_units().DefaultToTHz

    if dm_ws is None:
        dm_ws = DynMatWorkspace(dynamical_matrix,
                with_eigenvectors=with_eigenvectors)
    else:
        dm_ws.with_eigenvectors = False
    qp_d, eigvals, eigvecs, dynmat = dm_ws.solve_dm_on_qpoints(qpoints,
        nac_q_direction=nac_q_direction)
    freqs = frequencies(cupy.asarray(eigvals), factor)
    
    return freqs, eigvals, eigvecs, dynmat

def run_qpoints_phonon(qpp):
    """Accelerated version of QpointsPhonon.run()."""

    freqs, eigvals, eigvecs, dynmat = _run_qpoints_phonon(
        qpp._qpoints,
        qpp._dynamical_matrix,
        nac_q_direction=qpp._nac_q_direction,
        with_eigenvectors=qpp._with_eigenvectors,
        factor=qpp._factor,
    )

    ret_freqs = cupy.asnumpy(freqs)
    ret_eigvals = cupy.asnumpy(eigvals)

    if qpp._with_eigenvectors:
        ret_eigvecs = cupy.asnumpy(eigvecs)
    else:
        ret_eigvecs = None

    if qpp._with_dynamical_matrices:
        ret_dynmat = cupy.asnumpy(dynmat)
    else:
        ret_dynmat = None

    return ret_freqs, ret_eigvals, ret_eigvecs, ret_dynmat
