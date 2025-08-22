import math, numpy as np, numba
from numba import cuda

@cuda.jit(device=True, inline=True)
def transpose(A, A_T):
    """Store transpose of A in A_T."""
    for i in range(3):
        for j in range(3):
            A_T[i,j] = A[j,i]
    return

@cuda.jit(device=True, inline=True)
def outer(a, b, M):
    """Store outer product of a and b in M."""
    M[0,0] = a[0]*b[0]
    M[0,1] = a[0]*b[1]
    M[0,2] = a[0]*b[2]
    M[1,0] = a[1]*b[0]
    M[1,1] = a[1]*b[1]
    M[1,2] = a[1]*b[2]
    M[2,0] = a[2]*b[0]
    M[2,1] = a[2]*b[1]
    M[2,2] = a[2]*b[2]
    return

@cuda.jit(device=True, inline=True)
def matmul(A, B, C):
    """C = A * B."""
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    return

@cuda.jit(device=True, inline=True)
def matvecmul(A, v, vout):
    """Matrix-vector multiplication, store result in vout."""
    vout[0] = A[0,0]*v[0] + A[0,1]*v[1] + A[0,2]*v[2]
    vout[1] = A[1,0]*v[0] + A[1,1]*v[1] + A[1,2]*v[2]
    vout[2] = A[2,0]*v[0] + A[2,1]*v[1] + A[2,2]*v[2]
    return

@cuda.jit(device=True, inline=True)
def vecmatmul(v, A, vout):
    """Vector-matrix multiplication, store result in vout."""
    vout[0] = v[0]*A[0,0] + v[1]*A[1,0] + v[2]*A[2,0]
    vout[1] = v[0]*A[0,1] + v[1]*A[1,1] + v[2]*A[2,1]
    vout[2] = v[0]*A[0,2] + v[1]*A[1,2] + v[2]*A[2,2]
    return
