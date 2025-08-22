import math, numpy as np, numba
from numba import cuda

@cuda.jit(device=True, inline=True)
def dot(v1, v2):
    """Take dot product of two 3d vectors."""
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

@cuda.jit(device=True, inline=True)
def cross(v1, v2, vout):
    """Take cross product of two 3d vectors (store result in vout)."""
    vout[0] = v1[1]*v2[2]-v1[2]*v2[1]
    vout[1] = v1[2]*v2[0]-v1[0]*v2[2]
    vout[2] = v1[0]*v2[1]-v1[1]*v2[0]
    return

@cuda.jit(device=True, inline=True)
def length(v):
    """Compute magnitude of v."""
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

@cuda.jit(device=True, inline=True)
def length2(v):
    """Compute squared magnitude of v."""
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]

@cuda.jit(device=True, inline=True)
def norm(v):
    """Alias of length(v)."""
    return length(v)

@cuda.jit(device=True, inline=True)
def norm2(v):
    """Alias of length2(v)."""
    return length2(v)

@cuda.jit(device=True, inline=True)
def normalize(v):
    """Scale v to unit length."""
    scale(v, 1.0/norm(v))
    return

@cuda.jit(device=True, inline=True)
def scale(v, s):
    """Multiply by scalar."""
    v[0]*=s
    v[1]*=s
    v[2]*=s
    return

@cuda.jit(device=True, inline=True)
def copy(v1, v2):
    """Store copy of v1 in v2."""
    v2[0]=v1[0]
    v2[1]=v1[1]
    v2[2]=v1[2]
    return

@cuda.jit(device=True, inline=True)
def add(v1, v2, v3):
    """Add v1 + v2, store result in v3."""
    v3[0]=v1[0]+v2[0]
    v3[1]=v1[1]+v2[1]
    v3[2]=v1[2]+v2[2]
    return

@cuda.jit(device=True, inline=True)
def subtract(v1, v2, v3):
    """Subtract v1 - v2, store result in v3."""
    v3[0]=v1[0]-v2[0]
    v3[1]=v1[1]-v2[1]
    v3[2]=v1[2]-v2[2]
    return
