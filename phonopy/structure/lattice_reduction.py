# SPDX-License-Identifier: BSD-3-Clause
"""Niggli and Delaunay reductions of basis vectors.

This module is the reference implementation of the lattice reductions in
phonors.  The Rust implementation follows the same steps, with the same names
and the same order of the floating-point operations, so that both give equal
results.

The Niggli reduction follows I. Krivy and B. Gruber, Acta Cryst. A32, 297
(1976) with the tolerance of R. W. Grosse-Kunstleve, N. K. Sauter and P. D.
Adams, Acta Cryst. A60, 1 (2004).  The notation follows
https://atztogo.github.io/niggli/.  The Delaunay reduction follows
International Tables for Crystallography, Volume A.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

# Number of times the Niggli reduction is started again with half the
# tolerance after a cycle of its steps is found.
NIGGLI_NUM_RETRIES = 10

# Limit of the passes over the steps A1-A8 in one Niggli reduction.  A cycle
# is detected before this limit is reached; the limit only guards against a
# sequence of steps that never repeats a basis.
NIGGLI_MAX_NUM_PASSES = 100000

# Limit of the steps of the Delaunay reduction.  Each step decreases the sum
# of the squared lengths of the four vectors of the superbase by more than
# 2 * eps, so the reduction terminates; the limit only guards against
# unexpected input such as non-finite numbers.
DELAUNAY_MAX_NUM_STEPS = 1000000

# Tolerance of the comparison of squared lengths in the sort of the candidate
# vectors of the Delaunay reduction.
DELAUNAY_SORT_TOLERANCE = 1e-10

FloatMatrix = list[list[float]]
IntMatrix = list[list[int]]


def niggli_reduce(
    lattice: Sequence[Sequence[float]] | NDArray[np.double], eps: float = 1e-5
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return Niggli-reduced basis vectors and the transformation matrix.

    With some input, the steps of the reduction come back to a basis seen
    before and cycle, because the comparisons with eps disagree with the sums
    that the steps form.  When a cycle is found, the reduction starts again
    from the input with eps / 2, up to NIGGLI_NUM_RETRIES times.  The
    returned basis then satisfies the conditions of the reduction with the
    smaller tolerance.

    Parameters
    ----------
    lattice : array_like
        Basis vectors a, b, c in rows.
        shape=(3, 3), dtype='double'
    eps : float, optional
        Tolerance with which the parameters of the metric tensor,
        A = a.a, B = b.b, C = c.c, xi = 2 b.c, eta = 2 c.a, zeta = 2 a.b,
        and their sums are compared.  It has the unit of a squared length and
        is not scaled by the size of the lattice.  Default is 1e-5.

    Returns
    -------
    reduced_lattice : ndarray
        Reduced basis vectors a', b', c' in rows.
        shape=(3, 3), dtype='double', order='C'
    tmat : ndarray
        Integer matrix with determinant 1 or -1 that gives
        (a', b', c') = (a, b, c) tmat, i.e., reduced_lattice equals
        tmat.T @ lattice within rounding.
        shape=(3, 3), dtype='int64', order='C'

    Raises
    ------
    RuntimeError
        When the steps cycle also with eps / 2**NIGGLI_NUM_RETRIES.

    """
    columns = _get_columns(lattice)
    for retry in range(NIGGLI_NUM_RETRIES + 1):
        result = _run_niggli_reduce(columns, eps / 2**retry)
        if result is not None:
            reduced_columns, tmat = result
            return _get_rows(reduced_columns), np.array(tmat, dtype="int64")
    raise RuntimeError("Niggli reduction failed.")


def delaunay_reduce(
    lattice: Sequence[Sequence[float]] | NDArray[np.double], eps: float = 1e-5
) -> tuple[NDArray[np.double], NDArray[np.int64]]:
    """Return Delaunay-reduced basis vectors and the transformation matrix.

    The basis vectors a, b, c are extended to the superbase
    b_1 = a, b_2 = b, b_3 = c, b_4 = -(a + b + c).  While a pair i < j has
    b_i . b_j > eps, b_i is added to the other two vectors and reversed.
    The reduced basis is chosen among b_1, b_2, b_1 + b_2, b_3, b_4,
    b_2 + b_3, b_3 + b_1.

    Parameters
    ----------
    lattice : array_like
        Basis vectors a, b, c in rows.
        shape=(3, 3), dtype='double'
    eps : float, optional
        Tolerance of the scalar products b_i . b_j and of the volumes.  It is
        not scaled by the size of the lattice.  Default is 1e-5.

    Returns
    -------
    reduced_lattice : ndarray
        Reduced basis vectors a', b', c' in rows, with a positive volume.
        shape=(3, 3), dtype='double', order='C'
    tmat : ndarray
        Integer matrix with determinant 1 or -1 that gives
        (a', b', c') = (a, b, c) tmat, i.e., reduced_lattice equals
        tmat.T @ lattice within rounding.
        shape=(3, 3), dtype='int64', order='C'

    Raises
    ------
    RuntimeError
        When the volume of the reduced basis is below eps, when the chosen
        three vectors do not form a basis of the lattice, or when the number
        of steps exceeds DELAUNAY_MAX_NUM_STEPS.

    """
    columns = _get_columns(lattice)
    reduced_columns, tmat = _run_delaunay_reduce(columns, eps)
    return _get_rows(reduced_columns), np.array(tmat, dtype="int64")


# ---------------------------------------------------------------------------
# Niggli reduction
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _NiggliState:
    """State of the Niggli reduction.

    Attributes
    ----------
    lattice : list[list[float]]
        Current basis vectors in columns, lattice[i][j] is the Cartesian
        component i of the basis vector j.
    tmat : list[list[int]]
        Product of the transformation matrices of the steps, from the input
        basis.
    eps : float
        Tolerance.
    A, B, C : float
        a.a, b.b and c.c.
    xi, eta, zeta : float
        2 b.c, 2 c.a and 2 a.b.
    l, m, n : int
        -1, 0 or 1 for xi, eta and zeta below -eps, within +-eps, or above
        eps.

    """

    lattice: FloatMatrix
    tmat: IntMatrix
    eps: float
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    xi: float = 0.0
    eta: float = 0.0
    zeta: float = 0.0
    l: int = 0  # noqa: E741
    m: int = 0
    n: int = 0


def _run_niggli_reduce(
    columns: FloatMatrix, eps: float
) -> tuple[FloatMatrix, IntMatrix] | None:
    """Run the Niggli reduction; None when its steps cycle.

    A pass runs the steps A1-A8 in order and ends early when A2, A5, A6, A7
    or A8 changes the basis.  The reduction succeeds when a pass reaches its
    end.  The basis at the start of a pass determines all later steps, so a
    basis that is seen again at the start of a pass means a cycle.

    """
    state = _NiggliState(
        lattice=columns,
        tmat=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        eps=eps,
    )
    _set_niggli_parameters(state)
    seen_bases: set[tuple[float, ...]] = set()
    for _ in range(NIGGLI_MAX_NUM_PASSES):
        basis_key = tuple(x for row in state.lattice for x in row)
        if basis_key in seen_bases:
            return None
        seen_bases.add(basis_key)
        if _run_niggli_pass(state):
            return state.lattice, state.tmat
    return None


def _run_niggli_pass(state: _NiggliState) -> bool:
    """Run the steps A1-A8 once; True when the pass reaches its end."""
    for step, restarts in _NIGGLI_STEPS:
        tmat = step(state)
        if tmat is None:
            continue
        _transform_niggli_basis(state, tmat)
        if restarts:
            return False
    return True


def _transform_niggli_basis(state: _NiggliState, tmat: IntMatrix) -> None:
    """Replace the basis by (a, b, c) tmat and update the parameters."""
    state.lattice = _multiply_float_int(state.lattice, tmat)
    state.tmat = _multiply_int_int(state.tmat, tmat)
    _set_niggli_parameters(state)


def _set_niggli_parameters(s: _NiggliState) -> None:
    """Set the parameters of the metric tensor and their signs."""
    G = _get_metric(s.lattice)
    s.A, s.B, s.C = G[0][0], G[1][1], G[2][2]
    s.xi, s.eta, s.zeta = G[1][2] * 2, G[0][2] * 2, G[0][1] * 2
    s.l = _get_sign(s.xi, s.eps)
    s.m = _get_sign(s.eta, s.eps)
    s.n = _get_sign(s.zeta, s.eps)


def _get_sign(value: float, eps: float) -> int:
    """Return -1, 1 or 0 for value below -eps, above eps, or otherwise."""
    if value < -eps:
        return -1
    if value > eps:
        return 1
    return 0


def _niggli_step_a1(s: _NiggliState) -> IntMatrix | None:
    """A1: A > B, or A = B and |xi| > |eta|; swap a and b."""
    A, B, xi, eta, e = s.A, s.B, s.xi, s.eta, s.eps
    if A > B + e or (not (abs(A - B) > e) and abs(xi) > abs(eta) + e):
        return [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    return None


def _niggli_step_a2(s: _NiggliState) -> IntMatrix | None:
    """A2: B > C, or B = C and |eta| > |zeta|; swap b and c."""
    B, C, eta, zeta, e = s.B, s.C, s.eta, s.zeta, s.eps
    if B > C + e or (not (abs(B - C) > e) and abs(eta) > abs(zeta) + e):
        return [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    return None


def _niggli_step_a3(s: _NiggliState) -> IntMatrix | None:
    """A3: lmn = 1; make xi, eta, zeta all positive."""
    l, m, n = s.l, s.m, s.n  # noqa: E741
    if l * m * n == 1:
        i = -1 if l == -1 else 1
        j = -1 if m == -1 else 1
        k = -1 if n == -1 else 1
        return [[i, 0, 0], [0, j, 0], [0, 0, k]]
    return None


def _niggli_step_a4(s: _NiggliState) -> IntMatrix | None:
    """A4: lmn = 0 or -1 except l = m = n = -1; make xi, eta, zeta non-positive.

    The signs i, j, k of the basis vectors have the product 1.  When the
    product of those chosen from l, m, n is -1, the sign that belongs to a
    zero among l, m, n (the last one) is reversed as well.

    """
    l, m, n = s.l, s.m, s.n  # noqa: E741
    if l == -1 and m == -1 and n == -1:
        return None
    if l * m * n == 0 or l * m * n == -1:
        ijk = [1, 1, 1]
        z = -1
        for p, sign in enumerate((l, m, n)):
            if sign == 1:
                ijk[p] = -1
            if sign == 0:
                z = p
        if ijk[0] * ijk[1] * ijk[2] == -1 and z >= 0:
            ijk[z] = -1
        i, j, k = ijk
        return [[i, 0, 0], [0, j, 0], [0, 0, k]]
    return None


def _niggli_step_a5(s: _NiggliState) -> IntMatrix | None:
    """A5: |xi| > B, or the boundary cases; add -sign(xi) b to c."""
    B, xi, eta, zeta, e = s.B, s.xi, s.eta, s.zeta, s.eps
    if (
        abs(xi) > B + e
        or (not (abs(B - xi) > e) and 2 * eta < zeta - e)
        or (not (abs(B + xi) > e) and zeta < -e)
    ):
        return [[1, 0, 0], [0, 1, -_get_exact_sign(xi)], [0, 0, 1]]
    return None


def _niggli_step_a6(s: _NiggliState) -> IntMatrix | None:
    """A6: |eta| > A, or the boundary cases; add -sign(eta) a to c."""
    A, xi, eta, zeta, e = s.A, s.xi, s.eta, s.zeta, s.eps
    if (
        abs(eta) > A + e
        or (not (abs(A - eta) > e) and 2 * xi < zeta - e)
        or (not (abs(A + eta) > e) and zeta < -e)
    ):
        return [[1, 0, -_get_exact_sign(eta)], [0, 1, 0], [0, 0, 1]]
    return None


def _niggli_step_a7(s: _NiggliState) -> IntMatrix | None:
    """A7: |zeta| > A, or the boundary cases; add -sign(zeta) a to b."""
    A, xi, eta, zeta, e = s.A, s.xi, s.eta, s.zeta, s.eps
    if (
        abs(zeta) > A + e
        or (not (abs(A - zeta) > e) and 2 * xi < eta - e)
        or (not (abs(A + zeta) > e) and eta < -e)
    ):
        return [[1, -_get_exact_sign(zeta), 0], [0, 1, 0], [0, 0, 1]]
    return None


def _niggli_step_a8(s: _NiggliState) -> IntMatrix | None:
    """A8: xi + eta + zeta + A + B < 0, or the boundary case; add a + b to c."""
    A, B, xi, eta, zeta, e = s.A, s.B, s.xi, s.eta, s.zeta, s.eps
    t = xi + eta + zeta + A + B
    if t < -e or (not (abs(t) > e) and 2 * (A + eta) + zeta > e):
        return [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
    return None


def _get_exact_sign(value: float) -> int:
    """Return the sign of value without tolerance, 0 for zero."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# Steps A1-A8 and whether a change of the basis by the step ends the pass.
_NIGGLI_STEPS: tuple[tuple[Callable[[_NiggliState], IntMatrix | None], bool], ...] = (
    (_niggli_step_a1, False),
    (_niggli_step_a2, True),
    (_niggli_step_a3, False),
    (_niggli_step_a4, False),
    (_niggli_step_a5, True),
    (_niggli_step_a6, True),
    (_niggli_step_a7, True),
    (_niggli_step_a8, True),
)


# ---------------------------------------------------------------------------
# Delaunay reduction
# ---------------------------------------------------------------------------


def _run_delaunay_reduce(
    columns: FloatMatrix, eps: float
) -> tuple[FloatMatrix, IntMatrix]:
    """Run the Delaunay reduction of the basis vectors in columns.

    vecs[n] are the four vectors of the superbase in Cartesian coordinates,
    and coefs[n] are their integer coordinates in the input basis.

    """
    vecs = [[columns[i][j] for i in range(3)] for j in range(3)]
    vecs.append([-columns[i][0] - columns[i][1] - columns[i][2] for i in range(3)])
    coefs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]

    for _ in range(DELAUNAY_MAX_NUM_STEPS):
        if _reduce_superbase_once(vecs, coefs, eps):
            break
    else:
        raise RuntimeError("Delaunay reduction did not terminate.")

    basis, basis_coefs = _choose_delaunay_basis(vecs, coefs, eps)

    reduced_columns = [[basis[j][i] for j in range(3)] for i in range(3)]
    tmat = [[basis_coefs[j][i] for j in range(3)] for i in range(3)]
    volume = _get_determinant(reduced_columns)
    if abs(volume) < eps:
        raise RuntimeError("Delaunay-reduced basis has no volume.")
    if volume < 0:
        reduced_columns = [[-x for x in row] for row in reduced_columns]
        tmat = [[-x for x in row] for row in tmat]
    if abs(_get_determinant_int(tmat)) != 1:
        raise RuntimeError("Delaunay-reduced vectors do not form a basis.")
    return reduced_columns, tmat


def _reduce_superbase_once(vecs: FloatMatrix, coefs: IntMatrix, eps: float) -> bool:
    """Apply one step of the reduction; True when no step is needed.

    For the first pair i < j with b_i . b_j > eps, b_i is added to the two
    other vectors and b_i is reversed.  The four vectors still sum to zero.

    """
    for i in range(3):
        for j in range(i + 1, 4):
            dot_product = 0.0
            for k in range(3):
                dot_product += vecs[i][k] * vecs[j][k]
            if dot_product > eps:
                for other in range(4):
                    if not (other == i or other == j):
                        for k in range(3):
                            vecs[other][k] += vecs[i][k]
                            coefs[other][k] += coefs[i][k]
                for k in range(3):
                    vecs[i][k] = -vecs[i][k]
                    coefs[i][k] = -coefs[i][k]
                return False
    return True


def _choose_delaunay_basis(
    vecs: FloatMatrix, coefs: IntMatrix, eps: float
) -> tuple[FloatMatrix, IntMatrix]:
    """Choose three basis vectors among seven candidates.

    The candidates b_1, b_2, b_1 + b_2, b_3, b_4, b_2 + b_3, b_3 + b_1 are
    sorted by squared length with a bubble sort that swaps two neighbors only
    when the first is longer by more than DELAUNAY_SORT_TOLERANCE.  The first
    two sorted candidates are taken, and the third is the first of the others
    with which the volume exceeds eps.  When none does, b_1, b_2, b_3 are
    kept.

    """
    cands = [
        vecs[0],
        vecs[1],
        [vecs[0][k] + vecs[1][k] for k in range(3)],
        vecs[2],
        vecs[3],
        [vecs[1][k] + vecs[2][k] for k in range(3)],
        [vecs[2][k] + vecs[0][k] for k in range(3)],
    ]
    cand_coefs = [
        coefs[0],
        coefs[1],
        [coefs[0][k] + coefs[1][k] for k in range(3)],
        coefs[2],
        coefs[3],
        [coefs[1][k] + coefs[2][k] for k in range(3)],
        [coefs[2][k] + coefs[0][k] for k in range(3)],
    ]
    for _ in range(6):
        for j in range(6):
            if (
                _get_norm_squared(cands[j])
                > _get_norm_squared(cands[j + 1]) + DELAUNAY_SORT_TOLERANCE
            ):
                cands[j], cands[j + 1] = cands[j + 1], cands[j]
                cand_coefs[j], cand_coefs[j + 1] = cand_coefs[j + 1], cand_coefs[j]

    for n in range(2, 7):
        three_columns = [[cands[0][k], cands[1][k], cands[n][k]] for k in range(3)]
        if abs(_get_determinant(three_columns)) > eps:
            return (
                [cands[0], cands[1], cands[n]],
                [cand_coefs[0], cand_coefs[1], cand_coefs[n]],
            )
    return vecs[:3], coefs[:3]


# ---------------------------------------------------------------------------
# Small matrix and vector operations
# ---------------------------------------------------------------------------


def _get_columns(
    lattice: Sequence[Sequence[float]] | NDArray[np.double],
) -> FloatMatrix:
    """Return the basis vectors given in rows as a matrix with them in columns."""
    rows = np.array(lattice, dtype="double")
    if rows.shape != (3, 3):
        raise ValueError("lattice has to be a 3x3 matrix.")
    return [[float(rows[j][i]) for j in range(3)] for i in range(3)]


def _get_rows(columns: FloatMatrix) -> NDArray[np.double]:
    """Return the basis vectors given in columns as rows of an ndarray."""
    return np.array(
        [[columns[j][i] for j in range(3)] for i in range(3)], dtype="double", order="C"
    )


def _multiply_float_int(left: FloatMatrix, right: IntMatrix) -> FloatMatrix:
    """Return left @ right, summed in the order of k from 0."""
    prod = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            value = 0.0
            for k in range(3):
                value += left[i][k] * float(right[k][j])
            prod[i][j] = value
    return prod


def _multiply_int_int(left: IntMatrix, right: IntMatrix) -> IntMatrix:
    """Return left @ right of integer matrices."""
    return [
        [sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _get_metric(columns: FloatMatrix) -> FloatMatrix:
    """Return the metric tensor of the basis vectors in columns."""
    metric = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            value = 0.0
            for k in range(3):
                value += columns[k][i] * columns[k][j]
            metric[i][j] = value
    return metric


def _get_norm_squared(vec: list[float]) -> float:
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]


def _get_determinant(m: FloatMatrix) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _get_determinant_int(m: IntMatrix) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        + m[0][1] * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
