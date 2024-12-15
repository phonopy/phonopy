"""Calculation of Smith normal form of 3x3 matrix."""

# Copyright (C) 2020 Atsushi Togo
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


#
# Smith normal form for 3x3 integer matrix
# This code is maintained at https://github.com/atztogo/snf3x3.
#
class SNF3x3:
    """Smith normal form for 3x3 matrix.

    Input 3x3 interger matrix A is transformed to 3x3 diagonal interger
    matrix (D) by two 3x3 interger matrices P and Q, which is written as

        D = PAQ

    with abs(det(A)) = det(D), det(P) = 1, det(Q) = sgn(det(A)).

    The algorithm is implemented following
    https://en.wikipedia.org/wiki/Smith_normal_form,
    but the diagonal elements don't follow the rule of it.

    Usage
    -----
    snf = SNF3x3(int_mat)
    snf.run()

    Attributes
    ----------
    D, P, Q : ndarray
        These 3x3 interger matrices explained above.
        shape=(3, 3), dtype='long'

    """

    def __init__(self, A):
        """Init method.

        Parameters
        ----------
        A : array_like
            3x3 interger matrix.
            shape=(3, 3)

        """
        self._A_orig = np.array(A, dtype="long", order="C")
        self._A = np.array(A, dtype="long", order="C")
        self._Ps = []
        self._Qs = []
        self._L = []
        self._P = None
        self._Q = None
        self._D = None
        self._attempt = 0

    def run(self):
        """Calculate SNF."""
        for _ in self:
            pass

        A = np.dot(self._P, np.dot(self._A_orig, self._Q))
        assert (A == self._A).all()

    def __iter__(self):
        """Define iterator over calculation steps."""
        return self

    def __next__(self):
        """Calculate an algorithm step."""
        self._attempt += 1
        if self._first():
            if self._second():
                self._finalize()
                raise StopIteration
        return self._attempt

    @property
    def A(self):
        """Return A of D = PAQ."""
        return self._A

    @property
    def D(self):
        """Return D of D = PAQ."""
        return self._D

    @property
    def P(self):
        """Return P of D = PAQ."""
        return self._P

    @property
    def Q(self):
        """Return Q of D = PAQ."""
        return self._Q

    def _first(self):
        self._first_one_loop()
        A = self._A
        if A[1, 0] == 0 and A[2, 0] == 0:
            return True
        elif A[1, 0] % A[0, 0] == 0 and A[2, 0] % A[0, 0] == 0:
            self._first_finalize()
            self._Ps += self._L
            self._L = []
            return True
        else:
            return False

    def _first_one_loop(self):
        self._first_column()
        self._Ps += self._L
        self._L = []
        self._A[:] = self._A.T
        self._first_column()
        self._Qs += self._L
        self._L = []
        self._A[:] = self._A.T

    def _first_column(self):
        i = self._search_first_pivot()
        if i > 0:
            self._swap_rows(0, i)
        if i < 0:
            raise RuntimeError("Determinant is 0.")

        if self._A[1, 0] != 0:
            self._zero_first_column(1)
        if self._A[2, 0] != 0:
            self._zero_first_column(2)

    def _zero_first_column(self, j):
        A = self._A
        r, s, t = xgcd([A[0, 0], A[j, 0]])
        self._set_zero(0, j, A[0, 0], A[j, 0], r, s, t)

    def _search_first_pivot(self):
        for i in range(3):  # column index
            if self._A[i, 0] != 0:
                return i
        return -1

    def _first_finalize(self):
        """Set zeros along the first colomn except for A[0, 0].

        This is possible only when A[1,0] and A[2,0] are dividable by A[0,0].

        """
        A = self._A
        L = np.eye(3, dtype="long")
        L[1, 0] = -A[1, 0] // A[0, 0]
        L[2, 0] = -A[2, 0] // A[0, 0]
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _second(self):
        """Find Smith normal form for Right-low 2x2 matrix."""
        self._second_one_loop()
        A = self._A
        if A[2, 1] == 0:
            return True
        elif A[2, 1] % A[1, 1] == 0:
            self._second_finalize()
            self._Ps += self._L
            self._L = []
            return True
        else:
            return False

    def _second_one_loop(self):
        self._second_column()
        self._Ps += self._L
        self._L = []
        self._A[:] = self._A.T
        self._second_column()
        self._Qs += self._L
        self._L = []
        self._A[:] = self._A.T

    def _second_column(self):
        """Right-low 2x2 matrix.

        Assume elements in first row and column are all zero except for A[0,0].

        """
        if self._A[1, 1] == 0 and self._A[2, 1] != 0:
            self._swap_rows(1, 2)

        if self._A[2, 1] != 0:
            self._zero_second_column()

    def _zero_second_column(self):
        A = self._A
        r, s, t = xgcd([A[1, 1], A[2, 1]])
        self._set_zero(1, 2, A[1, 1], A[2, 1], r, s, t)

    def _second_finalize(self):
        """Set zero at A[2, 1].

        This is possible only when A[2,1] is dividable by A[1,1].

        """
        A = self._A
        L = np.eye(3, dtype="long")
        L[2, 1] = -A[2, 1] // A[1, 1]
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _finalize(self):
        for i in range(3):
            if self._A[i, i] < 0:
                self._flip_sign_row(i)
                self._Ps += self._L
                self._L = []
        self._finalize_sort()
        self._finalize_disturb(0, 1)
        self._first()
        self._finalize_sort()
        self._finalize_disturb(1, 2)
        self._second()
        self._set_PQ()

    def _finalize_sort(self):
        if self._A[0, 0] > self._A[1, 1]:
            self._swap_diag_elems(0, 1)
        if self._A[1, 1] > self._A[2, 2]:
            self._swap_diag_elems(1, 2)
        if self._A[0, 0] > self._A[1, 1]:
            self._swap_diag_elems(0, 1)

    def _finalize_disturb(self, i, j):
        if self._A[j, j] % self._A[i, i] != 0:
            self._A[:] = self._A.T
            self._disturb_rows(i, j)
            self._Qs += self._L
            self._L = []
            self._A[:] = self._A.T

    def _disturb_rows(self, i, j):
        L = np.eye(3, dtype="long")
        L[i, i] = 1
        L[i, j] = 1
        L[j, i] = 0
        L[j, j] = 1
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _swap_diag_elems(self, i, j):
        self._swap_rows(i, j)
        self._Ps += self._L
        self._L = []
        self._A[:] = self._A.T
        self._swap_rows(i, j)
        self._Qs += self._L
        self._L = []
        self._A[:] = self._A.T

    def _swap_rows(self, i, j):
        """Swap i and j rows.

        As the side effect, determinant flips.

        """
        L = np.eye(3, dtype="long")
        L[i, i] = 0
        L[j, j] = 0
        L[i, j] = 1
        L[j, i] = 1
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _flip_sign_row(self, i):
        """Multiply -1 for all elements in row."""
        L = np.eye(3, dtype="long")
        L[i, i] = -1
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _set_zero(self, i, j, a, b, r, s, t):
        """Let A[i, j] be zero based on Bezout's identity.

        [ii ij]
        [ji jj] is a (k,k) minor of original 3x3 matrix.

        """
        L = np.eye(3, dtype="long")
        L[i, i] = s
        L[i, j] = t
        L[j, i] = -b // r
        L[j, j] = a // r
        self._L.append(L.copy())
        self._A[:] = np.dot(L, self._A)

    def _set_PQ(self):
        P = np.eye(3, dtype="long")
        for _P in self._Ps:
            P = np.dot(_P, P)
        Q = np.eye(3, dtype="long")
        for _Q in self._Qs:
            Q = np.dot(Q, _Q.T)

        if self._det(P) < 0:
            P *= -1
            Q *= -1

        self._P = P
        self._Q = Q
        self._D = self._A.copy()

    def _det(self, m):
        return (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            + m[0, 1] * (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        )


def xgcd(vals):
    """Calculate extended greatest commond divisor."""
    _xgcd = Xgcd(vals)
    return _xgcd.run()


class Xgcd:
    """Class to implement extended Euclidean algorithm."""

    def __init__(self, vals):
        """Init method."""
        self._vals = np.array(vals, dtype="intc")

    def run(self):
        """Calculate extended GCD."""
        r0, r1 = self._vals
        s0 = 1
        s1 = 0
        t0 = 0
        t1 = 1
        for _ in range(1000):
            r0, r1, s0, s1, t0, t1 = self._step(r0, r1, s0, s1, t0, t1)
            if r1 == 0:
                break

        assert r0 == self._vals[0] * s0 + self._vals[1] * t0

        self._rst = np.array([r0, s0, t0], dtype="intc")

        return self._rst

    def _step(self, r0, r1, s0, s1, t0, t1):
        q, m = divmod(r0, r1)
        if m < 0:
            if r1 > 0:
                m += r1
                q -= 1
            if r1 < 0:
                m -= r1
                q += 1
        r2 = m
        s2 = s0 - q * s1
        t2 = t0 - q * t1
        return r1, r2, s1, s2, t1, t2

    def __str__(self):
        """Show GCD relation."""
        v = self._vals
        r, s, t = self._rst
        return "%d = %d * (%d) + %d * (%d)" % (r, v[0], s, v[1], t)
