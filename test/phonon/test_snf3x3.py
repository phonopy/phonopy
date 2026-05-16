"""Regression tests comparing Rust and C snf3x3 implementations."""

from __future__ import annotations

import numpy as np
import pytest

phonors = pytest.importorskip("phonors")
recgrid_c = pytest.importorskip("phonopy._recgrid")

FIXED_CASES: list[np.ndarray] = [
    np.array([[0, 16, 16], [16, 0, 16], [6, 6, 0]], dtype="int64"),
    np.array([[0, 5, 5], [2, 0, 2], [3, 3, 0]], dtype="int64"),
    np.array([[2, 0, 0], [0, 3, 0], [0, 0, 5]], dtype="int64"),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64"),
    np.array([[6, 0, 0], [0, 10, 0], [0, 0, 15]], dtype="int64"),
]


def _rust_snf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, p, q = phonors.snf3x3(np.ascontiguousarray(a, dtype="int64"))
    return (
        np.asarray(d, dtype="int64"),
        np.asarray(p, dtype="int64"),
        np.asarray(q, dtype="int64"),
    )


def _c_snf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.zeros(3, dtype="int64")
    p = np.zeros((3, 3), dtype="int64")
    q = np.zeros((3, 3), dtype="int64")
    ok = recgrid_c.snf3x3(d, p, q, a.copy())
    assert ok, "C snf3x3 failed"
    return d, p, q


def _assert_valid_snf(
    a: np.ndarray, d: np.ndarray, p: np.ndarray, q: np.ndarray
) -> None:
    """Check the diagonalization properties guaranteed by the C algorithm.

    Note: the strict SNF condition ``d_i | d_{i+1}`` is *not* enforced
    by the C implementation and therefore not checked here.
    """
    np.testing.assert_array_equal(p @ a @ q, np.diag(d))
    assert round(abs(np.linalg.det(p))) == 1
    assert round(abs(np.linalg.det(q))) == 1
    assert (d >= 0).all()


@pytest.mark.parametrize("a", FIXED_CASES)
def test_snf3x3_rust_valid(a: np.ndarray) -> None:
    """Rust snf3x3 output must satisfy the diagonalization properties."""
    d, p, q = _rust_snf(a)
    _assert_valid_snf(a, d, p, q)


@pytest.mark.parametrize("a", FIXED_CASES)
def test_snf3x3_rust_matches_c(a: np.ndarray) -> None:
    """Rust and C snf3x3 must return element-wise identical ``(d, p, q)``."""
    d_rs, p_rs, q_rs = _rust_snf(a)
    d_c, p_c, q_c = _c_snf(a)
    np.testing.assert_array_equal(d_rs, d_c)
    np.testing.assert_array_equal(p_rs, p_c)
    np.testing.assert_array_equal(q_rs, q_c)


def test_snf3x3_rust_random_matches_c() -> None:
    """Randomized cross-check: Rust and C snf3x3 agree on 200 random inputs."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        while True:
            a = rng.integers(-20, 21, size=(3, 3), dtype="int64")
            if round(np.linalg.det(a)) != 0:
                break
        d_rs, p_rs, q_rs = _rust_snf(a)
        d_c, p_c, q_c = _c_snf(a)
        _assert_valid_snf(a, d_rs, p_rs, q_rs)
        np.testing.assert_array_equal(d_rs, d_c)
        np.testing.assert_array_equal(p_rs, p_c)
        np.testing.assert_array_equal(q_rs, q_c)
