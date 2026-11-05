# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the Niggli and Delaunay reductions in lattice_reduction.py."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import spglib
from numpy.typing import NDArray

import phonopy.structure.lattice_reduction as lattice_reduction
from phonopy.structure.lattice_reduction import delaunay_reduce, niggli_reduce

EPS = 1e-5

# Period lattices for which the Niggli steps cycle with eps = 1e-5 when they
# start from the Delaunay-reduced basis, and for which spglib 2.8.0 fails.
# The values are given in full precision because the cycles depend on the
# last bits.
CYCLING_LATTICES = [
    [
        [-0.47210111545578537, 0.28152363220693777, -0.20266982387545368],
        [0.030722311169000455, -0.8977760625737515, -0.47187959786281586],
        [0.9396144571951511, 0.022311936767950243, 0.3823182013017702],
    ],
    [
        [0.029161922322695173, 0.36143648700700204, -0.024450810729602184],
        [0.2616762288632269, 0.5741701820246939, 0.1565479591091606],
        [0.10060316679454116, -1.4794715038536985, -0.14166619601351688],
    ],
    [
        [-0.3608909439538564, -0.1456881723927258, -0.11818308514139844],
        [-0.9553247211232809, -0.49463355528361874, 0.027477434419668923],
        [0.30751379516519195, -0.2553135053576626, 0.07559410382099435],
    ],
]


def _random_unimodular(rng: np.random.Generator) -> NDArray[np.int64]:
    """Return a random integer matrix with determinant 1 or -1."""
    umat = np.eye(3, dtype="int64")
    for _ in range(4):
        i, j = rng.choice(3, size=2, replace=False)
        shear = np.eye(3, dtype="int64")
        shear[i, j] = rng.integers(-2, 3)
        umat = umat @ shear
    if rng.random() < 0.5:
        umat = umat @ rng.permutation(np.eye(3, dtype="int64"))
    return umat


def _random_rotation(rng: np.random.Generator) -> NDArray[np.double]:
    qmat, rmat = np.linalg.qr(rng.normal(size=(3, 3)))
    qmat = qmat * np.sign(np.diag(rmat))
    if np.linalg.det(qmat) < 0:
        qmat[:, 0] = -qmat[:, 0]
    return qmat


def _general_lattices(num: int, seed: int) -> list[NDArray[np.double]]:
    """Return well-conditioned random lattices in random bases."""
    rng = np.random.default_rng(seed)
    lattices: list[NDArray[np.double]] = []
    while len(lattices) < num:
        lattice = rng.normal(size=(3, 3))
        if np.linalg.cond(lattice) > 20:
            continue
        lattices.append(_random_unimodular(rng) @ lattice * rng.uniform(3, 15))
    return lattices


def _symmetric_lattices(
    num: int, seed: int, strained: bool = True
) -> list[NDArray[np.double]]:
    """Return symmetric lattices in random bases, with or without small strain.

    Several parameters of the Niggli reduction of these lattices are nearly
    equal, so that the comparisons with eps decide the result.  Without
    strain, some candidate vectors of the Delaunay reduction have equal
    lengths, so that their order decides the result.

    """
    rng = np.random.default_rng(seed)
    hexagonal = [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 1.633]]
    bases = [
        np.eye(3),
        np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
        np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
        np.array(hexagonal),
        np.diag([1.0, 1.0, 2.0]),
    ]
    lattices = []
    for _ in range(num):
        lattice = bases[rng.integers(len(bases))]
        if strained:
            strain_size = 10 ** rng.uniform(-5, np.log10(3e-2))
            sym = rng.normal(size=(3, 3))
            lattice = lattice @ (np.eye(3) + strain_size * (sym + sym.T) / 2)
        lattice = lattice @ _random_rotation(rng).T
        lattices.append(_random_unimodular(rng) @ lattice * rng.uniform(3, 15))
    return lattices


def _satisfies_niggli_main_conditions(lattice: NDArray[np.double], eps: float) -> bool:
    """Return whether the main conditions of a Niggli-reduced basis hold.

    A <= B <= C, |xi| <= B, |eta| <= A, |zeta| <= A, and xi, eta, zeta are
    all positive or all non-positive.  In the latter case,
    A + B + xi + eta + zeta >= 0, i.e., |a + b + c|^2 >= C.

    """
    metric = lattice @ lattice.T
    A, B, C = metric[0, 0], metric[1, 1], metric[2, 2]
    xi, eta, zeta = 2 * metric[1, 2], 2 * metric[0, 2], 2 * metric[0, 1]
    if not (A <= B + eps and B <= C + eps):
        return False
    if not (abs(xi) <= B + eps and abs(eta) <= A + eps and abs(zeta) <= A + eps):
        return False
    if xi > eps and eta > eps and zeta > eps:
        return True
    if xi <= eps and eta <= eps and zeta <= eps:
        return A + B + xi + eta + zeta >= -eps
    return False


def _satisfies_niggli_main_conditions_with_retries(
    lattice: NDArray[np.double],
) -> bool:
    """Return whether the main conditions hold with eps / 2**r for some r.

    After a cycle of its steps, the Niggli reduction starts again with
    eps / 2, up to NIGGLI_NUM_RETRIES times.  The conditions are not
    monotonic in eps, e.g., xi between eps / 2 and eps is non-positive with
    eps and positive with eps / 2, so the result of a retry may not satisfy
    them with eps.

    """
    return any(
        _satisfies_niggli_main_conditions(lattice, EPS / 2**retry)
        for retry in range(lattice_reduction.NIGGLI_NUM_RETRIES + 1)
    )


def _error_from_exact_product(
    lattice: NDArray[np.double],
    reduced: NDArray[np.double],
    tmat: NDArray[np.int64],
) -> float:
    """Return max |reduced - tmat.T @ lattice| with the product in rationals."""
    error = Fraction(0)
    for i in range(3):
        for c in range(3):
            exact = sum(
                int(tmat[k, i]) * Fraction(float(lattice[k, c])) for k in range(3)
            )
            error = max(error, abs(Fraction(float(reduced[i, c])) - exact))
    return float(error)


def _check_tmat(
    lattice: NDArray[np.double],
    reduced: NDArray[np.double],
    tmat: NDArray[np.int64],
) -> None:
    assert tmat.dtype == np.int64
    assert reduced.shape == (3, 3)
    np.testing.assert_allclose(
        tmat.T @ lattice, reduced, rtol=0, atol=1e-12 * np.abs(lattice).max()
    )


@pytest.mark.parametrize(
    "lattices",
    [
        _general_lattices(200, 11),
        _symmetric_lattices(200, 12),
        _symmetric_lattices(200, 17, strained=False),
    ],
    ids=["general", "strained", "symmetric"],
)
def test_delaunay_reduce_agrees_with_spglib(lattices: list[NDArray[np.double]]):
    """Delaunay reduction chooses the same basis as spglib."""
    for lattice in lattices:
        reduced, tmat = delaunay_reduce(lattice, eps=EPS)
        _check_tmat(lattice, reduced, tmat)
        assert round(np.linalg.det(tmat)) in (1, -1)
        assert np.linalg.det(reduced) > 0
        ref = spglib.delaunay_reduce(lattice, eps=EPS)
        assert ref is not None
        ref_tmat = np.rint((ref @ np.linalg.inv(lattice)).T).astype("int64")
        np.testing.assert_array_equal(tmat, ref_tmat)
        np.testing.assert_allclose(
            reduced, ref, rtol=0, atol=1e-12 * np.abs(lattice).max()
        )


@pytest.mark.parametrize(
    "lattices",
    [_general_lattices(200, 13), _symmetric_lattices(200, 14)],
    ids=["general", "strained"],
)
def test_niggli_reduce(lattices: list[NDArray[np.double]]):
    """Niggli-reduced basis satisfies the main conditions.

    It keeps the handedness of the input, and for general lattices it agrees
    with the basis of spglib.  For strained symmetric lattices, the basis of
    spglib can differ by the comparisons with eps.

    """
    for lattice in lattices:
        reduced, tmat = niggli_reduce(lattice, eps=EPS)
        _check_tmat(lattice, reduced, tmat)
        assert round(np.linalg.det(tmat)) == 1
        assert np.sign(np.linalg.det(reduced)) == np.sign(np.linalg.det(lattice))
        assert _satisfies_niggli_main_conditions_with_retries(reduced)


def test_niggli_reduce_agrees_with_spglib():
    """Niggli-reduced basis of general lattices equals that of spglib."""
    for lattice in _general_lattices(200, 15):
        reduced, _ = niggli_reduce(lattice, eps=EPS)
        ref = spglib.niggli_reduce(lattice, eps=EPS)
        assert ref is not None
        np.testing.assert_allclose(reduced, ref, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "lattices",
    [_general_lattices(200, 18), _symmetric_lattices(200, 19)],
    ids=["general", "strained"],
)
def test_niggli_steps_from_input(lattices: list[NDArray[np.double]]):
    """Niggli steps alone reduce the input basis.

    niggli_reduce starts the Niggli steps from the Delaunay-reduced basis,
    for which the steps A5-A8 are rarely applied.  Here the steps start from
    the input basis.

    """
    for lattice in lattices:
        columns = lattice_reduction._get_columns(lattice)
        for retry in range(lattice_reduction.NIGGLI_NUM_RETRIES + 1):
            tmat_list = lattice_reduction._run_niggli_reduce(columns, EPS / 2**retry)
            if tmat_list is not None:
                break
        assert tmat_list is not None
        tmat = np.array(tmat_list, dtype="int64")
        assert round(np.linalg.det(tmat)) == 1
        assert _satisfies_niggli_main_conditions_with_retries(tmat.T @ lattice)


def test_niggli_step_a8():
    """Niggli steps reduce a basis that only A8 changes.

    The metric tensor A = B = 1, C = 1.2, xi = eta = zeta = -0.9 satisfies
    the main conditions except A + B + xi + eta + zeta >= 0, which A8 makes
    hold by replacing c with a + b + c.

    """
    metric = np.array([[1, -0.45, -0.45], [-0.45, 1, -0.45], [-0.45, -0.45, 1.2]])
    lattice = np.linalg.cholesky(metric)
    assert not _satisfies_niggli_main_conditions(lattice, EPS)

    columns = lattice_reduction._get_columns(lattice)
    tmat_list = lattice_reduction._run_niggli_reduce(columns, EPS)
    assert tmat_list is not None
    tmat = np.array(tmat_list, dtype="int64")
    assert round(np.linalg.det(tmat)) == 1
    assert _satisfies_niggli_main_conditions(tmat.T @ lattice, EPS)

    reduced, tmat = niggli_reduce(lattice, eps=EPS)
    _check_tmat(lattice, reduced, tmat)
    assert _satisfies_niggli_main_conditions(reduced, EPS)


@pytest.mark.parametrize("lattice", CYCLING_LATTICES)
def test_niggli_reduce_cycling_lattices(lattice: list[list[float]]):
    """Niggli reduction starts again with eps / 2 after a cycle."""
    lattice_array = np.array(lattice)
    columns, _ = lattice_reduction._run_delaunay_reduce(
        lattice_reduction._get_columns(lattice_array), EPS
    )
    assert lattice_reduction._run_niggli_reduce(columns, EPS) is None

    reduced, tmat = niggli_reduce(lattice_array, eps=EPS)
    _check_tmat(lattice_array, reduced, tmat)
    assert round(np.linalg.det(tmat)) == 1
    assert _satisfies_niggli_main_conditions(reduced, EPS / 2)


def test_niggli_reduce_pass_limit(monkeypatch: pytest.MonkeyPatch):
    """Niggli reduction fails without retries when the pass limit is reached."""
    epsilons: list[float] = []
    run_niggli_reduce = lattice_reduction._run_niggli_reduce

    def counting_run_niggli_reduce(columns, eps):
        epsilons.append(eps)
        return run_niggli_reduce(columns, eps)

    monkeypatch.setattr(lattice_reduction, "NIGGLI_MAX_NUM_PASSES", 1)
    monkeypatch.setattr(
        lattice_reduction, "_run_niggli_reduce", counting_run_niggli_reduce
    )
    with pytest.raises(RuntimeError, match="did not terminate"):
        niggli_reduce(CYCLING_LATTICES[0], eps=EPS)
    assert epsilons == [EPS]


@pytest.mark.parametrize("skew,max_error", [(100, 2e-12), (1000, 1e-9)])
def test_skewed_lattices(skew: int, max_error: float):
    """Strongly skewed lattices are reduced, and the bases are accurate.

    The error is that from the exact product of the input basis and tmat.
    With skew = 100 (582 Delaunay steps), the basis updated in floating
    point by the steps has the error 1.4e-11, and the returned basis,
    computed from tmat, 2.2e-13.  With skew = 1000 (5531 steps), the error
    without any recomputation is 2.3e-8 and that of the returned basis
    1.2e-10.  The recomputation of the superbase every 1000 steps does not
    change the returned basis and is not checked here.

    """
    general = np.array([[1.0, 0.1, 0.2], [0.0, 1.1, 0.3], [0.0, 0.0, 0.9]])
    lattice = np.array([[1, skew, 0], [0, 1, skew], [0, 0, 1]], dtype=float) @ general
    for reduce in (delaunay_reduce, niggli_reduce):
        reduced, tmat = reduce(lattice, eps=EPS)
        assert round(np.linalg.det(tmat)) in (1, -1)
        assert np.abs(reduced).max() < 2
        assert _error_from_exact_product(lattice, reduced, tmat) < max_error
        if reduce is niggli_reduce:
            assert _satisfies_niggli_main_conditions_with_retries(reduced)


def test_left_handed_input():
    """Delaunay basis has a positive volume, Niggli basis keeps the sign."""
    lattice = _general_lattices(1, 16)[0]
    if np.linalg.det(lattice) > 0:
        lattice = -lattice
    reduced, tmat = delaunay_reduce(lattice, eps=EPS)
    assert np.linalg.det(reduced) > 0
    assert round(np.linalg.det(tmat)) == -1
    reduced, tmat = niggli_reduce(lattice, eps=EPS)
    assert np.linalg.det(reduced) < 0
    assert round(np.linalg.det(tmat)) == 1


@pytest.mark.parametrize("reduce", [delaunay_reduce, niggli_reduce])
def test_invalid_input(reduce):
    """Wrong shape raises ValueError and no volume raises RuntimeError."""
    with pytest.raises(ValueError):
        reduce([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(RuntimeError, match="no volume"):
        reduce([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
