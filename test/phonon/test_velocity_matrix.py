# SPDX-License-Identifier: BSD-3-Clause
"""Tests for VelocityMatrix."""

import numpy as np
import pytest
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.phonon.grid import BZGrid
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.phonon.velocity_matrix import VelocityMatrix
from phonopy.physical_units import get_physical_units


def test_velocity_matrices_are_not_projected(ph_si: Phonopy):
    """Test that run does not overwrite the diagonal of velocity matrices.

    Group velocities are projected by the little group of q, but velocity
    matrices are not. The diagonal is compared with the group velocities of
    GroupVelocity without symmetry, which are not projected. The q-point is on
    Gamma-L, where the projection changes the group velocities of some bands,
    as the test asserts. At a q-point where the projection does nothing, this
    test would pass with the overwritten diagonal.

    """
    bz_grid = BZGrid(
        [9, 9, 9],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    assert ph_si.dynamical_matrix is not None
    group_velocity = GroupVelocity(ph_si.dynamical_matrix)
    q_point = np.array([1, 1, 1], dtype="double") / 9

    velocity.run([q_point])
    group_velocity.run([q_point])

    assert velocity.velocity_matrices is not None
    assert velocity.group_velocities is not None
    assert group_velocity.group_velocities is not None
    diagonal = np.einsum("aii->ia", velocity.velocity_matrices[0]).real
    expected_diagonal = group_velocity.group_velocities[0]
    projected = expected_diagonal @ velocity._get_projector(q_point).T

    assert not np.allclose(expected_diagonal, projected, atol=1e-2)
    np.testing.assert_allclose(diagonal, expected_diagonal, atol=1e-8)
    np.testing.assert_allclose(velocity.group_velocities[0], projected, atol=1e-8)


def test_velocity_matrices_degeneracy_lifting_direction(ph_si: Phonopy):
    """Test degeneracy_lifting_direction of VelocityMatrix.run by Si.

    The q-point is on Gamma-L, where transverse bands are doubly degenerate,
    so that velocity matrices depend on the lifting direction, as the test
    asserts. ``perturbation`` is a deprecated alias.

    """
    bz_grid = BZGrid(
        [9, 9, 9],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    q_points = [[0.1, 0.1, 0.1]]

    velocity.run(q_points, degeneracy_lifting_direction=[1, 0, 0])
    vm_100 = velocity.velocity_matrices
    velocity.run(q_points, degeneracy_lifting_direction=[0, 1, 0])
    vm_010 = velocity.velocity_matrices
    assert vm_100 is not None and vm_010 is not None
    assert not np.allclose(vm_100, vm_010, atol=1e-2)

    with pytest.warns(DeprecationWarning, match="degeneracy_lifting_direction"):
        velocity.run(q_points, perturbation=[1, 0, 0])
    np.testing.assert_array_equal(velocity.velocity_matrices, vm_100)


def test_group_velocities_agree_with_group_velocity(ph_si: Phonopy):
    """Test group velocities of VelocityMatrix against GroupVelocity by Si.

    Both classes lift the degeneracy along the same direction and project
    the group velocities by the little group of q. The q-points on Gamma-L
    and at W have degenerate bands whose group velocities are changed by the
    projection, as the test asserts.

    """
    bz_grid = BZGrid(
        [9, 9, 9],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    group_velocity = GroupVelocity(
        ph_si.dynamical_matrix, symmetry=ph_si.primitive_symmetry
    )
    q_points = [[0.1, 0, 0], [0.5, 0.25, 0.75], [0.1, 0.2, 0.3]]

    velocity.run(q_points)
    group_velocity.run(q_points)

    assert velocity.velocity_matrices is not None
    assert velocity.group_velocities is not None
    assert group_velocity.group_velocities is not None
    diagonal = np.einsum("qaii->qia", velocity.velocity_matrices).real
    for i in (0, 1):
        assert not np.allclose(diagonal[i], velocity.group_velocities[i], atol=1e-2)
    np.testing.assert_allclose(
        velocity.group_velocities, group_velocity.group_velocities, atol=1e-8
    )


# Block sums at the H point of Zr3N4, sum_a sum_{s in I, s' in J}
# |V^a_ss'|^2, where I and J run over the nine degenerate sets of bands.
block_sums_zr3n4_H = [
    [
        10873.2519,
        5022.3255,
        515.4673,
        2444.1033,
        1484.1782,
        164.5771,
        667.4625,
        2950.4921,
        207.7963,
    ],
    [5022.3255, 0, 0, 1339.5650, 0, 0, 1841.2783, 579.4365, 151.5381],
    [515.4673, 0, 0, 2711.7525, 0, 0, 676.6531, 249.7709, 198.4930],
    [
        2444.1033,
        1339.5650,
        2711.7525,
        3391.4226,
        610.6101,
        871.4365,
        259.4689,
        2161.7734,
        2648.9102,
    ],
    [1484.1782, 0, 0, 610.6101, 0, 0, 518.6723, 9726.4503, 2270.4073],
    [164.5771, 0, 0, 871.4365, 0, 0, 3215.5155, 3101.3027, 2973.7222],
    [
        667.4625,
        1841.2783,
        676.6531,
        259.4689,
        518.6723,
        3215.5155,
        1684.2382,
        4033.1165,
        8351.5520,
    ],
    [
        2950.4921,
        579.4365,
        249.7709,
        2161.7734,
        9726.4503,
        3101.3027,
        4033.1165,
        3366.1851,
        1226.5734,
    ],
    [
        207.7963,
        151.5381,
        198.4930,
        2648.9102,
        2270.4073,
        2973.7222,
        8351.5520,
        1226.5734,
        1084.7836,
    ],
]


def _get_velocity_matrix(ph: Phonopy) -> VelocityMatrix:
    assert ph.dynamical_matrix is not None
    bz_grid = BZGrid(
        [4, 4, 4],
        lattice=ph.primitive.cell,
        symmetry_dataset=ph.primitive_symmetry.dataset,
    )
    return VelocityMatrix(
        ph.dynamical_matrix,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )


def _get_block_sums(
    velocity_matrix: NDArray[np.cdouble], freqs: NDArray[np.double]
) -> NDArray[np.cdouble]:
    """Return sums of V^a_ss' conj(V^b_ss') over pairs of degenerate sets.

    These sums do not depend on the choice of eigenvectors in degenerate
    sets of bands, hence not on the eigensolver. The products
    V^a_ss' conj(V^b_ss') of individual elements depend on it. The sums are
    independent of the choice only as far as the frequencies in each set
    agree, because each band is scaled by its own frequency.

    Returns
    -------
    ndarray
        shape=(num_sets, num_sets, 3, 3), dtype='cdouble'

    """
    products = np.einsum("aij,bij->ijab", velocity_matrix, velocity_matrix.conj())
    sets = degenerate_sets(freqs)
    return np.array(
        [
            [products[np.ix_(set_i, set_j)].sum(axis=(0, 1)) for set_j in sets]
            for set_i in sets
        ]
    )


def _get_reference_velocity_matrix(
    ph: Phonopy, q_point: list[float], cutoff_frequency: float = 1e-4
) -> tuple[NDArray[np.cdouble], NDArray[np.double]]:
    """Return velocity matrix and frequencies by numpy.linalg.eigh.

    Degeneracy is not lifted. The eigenvectors in degenerate sets of bands
    are those returned by the eigensolver.

    """
    factor = get_physical_units().DefaultToTHz
    dynmat = ph.dynamical_matrix
    assert dynmat is not None
    dynmat.run(q_point)
    assert dynmat.dynamical_matrix is not None
    eigvals, eigvecs = np.linalg.eigh(dynmat.dynamical_matrix)
    freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor
    ddynmat = DerivativeOfDynamicalMatrix(dynmat)
    ddynmat.run(q_point)
    assert ddynmat.d_dynamical_matrix is not None
    scale = np.zeros_like(freqs)
    mask = freqs > cutoff_frequency
    scale[mask] = 1 / np.sqrt(2 * freqs[mask])
    scaled_eigvecs = eigvecs * scale
    vm = np.array(
        [
            scaled_eigvecs.T.conj() @ dD @ scaled_eigvecs
            for dD in ddynmat.d_dynamical_matrix
        ]
    )
    vm = (vm + vm.transpose(0, 2, 1).conj()) / 2
    return vm * factor**2, freqs


@pytest.mark.parametrize(
    "fixture_name,q_point,num_sets",
    [
        ("ph_si", [0.1, 0, 0], 4),
        ("ph_si", [0.5, 0.25, 0.75], 3),
        ("ph_si", [0.1, 0.2, 0.3], 6),
        ("ph_zr3n4", [0, 0, 0], 17),
        ("ph_zr3n4", [-0.5, -0.5, 0.5], 9),
        ("ph_zr3n4", [0, 0, -0.5], 21),
        ("ph_zr3n4", [0.25, 0.25, 0.25], 14),
        ("ph_zr3n4", [0.1, 0.2, 0.3], 42),
    ],
)
def test_velocity_matrices_block_sums(
    request: pytest.FixtureRequest,
    fixture_name: str,
    q_point: list[float],
    num_sets: int,
):
    """Test velocity matrices by sums over pairs of degenerate sets of bands.

    The sums are compared with those computed from eigenvectors of
    numpy.linalg.eigh without lifting degeneracy. Velocity matrices
    themselves and the products of their elements are not compared, because
    they depend on the eigensolver. In Zr3N4, the Gamma point, the H point
    [-0.5, -0.5, 0.5] and the P point [0.25, 0.25, 0.25] have degenerate sets
    in which the products of elements were observed to depend on the
    eigensolver.

    The tolerance is determined by the numerical splitting of frequencies in
    degenerate sets. With relative splittings of about 1e-8 at the P point,
    the sums of about 1e3 were observed to differ by 1e-6.

    """
    ph: Phonopy = request.getfixturevalue(fixture_name)
    velocity = _get_velocity_matrix(ph)
    # The second q-point checks that the results are stored in the order of
    # the q-points.
    velocity.run([q_point, [0.3, 0.1, 0.2]])
    assert velocity.velocity_matrices is not None
    num_band = len(ph.primitive) * 3
    assert velocity.velocity_matrices.shape == (2, 3, num_band, num_band)

    for i, q in enumerate([q_point, [0.3, 0.1, 0.2]]):
        vm_ref, freqs = _get_reference_velocity_matrix(ph, q)
        block_sums_ref = _get_block_sums(vm_ref, freqs)
        if i == 0:
            assert len(block_sums_ref) == num_sets
        block_sums = _get_block_sums(velocity.velocity_matrices[i], freqs)
        assert np.abs(block_sums_ref).max() > 1
        np.testing.assert_allclose(block_sums, block_sums_ref, rtol=0, atol=1e-4)


def test_velocity_matrices_block_sums_zr3n4_H(ph_zr3n4: Phonopy):
    """Test block sums of velocity matrices at the H point of Zr3N4.

    See _get_block_sums. The traces over the Cartesian directions are
    compared with the reference values.

    """
    q_point = [-0.5, -0.5, 0.5]
    ph_zr3n4.run_qpoints([q_point])
    assert ph_zr3n4.qpoints is not None
    freqs = ph_zr3n4.qpoints.frequencies[0]
    assert [len(s) for s in degenerate_sets(freqs)] == [6, 4, 2, 6, 4, 2, 6, 6, 6]

    velocity = _get_velocity_matrix(ph_zr3n4)
    velocity.run([q_point])
    assert velocity.velocity_matrices is not None
    block_sums = _get_block_sums(velocity.velocity_matrices[0], freqs)
    traces = np.einsum("ijaa->ij", block_sums)
    np.testing.assert_allclose(traces.imag, 0, atol=1e-8)
    np.testing.assert_allclose(traces.real, block_sums_zr3n4_H, rtol=0, atol=1e-3)


@pytest.mark.parametrize("q_length", [None, 1e-5])
def test_velocity_matrices_many_qpoints(ph_si: Phonopy, q_length: float | None):
    """Test that VelocityMatrix.run at many q-points agrees with one by one.

    Derivatives of dynamical matrices of all q-points are computed before
    the loop over q-points. The products V^a_ss' conj(V^b_ss') are compared,
    because they do not depend on the phases of eigenvectors. They differ
    among the q-points, as the test asserts, so that results stored in a
    wrong order are detected. The first two q-points have degenerate bands.

    """
    assert ph_si.dynamical_matrix is not None
    bz_grid = BZGrid(
        [4, 4, 4],
        lattice=ph_si.primitive.cell,
        symmetry_dataset=ph_si.primitive_symmetry.dataset,
    )
    velocity = VelocityMatrix(
        ph_si.dynamical_matrix,
        q_length=q_length,
        rotations_cartesian=bz_grid.rotations_cartesian,
        reciprocal_operations=bz_grid.reciprocal_operations,
    )
    q_points = [[0.1, 0.1, 0.1], [0.5, 0.25, 0.75], [0.1, 0.2, 0.3]]
    velocity.run(q_points)
    assert velocity.velocity_matrices is not None
    assert velocity.group_velocities is not None
    assert velocity.velocity_matrices.shape == (3, 3, 6, 6)
    vm_at_once = velocity.velocity_matrices
    products_at_once = np.einsum("qaij,qbij->qijab", vm_at_once, vm_at_once.conj())
    gv_at_once = velocity.group_velocities
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert not np.allclose(products_at_once[i], products_at_once[j], atol=1)

    for i, q_point in enumerate(q_points):
        velocity.run([q_point])
        assert velocity.velocity_matrices is not None
        assert velocity.group_velocities is not None
        vm = velocity.velocity_matrices[0]
        products = np.einsum("aij,bij->ijab", vm, vm.conj())
        np.testing.assert_allclose(products_at_once[i], products, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            gv_at_once[i], velocity.group_velocities[0], rtol=0, atol=1e-8
        )
