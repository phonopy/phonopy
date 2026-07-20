# SPDX-License-Identifier: BSD-3-Clause
"""Tests for MLP SSCHA."""

import pathlib

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import MLPSSCHA

cwd = pathlib.Path(__file__).parent


@pytest.fixture(scope="module")
def mlp_kcl():
    """Return PhonopyMLP instance for KCl."""
    pytest.importorskip("pypolymlp")
    return PhonopyMLP().load(cwd / ".." / "polymlp_KCL-120.yaml")


@pytest.fixture(scope="module")
def sscha_result(ph_kcl: Phonopy, mlp_kcl):
    """Return MLPSSCHA instance after 3 iterations with fixed random seed."""
    sscha = MLPSSCHA(
        ph_kcl,
        mlp_kcl,
        number_of_snapshots=50,
        max_iterations=3,
        temperature=300,
        random_seed=42,
        log_level=1,
    )
    sscha.run()
    return sscha


def test_MLPSSCHA(ph_kcl: Phonopy, mlp_kcl):
    """Test MLPSSCHA class."""
    sscha = MLPSSCHA(
        ph_kcl,
        mlp_kcl,
        number_of_snapshots=2,
        max_iterations=2,
        temperature=300,
        log_level=2,
    )
    sscha.run()

    sscha = MLPSSCHA(
        ph_kcl,
        mlp_kcl,
        number_of_snapshots=2,
        max_iterations=2,
        temperature=300,
        log_level=1,
    )
    count = 0
    for i, _ in enumerate(sscha):
        count = i
    assert count == 2


def test_MLPSSCHA_force_constants_shape(sscha_result: MLPSSCHA):
    """Force constants shape should be (n_atoms, n_atoms, 3, 3)."""
    fc = sscha_result.force_constants
    n_atoms = len(sscha_result.phonopy.supercell)
    assert fc.shape == (n_atoms, n_atoms, 3, 3)


def test_MLPSSCHA_force_constants_symmetry(sscha_result: MLPSSCHA):
    """Force constants should satisfy fc[i,j,a,b] == fc[j,i,b,a]."""
    fc = sscha_result.force_constants
    np.testing.assert_allclose(fc, fc.transpose(1, 0, 3, 2), atol=1e-10)


def test_MLPSSCHA_force_constants_diagonal_block(sscha_result: MLPSSCHA):
    """Self-interaction block fc[i,i] should be isotropic and positive."""
    fc = sscha_result.force_constants
    # KCl has cubic symmetry: diagonal block is proportional to identity
    np.testing.assert_allclose(fc[0, 0, 0, 0], fc[0, 0, 1, 1], rtol=1e-6)
    np.testing.assert_allclose(fc[0, 0, 0, 0], fc[0, 0, 2, 2], rtol=1e-6)
    np.testing.assert_allclose(fc[0, 0, 0, 1], 0.0, atol=1e-10)
    # Self-interaction eigenvalue must be positive (stable structure)
    assert fc[0, 0, 0, 0] > 0


def test_MLPSSCHA_force_constants_values(sscha_result: MLPSSCHA):
    """Force constants should match values reproduced by random_seed=42."""
    fc = sscha_result.force_constants
    # Values differ slightly between platforms (Mac/Windows vs Linux) due to
    # differences in BLAS implementations and floating-point ordering.
    fc00 = np.eye(3) * 2.1
    assert np.allclose(fc[0, 0], fc00, atol=0.05)
    print(fc[0, 0])


def test_MLPSSCHA_free_energy(sscha_result: MLPSSCHA):
    """free_energy should be finite and match value reproduced by random_seed=42."""
    sscha_result.calculate_free_energy()
    assert isinstance(sscha_result.free_energy, float)
    assert np.isfinite(sscha_result.free_energy)
    # Values differ slightly between platforms due to differences in BLAS
    # implementations and floating-point ordering.
    assert np.isclose(sscha_result.free_energy, -0.0986, atol=1e-3)
    print(sscha_result.free_energy)
    # The error is a standard error of the mean over the sampled supercells,
    # so its value depends on the random numbers.
    assert np.isfinite(sscha_result.free_energy_error)
    assert sscha_result.free_energy_error > 0


def test_MLPSSCHA_sample_supercells(ph_kcl: Phonopy, mlp_kcl, sscha_result: MLPSSCHA):
    """Force constants must survive the sampling to give the free energy.

    Mutating the displacement dataset clears the force constants of the
    Phonopy instance. In the iteration they are computed again just after the
    sampling, but the free energy of given force constants can be evaluated
    only if the sampling keeps them.

    """
    ph = ph_kcl.replicate()
    ph.force_constants = sscha_result.force_constants
    sscha = MLPSSCHA(
        ph, mlp_kcl, number_of_snapshots=10, temperature=300, random_seed=42
    )
    sscha.sample_supercells()

    assert sscha.phonopy.force_constants is not None
    np.testing.assert_allclose(
        sscha.phonopy.force_constants, sscha_result.force_constants, atol=1e-10
    )

    sscha.calculate_free_energy()
    assert np.isfinite(sscha.free_energy)


def test_MLPSSCHA_compact_force_constants(
    ph_kcl: Phonopy, mlp_kcl, sscha_result: MLPSSCHA
):
    """Compact force constants must be expanded to the full form.

    The harmonic potential energy is computed with the full force constants.

    """
    fc = sscha_result.force_constants
    ph = ph_kcl.replicate()
    p2s_map = ph.primitive.p2s_map
    ph.force_constants = fc[p2s_map]
    assert ph.force_constants.shape[0] != ph.force_constants.shape[1]

    sscha = MLPSSCHA(ph, mlp_kcl, number_of_snapshots=2, temperature=300)
    n_atoms = len(sscha.phonopy.supercell)
    assert sscha.phonopy.force_constants.shape == (n_atoms, n_atoms, 3, 3)
    np.testing.assert_allclose(
        sscha.phonopy.force_constants[p2s_map], fc[p2s_map], atol=1e-10
    )
