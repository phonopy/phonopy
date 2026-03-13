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
    fc00_mac = [[2.08849783, 0.0, 0.0], [0.0, 2.08849783, 0.0], [0.0, 0.0, 2.08849783]]
    fc00_linux = [
        [2.11251410, 0.0, 0.0],
        [0.0, 2.11251410, 0.0],
        [0.0, 0.0, 2.11251410],
    ]
    fc01_mac = [[-0.21230723, 0.0, 0.0], [0.0, 0.01625349, 0.0], [0.0, 0.0, 0.01625349]]
    assert np.allclose(fc[0, 0], fc00_mac, atol=1e-5) or np.allclose(
        fc[0, 0], fc00_linux, atol=1e-5
    )
    if np.allclose(fc[0, 0], fc00_mac, atol=1e-5):
        np.testing.assert_allclose(fc[0, 1], fc01_mac, atol=1e-5)


def test_MLPSSCHA_free_energy(sscha_result: MLPSSCHA):
    """free_energy should be finite and match value reproduced by random_seed=42."""
    sscha_result.calculate_free_energy()
    assert isinstance(sscha_result.free_energy, float)
    assert np.isfinite(sscha_result.free_energy)
    # Values differ slightly between platforms due to differences in BLAS
    # implementations and floating-point ordering.
    free_energy_mac = -0.09877
    free_energy_linux = -0.09851
    assert np.isclose(
        sscha_result.free_energy, free_energy_mac, atol=1e-4
    ) or np.isclose(sscha_result.free_energy, free_energy_linux, atol=1e-4)
