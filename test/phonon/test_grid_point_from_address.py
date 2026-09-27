# SPDX-License-Identifier: BSD-3-Clause
"""Tests for get_grid_point_from_address."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.phonon.grid import get_grid_point_from_address

pytest.importorskip("phonors")


def test_get_grid_point_from_address_python_matches_rust():
    """The Rust path gives the indices of the Python reference."""
    D_diag = [4, 5, 6]
    rng = np.random.default_rng(0)
    addresses = rng.integers(-20, 20, size=(1000, 3))
    np.testing.assert_array_equal(
        get_grid_point_from_address(addresses, D_diag),
        get_grid_point_from_address(addresses, D_diag, lang="Python"),
    )
    for address in addresses[:10]:
        gp = get_grid_point_from_address(address, D_diag)
        assert np.ndim(gp) == 0
        assert gp == get_grid_point_from_address(address, D_diag, lang="Python")
