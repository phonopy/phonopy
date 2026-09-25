# SPDX-License-Identifier: BSD-3-Clause
"""Tests for routines in tetrahedron_method.py."""

import numpy as np
import pytest

from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_grid_address

# The C/Rust parity check requires phonopy._phonopy.
pytest.importorskip("phonopy._phonopy")


# Microzone lattices spanning a few cell shapes (cubic, distorted, monoclinic-ish).
# get_tetrahedra_relative_grid_address picks a main diagonal whose orientation
# depends on the metric, so different lattices exercise different code paths.
_MICROZONE_LATTICES = [
    np.eye(3) / 4,
    np.diag([1.0, 1.5, 2.0]) / 4,
    np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]]) / 4,
    np.array(
        [[1.0, 0.5, 0.0], [0.0, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]],
    )
    / 4,
]


@pytest.mark.parametrize("lat", _MICROZONE_LATTICES)
def test_get_tetrahedra_relative_grid_address_rust_matches_c(
    lat: np.ndarray,
) -> None:
    """phonors.tetrahedra_relative_grid_address must agree with the C kernel."""
    pytest.importorskip("phonors")
    ga_c = get_tetrahedra_relative_grid_address(lat, lang="C")
    ga_r = get_tetrahedra_relative_grid_address(lat, lang="Rust")
    np.testing.assert_array_equal(ga_c, ga_r)
