# SPDX-License-Identifier: BSD-3-Clause
"""Tests for BZGrid.gp_Gamma.

Kept out of test_grid.py, which skips itself without the C extension because
most of its tests drive the C kernels directly. These run with the default
kernel.

"""

from __future__ import annotations

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.grid import BZGrid


def _qpoint(bzgrid: BZGrid, gp: int) -> np.ndarray:
    """Return the q-point of a BZ grid point in reduced coordinates."""
    return bzgrid.QDinv @ (bzgrid.addresses[gp] + bzgrid.PS / 2)


@pytest.mark.parametrize("store_dense_gp_map", [False, True])
def test_BZGrid_gp_Gamma(ph_si: Phonopy, store_dense_gp_map: bool):
    """gp_Gamma is the grid point at q = 0, and None on a shifted grid."""
    lat = ph_si.primitive.cell
    for kwargs in (
        {"mesh": [4, 4, 4]},
        {"mesh": [5, 5, 5]},
        {
            "mesh": 10,
            "symmetry_dataset": ph_si.primitive_symmetry.dataset,
            "use_grg": True,
        },
    ):
        bzgrid = BZGrid(lattice=lat, store_dense_gp_map=store_dense_gp_map, **kwargs)
        assert bzgrid.gp_Gamma is not None
        np.testing.assert_allclose(_qpoint(bzgrid, bzgrid.gp_Gamma), 0, atol=1e-12)

    shifted = BZGrid(
        [4, 4, 4],
        lattice=lat,
        is_shift=[1, 1, 1],
        store_dense_gp_map=store_dense_gp_map,
    )
    assert shifted.gp_Gamma is None


@pytest.mark.filterwarnings("ignore::phonopy.phonon.mesh.MeshSymmetryFallbackWarning")
def test_Mesh_gamma_index(ph_si: Phonopy):
    """Mesh.gamma_index is the row of q = 0, and None on a shifted mesh."""
    ph = Phonopy(
        ph_si.unitcell,
        supercell_matrix=ph_si.supercell_matrix,
        primitive_matrix=ph_si.primitive_matrix,
        log_level=0,
    )
    ph.force_constants = ph_si.force_constants
    for kwargs in (
        {"mesh": [4, 4, 4], "is_gamma_center": True},
        {"mesh": [5, 5, 5]},
        {"mesh": 30.0},
    ):
        ph.run_mesh(**kwargs)
        index = ph.mesh.gamma_index
        assert index is not None
        np.testing.assert_allclose(ph.mesh.qpoints[index], 0, atol=1e-12)

    # phonopy's default for an even mesh is shifted by half a grid spacing.
    ph.run_mesh([4, 4, 4])
    assert ph.mesh.gamma_index is None
