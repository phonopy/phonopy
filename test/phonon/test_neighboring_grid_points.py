# SPDX-License-Identifier: BSD-3-Clause
"""Tests for get_neighboring_grid_points."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.phonon.grid import BZGrid, get_neighboring_grid_points
from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_gr_grid_address
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

pytest.importorskip("phonors")


@pytest.mark.parametrize(
    "cell_name,mesh,use_grg", [("aln_cell", [4, 4, 2], False), ("nacl", 20, True)]
)
def test_get_neighboring_grid_points_python_matches_rust(
    request: pytest.FixtureRequest, cell_name: str, mesh, use_grg: bool
):
    """The Python path picks the same BZ-surface images as phonors."""
    if cell_name == "nacl":
        a = 5.69
        cell = PhonopyAtoms(
            ["Na", "Cl"],
            cell=(np.ones((3, 3)) - np.eye(3)) * a / 2,
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )
    else:
        cell = request.getfixturevalue(cell_name)
    bz_grid = BZGrid(
        mesh,
        lattice=cell.cell,
        symmetry_dataset=Symmetry(cell).dataset,
        use_grg=use_grg,
    )
    relative_grid_address = get_tetrahedra_relative_gr_grid_address(
        bz_grid, symmetrize_tetrahedra=True
    )
    num_not_first_image = 0
    for gp in range(len(bz_grid.addresses)):
        vertices = get_neighboring_grid_points(
            gp, relative_grid_address, bz_grid, lang="Python"
        )
        np.testing.assert_array_equal(
            vertices,
            get_neighboring_grid_points(gp, relative_grid_address, bz_grid),
        )
        num_not_first_image += (
            vertices != bz_grid.grg2bzg[bz_grid.bzg2grg[vertices]]
        ).sum()
    # BZ-surface images other than the first one are chosen.
    assert num_not_first_image > 0


def test_get_neighboring_grid_points_sparse_gp_map(aln_cell: PhonopyAtoms):
    """The Python path supports only store_dense_gp_map=True."""
    bz_grid = BZGrid(
        [4, 4, 2],
        lattice=aln_cell.cell,
        symmetry_dataset=Symmetry(aln_cell).dataset,
        store_dense_gp_map=False,
    )
    with pytest.raises(NotImplementedError):
        get_neighboring_grid_points(
            0, np.zeros((1, 3), dtype="int64"), bz_grid, lang="Python"
        )
