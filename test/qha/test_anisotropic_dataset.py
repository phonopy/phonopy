"""Tests for the anisotropic QHA intermediate dataset I/O."""

from __future__ import annotations

import numpy as np

from phonopy.qha.anisotropic_dataset import (
    AnisoQHADataset,
    AnisoQHAGridPoint,
    read_aniso_qha_dataset,
    write_aniso_qha_dataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.structure.atoms import PhonopyAtoms


def _grid_point(index: int, with_electronic: bool) -> AnisoQHAGridPoint:
    """Build a small grid point with deterministic random arrays."""
    cell = PhonopyAtoms(
        symbols=["Ti", "Ti"],
        cell=np.diag([2.95 + 0.01 * index, 2.95 + 0.01 * index, 4.68]),
        scaled_positions=[[0.0, 0.0, 0.0], [1.0 / 3, 2.0 / 3, 0.5]],
    )
    n_satom = 16  # 2 atoms x diag(2, 2, 2) supercell
    rng = np.random.default_rng(index)
    displacements = rng.normal(size=(3, n_satom, 3))
    forces = rng.normal(size=(3, n_satom, 3))
    electronic_states = None
    if with_electronic:
        electronic_states = ElectronicStates(
            eigenvalues=rng.normal(size=(1, 5, 8)),
            weights=np.ones(5, dtype="int64"),
            n_electrons=8.0,
        )
    return AnisoQHAGridPoint(
        index=index,
        cell=cell,
        supercell_matrix=np.diag([2, 2, 2]).astype("int64"),
        primitive_matrix=np.eye(3),
        displacements=displacements,
        forces=forces,
        internal_energy=-15.0 - index,
        electronic_states=electronic_states,
    )


def test_dataset_roundtrip(tmp_path):
    """Write then read a dataset and recover every field."""
    dataset = AnisoQHADataset(
        grid_points=(_grid_point(0, True), _grid_point(1, False)),
        calculator="vasp",
        length_unit="angstrom",
        free_dof=("a", "c"),
        crystal_system="hexagonal",
        tie_description="b = a",
        phonopy_version="0.0.0",
    )
    path = tmp_path / "aniso_qha_dataset.hdf5"
    write_aniso_qha_dataset(dataset, path)
    out = read_aniso_qha_dataset(path)

    assert out.calculator == "vasp"
    assert out.length_unit == "angstrom"
    assert out.free_dof == ("a", "c")
    assert out.crystal_system == "hexagonal"
    assert out.tie_description == "b = a"
    assert out.phonopy_version == "0.0.0"
    assert len(out.grid_points) == 2

    for p_in, p_out in zip(dataset.grid_points, out.grid_points, strict=True):
        assert p_out.index == p_in.index
        np.testing.assert_allclose(p_out.cell.cell, p_in.cell.cell)
        np.testing.assert_allclose(
            p_out.cell.scaled_positions, p_in.cell.scaled_positions
        )
        np.testing.assert_array_equal(p_out.cell.numbers, p_in.cell.numbers)
        np.testing.assert_allclose(p_out.cell.masses, p_in.cell.masses)
        np.testing.assert_array_equal(p_out.supercell_matrix, p_in.supercell_matrix)
        np.testing.assert_allclose(p_out.primitive_matrix, p_in.primitive_matrix)
        np.testing.assert_allclose(p_out.displacements, p_in.displacements)
        np.testing.assert_allclose(p_out.forces, p_in.forces)
        assert p_out.internal_energy == p_in.internal_energy


def test_dataset_electronic_states_optional(tmp_path):
    """The electronic states are preserved when present and absent."""
    dataset = AnisoQHADataset(
        grid_points=(_grid_point(0, True), _grid_point(1, False)),
        free_dof=("a", "c"),
    )
    path = tmp_path / "aniso_qha_dataset.hdf5"
    write_aniso_qha_dataset(dataset, path)
    out = read_aniso_qha_dataset(path)

    assert out.grid_points[0].electronic_states is not None
    np.testing.assert_allclose(
        out.grid_points[0].electronic_states.eigenvalues,
        dataset.grid_points[0].electronic_states.eigenvalues,
    )
    np.testing.assert_array_equal(
        out.grid_points[0].electronic_states.weights,
        dataset.grid_points[0].electronic_states.weights,
    )
    assert out.grid_points[0].electronic_states.n_electrons == 8.0
    assert out.grid_points[1].electronic_states is None


def test_dataset_index_order_preserved(tmp_path):
    """Grid points are read back in ascending index order."""
    dataset = AnisoQHADataset(
        grid_points=(
            _grid_point(2, False),
            _grid_point(0, False),
            _grid_point(1, False),
        ),
    )
    path = tmp_path / "aniso_qha_dataset.hdf5"
    write_aniso_qha_dataset(dataset, path)
    out = read_aniso_qha_dataset(path)

    assert [p.index for p in out.grid_points] == [0, 1, 2]
