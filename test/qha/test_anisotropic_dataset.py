# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the anisotropic QHA intermediate dataset I/O."""

from __future__ import annotations

import numpy as np

from phonopy import Phonopy
from phonopy.qha.anisotropic_dataset import (
    AnisoQHADataset,
    AnisoQHAGridPoint,
    read_aniso_qha_dataset,
    write_aniso_qha_dataset,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.structure.atoms import PhonopyAtoms


def _grid_point(index: int, with_electronic: bool) -> AnisoQHAGridPoint:
    """Build a small type-2 grid point with deterministic random arrays."""
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
        dataset={"displacements": displacements, "forces": forces},
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
        assert "first_atoms" not in p_out.dataset  # type-2 preserved
        np.testing.assert_allclose(
            p_out.dataset["displacements"], p_in.dataset["displacements"]
        )
        np.testing.assert_allclose(p_out.dataset["forces"], p_in.dataset["forces"])
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


def _omega_ti_cell() -> PhonopyAtoms:
    """Return the omega-Ti (P6/mmm) primitive cell, no internal DOF."""
    return PhonopyAtoms(
        symbols=["Ti", "Ti", "Ti"],
        cell=[
            [4.564873370841839, 0.0, 0.0],
            [-2.282436685420920, 3.953296304208135, 0.0],
            [0.0, 0.0, 2.824656473355021],
        ],
        scaled_positions=[
            [0.0, 0.0, 0.5],
            [2.0 / 3, 1.0 / 3, 0.0],
            [1.0 / 3, 2.0 / 3, 0.0],
        ],
    )


def _omega_ti_type1_phonon() -> Phonopy:
    """Return an omega-Ti Phonopy with a type-1 dataset and random forces."""
    ph = Phonopy(_omega_ti_cell(), supercell_matrix=np.diag([2, 2, 4]), log_level=0)
    ph.generate_displacements(distance=0.03)
    assert len(ph.displacements) == 2  # symmetry reduces omega-Ti to 2 disps
    assert "first_atoms" in ph.dataset  # type-1 signature
    rng = np.random.default_rng(0)
    ph.forces = rng.normal(scale=0.1, size=(2, len(ph.supercell), 3))
    return ph


def test_type1_dataset_roundtrip_and_fc(tmp_path):
    """A type-1 grid point survives HDF5 I/O and yields identical FC.

    The stored dataset keeps its type-1 form, and force constants from
    AnisoQHAGridPoint.to_phonopy match those computed directly.

    """
    ph = _omega_ti_type1_phonon()
    ph.produce_force_constants()
    fc_direct = np.array(ph.force_constants)

    point = AnisoQHAGridPoint(
        index=0,
        cell=ph.unitcell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        dataset=ph.dataset,
        internal_energy=0.0,
    )

    path = tmp_path / "aniso_qha_dataset.hdf5"
    write_aniso_qha_dataset(AnisoQHADataset(grid_points=(point,)), path)
    out = read_aniso_qha_dataset(path).grid_points[0]

    assert "first_atoms" in out.dataset  # type-1 preserved across I/O
    assert out.n_displacements == 2
    fc_roundtrip = np.array(out.to_phonopy().force_constants)
    np.testing.assert_allclose(fc_direct, fc_roundtrip, atol=1e-12)


def test_type2_to_phonopy_fc(tmp_path):
    """A type-2 grid point yields FC via the symfc solver after I/O."""
    ph = Phonopy(_omega_ti_cell(), supercell_matrix=np.diag([2, 2, 4]), log_level=0)
    ph.generate_displacements(distance=0.03, number_of_snapshots=4, random_seed=1)
    assert "first_atoms" not in ph.dataset  # type-2 signature
    rng = np.random.default_rng(2)
    ph.forces = rng.normal(scale=0.1, size=(4, len(ph.supercell), 3))
    ph.produce_force_constants(fc_calculator="symfc")
    fc_direct = np.array(ph.force_constants)

    point = AnisoQHAGridPoint(
        index=0,
        cell=ph.unitcell,
        supercell_matrix=np.array(ph.supercell_matrix, dtype="int64"),
        primitive_matrix=np.array(ph.primitive_matrix, dtype="double"),
        dataset=ph.dataset,
        internal_energy=0.0,
    )

    path = tmp_path / "aniso_qha_dataset.hdf5"
    write_aniso_qha_dataset(AnisoQHADataset(grid_points=(point,)), path)
    out = read_aniso_qha_dataset(path).grid_points[0]

    assert "first_atoms" not in out.dataset
    fc_roundtrip = np.array(out.to_phonopy(fc_calculator="symfc").force_constants)
    np.testing.assert_allclose(fc_direct, fc_roundtrip, atol=1e-12)
