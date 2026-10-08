# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.cui.load_helper."""

from __future__ import annotations

import pathlib
import warnings
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.cui.load_helper import (
    _load_pypolymlp,
    move_force_dataset_to_mlp_dataset,
    read_force_sets,
    select_and_load_dataset,
)
from phonopy.file_IO import write_FORCE_SETS
from phonopy.harmonic.displacement import (
    Type1DisplacementDataset,
    Type2DisplacementDataset,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, merge_weighted_species


def test_move_force_dataset_to_mlp_dataset_no_dataset():
    """No dataset means nothing to move."""
    phonon = SimpleNamespace(dataset=None, mlp_dataset=None)
    move_force_dataset_to_mlp_dataset(phonon)
    assert phonon.dataset is None
    assert phonon.mlp_dataset is None


def test_move_force_dataset_to_mlp_dataset_with_forces():
    """A dataset with forces becomes the MLP training dataset."""
    dataset = {
        "displacements": [[[0.01, 0.0, 0.0]]],
        "forces": [[[0.1, 0.0, 0.0]]],
    }
    phonon = SimpleNamespace(dataset=dataset, mlp_dataset=None)
    move_force_dataset_to_mlp_dataset(phonon)
    assert phonon.mlp_dataset is dataset
    assert phonon.dataset is None


def test_move_force_dataset_to_mlp_dataset_displacement_only():
    """A displacement-only dataset is discarded, not used for training.

    An existing MLP (polymlp.yaml) is loaded instead of triggering training,
    so mlp_dataset must stay unset.

    """
    dataset = {"displacements": [[[0.01, 0.0, 0.0]]]}
    phonon = SimpleNamespace(dataset=dataset, mlp_dataset=None)
    move_force_dataset_to_mlp_dataset(phonon)
    assert phonon.mlp_dataset is None
    assert phonon.dataset is None


def test_load_pypolymlp_ignores_unsupported_suffix(monkeypatch, tmp_path):
    """A file such as polymlp.yaml.bak must not be loaded as MLPs.

    Its name matches the glob of the default MLP filename, but its suffix is
    not supported. Loading it would break the development of new MLPs after
    renaming polymlp.yaml to keep it aside.

    """
    (tmp_path / "polymlp.yaml.bak").write_text("dummy")
    monkeypatch.chdir(tmp_path)
    loaded = []
    _load_pypolymlp(SimpleNamespace(load_mlp=loaded.append))
    assert loaded == []


def test_load_pypolymlp_selects_supported_suffix(monkeypatch, tmp_path):
    """A supported file is found even when an unsupported one also matches."""
    (tmp_path / "polymlp.yaml.bak").write_text("dummy")
    (tmp_path / "polymlp.yaml").write_text("dummy")
    monkeypatch.chdir(tmp_path)
    loaded = []
    _load_pypolymlp(SimpleNamespace(load_mlp=loaded.append))
    assert [path.name for path in loaded] == ["polymlp.yaml"]


def _get_phonon_of_two_atoms() -> Phonopy:
    """Return Phonopy of a cubic cell with two atoms, the unit cell as supercell."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )
    return Phonopy(cell, supercell_matrix=[1, 1, 1], primitive_matrix="P")


def _type1_dataset(displacement: list[float]) -> Type1DisplacementDataset:
    return {
        "natom": 2,
        "first_atoms": [{"number": 0, "displacement": np.array(displacement)}],
    }


def _write_type1_force_sets(path, displacement: list[float]) -> None:
    dataset = _type1_dataset(displacement)
    dataset["first_atoms"][0]["forces"] = np.zeros((2, 3))
    write_FORCE_SETS(dataset, filename=path)


def test_select_and_load_dataset_force_sets_matches_yaml(tmp_path):
    """No warning when FORCE_SETS has the displacements of the yaml."""
    force_sets = tmp_path / "FORCE_SETS"
    _write_type1_force_sets(force_sets, [0.01, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dataset = select_and_load_dataset(
            _get_phonon_of_two_atoms(),
            yaml_dataset=_type1_dataset([0.01, 0, 0]),
            force_sets_filename=force_sets,
        )
    assert dataset is not None
    assert "forces" in dataset["first_atoms"][0]


def test_select_and_load_dataset_force_sets_differs_from_yaml(tmp_path):
    """FORCE_SETS with other displacements is used with a warning."""
    force_sets = tmp_path / "FORCE_SETS"
    _write_type1_force_sets(force_sets, [0, 0.01, 0])
    with pytest.warns(UserWarning, match="do not match"):
        dataset = select_and_load_dataset(
            _get_phonon_of_two_atoms(),
            yaml_dataset=_type1_dataset([0.01, 0, 0]),
            yaml_filename="phonopy_disp.yaml",
            force_sets_filename=force_sets,
        )
    assert dataset is not None
    np.testing.assert_allclose(dataset["first_atoms"][0]["displacement"], [0, 0.01, 0])


@pytest.mark.parametrize("n_force_sets,is_warned", [(2, False), (4, True)])
def test_select_and_load_dataset_type2_force_sets(
    tmp_path: pathlib.Path, n_force_sets: int, is_warned: bool
) -> None:
    """Type-2 FORCE_SETS can have the first part of the displacements of yaml."""
    rng = np.random.default_rng(1)
    disps = rng.standard_normal((4, 2, 3)) * 0.01
    force_sets = tmp_path / "FORCE_SETS"
    force_sets_dataset: Type2DisplacementDataset = {
        "displacements": disps[:n_force_sets],
        "forces": np.zeros((n_force_sets, 2, 3)),
    }
    write_FORCE_SETS(force_sets_dataset, filename=force_sets)
    yaml_dataset: Type2DisplacementDataset = {"displacements": disps[:3]}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataset = select_and_load_dataset(
            _get_phonon_of_two_atoms(),
            yaml_dataset=yaml_dataset,
            force_sets_filename=force_sets,
        )
    assert dataset is not None
    assert len(cast(Type2DisplacementDataset, dataset)["displacements"]) == (
        n_force_sets
    )
    assert any("do not match" in str(x.message) for x in w) is is_warned


def test_select_and_load_dataset_type2_merge_different_site_displacements(
    tmp_path: pathlib.Path,
) -> None:
    """Atoms of a site in type-2 FORCE_SETS of the merge scheme move together."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4,
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        symbols=["Ge", "Sn"],
    )
    unmerged = apply_site_mixture(cell, [0.5, 0.5])
    force_sets_dataset: Type2DisplacementDataset = {
        "displacements": np.array([[[0.01, 0, 0], [0.02, 0, 0]]]),
        "forces": np.zeros((1, 2, 3)),
    }
    force_sets = tmp_path / "FORCE_SETS"
    write_FORCE_SETS(force_sets_dataset, filename=force_sets)
    with pytest.raises(RuntimeError, match="Atoms of a site have different"):
        read_force_sets(
            force_sets,
            supercell=merge_weighted_species(unmerged)[0],
            unmerged_supercell=unmerged,
        )


def test_read_force_sets_without_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None without filename and without FORCE_SETS in the current directory."""
    monkeypatch.chdir(tmp_path)
    supercell = _get_phonon_of_two_atoms().supercell
    assert read_force_sets(supercell=supercell) is None
    _write_type1_force_sets(tmp_path / "FORCE_SETS", [0.01, 0, 0])
    dataset = read_force_sets(supercell=supercell)
    assert dataset is not None
    assert "first_atoms" in dataset


def test_read_force_sets_type1_number_of_atoms(tmp_path: pathlib.Path) -> None:
    """Type-1 FORCE_SETS has to have forces on the atoms of the supercell."""
    force_sets = tmp_path / "FORCE_SETS"
    _write_type1_force_sets(force_sets, [0.01, 0, 0])
    phonon = _get_phonon_of_two_atoms()
    dataset = read_force_sets(force_sets, supercell=phonon.supercell)
    assert dataset is not None
    larger_supercell = Phonopy(
        phonon.unitcell, supercell_matrix=[2, 2, 2], primitive_matrix="P"
    ).supercell
    with pytest.raises(RuntimeError, match="has forces on 2 atoms"):
        read_force_sets(force_sets, supercell=larger_supercell)


def test_read_force_sets_type1_merge(tmp_path: pathlib.Path) -> None:
    """Type-1 FORCE_SETS of the merge scheme has forces on the unmerged atoms."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4,
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        symbols=["Ge", "Sn"],
    )
    unmerged = apply_site_mixture(cell, [0.5, 0.5])
    force_sets = tmp_path / "FORCE_SETS"
    dataset: Type1DisplacementDataset = {
        "natom": 1,
        "first_atoms": [
            {
                "number": 0,
                "displacement": np.array([0.01, 0, 0]),
                "forces": np.zeros((2, 3)),
            }
        ],
    }
    write_FORCE_SETS(dataset, filename=force_sets)
    read = cast(
        Type1DisplacementDataset,
        read_force_sets(
            force_sets,
            supercell=merge_weighted_species(unmerged)[0],
            unmerged_supercell=unmerged,
        ),
    )
    assert read["natom"] == 1
    assert read["first_atoms"][0]["forces"].shape == (2, 3)

    dataset["first_atoms"][0]["forces"] = np.zeros((1, 3))
    write_FORCE_SETS(dataset, filename=force_sets)
    with pytest.raises(RuntimeError, match="has forces on 1 atoms"):
        read_force_sets(
            force_sets,
            supercell=merge_weighted_species(unmerged)[0],
            unmerged_supercell=unmerged,
        )
