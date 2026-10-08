# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.cui.load_helper."""

from __future__ import annotations

import pathlib
import warnings
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from phonopy.cui.load_helper import (
    _load_pypolymlp,
    move_force_dataset_to_mlp_dataset,
    select_and_load_dataset,
)
from phonopy.file_IO import write_FORCE_SETS
from phonopy.harmonic.displacement import (
    Type1DisplacementDataset,
    Type2DisplacementDataset,
)


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
            2, _type1_dataset([0.01, 0, 0]), force_sets_filename=force_sets
        )
    assert dataset is not None
    assert "forces" in dataset["first_atoms"][0]


def test_select_and_load_dataset_force_sets_differs_from_yaml(tmp_path):
    """FORCE_SETS with other displacements is used with a warning."""
    force_sets = tmp_path / "FORCE_SETS"
    _write_type1_force_sets(force_sets, [0, 0.01, 0])
    with pytest.warns(UserWarning, match="do not match"):
        dataset = select_and_load_dataset(
            2,
            _type1_dataset([0.01, 0, 0]),
            phonopy_yaml_filename="phonopy_disp.yaml",
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
            2, yaml_dataset, force_sets_filename=force_sets
        )
    assert dataset is not None
    assert len(cast(Type2DisplacementDataset, dataset)["displacements"]) == (
        n_force_sets
    )
    assert any("do not match" in str(x.message) for x in w) is is_warned
