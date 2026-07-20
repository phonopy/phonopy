# SPDX-License-Identifier: BSD-3-Clause
"""Tests for phonopy.cui.load_helper."""

from __future__ import annotations

from types import SimpleNamespace

from phonopy.cui.load_helper import (
    _load_pypolymlp,
    move_force_dataset_to_mlp_dataset,
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
