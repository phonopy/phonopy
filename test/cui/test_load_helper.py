"""Tests for phonopy.cui.load_helper."""

from __future__ import annotations

from types import SimpleNamespace

from phonopy.cui.load_helper import move_force_dataset_to_mlp_dataset


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
