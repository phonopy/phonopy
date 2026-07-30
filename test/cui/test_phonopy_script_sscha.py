# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the SSCHA path of the phonopy command line script."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from phonopy import Phonopy
from phonopy.cui import phonopy_script
from phonopy.cui.settings import PhonopySettings

cwd = pathlib.Path(__file__).parent


@pytest.fixture
def mlp_in_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Put a trained MLP where the loader looks for it, and work there."""
    pytest.importorskip("pypolymlp")
    shutil.copy(cwd / ".." / "polymlp_KCL-120.yaml", tmp_path / "polymlp.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_prepare_pypolymlp_without_dataset(ph_kcl: Phonopy, mlp_in_cwd) -> None:
    """An MLP is loaded when no dataset is available.

    This is the --readfc situation: force constants come from a file, so the
    block that would normally load the dataset and prepare the MLP is skipped
    and no training data exists. An existing polymlp.yaml has to be enough.

    """
    phonon = Phonopy(
        ph_kcl.unitcell,
        supercell_matrix=ph_kcl.supercell_matrix,
        primitive_matrix=ph_kcl.primitive_matrix,
        log_level=0,
    )
    assert phonon.dataset is None
    assert phonon.mlp is None

    settings = PhonopySettings()
    settings.use_pypolymlp = True
    phonopy_script._prepare_pypolymlp(phonon, settings, 0)

    assert phonon.mlp is not None


class _FakeSSCHA:
    """Records the arguments MLPSSCHA was constructed with."""

    last_kwargs: dict = {}

    def __init__(self, phonon, mlp, **kwargs):
        _FakeSSCHA.last_kwargs = dict(kwargs)
        self._phonon = phonon

    def __iter__(self):
        # One iteration, so that _run_MLPSSCHA reaches its final assignment.
        return iter([0])

    @property
    def phonopy(self):
        return self._phonon


def test_run_MLPSSCHA_passes_random_seed(
    ph_kcl: Phonopy, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command-line random seed reaches MLPSSCHA.

    Without it a recorded seed does not reproduce a run, and independently
    seeded runs cannot be compared.

    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phonopy_script, "MLPSSCHA", _FakeSSCHA)

    settings = PhonopySettings()
    settings.use_pypolymlp = True
    settings.sscha_iterations = 4
    settings.random_displacements = 123
    settings.random_displacement_temperature = 250.0
    settings.random_seed = 987

    phonon = ph_kcl
    phonon.mlp = object()  # only passed through by _run_MLPSSCHA
    phonopy_script._run_MLPSSCHA(phonon, settings, 0)

    assert _FakeSSCHA.last_kwargs["random_seed"] == 987
    assert _FakeSSCHA.last_kwargs["number_of_snapshots"] == 123
    assert _FakeSSCHA.last_kwargs["max_iterations"] == 4
    assert _FakeSSCHA.last_kwargs["temperature"] == 250.0
