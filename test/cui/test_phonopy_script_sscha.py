# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the SSCHA path of the phonopy command line script."""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Iterator
from typing import Any, cast

import pytest
import yaml

from phonopy import Phonopy
from phonopy.cui import phonopy_script
from phonopy.cui.settings import PhonopySettings
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import SSCHAIterationResult

cwd = pathlib.Path(__file__).parent


@pytest.fixture
def mlp_in_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Put a trained MLP where the loader looks for it, and work there."""
    pytest.importorskip("pypolymlp")
    shutil.copy(cwd / ".." / "polymlp_KCL-120.yaml", tmp_path / "polymlp.yaml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_prepare_pypolymlp_without_dataset(
    ph_kcl: Phonopy, mlp_in_cwd: pathlib.Path
) -> None:
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
    """Records the arguments MLPSSCHA was constructed with.

    The settings properties are those the yaml writer reads back.

    """

    last_kwargs: dict[str, Any] = {}

    def __init__(self, phonon: Phonopy, mlp: Any, **kwargs: Any) -> None:
        _FakeSSCHA.last_kwargs = dict(kwargs)
        self._phonon = phonon
        self.history: tuple[SSCHAIterationResult, ...] = ()
        self.temperature: float = kwargs.get("temperature") or 300.0
        self.number_of_snapshots: int = kwargs.get("number_of_snapshots") or 1000
        self.max_iterations: int = kwargs.get("max_iterations") or 10
        mesh = kwargs.get("mesh")
        self.mesh: float | list[int] = 100.0 if mesh is None else mesh
        self.random_seed: int | None = kwargs.get("random_seed")
        self.fc_calculator = "symfc"
        self.distance = 0.01
        self.initial_force_constants_provided = True
        self.supercell_energy = -1234.5

    def __iter__(self) -> Iterator[int]:
        # One iteration, so that _run_MLPSSCHA reaches its final assignment.
        return iter([0])

    @property
    def phonopy(self) -> Phonopy:
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
    # Only passed through by _run_MLPSSCHA, never used as an MLP here.
    phonon.mlp = cast(PhonopyMLP, object())
    phonopy_script._run_MLPSSCHA(phonon, settings, 0)

    assert _FakeSSCHA.last_kwargs["random_seed"] == 987
    assert _FakeSSCHA.last_kwargs["number_of_snapshots"] == 123
    assert _FakeSSCHA.last_kwargs["max_iterations"] == 4
    assert _FakeSSCHA.last_kwargs["temperature"] == 250.0


def test_run_MLPSSCHA_passes_mesh(
    ph_kcl: Phonopy, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command-line mesh reaches MLPSSCHA.

    The free energy is sampled on it, so leaving it behind silently reports a
    value computed on a mesh the user did not ask for.

    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phonopy_script, "MLPSSCHA", _FakeSSCHA)

    settings = PhonopySettings()
    settings.use_pypolymlp = True
    settings.sscha_iterations = 1
    settings.mesh_numbers = [20, 20, 20]

    phonon = ph_kcl
    # Only passed through by _run_MLPSSCHA, never used as an MLP here.
    phonon.mlp = cast(PhonopyMLP, object())
    phonopy_script._run_MLPSSCHA(phonon, settings, 0)

    assert _FakeSSCHA.last_kwargs["mesh"] == [20, 20, 20]


def test_run_MLPSSCHA_writes_yaml_without_log(
    ph_kcl: Phonopy, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free energies are saved even when nothing is printed.

    They used to exist only as a line of log, so a quiet run left no record of
    them at all.

    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phonopy_script, "MLPSSCHA", _FakeSSCHA)

    settings = PhonopySettings()
    settings.use_pypolymlp = True
    settings.sscha_iterations = 1

    phonon = ph_kcl
    # Only passed through by _run_MLPSSCHA, never used as an MLP here.
    phonon.mlp = cast(PhonopyMLP, object())
    phonopy_script._run_MLPSSCHA(phonon, settings, 0)

    with open(tmp_path / "sscha_free_energies.yaml") as f:
        data = yaml.safe_load(f)
    assert data["sscha"]["max_iterations"] == 1
    assert data["iterations"] == []
