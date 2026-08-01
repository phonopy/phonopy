# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the yaml output of SSCHA calculations."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any, cast

import yaml

from phonopy.sscha.core import MLPSSCHA, SSCHAIterationResult
from phonopy.sscha.output import write_sscha_yaml


class _StubSSCHA:
    """Settings and history of an MLPSSCHA run, without running one.

    The writer reads only these, so a stub keeps the test free of pypolymlp
    and of the sampling cost.

    """

    def __init__(
        self,
        history: tuple[SSCHAIterationResult, ...] = (),
        initial_force_constants_provided: bool = True,
    ) -> None:
        self.history = history
        self.initial_force_constants_provided = initial_force_constants_provided
        self.temperature = 300.0
        self.number_of_snapshots = 1000
        self.max_iterations = 10
        self.mesh = [4, 4, 4]
        self.random_seed = 42
        self.fc_calculator = "symfc"
        self.distance = 0.01


def _history() -> tuple[SSCHAIterationResult, ...]:
    return (
        SSCHAIterationResult(
            iteration=1,
            free_energy=-0.098107,
            free_energy_error=8.9e-5,
            harmonic=-0.098446,
            anharmonic=0.000339,
        ),
        SSCHAIterationResult(
            iteration=2,
            free_energy=-0.098193,
            free_energy_error=6.9e-5,
            harmonic=-0.098531,
            anharmonic=0.000338,
        ),
    )


def _write(
    tmp_path: pathlib.Path,
    sscha: _StubSSCHA,
    filenames: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    filename = tmp_path / "phonopy_sscha.yaml"
    # The writer reads settings and history only, so the stub stands in for an
    # MLPSSCHA instance here.
    written = write_sscha_yaml(cast(MLPSSCHA, sscha), filenames, filename=filename)
    assert written == str(filename)
    with open(filename) as f:
        return yaml.safe_load(f)


def test_write_sscha_yaml_settings(tmp_path: pathlib.Path) -> None:
    """The settings that determine the free energies are recorded."""
    data = _write(tmp_path, _StubSSCHA(history=_history()))

    assert data["sscha"] == {
        "temperature": 300.0,
        "number_of_snapshots": 1000,
        "max_iterations": 10,
        "mesh": [4, 4, 4],
        "random_seed": 42,
        "fc_calculator": "symfc",
        "initial_force_constants": "provided",
    }
    assert data["free_energy_unit"] == "meV"
    assert "phonopy_version" in data


def test_write_sscha_yaml_displacement_distance(tmp_path: pathlib.Path) -> None:
    """The initialization distance is recorded only when it is used."""
    data = _write(
        tmp_path,
        _StubSSCHA(history=_history(), initial_force_constants_provided=False),
    )

    assert data["sscha"]["initial_force_constants"] == "generated"
    assert data["sscha"]["distance"] == 0.01


def test_write_sscha_yaml_free_energies(tmp_path: pathlib.Path) -> None:
    """Free energies are written per iteration, in meV per primitive cell."""
    data = _write(tmp_path, _StubSSCHA(history=_history()))

    assert [entry["iteration"] for entry in data["iterations"]] == [1, 2]
    entry = data["iterations"][0]
    assert entry["free_energy"] == -98.107
    assert entry["free_energy_error"] == 0.089
    assert entry["harmonic"] == -98.446
    assert entry["anharmonic"] == 0.339


def test_write_sscha_yaml_force_constants_filenames(tmp_path: pathlib.Path) -> None:
    """An iteration samples the force constants the previous one produced.

    The free energy belongs to the sampled ones, so the pairing has to be
    readable from the file rather than inferred from the iteration number.

    """
    filenames = {
        0: "phonopy_sscha_fc_0.yaml.xz",
        1: "phonopy_sscha_fc_1.yaml.xz",
        2: "phonopy_sscha_fc_2.yaml.xz",
    }
    data = _write(tmp_path, _StubSSCHA(history=_history()), filenames)

    first, second = data["iterations"]
    assert first["sampled_force_constants"] == "phonopy_sscha_fc_0.yaml.xz"
    assert first["produced_force_constants"] == "phonopy_sscha_fc_1.yaml.xz"
    assert second["sampled_force_constants"] == "phonopy_sscha_fc_1.yaml.xz"
    assert second["produced_force_constants"] == "phonopy_sscha_fc_2.yaml.xz"


def test_write_sscha_yaml_without_force_constants_filenames(
    tmp_path: pathlib.Path,
) -> None:
    """Force constants of the first iteration may come from no file of ours."""
    data = _write(tmp_path, _StubSSCHA(history=_history()))

    assert data["iterations"][0]["sampled_force_constants"] is None
    assert data["iterations"][0]["produced_force_constants"] is None


def test_write_sscha_yaml_before_any_iteration(tmp_path: pathlib.Path) -> None:
    """The file is valid after the initialization step, which has no value."""
    data = _write(tmp_path, _StubSSCHA())

    assert data["iterations"] == []
