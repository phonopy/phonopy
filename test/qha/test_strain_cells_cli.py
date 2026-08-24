# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the phonopy-strain-cells command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.scripts.phonopy_strain_cells import run
from phonopy.structure.atoms import PhonopyAtoms


def _write_disp_yaml(directory: Path) -> None:
    """Write a phonopy_disp.yaml for a tetragonal cell in the directory."""
    cell = PhonopyAtoms(
        symbols=["Cu"], cell=np.diag([4.0, 4.0, 6.0]), scaled_positions=[[0, 0, 0]]
    )
    phonon = Phonopy(cell, supercell_matrix=np.diag([2, 2, 2]), log_level=0)
    phonon.generate_displacements()
    phonon.save(directory / "phonopy_disp.yaml")


def test_cli_dof_display(tmp_path, monkeypatch, capsys) -> None:
    """Without ranges the command prints the free lattice DOF."""
    _write_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["phonopy-strain-cells", "phonopy_disp.yaml"])

    run()

    out = capsys.readouterr().out
    assert "tetragonal" in out
    assert "Free lattice parameter(s): a, c" in out
    # Reference strains and the spanned cell volume are shown.
    for percent in ("+/-1%", "+/-2%", "+/-3%"):
        assert percent in out
    assert "volume" in out
    # +/-2% keeps the previous bracket (0.98 / 1.02 of a = 4.0).
    assert "--a 3.9200 4.0800" in out


def test_cli_grid_sampling(tmp_path, monkeypatch, capsys) -> None:
    """--grid writes a tensor grid and records a deterministic (seedless) run."""
    yaml = pytest.importorskip("yaml")
    _write_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-strain-cells",
            "phonopy_disp.yaml",
            "--a",
            "3.92",
            "4.08",
            "--c",
            "5.88",
            "6.12",
            "--grid",
            "5",
        ],
    )

    run()

    out = capsys.readouterr().out
    files = sorted(tmp_path.glob("unitcell-*"))
    assert len(files) == 25  # 5 x 5
    assert "Grid sampling: 5 x 5" in out
    # The selected volume path is shown, with the c/a shape column.
    assert "Main diagonal (5 cells)" in out
    assert "c/a" in out

    manifest = yaml.safe_load((tmp_path / "strain_cells.yaml").read_text())
    assert manifest["parameters"]["grid_shape"] == [5, 5]
    assert manifest["output"]["num_cells"] == 25


def test_cli_grid_rectangular(tmp_path, monkeypatch, capsys) -> None:
    """--grid with one value per free DOF makes a rectangular grid."""
    yaml = pytest.importorskip("yaml")
    _write_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-strain-cells",
            "phonopy_disp.yaml",
            "--a",
            "3.92",
            "4.08",
            "--c",
            "5.88",
            "6.12",
            "--grid",
            "5",
            "6",
        ],
    )

    run()

    out = capsys.readouterr().out
    assert len(sorted(tmp_path.glob("unitcell-*"))) == 30  # 5 x 6
    assert "Grid sampling: 5 x 6" in out
    # The diagonal is min(5, 6) = 5 cells; the path is shown either way.
    assert "Main diagonal (5 cells)" in out
    manifest = yaml.safe_load((tmp_path / "strain_cells.yaml").read_text())
    assert manifest["parameters"]["grid_shape"] == [5, 6]
    assert manifest["free_dof"] == ["a", "c"]
    assert manifest["parameters"]["ranges"] == {"a": [3.92, 4.08], "c": [5.88, 6.12]}
    assert manifest["output"]["num_cells"] == 30


def test_cli_requires_a_grid(tmp_path, monkeypatch) -> None:
    """Ranges without --grid stop the command rather than sampling somehow."""
    _write_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-strain-cells",
            "phonopy_disp.yaml",
            "--a",
            "3.9",
            "4.1",
            "--c",
            "5.8",
            "6.2",
        ],
    )
    with pytest.raises(SystemExit):
        run()


def test_cli_rejects_non_free_parameter(tmp_path, monkeypatch) -> None:
    """Giving a range for a tied parameter exits with an error."""
    _write_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-strain-cells",
            "phonopy_disp.yaml",
            "--a",
            "3.9",
            "4.1",
            "--b",
            "3.9",
            "4.1",
            "--c",
            "5.8",
            "6.2",
        ],
    )

    with pytest.raises(SystemExit):
        run()
