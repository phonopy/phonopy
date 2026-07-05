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


def test_cli_sample_unitcells(tmp_path, monkeypatch) -> None:
    """Ranges produce the requested number of strained unit cells."""
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
            "-n",
            "4",
            "--seed",
            "0",
        ],
    )

    run()

    files = sorted(tmp_path.glob("unitcell-*"))
    assert len(files) == 4


def test_cli_sample_rd_supercells(tmp_path, monkeypatch) -> None:
    """--rd produces random-displacement supercells instead of unit cells."""
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
            "-n",
            "3",
            "--seed",
            "0",
            "--rd",
            "0.1",
        ],
    )

    run()

    assert len(sorted(tmp_path.glob("supercell-*"))) == 3
    assert not sorted(tmp_path.glob("unitcell-*"))


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
