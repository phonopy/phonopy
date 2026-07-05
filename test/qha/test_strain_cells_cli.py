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


def test_cli_writes_manifest(tmp_path, monkeypatch, capsys) -> None:
    """Sampling writes a provenance manifest recording a resolved seed."""
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
            "3.9",
            "4.1",
            "--c",
            "5.8",
            "6.2",
            "-n",
            "5",
        ],
    )

    run()

    out = capsys.readouterr().out
    assert "Random seed:" in out
    manifest = yaml.safe_load((tmp_path / "strain_cells.yaml").read_text())
    # No --seed was given, so a concrete integer seed must have been recorded.
    assert isinstance(manifest["parameters"]["seed"], int)
    assert manifest["free_dof"] == ["a", "c"]
    assert manifest["parameters"]["ranges"] == {"a": [3.9, 4.1], "c": [5.8, 6.2]}
    assert manifest["parameters"]["num"] == 5
    cells = manifest["output"]["cells"]
    assert manifest["output"]["num_cells"] == 5
    assert len(cells) == 5
    assert all("a" in entry and "c" in entry for entry in cells)


def test_cli_manifest_seed_is_reproducible(tmp_path, monkeypatch) -> None:
    """Replaying the recorded seed reproduces the sampled cells."""
    yaml = pytest.importorskip("yaml")
    argv = [
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
    ]

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    for directory in (run1, run2):
        directory.mkdir()
        _write_disp_yaml(directory)

    monkeypatch.chdir(run1)
    monkeypatch.setattr(sys, "argv", argv)
    run()
    manifest1 = yaml.safe_load((run1 / "strain_cells.yaml").read_text())
    seed = manifest1["parameters"]["seed"]

    monkeypatch.chdir(run2)
    monkeypatch.setattr(sys, "argv", argv + ["--seed", str(seed)])
    run()
    manifest2 = yaml.safe_load((run2 / "strain_cells.yaml").read_text())

    assert manifest2["parameters"]["seed"] == seed
    assert manifest1["output"]["cells"] == manifest2["output"]["cells"]


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
