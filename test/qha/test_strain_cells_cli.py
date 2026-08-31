# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the phonopy-strain-cells command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
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


def _write_fcc_disp_yaml(directory: Path) -> None:
    """Write a phonopy_disp.yaml for the FCC conventional cell of NaCl.

    The unit cell holds four primitive cells, which is the case where the two
    differ and phonopy-strain-cells writes both.

    """
    cell = PhonopyAtoms(
        symbols=["Na"] * 4 + ["Cl"] * 4,
        cell=np.diag([5.7, 5.7, 5.7]),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
        ],
    )
    phonon = Phonopy(
        cell,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix="auto",
        log_level=0,
    )
    phonon.generate_displacements()
    phonon.save(directory / "phonopy_disp.yaml")


def test_cli_writes_the_primitive_cell_of_a_centred_lattice(
    tmp_path, monkeypatch, capsys
) -> None:
    """A centred lattice gets both cells, and the atom order is the reference's.

    U may be computed on either cell, so both are written. The primitive cells
    take their atoms from phonopy_disp.yaml rather than from a fresh symmetry
    search, which keeps one atom order across the whole grid.

    """
    import phonopy

    yaml = pytest.importorskip("yaml")
    _write_fcc_disp_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phonopy-strain-cells",
            "phonopy_disp.yaml",
            "--a",
            "5.6",
            "5.8",
            "--grid",
            "3",
        ],
    )

    run()

    out = capsys.readouterr().out
    assert "primitive 2 atoms, unit cell 8 atoms" in out
    unitcells = sorted(tmp_path.glob("unitcell-*"))
    primcells = sorted(tmp_path.glob("primcell-*"))
    assert len(unitcells) == 3
    assert len(primcells) == 3

    ph = phonopy.load(tmp_path / "phonopy_disp.yaml", produce_fc=False, log_level=0)
    pmat = np.array(ph.primitive_matrix)
    for unitcell_path, primcell_path in zip(unitcells, primcells, strict=True):
        unitcell = read_vasp(unitcell_path)
        primcell = read_vasp(primcell_path)
        # The atoms are the reference primitive cell, untouched.
        assert np.array_equal(primcell.numbers, ph.primitive.numbers)
        np.testing.assert_allclose(
            primcell.scaled_positions, ph.primitive.scaled_positions, atol=1e-8
        )
        # Only the lattice follows the strain.
        np.testing.assert_allclose(
            np.array(primcell.cell), pmat.T @ np.array(unitcell.cell), atol=1e-8
        )
        assert unitcell.volume / primcell.volume == pytest.approx(4.0)

    manifest = yaml.safe_load((tmp_path / "strain_cells.yaml").read_text())
    assert [c["primitive_file"] for c in manifest["output"]["cells"]] == [
        "primcell-001",
        "primcell-002",
        "primcell-003",
    ]


def test_cli_writes_the_primitive_cell_of_an_unusual_setting(
    tmp_path, monkeypatch, capsys
) -> None:
    """A primitive matrix that only changes the setting still gets both cells.

    The test is on the primitive matrix, not on the atom counts: a primitive
    cell may be taken in an unusual setting and hold the same atoms as the
    unit cell. U then needs no conversion, but the cell is still a different
    cell and is written.

    """
    cell = PhonopyAtoms(
        symbols=["Cu"], cell=np.diag([4.0, 4.0, 6.0]), scaled_positions=[[0, 0, 0]]
    )
    phonon = Phonopy(
        cell,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix=[[0, 1, 0], [0, 0, 1], [1, 0, 0]],  # det = 1
        log_level=0,
    )
    phonon.generate_displacements()
    phonon.save(tmp_path / "phonopy_disp.yaml")
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
            "5.9",
            "6.1",
            "--grid",
            "3",
        ],
    )

    run()

    out = capsys.readouterr().out
    assert "primitive 1 atoms, unit cell 1 atoms" in out
    assert len(sorted(tmp_path.glob("primcell-*"))) == 9


def test_cli_writes_no_primitive_cell_for_a_primitive_lattice(
    tmp_path, monkeypatch, capsys
) -> None:
    """With one cell there is nothing to write twice."""
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
            "5.9",
            "6.1",
            "--grid",
            "3",
        ],
    )

    run()

    assert sorted(tmp_path.glob("primcell-*")) == []
    assert "primitive cell of each" not in capsys.readouterr().out
    manifest = yaml.safe_load((tmp_path / "strain_cells.yaml").read_text())
    assert "primitive_file" not in manifest["output"]["cells"][0]


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
