"""Tests for collect_cell_info."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from phonopy.cui.collect_cell_info import collect_cell_info
from phonopy.exception import CellNotFoundError

_test_dir = pathlib.Path(__file__).parent.parent
_poscar_nacl = _test_dir / "POSCAR_NaCl"
# phonopy_NaCl_unitcell1.yaml has unit_cell but no supercell_matrix.
_phonopy_yaml_nacl = _test_dir / "phonopy_NaCl_unitcell1.yaml"

_supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


# ---------------------------------------------------------------------------
# Normal success cases
# ---------------------------------------------------------------------------


def test_poscar_with_supercell_matrix(monkeypatch, tmp_path):
    """Read unit cell from POSCAR with explicit supercell_matrix."""
    monkeypatch.chdir(tmp_path)
    result = collect_cell_info(
        supercell_matrix=_supercell_matrix,
        cell_filename=_poscar_nacl,
    )
    assert result.unitcell is not None
    assert len(result.unitcell) == 8  # NaCl has 8 atoms
    np.testing.assert_array_equal(result.supercell_matrix, _supercell_matrix)
    assert result.phonopy_yaml is None


def test_phonopy_yaml_with_supercell_matrix(monkeypatch, tmp_path):
    """Read unit cell from phonopy.yaml with explicit supercell_matrix."""
    monkeypatch.chdir(tmp_path)
    result = collect_cell_info(
        supercell_matrix=_supercell_matrix,
        cell_filename=_phonopy_yaml_nacl,
        load_phonopy_yaml=True,
    )
    assert result.unitcell is not None
    assert len(result.unitcell) == 8
    assert result.phonopy_yaml is not None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_fallback_no_poscar_no_dim(monkeypatch, tmp_path):
    """No POSCAR and no supercell_matrix: error message mentions supercell_matrix."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CellNotFoundError) as exc_info:
        collect_cell_info()
    msg = str(exc_info.value)
    assert "Supercell matrix" in msg or "DIM" in msg


def test_explicit_missing_file_vasp(monkeypatch, tmp_path):
    """Explicit cell_filename that does not exist raises CellNotFoundError."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "POSCAR_missing"
    with pytest.raises(CellNotFoundError) as exc_info:
        collect_cell_info(
            supercell_matrix=_supercell_matrix,
            cell_filename=missing,
        )
    assert "was not found" in str(exc_info.value)


def test_non_vasp_interface_missing_file(monkeypatch, tmp_path):
    """Non-VASP interface with missing file raises CellNotFoundError."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.pw.in"
    with pytest.raises(CellNotFoundError) as exc_info:
        collect_cell_info(
            supercell_matrix=_supercell_matrix,
            interface_mode="qe",
            cell_filename=missing,
        )
    assert "was not found" in str(exc_info.value)
