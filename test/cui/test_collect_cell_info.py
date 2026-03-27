"""Tests for collect_cell_info."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from phonopy.cui.collect_cell_info import collect_cell_info, get_cell_info
from phonopy.cui.settings import Settings
from phonopy.exception import CellNotFoundError, MagmomValueError

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


def _settings(
    supercell_matrix: list[list[int]] | None = None,
    primitive_matrix: str | list[list[float]] | None = None,
    magnetic_moments: list[float] | None = None,
) -> Settings:
    settings = Settings()
    settings.supercell_matrix = (
        None
        if supercell_matrix is None
        else np.array(supercell_matrix, dtype="int64", order="C")
    )
    settings.primitive_matrix = primitive_matrix
    settings.calculator = None
    settings.chemical_symbols = None
    settings.magnetic_moments = magnetic_moments
    return settings


def test_get_cell_info_poscar_success(monkeypatch, tmp_path):
    """Read unit cell from POSCAR and return expected values."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(supercell_matrix=_supercell_matrix)

    result = get_cell_info(settings=settings, cell_filename=_poscar_nacl)

    assert result.unitcell is not None
    assert len(result.unitcell) == 8
    np.testing.assert_array_equal(result.supercell_matrix, _supercell_matrix)
    assert result.phonopy_yaml is None


def test_get_cell_info_enforce_primitive_matrix_auto(monkeypatch, tmp_path):
    """enforce_primitive_matrix_auto=True forces primitive_matrix='auto'."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(
        supercell_matrix=_supercell_matrix,
        primitive_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    result = get_cell_info(
        settings=settings,
        cell_filename=_poscar_nacl,
        enforce_primitive_matrix_auto=True,
    )

    assert result.primitive_matrix == "auto"


def test_get_cell_info_invalid_magnetic_moments_raises(monkeypatch, tmp_path):
    """Invalid MAGMOM length raises MagmomValueError."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(
        supercell_matrix=_supercell_matrix,
        magnetic_moments=[1.0, -1.0],
    )

    with pytest.raises(MagmomValueError):
        get_cell_info(settings=settings, cell_filename=_poscar_nacl)


def test_get_cell_info_missing_file_raises(monkeypatch, tmp_path):
    """Missing input file raises CellNotFoundError."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(supercell_matrix=_supercell_matrix)
    missing = tmp_path / "missing_POSCAR"

    with pytest.raises(CellNotFoundError):
        get_cell_info(settings=settings, cell_filename=missing)


def test_get_cell_info_prints_primitive_overwrite_message(
    monkeypatch, tmp_path, capsys
):
    """Mismatched primitive matrix between YAML and settings is reported."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(supercell_matrix=_supercell_matrix, primitive_matrix="F")

    get_cell_info(
        settings=settings,
        cell_filename=_phonopy_yaml_nacl,
        load_phonopy_yaml=True,
        log_level=1,
    )

    captured = capsys.readouterr()
    assert "Primitive matrix is not specified" in captured.out
    assert "But it is overwritten by" in captured.out


def test_get_cell_info_no_primitive_overwrite_message_at_log_level_0(
    monkeypatch, tmp_path, capsys
):
    """Primitive overwrite message is suppressed at log_level=0."""
    monkeypatch.chdir(tmp_path)
    settings = _settings(supercell_matrix=_supercell_matrix, primitive_matrix="F")

    get_cell_info(
        settings=settings,
        cell_filename=_phonopy_yaml_nacl,
        load_phonopy_yaml=True,
        log_level=0,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
