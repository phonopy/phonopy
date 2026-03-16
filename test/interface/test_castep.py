"""Tests for CASTEP calculator interface."""

import pathlib

import numpy as np
import pytest

from phonopy.interface.castep import (
    CastepIn,
    get_castep_structure,
    read_castep,
    write_castep,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


def test_read_castep() -> None:
    """Test read CASTEP file."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    cell_ref = read_cell_yaml(cwd / "NaCl-castep.yaml")
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_read_castep_no_magnetic_moments() -> None:
    """NaCl test file has no spin → magnetic_moments is None."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    assert cell.magnetic_moments is None


# ---------------------------------------------------------------------------
# get_castep_structure content tests
# ---------------------------------------------------------------------------


def test_get_castep_structure_keywords() -> None:
    """Required block keywords must appear in the output."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    text = get_castep_structure(cell)

    assert "%BLOCK LATTICE_CART" in text
    assert "%ENDBLOCK LATTICE_CART" in text
    assert "%BLOCK POSITIONS_FRAC" in text
    assert "%ENDBLOCK POSITIONS_FRAC" in text


def test_get_castep_structure_lattice_vectors() -> None:
    """Lattice vector block must match cell.cell rows."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    text = get_castep_structure(cell)
    lines = text.splitlines()
    idx = lines.index("%BLOCK LATTICE_CART")
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(3)]
    )
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_castep_structure_atom_count() -> None:
    """Number of position lines must equal the number of atoms."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    text = get_castep_structure(cell)
    lines = text.splitlines()
    start = lines.index("%BLOCK POSITIONS_FRAC") + 1
    end = lines.index("%ENDBLOCK POSITIONS_FRAC")
    pos_lines = [ln for ln in lines[start:end] if ln.strip()]
    assert len(pos_lines) == len(cell)


def test_get_castep_structure_no_spin_without_magmoms() -> None:
    """spin= keyword must not appear when no magnetic moments are set."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    text = get_castep_structure(cell)
    assert "spin" not in text.lower()


def test_get_castep_structure_spin_with_magmoms() -> None:
    """spin= keyword appears for atoms with non-zero magnetic moment."""
    cell = PhonopyAtoms(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        symbols=["Fe", "Fe"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[2.0, -2.0],
    )
    text = get_castep_structure(cell)
    assert text.count("spin=") == 2


def test_get_castep_structure_zero_magmom_no_spin() -> None:
    """Atoms with magnetic_moment == 0.0 do not get a spin= tag."""
    cell = PhonopyAtoms(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        symbols=["Fe", "Fe"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[0.0, 0.0],
    )
    text = get_castep_structure(cell)
    assert "spin" not in text.lower()


# ---------------------------------------------------------------------------
# write_castep round-trip
# ---------------------------------------------------------------------------


def test_write_castep_roundtrip(tmp_path: pathlib.Path) -> None:
    """write_castep then read_castep must reproduce the original cell."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    fpath = tmp_path / "out.cell"
    write_castep(str(fpath), cell)
    cell2 = read_castep(str(fpath))

    assert isclose(cell, cell2)
    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-10)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == list(cell2.symbols)


def test_write_castep_roundtrip_with_magmoms(tmp_path: pathlib.Path) -> None:
    """Round-trip preserves magnetic moments."""
    cell = PhonopyAtoms(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        symbols=["Fe", "Fe"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[2.0, -2.0],
    )
    fpath = tmp_path / "mag.cell"
    write_castep(str(fpath), cell)
    cell2 = read_castep(str(fpath))

    assert cell2.magnetic_moments is not None
    np.testing.assert_allclose(
        np.ravel(cell.magnetic_moments), np.ravel(cell2.magnetic_moments), atol=1e-2
    )


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """supercell.cell and supercell-001.cell, … must be created."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    ids = np.array([1, 2])
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
    assert (tmp_path / "supercell.cell").exists()
    assert (tmp_path / "supercell-001.cell").exists()
    assert (tmp_path / "supercell-002.cell").exists()


def test_write_supercells_custom_prefix(tmp_path: pathlib.Path) -> None:
    """Custom pre_filename is used for all output files."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    pre = str(tmp_path / "disp")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    assert (tmp_path / "disp.cell").exists()
    assert (tmp_path / "disp-001.cell").exists()


def test_write_supercells_content_readable(tmp_path: pathlib.Path) -> None:
    """Written supercell file must be parseable by read_castep."""
    cell = read_castep(cwd / "NaCl-castep.cell")
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    cell2 = read_castep(str(tmp_path / "supercell.cell"))
    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# CastepIn parsing corner cases
# ---------------------------------------------------------------------------

_BASE_CELL = """\
%BLOCK LATTICE_CART
 5.0  0.0  0.0
 0.0  5.0  0.0
 0.0  0.0  5.0
%ENDBLOCK LATTICE_CART
%BLOCK POSITIONS_FRAC
"""


def test_castep_in_basic_parse() -> None:
    """Basic two-atom cell is parsed correctly."""
    src = (
        _BASE_CELL + "Na  0.0  0.0  0.0\nCl  0.5  0.5  0.5\n%ENDBLOCK POSITIONS_FRAC\n"
    )
    tags = CastepIn(src.splitlines(keepends=True)).get_tags()
    assert tags["atomic_species"] == ["Na", "Cl"]
    np.testing.assert_allclose(tags["coordinates"][0], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(tags["coordinates"][1], [0.5, 0.5, 0.5], atol=1e-10)
    assert tags["magnetic_moments"] is None


def test_castep_in_bohr_units() -> None:
    """BOHR keyword in LATTICE_CART triggers unit conversion."""
    src = (
        "%BLOCK LATTICE_CART\n"
        " BOHR\n"
        " 1.0  0.0  0.0\n"
        " 0.0  1.0  0.0\n"
        " 0.0  0.0  1.0\n"
        "%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\n"
        "Na  0.0  0.0  0.0\n"
        "%ENDBLOCK POSITIONS_FRAC\n"
    )
    tags = CastepIn(src.splitlines(keepends=True)).get_tags()
    # 1 Bohr = 0.529177211 Å
    np.testing.assert_allclose(
        np.diag(tags["lattice_vectors"]), [0.529177211] * 3, atol=1e-6
    )


def test_castep_in_spin_polarized() -> None:
    """Atoms with spin= tag populate magnetic_moments list."""
    src = (
        _BASE_CELL
        + "Fe  0.0  0.0  0.0  spin= 2.00\n"
        + "Fe  0.5  0.5  0.5  spin=-2.00\n"
        + "%ENDBLOCK POSITIONS_FRAC\n"
    )
    tags = CastepIn(src.splitlines(keepends=True)).get_tags()
    assert tags["magnetic_moments"] is not None
    assert len(tags["magnetic_moments"]) == 2
    assert tags["magnetic_moments"][0] == pytest.approx(2.0, abs=1e-2)
    assert tags["magnetic_moments"][1] == pytest.approx(-2.0, abs=1e-2)


def test_castep_in_partial_spin_sets_zero() -> None:
    """If only some atoms have spin=, others get 0.0 in magnetic_moments."""
    src = (
        _BASE_CELL
        + "Fe  0.0  0.0  0.0  spin= 2.00\n"
        + "Fe  0.5  0.5  0.5\n"
        + "%ENDBLOCK POSITIONS_FRAC\n"
    )
    tags = CastepIn(src.splitlines(keepends=True)).get_tags()
    assert tags["magnetic_moments"] is not None
    assert tags["magnetic_moments"][0] == pytest.approx(2.0, abs=1e-2)
    assert tags["magnetic_moments"][1] == pytest.approx(0.0, abs=1e-10)
