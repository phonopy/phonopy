"""Tests for CRYSTAL calculator interface."""

import pathlib

import numpy as np
import pytest

from phonopy.interface.crystal import (
    get_crystal_structure,
    read_crystal,
    write_crystal,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

cwd = pathlib.Path(__file__).parent


def test_read_crystal() -> None:
    """Test of read_crystal."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    cell_ref = read_cell_yaml(cwd / "Si-CRYSTAL.yaml")
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_read_crystal_conv_numbers() -> None:
    """conv_numbers must be a list of integers matching the number of atoms."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    assert conv_numbers is not None
    assert len(conv_numbers) == len(cell)
    assert all(isinstance(n, int) for n in conv_numbers)


def test_read_crystal_no_magnetic_moments() -> None:
    """Si CRYSTAL output has no ATOMSPIN → magnetic_moments is None."""
    cell, _ = read_crystal(cwd / "Si-CRYSTAL.o")
    assert cell.magnetic_moments is None


# ---------------------------------------------------------------------------
# get_crystal_structure content tests
# ---------------------------------------------------------------------------


def _make_simple_cell() -> tuple[PhonopyAtoms, list[int]]:
    """Return a simple 2-atom Si-like cell with conv_numbers."""
    a = 3.84
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        symbols=["Si", "Si"],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    conv_numbers = [14, 14]
    return cell, conv_numbers


def test_get_crystal_structure_header() -> None:
    """First line must be '3 1 1' (dimensionality / centring / crystal type)."""
    cell, conv_numbers = _make_simple_cell()
    text = get_crystal_structure(cell, conv_numbers)
    assert text.splitlines()[0].strip() == "3 1 1"


def test_get_crystal_structure_lattice_vectors() -> None:
    """Three lattice vector lines must follow the header line."""
    cell, conv_numbers = _make_simple_cell()
    text = get_crystal_structure(cell, conv_numbers)
    lines = text.splitlines()
    parsed = np.array([[float(x) for x in lines[i + 1].split()] for i in range(3)])
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-8)


def test_get_crystal_structure_atom_count_line() -> None:
    """The atom-count line must equal len(cell)."""
    cell, conv_numbers = _make_simple_cell()
    text = get_crystal_structure(cell, conv_numbers)
    lines = text.splitlines()
    # After header (1) + lattice (3) + symmetry block (4, identity) → atom count line
    # identity block: "1\n  1.00  0.00 ...\n" * 3 rows + translation → 4 lines
    idx_nsym = 4  # line index of the symm-op count ("1")
    natom_line_idx = idx_nsym + 1 + 4  # skip "1" + 4 symm-op lines
    assert int(lines[natom_line_idx]) == len(cell)


def test_get_crystal_structure_conv_numbers_in_output() -> None:
    """Each atom line starts with its conventional atomic number."""
    cell, conv_numbers = _make_simple_cell()
    text = get_crystal_structure(cell, conv_numbers)
    lines = text.splitlines()
    # Find the first atom line: after natom line
    idx_nsym = 4
    natom_line_idx = idx_nsym + 1 + 4
    for _i, (cn, line) in enumerate(
        zip(conv_numbers, lines[natom_line_idx + 1 :], strict=True)
    ):
        assert int(line.split()[0]) == cn


# ---------------------------------------------------------------------------
# write_crystal file creation tests
# ---------------------------------------------------------------------------


def test_write_crystal_creates_ext_and_d12(tmp_path: pathlib.Path) -> None:
    """write_crystal must create both .ext and .d12 files."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    pre = str(tmp_path / "supercell")
    write_crystal(pre, cell, conv_numbers)
    assert (tmp_path / "supercell.ext").exists()
    assert (tmp_path / "supercell.d12").exists()


def test_write_crystal_d12_keywords(tmp_path: pathlib.Path) -> None:
    """The .d12 file must contain required CRYSTAL input keywords."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    pre = str(tmp_path / "sc")
    write_crystal(pre, cell, conv_numbers)
    text = (tmp_path / "sc.d12").read_text()
    assert "EXTERNAL" in text
    assert "ENDGEOM" in text
    assert "GRADCAL" in text
    assert "END" in text


def test_write_crystal_no_conv_numbers_warns(tmp_path: pathlib.Path) -> None:
    """Passing conv_numbers=None must raise a UserWarning."""
    cell, _ = read_crystal(cwd / "Si-CRYSTAL.o")
    pre = str(tmp_path / "sc")
    with pytest.warns(UserWarning, match="conventional atomic numbers"):
        write_crystal(pre, cell, None)
    assert (tmp_path / "sc.ext").exists()


def test_write_crystal_conv_numbers_length_mismatch_raises(
    tmp_path: pathlib.Path,
) -> None:
    """Mismatched conv_numbers length must raise ValueError."""
    cell, _ = read_crystal(cwd / "Si-CRYSTAL.o")
    pre = str(tmp_path / "sc")
    with pytest.raises(ValueError, match="conv_numbers"):
        write_crystal(pre, cell, [14])  # Si has 2 atoms, only 1 number given


def test_write_crystal_atomspin_in_d12(tmp_path: pathlib.Path) -> None:
    """Magnetic moments must produce ATOMSPIN block in .d12."""
    cell = PhonopyAtoms(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        symbols=["Fe", "Fe"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=[1, -1],
    )
    conv_numbers = [26, 26]
    pre = str(tmp_path / "mag")
    write_crystal(pre, cell, conv_numbers)
    text = (tmp_path / "mag.d12").read_text()
    assert "ATOMSPIN" in text


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """supercell.ext/.d12 and supercell-001.ext/.d12 must be created."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    ids = np.array([1, 2])
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(
        cell,
        [cell, cell],
        ids,
        conv_numbers,
        num_unitcells_in_supercell=1,
        pre_filename=pre,
    )
    assert (tmp_path / "supercell.ext").exists()
    assert (tmp_path / "supercell.d12").exists()
    assert (tmp_path / "supercell-001.ext").exists()
    assert (tmp_path / "supercell-001.d12").exists()
    assert (tmp_path / "supercell-002.ext").exists()
    assert (tmp_path / "supercell-002.d12").exists()


def test_write_supercells_content_has_correct_natom(tmp_path: pathlib.Path) -> None:
    """The .ext supercell file must contain the correct number of atoms."""
    cell, conv_numbers = read_crystal(cwd / "Si-CRYSTAL.o")
    pre = str(tmp_path / "sc")
    write_supercells_with_displacements(
        cell,
        [cell],
        np.array([1]),
        conv_numbers,
        num_unitcells_in_supercell=1,
        pre_filename=pre,
    )
    text = (tmp_path / "sc.ext").read_text()
    lines = text.splitlines()
    # header=1 line, lattice=3 lines, nsym line at idx 4, symm block=4 lines,
    # then natom line
    natom_line_idx = 4 + 1 + 4
    assert int(lines[natom_line_idx]) == len(cell)
