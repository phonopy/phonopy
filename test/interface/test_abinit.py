"""Tests of Abinit calculator interface."""

import io
import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.interface.abinit import (
    get_abinit_structure,
    read_abinit,
    write_abinit,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


def test_read_abinit() -> None:
    """Test of read_abinit."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    filename = cwd / "NaCl-abinit-pwscf.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_get_abinit_structure() -> None:
    """Round-trip: read → get_abinit_structure → read back preserves cell."""
    cell_ref = read_abinit(cwd / "NaCl-abinit.in")
    cell = read_abinit(io.StringIO(get_abinit_structure(cell_ref)))
    assert isclose(cell_ref, cell)


# ---------------------------------------------------------------------------
# Regression tests (written before refactoring to capture current behaviour)
# ---------------------------------------------------------------------------


# ── get_abinit_structure content ──────────────────────────────────────────


def test_get_abinit_structure_content() -> None:
    """Key tokens must be present in the output text."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    text = get_abinit_structure(cell)

    assert "natom 8" in text
    assert "ntypat 2" in text
    assert "znucl 11 17" in text
    assert "acell 1 1 1" in text
    assert "rprim" in text
    assert "xred" in text
    # xcart / xangst must not appear (always writes xred)
    assert "xcart" not in text
    assert "xangst" not in text


def test_get_abinit_structure_lattice_vectors() -> None:
    """Rprim block must match cell.cell values."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    text = get_abinit_structure(cell)
    lines = text.splitlines()
    idx = lines.index("rprim")
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(3)]
    )
    # get_abinit_structure writes cell rows as rprim rows
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_abinit_structure_positions() -> None:
    """Xred block must match cell.scaled_positions."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    text = get_abinit_structure(cell)
    lines = text.splitlines()
    idx = lines.index("xred")
    natom = len(cell)
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(natom)]
    )
    np.testing.assert_allclose(parsed, cell.scaled_positions, atol=1e-10)


# ── write_abinit ──────────────────────────────────────────────────────────


def test_write_abinit_roundtrip() -> None:
    """write_abinit then read_abinit must reproduce the original cell."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "out.in"
        write_abinit(fpath, cell)
        cell2 = read_abinit(fpath)

    assert isclose(cell, cell2)


# ── write_supercells_with_displacements ───────────────────────────────────


def test_write_supercells_filenames() -> None:
    """supercell.in and supercell-001.in, … must be created."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    ids = np.array([1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
        assert pathlib.Path(tmpdir, "supercell.in").exists()
        assert pathlib.Path(tmpdir, "supercell-001.in").exists()
        assert pathlib.Path(tmpdir, "supercell-002.in").exists()


def test_write_supercells_custom_prefix() -> None:
    """Custom pre_filename is used for all output files."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "disp")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "disp.in").exists()
        assert pathlib.Path(tmpdir, "disp-001.in").exists()


def test_write_supercells_content_readable() -> None:
    """Written supercell file must be parseable by read_abinit."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        cell2 = read_abinit(pathlib.Path(tmpdir, "supercell.in"))
    assert isclose(cell, cell2)


# ── AbinitIn parsing corner cases ────────────────────────────────────────

_BASE = """\
natom 1
ntypat 1
znucl 11
typat 1
rprim
1 0 0
0 1 0
0 0 1
"""


def test_abinit_in_acell_repeat_syntax() -> None:
    """Acell with Fortran repeat syntax (3*value) is parsed correctly."""
    src = _BASE + "acell 3*10.0\nxred\n0 0 0\n"
    cell = read_abinit(io.StringIO(src))
    np.testing.assert_allclose(np.diag(cell.cell), [10.0, 10.0, 10.0], atol=1e-10)


def test_abinit_in_acell_angstrom() -> None:
    """Acell with Angstrom keyword is converted to Bohr-based lattice correctly."""
    bohr = get_physical_units().Bohr
    src = _BASE + f"acell 3*{bohr} Angstrom\nxred\n0 0 0\n"
    cell = read_abinit(io.StringIO(src))
    # acell in Angstrom → divided by Bohr → rprim(identity) → cell in Bohr → but
    # phonopy stores in Angstrom? No: cell is stored as lattice * acell where
    # acell is already in Bohr. The result is the identity * 1 Angstrom.
    np.testing.assert_allclose(np.diag(cell.cell), [1.0, 1.0, 1.0], atol=1e-10)


def test_abinit_in_xcart() -> None:
    """Xcart (Cartesian in Bohr) is converted to reduced coordinates."""
    src = _BASE + "acell 3*10.0\nxcart\n5.0 0.0 0.0\n"
    cell = read_abinit(io.StringIO(src))
    np.testing.assert_allclose(cell.scaled_positions[0], [0.5, 0.0, 0.0], atol=1e-10)


def test_abinit_in_xangst() -> None:
    """Xangst (Cartesian in Angstrom) is converted to reduced coordinates."""
    bohr = get_physical_units().Bohr
    src = _BASE + f"acell 3*10.0\nxangst\n{5.0 * bohr} 0.0 0.0\n"
    cell = read_abinit(io.StringIO(src))
    np.testing.assert_allclose(cell.scaled_positions[0], [0.5, 0.0, 0.0], atol=1e-10)


def test_abinit_in_scalecart() -> None:
    """Scalecart scales each Cartesian axis of the lattice independently."""
    src = _BASE + "acell 3*1.0\nscalecart 2.0 3.0 4.0\nxred\n0 0 0\n"
    cell = read_abinit(io.StringIO(src))
    np.testing.assert_allclose(np.diag(cell.cell), [2.0, 3.0, 4.0], atol=1e-10)


def test_abinit_in_comment_stripping() -> None:
    """Lines with # and ! comments are ignored."""
    src = (
        "natom 1  # one atom\n"
        "ntypat 1 ! one type\n"
        "znucl 11\ntypat 1\n"
        "acell 3*5.0\n"
        "rprim\n1 0 0\n0 1 0\n0 0 1\n"
        "xred\n0 0 0\n"
    )
    cell = read_abinit(io.StringIO(src))
    assert len(cell) == 1
    assert cell.symbols[0] == "Na"


@pytest.mark.parametrize(
    "typat_str,expected",
    [
        ("1 1 1 1 2 2 2 2", ["Na"] * 4 + ["Cl"] * 4),
        ("4*1 4*2", ["Na"] * 4 + ["Cl"] * 4),
    ],
)
def test_abinit_in_typat_repeat_syntax(typat_str: str, expected: list[str]) -> None:
    """Typat with repeat syntax gives the same result as explicit listing."""
    src = (
        f"natom 8\nntypat 2\nznucl 11 17\ntypat {typat_str}\n"
        "acell 3*10.0\nrprim\n1 0 0\n0 1 0\n0 0 1\n"
        "xred\n" + "0 0 0\n" * 4 + "0.5 0.5 0.5\n" * 4
    )
    cell = read_abinit(io.StringIO(src))
    assert list(cell.symbols) == expected
