"""Tests for Elk calculator interface."""

import pathlib

import numpy as np
import pytest

from phonopy.interface.elk import (
    ElkIn,
    get_elk_structure,
    read_elk,
    write_elk,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# read_elk
# ---------------------------------------------------------------------------


def test_read_elk_natoms() -> None:
    """elk.in has 64 Si atoms."""
    cell, spfnames = read_elk(cwd / "elk.in")
    assert len(cell) == 64


def test_read_elk_symbols() -> None:
    """All atoms should be Si."""
    cell, spfnames = read_elk(cwd / "elk.in")
    assert all(s == "Si" for s in cell.symbols)


def test_read_elk_spfnames() -> None:
    """Spfnames must contain 'Si.in'."""
    _cell, spfnames = read_elk(cwd / "elk.in")
    assert spfnames == ["Si.in"]


def test_read_elk_cell_shape() -> None:
    """Lattice must be a 3×3 array."""
    cell, _spfnames = read_elk(cwd / "elk.in")
    assert cell.cell.shape == (3, 3)


def test_read_elk_scaled_positions_shape() -> None:
    """scaled_positions must be (natoms, 3)."""
    cell, _spfnames = read_elk(cwd / "elk.in")
    assert cell.scaled_positions.shape == (64, 3)


# ---------------------------------------------------------------------------
# write_elk / read_elk round-trip
# ---------------------------------------------------------------------------


def test_write_elk_roundtrip(tmp_path: pathlib.Path) -> None:
    """write_elk then read_elk must reproduce the original cell."""
    cell, spfnames = read_elk(cwd / "elk.in")
    fpath = tmp_path / "out.in"
    write_elk(str(fpath), cell, spfnames)
    cell2, spfnames2 = read_elk(str(fpath))

    assert isclose(cell, cell2)
    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-10)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == list(cell2.symbols)
    assert spfnames2 == spfnames


# ---------------------------------------------------------------------------
# get_elk_structure content tests
# ---------------------------------------------------------------------------


def _make_nacl_cell() -> PhonopyAtoms:
    """Return a simple 2-species NaCl-like cell."""
    a = 5.0
    return PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        symbols=["Na", "Na", "Cl", "Cl"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )


def test_get_elk_structure_keywords() -> None:
    """Output must contain 'avec' and 'atoms' keywords."""
    cell, spfnames = read_elk(cwd / "elk.in")
    text = get_elk_structure(cell, spfnames)
    assert "avec" in text
    assert "atoms" in text


def test_get_elk_structure_lattice_vectors() -> None:
    """Avec block must match cell.cell rows."""
    cell, spfnames = read_elk(cwd / "elk.in")
    text = get_elk_structure(cell, spfnames)
    lines = text.splitlines()
    idx = lines.index("avec")
    parsed = np.array(
        [[float(x) for x in lines[idx + i + 1].split()] for i in range(3)]
    )
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_elk_structure_nspecies_line() -> None:
    """Number of species in the atoms block must equal unique symbols."""
    cell = _make_nacl_cell()
    spfnames = ["Na.in", "Cl.in"]
    text = get_elk_structure(cell, spfnames)
    lines = text.splitlines()
    idx = lines.index("atoms")
    nspecies = int(lines[idx + 1].strip())
    assert nspecies == 2


def test_get_elk_structure_auto_spfnames() -> None:
    """Without sp_filenames, spfnames are '<symbol>.in'."""
    cell = _make_nacl_cell()
    text = get_elk_structure(cell)
    assert "'Na.in'" in text
    assert "'Cl.in'" in text


def test_get_elk_structure_spfnames_in_output() -> None:
    """Custom sp_filenames appear quoted in the atoms block."""
    cell, spfnames = read_elk(cwd / "elk.in")
    text = get_elk_structure(cell, spfnames)
    assert "'Si.in'" in text


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """supercell.in and supercell-001.in, … must be created."""
    cell, spfnames = read_elk(cwd / "elk.in")
    ids = np.array([1, 2])
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(
        cell, [cell, cell], ids, spfnames, pre_filename=pre
    )
    assert (tmp_path / "supercell.in").exists()
    assert (tmp_path / "supercell-001.in").exists()
    assert (tmp_path / "supercell-002.in").exists()


def test_write_supercells_custom_prefix(tmp_path: pathlib.Path) -> None:
    """Custom pre_filename is used for all output files."""
    cell, spfnames = read_elk(cwd / "elk.in")
    pre = str(tmp_path / "disp")
    write_supercells_with_displacements(
        cell, [cell], np.array([1]), spfnames, pre_filename=pre
    )
    assert (tmp_path / "disp.in").exists()
    assert (tmp_path / "disp-001.in").exists()


def test_write_supercells_content_readable(tmp_path: pathlib.Path) -> None:
    """Written supercell file must be parseable by read_elk."""
    cell, spfnames = read_elk(cwd / "elk.in")
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(
        cell, [cell], np.array([1]), spfnames, pre_filename=pre
    )
    cell2, _ = read_elk(str(tmp_path / "supercell.in"))
    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# ElkIn parsing corner cases
# ---------------------------------------------------------------------------

_AVEC = """\
avec
 5.0  0.0  0.0
 0.0  5.0  0.0
 0.0  0.0  5.0
atoms
 1
 'Na.in'
 1
 0.0  0.0  0.0
"""


def test_elkin_basic_parse() -> None:
    """Basic single-atom input is parsed correctly."""
    tags = ElkIn(_AVEC.splitlines(keepends=True)).get_variables()
    assert tags["avec"] is not None
    np.testing.assert_allclose(np.diag(tags["avec"]), [5.0, 5.0, 5.0], atol=1e-10)
    assert tags["atoms"]["spfnames"] == ["Na.in"]
    assert len(tags["atoms"]["positions"][0]) == 1


def test_elkin_scale() -> None:
    """'scale' keyword sets all three scale factors."""
    src = "scale\n 2.0\n" + _AVEC
    tags = ElkIn(src.splitlines(keepends=True)).get_variables()
    assert tags["scale"] == pytest.approx([2.0, 2.0, 2.0])


def test_elkin_scale123() -> None:
    """'scale1', 'scale2', 'scale3' set individual scale factors."""
    src = "scale1\n 2.0\nscale2\n 3.0\nscale3\n 4.0\n" + _AVEC
    tags = ElkIn(src.splitlines(keepends=True)).get_variables()
    assert tags["scale"] == pytest.approx([2.0, 3.0, 4.0])


def test_elkin_comment_ignored() -> None:
    """Lines starting with '!' are ignored."""
    src = "! this is a comment\n" + _AVEC
    tags = ElkIn(src.splitlines(keepends=True)).get_variables()
    assert tags["avec"] is not None


def test_elkin_multispecies() -> None:
    """Two-species atoms block is parsed into two separate position lists."""
    src = (
        "avec\n"
        " 5.0  0.0  0.0\n"
        " 0.0  5.0  0.0\n"
        " 0.0  0.0  5.0\n"
        "atoms\n"
        " 2\n"
        " 'Na.in'\n"
        " 1\n"
        " 0.0  0.0  0.0\n"
        " 'Cl.in'\n"
        " 1\n"
        " 0.5  0.5  0.5\n"
    )
    tags = ElkIn(src.splitlines(keepends=True)).get_variables()
    assert tags["atoms"]["spfnames"] == ["Na.in", "Cl.in"]
    assert len(tags["atoms"]["positions"]) == 2
    np.testing.assert_allclose(
        tags["atoms"]["positions"][1][0], [0.5, 0.5, 0.5], atol=1e-10
    )
