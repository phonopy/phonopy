"""Tests for Fleur calculator interface."""

import pathlib

import numpy as np
import pytest

from phonopy.interface.fleur import (
    FleurIn,
    get_fleur_structure,
    read_fleur,
    write_fleur,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# read_fleur
# ---------------------------------------------------------------------------


def test_read_fleur_natoms() -> None:
    """fleur_inpgen has 1 Al atom."""
    cell, _speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    assert len(cell) == 1


def test_read_fleur_symbols() -> None:
    """The single atom should be Al (atomic number 13)."""
    cell, _speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    assert list(cell.symbols) == ["Al"]


def test_read_fleur_speci() -> None:
    """Speci must be ['13.1'] as in the input file."""
    _cell, speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    assert speci == ["13.1"]


def test_read_fleur_cell_shape() -> None:
    """Lattice must be a 3×3 array."""
    cell, _speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    assert cell.cell.shape == (3, 3)


def test_read_fleur_scaled_positions_shape() -> None:
    """scaled_positions must be (1, 3)."""
    cell, _speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    assert cell.scaled_positions.shape == (1, 3)


def test_read_fleur_lattice_vectors() -> None:
    """Lattice constant 7.656 and scale 0.5 are applied to raw vectors."""
    cell, _speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    # raw vectors: [[0,1,1],[1,0,1],[1,1,0]], lattcon=7.656, scale=[0.5,0.5,0.5]
    expected = np.array(
        [
            [0.0, 3.828, 3.828],
            [3.828, 0.0, 3.828],
            [3.828, 3.828, 0.0],
        ]
    )
    np.testing.assert_allclose(cell.cell, expected, atol=1e-10)


def test_read_fleur_restlines() -> None:
    """Restlines must start with the title line."""
    _cell, _speci, restlines = read_fleur(cwd / "fleur_inpgen")
    assert restlines[0] == "Aluminium test Fleur"


# ---------------------------------------------------------------------------
# write_fleur / read_fleur round-trip
# ---------------------------------------------------------------------------


def test_write_fleur_roundtrip(tmp_path: pathlib.Path) -> None:
    """write_fleur then read_fleur must reproduce the original cell."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    fpath = tmp_path / "out.in"
    write_fleur(str(fpath), cell, speci, restlines)
    cell2, speci2, _restlines2 = read_fleur(str(fpath))

    assert isclose(cell, cell2)
    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-10)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == list(cell2.symbols)
    assert speci2 == speci


# ---------------------------------------------------------------------------
# get_fleur_structure content tests
# ---------------------------------------------------------------------------


def test_get_fleur_structure_title_line() -> None:
    """First line of output must be the title from restlines[0]."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    text = get_fleur_structure(cell, speci, restlines)
    assert text.splitlines()[0] == restlines[0]


def test_get_fleur_structure_lattice_vectors() -> None:
    """Three lattice-vector lines must match cell.cell rows."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    text = get_fleur_structure(cell, speci, restlines)
    lines = text.splitlines()
    parsed = np.array([[float(x) for x in lines[i + 1].split()] for i in range(3)])
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_fleur_structure_atom_count() -> None:
    """Atom count line must equal the number of atoms in the cell."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    text = get_fleur_structure(cell, speci, restlines)
    lines = text.splitlines()
    # line 0: title, 1-3: lattice vecs, 4: "1.0", 5: "1.0 1.0 1.0", 6: "", 7: natoms
    assert int(lines[7]) == len(cell)


def test_get_fleur_structure_speci_in_atom_lines() -> None:
    """Atom lines must begin with the speci identifier."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    text = get_fleur_structure(cell, speci, restlines)
    lines = text.splitlines()
    # atom lines start at index 8
    natom = len(cell)
    for i in range(natom):
        assert lines[8 + i].split()[0] == speci[0]


def test_get_fleur_structure_no_restlines_warns() -> None:
    """Passing restlines=None must emit a UserWarning."""
    cell, speci, _restlines = read_fleur(cwd / "fleur_inpgen")
    with pytest.warns(UserWarning, match="job info"):
        get_fleur_structure(cell, speci, None)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """supercell.in and supercell-001.in, … must be created."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    ids = np.array([1, 2])
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(
        cell,
        [cell, cell],
        ids,
        speci,
        n_repeat=1,
        restlines=restlines,
        pre_filename=pre,
    )
    assert (tmp_path / "supercell.in").exists()
    assert (tmp_path / "supercell-001.in").exists()
    assert (tmp_path / "supercell-002.in").exists()


def test_write_supercells_custom_prefix(tmp_path: pathlib.Path) -> None:
    """Custom pre_filename is used for all output files."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    pre = str(tmp_path / "disp")
    write_supercells_with_displacements(
        cell,
        [cell],
        np.array([1]),
        speci,
        n_repeat=1,
        restlines=restlines,
        pre_filename=pre,
    )
    assert (tmp_path / "disp.in").exists()
    assert (tmp_path / "disp-001.in").exists()


def test_write_supercells_content_readable(tmp_path: pathlib.Path) -> None:
    """Written supercell file must be parseable by read_fleur."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(
        cell,
        [cell],
        np.array([1]),
        speci,
        n_repeat=1,
        restlines=restlines,
        pre_filename=pre,
    )
    cell2, _speci2, _restlines2 = read_fleur(str(tmp_path / "supercell.in"))
    assert isclose(cell, cell2)


def test_write_supercells_n_repeat(tmp_path: pathlib.Path) -> None:
    """n_repeat multiplies speci entries for the supercell."""
    cell, speci, restlines = read_fleur(cwd / "fleur_inpgen")
    # Build a 2-atom supercell manually
    sc_cell = PhonopyAtoms(
        cell=cell.cell * 2,
        symbols=["Al", "Al"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    pre = str(tmp_path / "sc")
    write_supercells_with_displacements(
        sc_cell,
        [sc_cell],
        np.array([1]),
        speci,
        n_repeat=2,
        restlines=restlines,
        pre_filename=pre,
    )
    assert (tmp_path / "sc.in").exists()
    sc2, speci2, _ = read_fleur(str(tmp_path / "sc.in"))
    assert len(sc2) == 2


# ---------------------------------------------------------------------------
# FleurIn parsing corner cases
# ---------------------------------------------------------------------------

_MINIMAL = """\
Title
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
1.0
1.0 1.0 1.0

1
11.0 0.0 0.0 0.0
"""


def test_fleurin_basic_parse() -> None:
    """Basic single-atom input is parsed correctly."""
    tags = FleurIn(_MINIMAL.splitlines(keepends=True)).get_variables()
    assert tags["avec"] is not None
    np.testing.assert_allclose(np.diag(tags["avec"]), [1.0, 1.0, 1.0], atol=1e-10)
    assert tags["atoms"]["speci"] == ["11.0"]
    assert len(tags["atoms"]["positions"]) == 1
    np.testing.assert_allclose(
        tags["atoms"]["positions"][0], [0.0, 0.0, 0.0], atol=1e-10
    )


def test_fleurin_lattcon_applied() -> None:
    """Lattice constant scales all vectors uniformly."""
    src = (
        "Title\n"
        " 1.0  0.0  0.0\n"
        " 0.0  1.0  0.0\n"
        " 0.0  0.0  1.0\n"
        "2.5\n"
        "1.0 1.0 1.0\n"
        "\n"
        "1\n"
        "11.0 0.0 0.0 0.0\n"
    )
    tags = FleurIn(src.splitlines(keepends=True)).get_variables()
    np.testing.assert_allclose(np.diag(tags["avec"]), [2.5, 2.5, 2.5], atol=1e-10)


def test_fleurin_negative_scale() -> None:
    """Negative scale values are replaced by sqrt(|scale|)."""
    src = (
        "Title\n"
        " 1.0  0.0  0.0\n"
        " 0.0  1.0  0.0\n"
        " 0.0  0.0  1.0\n"
        "1.0\n"
        "-4.0 -4.0 -4.0\n"
        "\n"
        "1\n"
        "11.0 0.0 0.0 0.0\n"
    )
    tags = FleurIn(src.splitlines(keepends=True)).get_variables()
    # sqrt(4.0) = 2.0 for each axis
    np.testing.assert_allclose(np.diag(tags["avec"]), [2.0, 2.0, 2.0], atol=1e-10)


def test_fleurin_zero_scale() -> None:
    """Zero scale is treated as 1.0."""
    src = (
        "Title\n"
        " 3.0  0.0  0.0\n"
        " 0.0  3.0  0.0\n"
        " 0.0  0.0  3.0\n"
        "1.0\n"
        "0.0 0.0 0.0\n"
        "\n"
        "1\n"
        "11.0 0.0 0.0 0.0\n"
    )
    tags = FleurIn(src.splitlines(keepends=True)).get_variables()
    np.testing.assert_allclose(np.diag(tags["avec"]), [3.0, 3.0, 3.0], atol=1e-10)


def test_fleurin_restlines_title() -> None:
    """The title line is stored as restlines[0]."""
    fi = FleurIn(_MINIMAL.splitlines(keepends=True))
    assert fi.restlines[0] == "Title"


def test_fleurin_multispecies() -> None:
    """Two-species atoms block is parsed into two separate entries."""
    src = (
        "Title\n"
        " 5.0  0.0  0.0\n"
        " 0.0  5.0  0.0\n"
        " 0.0  0.0  5.0\n"
        "1.0\n"
        "1.0 1.0 1.0\n"
        "\n"
        "2\n"
        "11.0 0.0 0.0 0.0\n"
        "17.0 0.5 0.5 0.5\n"
    )
    tags = FleurIn(src.splitlines(keepends=True)).get_variables()
    assert tags["atoms"]["speci"] == ["11.0", "17.0"]
    assert len(tags["atoms"]["positions"]) == 2
    np.testing.assert_allclose(
        tags["atoms"]["positions"][1], [0.5, 0.5, 0.5], atol=1e-10
    )
