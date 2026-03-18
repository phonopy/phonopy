"""Tests for DFTB+ calculator interface."""

import pathlib

import numpy as np

from phonopy.interface.dftbp import (
    dftbpToBohr,
    get_reduced_symbols,
    read_dftbp,
    write_dftbp,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_nacl_cell() -> PhonopyAtoms:
    """Return a simple 2-atom NaCl-like cell in Angstrom."""
    a = 5.0
    return PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]],
    )


# ---------------------------------------------------------------------------
# get_reduced_symbols
# ---------------------------------------------------------------------------


def test_get_reduced_symbols_basic() -> None:
    """Duplicate symbols are reduced to unique list in order of first occurrence."""
    result = get_reduced_symbols(["Na", "Na", "Cl", "Cl"])
    assert result == ["Na", "Cl"]


def test_get_reduced_symbols_single() -> None:
    """Single-element list stays unchanged."""
    assert get_reduced_symbols(["Si"]) == ["Si"]


def test_get_reduced_symbols_mixed_order() -> None:
    """Order of first appearance is preserved."""
    result = get_reduced_symbols(["Fe", "O", "Fe", "O", "Fe"])
    assert result == ["Fe", "O"]


# ---------------------------------------------------------------------------
# write_dftbp / read_dftbp round-trip
# ---------------------------------------------------------------------------


def test_write_dftbp_roundtrip(tmp_path: pathlib.Path) -> None:
    """write_dftbp then read_dftbp must reproduce the original cell."""
    cell = _make_nacl_cell()
    fpath = str(tmp_path / "out.gen")
    write_dftbp(fpath, cell)
    cell2 = read_dftbp(fpath)

    assert isclose(cell, cell2)


def test_write_dftbp_roundtrip_multispecies(tmp_path: pathlib.Path) -> None:
    """Round-trip preserves symbol ordering for a 4-species cell."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        symbols=["Si", "Si", "Ge", "Ge"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )
    fpath = str(tmp_path / "out.gen")
    write_dftbp(fpath, cell)
    cell2 = read_dftbp(fpath)

    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# write_dftbp content tests
# ---------------------------------------------------------------------------


def test_write_dftbp_format_s(tmp_path: pathlib.Path) -> None:
    """Output gen file must start with natom and 'S' format marker."""
    cell = _make_nacl_cell()
    fpath = tmp_path / "out.gen"
    write_dftbp(str(fpath), cell)
    first_line = fpath.read_text().splitlines()[0]
    parts = first_line.split()
    assert int(parts[0]) == len(cell)
    assert parts[1].upper() == "S"


def test_write_dftbp_species_line(tmp_path: pathlib.Path) -> None:
    """Second line must list unique species in order of first appearance."""
    cell = _make_nacl_cell()
    fpath = tmp_path / "out.gen"
    write_dftbp(str(fpath), cell)
    second_line = fpath.read_text().splitlines()[1]
    assert second_line.strip() == "Na Cl"


def test_write_dftbp_positions_scaled_by_conversion(tmp_path: pathlib.Path) -> None:
    """Cartesian positions in the gen file are divided by dftbpToBohr."""
    cell = _make_nacl_cell()
    fpath = tmp_path / "out.gen"
    write_dftbp(str(fpath), cell)
    lines = fpath.read_text().splitlines()
    # Cl is at position [2.5, 2.5, 2.5] Å; in file it should be 2.5 / dftbpToBohr
    cl_line = lines[3]  # 0-indexed: header, species, Na, Cl
    x = float(cl_line.split()[2])
    np.testing.assert_allclose(x, 2.5 / dftbpToBohr, atol=1e-10)


def test_write_dftbp_cell_scaled_by_conversion(tmp_path: pathlib.Path) -> None:
    """Lattice vectors in the gen file are divided by dftbpToBohr."""
    cell = _make_nacl_cell()
    fpath = tmp_path / "out.gen"
    write_dftbp(str(fpath), cell)
    lines = fpath.read_text().splitlines()
    # origin line at index natom+2 (=4), then 3 cell lines
    cell_line = lines[5]  # first lattice vector line
    a_stored = float(cell_line.split()[0])
    np.testing.assert_allclose(a_stored, 5.0 / dftbpToBohr, atol=1e-10)


# ---------------------------------------------------------------------------
# read_dftbp with manually constructed F-format gen
# ---------------------------------------------------------------------------


def test_read_dftbp_fractional_format(tmp_path: pathlib.Path) -> None:
    """F-format (fractional) gen file is parsed correctly."""
    a = 5.0
    a_stored = a / dftbpToBohr  # cell stored in file
    src = (
        f"2 F\n"
        f"Na Cl\n"
        f"1 1  0.0  0.0  0.0\n"
        f"2 2  0.5  0.5  0.5\n"
        f"0.0 0.0 0.0\n"
        f"{a_stored:.15f} 0.0 0.0\n"
        f"0.0 {a_stored:.15f} 0.0\n"
        f"0.0 0.0 {a_stored:.15f}\n"
    )
    fpath = tmp_path / "frac.gen"
    fpath.write_text(src)
    cell = read_dftbp(str(fpath))

    np.testing.assert_allclose(np.diag(cell.cell), [a, a, a], atol=1e-10)
    diff = cell.scaled_positions - np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == ["Na", "Cl"]


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """Supercell (geoS) and displacement (geoS-001, …) files must be created."""
    cell = _make_nacl_cell()
    ids = np.array([1, 2])
    pre = str(tmp_path / "geo.gen")
    write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
    assert (tmp_path / "geo.genS").exists()
    assert (tmp_path / "geo.genS-001").exists()
    assert (tmp_path / "geo.genS-002").exists()


def test_write_supercells_custom_prefix(tmp_path: pathlib.Path) -> None:
    """Custom pre_filename is used for all output files."""
    cell = _make_nacl_cell()
    pre = str(tmp_path / "sc")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    assert (tmp_path / "scS").exists()
    assert (tmp_path / "scS-001").exists()


def test_write_supercells_content_readable(tmp_path: pathlib.Path) -> None:
    """Written supercell file must be parseable by read_dftbp."""
    cell = _make_nacl_cell()
    pre = str(tmp_path / "geo.gen")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    cell2 = read_dftbp(str(tmp_path / "geo.genS"))
    assert isclose(cell, cell2)
