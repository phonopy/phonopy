"""Tests for FHI-aims calculator interface."""

import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.interface.aims import (
    read_aims,
    write_aims,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


def _aims_cell_from_str(src: str) -> PhonopyAtoms:
    """Write src to a temp file and read it back via read_aims."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        fpath.write_text(src)
        return read_aims(str(fpath))


def test_read_aims() -> None:
    """Read geometry.in and compare against reference YAML."""
    cell = read_aims(cwd / "NaCl-aims.in")
    cell_ref = read_cell_yaml(cwd / "NaCl-abinit-pwscf.yaml")
    assert isclose(cell, cell_ref)
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_read_aims_no_magnetic_moments() -> None:
    """Cell without initial_moment lines has no magnetic_moments."""
    cell = read_aims(cwd / "NaCl-aims.in")
    assert cell.magnetic_moments is None


def test_read_aims_atom_frac() -> None:
    """atom_frac (fractional) coordinates are converted to Cartesian correctly."""
    a = 5.0
    src = (
        f"lattice_vector {a} 0.0 0.0\n"
        f"lattice_vector 0.0 {a} 0.0\n"
        f"lattice_vector 0.0 0.0 {a}\n"
        "atom_frac 0.5 0.5 0.5 Na\n"
    )
    cell = _aims_cell_from_str(src)
    np.testing.assert_allclose(cell.positions[0], [2.5, 2.5, 2.5], atol=1e-10)


def test_read_aims_mixed_atom_and_atom_frac() -> None:
    """Files mixing atom and atom_frac lines are parsed correctly."""
    a = 10.0
    src = (
        f"lattice_vector {a} 0.0 0.0\n"
        f"lattice_vector 0.0 {a} 0.0\n"
        f"lattice_vector 0.0 0.0 {a}\n"
        "atom 0.0 0.0 0.0 Na\n"
        "atom_frac 0.5 0.5 0.5 Cl\n"
    )
    cell = _aims_cell_from_str(src)
    np.testing.assert_allclose(cell.positions[0], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(cell.positions[1], [5.0, 5.0, 5.0], atol=1e-10)
    assert list(cell.symbols) == ["Na", "Cl"]


def test_read_aims_partial_moment_ignored() -> None:
    """When only some atoms have initial_moment, magnetic_moments is None."""
    a = 5.0
    src = (
        f"lattice_vector {a} 0.0 0.0\n"
        f"lattice_vector 0.0 {a} 0.0\n"
        f"lattice_vector 0.0 0.0 {a}\n"
        "atom 0.0 0.0 0.0 Na\n"
        "initial_moment 1.0\n"
        "atom 2.5 2.5 2.5 Cl\n"
    )
    cell = _aims_cell_from_str(src)
    # Only Na has an initial_moment; Cl does not → None in list → no magnetic_moments
    assert cell.magnetic_moments is None


def test_read_aims_all_moments() -> None:
    """When ALL atoms have initial_moment, magnetic_moments is set correctly."""
    a = 5.0
    src = (
        f"lattice_vector {a} 0.0 0.0\n"
        f"lattice_vector 0.0 {a} 0.0\n"
        f"lattice_vector 0.0 0.0 {a}\n"
        "atom 0.0 0.0 0.0 Fe\n"
        "initial_moment 2.5\n"
        "atom 2.5 2.5 2.5 Fe\n"
        "initial_moment -2.5\n"
    )
    cell = _aims_cell_from_str(src)
    assert cell.magnetic_moments is not None
    np.testing.assert_allclose(cell.magnetic_moments, [2.5, -2.5], atol=1e-10)


# ---------------------------------------------------------------------------
# write_aims round-trip and content tests
# ---------------------------------------------------------------------------


def test_write_aims_roundtrip() -> None:
    """write_aims then read_aims must reproduce the original cell."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        write_aims(str(fpath), cell)
        cell2 = read_aims(str(fpath))

    assert isclose(cell, cell2)
    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-10)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-10
    assert list(cell.symbols) == list(cell2.symbols)


def test_write_aims_content_keywords() -> None:
    """Output must contain lattice_vector and atom keywords."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        write_aims(str(fpath), cell)
        text = fpath.read_text()

    assert "lattice_vector" in text
    assert "atom" in text
    assert "initial_moment" not in text


def test_write_aims_lattice_vectors() -> None:
    """Written lattice_vector lines must match cell.cell rows."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        write_aims(str(fpath), cell)
        lines = fpath.read_text().splitlines()

    lv_lines = [ln for ln in lines if ln.startswith("lattice_vector")]
    assert len(lv_lines) == 3
    parsed = np.array([[float(x) for x in ln.split()[1:4]] for ln in lv_lines])
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_write_aims_atom_count() -> None:
    """Number of atom lines must equal the number of atoms."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        write_aims(str(fpath), cell)
        lines = fpath.read_text().splitlines()

    atom_lines = [ln for ln in lines if ln.startswith("atom ")]
    assert len(atom_lines) == len(cell)


def test_write_aims_magnetic_moments() -> None:
    """initial_moment lines are written when magnetic_moments is set."""
    cell = PhonopyAtoms(
        cell=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        symbols=["Fe", "Fe"],
        positions=[[0, 0, 0], [2.5, 2.5, 2.5]],
        magnetic_moments=[2.5, -2.5],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "geometry.in"
        write_aims(str(fpath), cell)
        text = fpath.read_text()

    assert text.count("initial_moment") == 2
    lines = text.splitlines()
    mom_lines = [ln for ln in lines if "initial_moment" in ln]
    assert float(mom_lines[0].split()[1]) == pytest.approx(2.5)
    assert float(mom_lines[1].split()[1]) == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements tests
# ---------------------------------------------------------------------------


def test_write_supercells_filenames() -> None:
    """Supercell (.supercell) and displacement (-001, -002, …) files are created."""
    cell = read_aims(cwd / "NaCl-aims.in")
    ids = np.array([1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "geometry.in")
        write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
        assert pathlib.Path(tmpdir, "geometry.in.supercell").exists()
        assert pathlib.Path(tmpdir, "geometry.in-001").exists()
        assert pathlib.Path(tmpdir, "geometry.in-002").exists()


def test_write_supercells_custom_prefix() -> None:
    """Custom pre_filename is used for all output files."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "sc")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "sc.supercell").exists()
        assert pathlib.Path(tmpdir, "sc-001").exists()


def test_write_supercells_content_readable() -> None:
    """Written supercell file must be parseable by read_aims."""
    cell = read_aims(cwd / "NaCl-aims.in")
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "geometry.in")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        cell2 = read_aims(str(pathlib.Path(tmpdir, "geometry.in.supercell")))
    assert isclose(cell, cell2)
