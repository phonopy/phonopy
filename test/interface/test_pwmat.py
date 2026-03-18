"""Tests for PWmat calculator interface."""

import pathlib
import tempfile

import numpy as np
import pytest

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.pwmat import (
    get_pwmat_structure,
    read_atom_config,
    write_atom_config,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# read_atom_config tests
# ---------------------------------------------------------------------------


def test_read_pwmat() -> None:
    """Test that read_atom_config returns cell matching the reference yaml.

    read_atom_config does not read magnetic moments, so only geometry
    (cell, positions, symbols) is compared.

    """
    cell = read_atom_config(cwd / "Si-pwmat.config")
    cell_ref = read_cell_yaml(cwd / "Si-pwmat.yaml")
    cell_ref.magnetic_moments = None  # Ignore magnetic moments for this test
    assert isclose(cell, cell_ref)


def test_read_pwmat_no_magnetic_moments() -> None:
    """Si-pwmat.config has no MAGNETIC block → magnetic_moments is None."""
    cell = read_atom_config(cwd / "Si-pwmat.config")
    assert cell.magnetic_moments is None


def test_read_pwmat_atom_count() -> None:
    """Si conventional cell has 8 atoms."""
    cell = read_atom_config(cwd / "Si-pwmat.config")
    assert len(cell) == 8


def test_read_pwmat_all_silicon() -> None:
    """All atoms in Si-pwmat.config are silicon."""
    cell = read_atom_config(cwd / "Si-pwmat.config")
    assert all(s == "Si" for s in cell.symbols)


def test_read_pwmat_lattice() -> None:
    """Lattice of Si-pwmat.config is cubic with a ≈ 5.47022 Å."""
    cell = read_atom_config(cwd / "Si-pwmat.config")
    a = 5.47022038
    expected = np.diag([a, a, a])
    np.testing.assert_allclose(cell.cell, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# get_pwmat_structure content tests
# ---------------------------------------------------------------------------


def _si_cell() -> PhonopyAtoms:
    """Return the Si 8-atom conventional cell."""
    return read_atom_config(cwd / "Si-pwmat.config")


def test_get_pwmat_structure_keywords() -> None:
    """Required section headers must appear in the output."""
    text = get_pwmat_structure(_si_cell())
    assert "Lattice vector" in text
    assert "Position" in text


def test_get_pwmat_structure_atom_count_header() -> None:
    """First non-empty line must be the atom count."""
    cell = _si_cell()
    text = get_pwmat_structure(cell)
    first_line = text.splitlines()[0].strip()
    assert int(first_line) == len(cell)


def test_get_pwmat_structure_lattice_vectors() -> None:
    """Lattice vectors in the output must match cell.cell."""
    cell = _si_cell()
    text = get_pwmat_structure(cell)
    lines = text.splitlines()
    lat_idx = next(i for i, ln in enumerate(lines) if "lattice" in ln.lower())
    parsed = np.array(
        [[float(x) for x in lines[lat_idx + i + 1].split()[:3]] for i in range(3)]
    )
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_pwmat_structure_position_count() -> None:
    """Number of position lines must equal the number of atoms."""
    cell = _si_cell()
    text = get_pwmat_structure(cell)
    lines = text.splitlines()
    pos_idx = next(i for i, ln in enumerate(lines) if "position" in ln.lower())
    pos_lines = [
        ln
        for ln in lines[pos_idx + 1 :]
        if ln.strip() and not ln.strip().lower().startswith("magnetic")
    ]
    assert len(pos_lines) == len(cell)


def test_get_pwmat_structure_move_flags() -> None:
    """Each position line must end with the '1   1   1' move flags."""
    cell = _si_cell()
    text = get_pwmat_structure(cell)
    lines = text.splitlines()
    pos_idx = next(i for i, ln in enumerate(lines) if "position" in ln.lower())
    pos_lines = [
        ln
        for ln in lines[pos_idx + 1 :]
        if ln.strip() and not ln.strip().lower().startswith("magnetic")
    ]
    for ln in pos_lines:
        parts = ln.split()
        assert parts[-3:] == ["1", "1", "1"]


def test_get_pwmat_structure_no_magnetic_section_without_magmoms() -> None:
    """'magnetic' keyword must not appear when no magnetic moments are set."""
    cell = _si_cell()
    assert cell.magnetic_moments is None
    text = get_pwmat_structure(cell)
    assert "magnetic" not in text.lower()


def test_get_pwmat_structure_colinear_magnetic() -> None:
    """Colinear magnetic moments produce a 'magnetic' section."""
    cell = _si_cell()
    cell.magnetic_moments = np.array([2.0] * len(cell))
    text = get_pwmat_structure(cell)
    assert "magnetic" in text.lower()
    assert "magnetic_xyz" not in text.lower()
    # Each mag line: atomic_number  mag_value
    lines = text.splitlines()
    mag_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "magnetic")
    mag_lines = lines[mag_idx + 1 : mag_idx + 1 + len(cell)]
    assert len(mag_lines) == len(cell)
    for ln in mag_lines:
        parts = ln.split()
        assert len(parts) == 2
        assert float(parts[1]) == pytest.approx(2.0)


def test_get_pwmat_structure_noncolinear_magnetic() -> None:
    """Non-colinear (3-vector) magnetic moments produce a 'magnetic_xyz' section."""
    cell = _si_cell()
    cell.magnetic_moments = np.tile([1.0, 0.5, 0.0], (len(cell), 1))
    text = get_pwmat_structure(cell)
    assert "magnetic_xyz" in text.lower()
    lines = text.splitlines()
    mag_idx = next(
        i for i, ln in enumerate(lines) if ln.strip().lower() == "magnetic_xyz"
    )
    mag_lines = lines[mag_idx + 1 : mag_idx + 1 + len(cell)]
    assert len(mag_lines) == len(cell)
    for ln in mag_lines:
        parts = ln.split()
        # atomic_number mx my mz
        assert len(parts) == 4
        assert float(parts[1]) == pytest.approx(1.0)
        assert float(parts[2]) == pytest.approx(0.5)
        assert float(parts[3]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# write_atom_config round-trip tests
# ---------------------------------------------------------------------------


def test_write_atom_config_roundtrip() -> None:
    """Write and re-read atom.config must preserve cell, positions, and symbols."""
    cell = _si_cell()
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "atom.config"
        write_atom_config(fpath, cell)
        cell2 = read_atom_config(fpath)

    assert isclose(cell, cell2)


def test_write_atom_config_colinear_magmom_roundtrip() -> None:
    """Colinear magnetic moments are preserved through write → read."""
    cell = _si_cell()
    cell.magnetic_moments = np.array([1.5] * len(cell))
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = pathlib.Path(tmpdir) / "atom.config"
        write_atom_config(fpath, cell)
        # Check that the file contains the 'magnetic' section
        text = fpath.read_text()
    assert "magnetic" in text.lower()
    assert "1.5" in text


# ---------------------------------------------------------------------------
# write_supercells_with_displacements tests
# ---------------------------------------------------------------------------


def test_write_supercells_filenames() -> None:
    """Supercell file and displacement files are created with correct names."""
    cell = _si_cell()
    ids = np.array([1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
        assert pathlib.Path(tmpdir, "supercell.config").exists()
        assert pathlib.Path(tmpdir, "supercell-001.config").exists()
        assert pathlib.Path(tmpdir, "supercell-002.config").exists()


def test_write_supercells_custom_prefix() -> None:
    """Custom pre_filename prefix is used for all output files."""
    cell = _si_cell()
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "PWMAT")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        assert pathlib.Path(tmpdir, "PWMAT.config").exists()
        assert pathlib.Path(tmpdir, "PWMAT-001.config").exists()


def test_write_supercells_content_readable() -> None:
    """Written supercell file must be parseable by read_atom_config."""
    cell = _si_cell()
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "supercell")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre
        )
        cell2 = read_atom_config(pathlib.Path(tmpdir) / "supercell.config")
    assert isclose(cell, cell2)


def test_write_supercells_custom_width() -> None:
    """Custom width parameter controls zero-padding of displacement file indices."""
    cell = _si_cell()
    with tempfile.TemporaryDirectory() as tmpdir:
        pre = str(pathlib.Path(tmpdir) / "sc")
        write_supercells_with_displacements(
            cell, [cell], np.array([1]), pre_filename=pre, width=4
        )
        assert pathlib.Path(tmpdir, "sc-0001.config").exists()
