"""Tests for the SIESTA calculator interface."""

import pathlib

import numpy as np

from phonopy.interface.siesta import (
    get_siesta_structure,
    parse_set_of_forces,
    read_siesta,
    write_siesta,
    write_supercells_with_displacements,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Minimal NaCl 2-atom conventional cell in fractional format.
# LatticeConstant 1.0 Bohr means the LatticeVectors are used as-is (no scaling).
_NACL_FDF = """\
LatticeConstant 1.0 Bohr

AtomicCoordinatesFormat Fractional

NumberOfSpecies 2
%block ChemicalSpeciesLabel
 1  11 Na
 2  17 Cl
%endblock ChemicalSpeciesLabel

%block LatticeVectors
 5.64  0.00  0.00
 0.00  5.64  0.00
 0.00  0.00  5.64
%endblock LatticeVectors

%block AtomicCoordinatesAndAtomicSpecies
 0.00  0.00  0.00  1
 0.50  0.50  0.50  2
%endblock AtomicCoordinatesAndAtomicSpecies
"""

# 8-atom NaCl conventional supercell in fractional format.
_NACL8_FDF = """\
LatticeConstant 1.0 Bohr

AtomicCoordinatesFormat Fractional

NumberOfSpecies 2
%block ChemicalSpeciesLabel
 1  11 Na
 2  17 Cl
%endblock ChemicalSpeciesLabel

%block LatticeVectors
 5.64  0.00  0.00
 0.00  5.64  0.00
 0.00  0.00  5.64
%endblock LatticeVectors

%block AtomicCoordinatesAndAtomicSpecies
 0.00  0.00  0.00  1
 0.00  0.50  0.50  1
 0.50  0.00  0.50  1
 0.50  0.50  0.00  1
 0.50  0.50  0.50  2
 0.50  0.00  0.00  2
 0.00  0.50  0.00  2
 0.00  0.00  0.50  2
%endblock AtomicCoordinatesAndAtomicSpecies
"""

# SIESTA forces file: first line is consumed as the hook ("" matches any line),
# then columns 1-3 (0-indexed) are the force components.
_FORCES_2ATOM = """\
# SIESTA forces [eV/Ang]
1   0.01000  -0.01000   0.00000
2  -0.01000   0.01000   0.00000
"""


def _write_tmp(content: str, tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# read_siesta
# ---------------------------------------------------------------------------


def test_read_siesta_symbols(tmp_path):
    """read_siesta returns correct chemical symbols."""
    fpath = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath)
    assert cell.symbols == ["Na", "Cl"]


def test_read_siesta_cell(tmp_path):
    """read_siesta returns correct lattice vectors."""
    fpath = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath)
    expected = np.diag([5.64, 5.64, 5.64])
    np.testing.assert_allclose(cell.cell, expected, atol=1e-10)


def test_read_siesta_positions_fractional(tmp_path):
    """read_siesta with Fractional format sets correct scaled positions."""
    fpath = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath)
    np.testing.assert_allclose(cell.scaled_positions[0], [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(cell.scaled_positions[1], [0.5, 0.5, 0.5], atol=1e-10)


def test_read_siesta_scaledcartesian(tmp_path):
    """read_siesta with ScaledCartesian format sets Cartesian positions."""
    # With ScaledCartesian, cell.positions = positions * alat.
    # alat = 1.0 (Bohr), so positions are stored as-is.
    fdf = _NACL_FDF.replace("Fractional", "ScaledCartesian")
    fpath = _write_tmp(fdf, tmp_path, "NaCl_sc.fdf")
    cell = read_siesta(fpath)
    # Just verify the cell is created and has the right size.
    assert len(cell) == 2
    assert cell.symbols == ["Na", "Cl"]


def test_read_siesta_bohr(tmp_path):
    """read_siesta with NotScaledCartesianBohr sets positions directly."""
    fdf = _NACL_FDF.replace("Fractional", "NotScaledCartesianBohr")
    fpath = _write_tmp(fdf, tmp_path, "NaCl_bohr.fdf")
    cell = read_siesta(fpath)
    assert len(cell) == 2
    assert cell.symbols == ["Na", "Cl"]


# ---------------------------------------------------------------------------
# get_siesta_structure – content tests
# ---------------------------------------------------------------------------


def test_get_siesta_structure_keywords(tmp_path):
    """Required block keywords must appear in the output."""
    fpath = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath)
    text = get_siesta_structure(cell)

    assert "NumberOfSpecies" in text
    assert "%block ChemicalSpeciesLabel" in text
    assert "%endblock ChemicalSpeciesLabel" in text
    assert "%block LatticeVectors" in text
    assert "%endblock LatticeVectors" in text
    assert "AtomicCoordinatesFormat  Fractional" in text
    assert "LatticeConstant 1.0 Bohr" in text
    assert "%block AtomicCoordinatesAndAtomicSpecies" in text
    assert "%endblock AtomicCoordinatesAndAtomicSpecies" in text


def test_get_siesta_structure_species_count(tmp_path):
    """NumberOfSpecies must equal the number of distinct elements."""
    fpath = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath)
    text = get_siesta_structure(cell)
    assert "NumberOfSpecies 2" in text


def test_get_siesta_structure_lattice_vectors(tmp_path):
    """Lattice vectors in the output match cell.cell."""
    fpath = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath)
    text = get_siesta_structure(cell)
    lines = text.splitlines()
    start = lines.index("%block LatticeVectors") + 1
    parsed = np.array([[float(x) for x in lines[start + i].split()] for i in range(3)])
    np.testing.assert_allclose(parsed, cell.cell, atol=1e-10)


def test_get_siesta_structure_atom_count(tmp_path):
    """Number of coordinate lines equals the number of atoms."""
    fpath = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath)
    text = get_siesta_structure(cell)
    lines = text.splitlines()
    start = lines.index("%block AtomicCoordinatesAndAtomicSpecies") + 1
    end = lines.index("%endblock AtomicCoordinatesAndAtomicSpecies")
    coord_lines = [ln for ln in lines[start:end] if ln.strip()]
    assert len(coord_lines) == len(cell)


def test_get_siesta_structure_symbol_in_species_block(tmp_path):
    """ChemicalSpeciesLabel block contains element symbols."""
    fpath = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath)
    text = get_siesta_structure(cell)
    assert "Na" in text
    assert "Cl" in text


# ---------------------------------------------------------------------------
# Round-trip: get_siesta_structure → read_siesta
# ---------------------------------------------------------------------------


def test_get_siesta_structure_roundtrip(tmp_path):
    """Write → read back preserves cell, positions, and symbols."""
    fpath_in = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath_in)

    fpath_out = _write_tmp(get_siesta_structure(cell), tmp_path, "NaCl8_out.fdf")
    cell2 = read_siesta(fpath_out)

    assert isclose(cell, cell2)


def test_get_siesta_structure_roundtrip_2atom(tmp_path):
    """Round-trip for 2-atom cell preserves cell, positions, and symbols."""
    fpath_in = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath_in)

    fpath_out = _write_tmp(get_siesta_structure(cell), tmp_path, "NaCl_out.fdf")
    cell2 = read_siesta(fpath_out)

    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# write_siesta
# ---------------------------------------------------------------------------


def test_write_siesta_creates_file(tmp_path):
    """write_siesta creates a parseable file that round-trips correctly."""
    fpath_in = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath_in)

    fpath_out = tmp_path / "NaCl_written.fdf"
    write_siesta(str(fpath_out), cell)

    assert fpath_out.exists()
    cell2 = read_siesta(str(fpath_out))
    assert isclose(cell, cell2)


def test_write_siesta_from_phonopyatoms(tmp_path):
    """write_siesta accepts a PhonopyAtoms and produces a readable file."""
    cell = PhonopyAtoms(
        cell=np.diag([5.64, 5.64, 5.64]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )
    fpath = tmp_path / "direct.fdf"
    write_siesta(str(fpath), cell)
    assert fpath.exists()
    cell2 = read_siesta(str(fpath))
    assert isclose(cell, cell2)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_default_filenames(tmp_path):
    """supercell.fdf and supercell-001.fdf … are created."""
    fpath_in = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath_in)
    pre = str(tmp_path / "supercell")

    write_supercells_with_displacements(cell, [cell, cell], [1, 2], pre_filename=pre)

    assert (tmp_path / "supercell.fdf").exists()
    assert (tmp_path / "supercell-001.fdf").exists()
    assert (tmp_path / "supercell-002.fdf").exists()


def test_write_supercells_content_readable(tmp_path):
    """Written displacement files are parseable and preserve cell and positions."""
    fpath_in = _write_tmp(_NACL8_FDF, tmp_path, "NaCl8.fdf")
    cell = read_siesta(fpath_in)
    pre = str(tmp_path / "supercell")

    write_supercells_with_displacements(cell, [cell], [1], pre_filename=pre)

    cell2 = read_siesta(str(tmp_path / "supercell-001.fdf"))
    assert isclose(cell, cell2)


def test_write_supercells_custom_width(tmp_path):
    """Width parameter controls zero-padding in displacement filenames."""
    fpath_in = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath_in)
    pre = str(tmp_path / "sc")

    write_supercells_with_displacements(cell, [cell], [3], pre_filename=pre, width=4)

    assert (tmp_path / "sc-0003.fdf").exists()


def test_write_supercells_custom_prefix(tmp_path):
    """Custom pre_filename is used for all output files."""
    fpath_in = _write_tmp(_NACL_FDF, tmp_path, "NaCl.fdf")
    cell = read_siesta(fpath_in)
    pre = str(tmp_path / "SIESTA")

    write_supercells_with_displacements(cell, [cell], [1], pre_filename=pre)

    assert (tmp_path / "SIESTA.fdf").exists()
    assert (tmp_path / "SIESTA-001.fdf").exists()


# ---------------------------------------------------------------------------
# parse_set_of_forces
# ---------------------------------------------------------------------------


def test_parse_set_of_forces_single(tmp_path):
    """parse_set_of_forces returns correct forces for a single file."""
    fpath = _write_tmp(_FORCES_2ATOM, tmp_path, "forces.FA")

    frs = parse_set_of_forces(2, [str(fpath)], verbose=False)

    assert len(frs) == 1
    assert frs[0].shape == (2, 3)
    # After drift correction the sum of forces must be zero.
    np.testing.assert_allclose(frs[0].sum(axis=0), 0.0, atol=1e-10)


def test_parse_set_of_forces_values(tmp_path):
    """parse_set_of_forces returns forces matching the file (after drift correction)."""
    forces_content = """\
header
1   0.01   -0.01   0.00
2  -0.01    0.01   0.00
"""
    fpath = _write_tmp(forces_content, tmp_path, "f.FA")
    frs = parse_set_of_forces(2, [str(fpath)], verbose=False)

    # Drift is exactly zero, so forces are unchanged.
    np.testing.assert_allclose(frs[0][0], [0.01, -0.01, 0.00], atol=1e-10)
    np.testing.assert_allclose(frs[0][1], [-0.01, 0.01, 0.00], atol=1e-10)


def test_parse_set_of_forces_multiple_files(tmp_path):
    """parse_set_of_forces collects forces from multiple files."""
    row1 = "hdr\n1  0.02  0.00  0.00\n2  -0.02  0.00  0.00\n"
    row2 = "hdr\n1  0.00  0.03  0.00\n2  0.00  -0.03  0.00\n"
    f1 = _write_tmp(row1, tmp_path, "f1.FA")
    f2 = _write_tmp(row2, tmp_path, "f2.FA")

    frs = parse_set_of_forces(2, [str(f1), str(f2)], verbose=False)

    assert len(frs) == 2
    for f in frs:
        assert f.shape == (2, 3)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
