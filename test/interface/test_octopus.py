# SPDX-License-Identifier: BSD-3-Clause
"""Tests of Octopus calculator interface."""

import numpy as np
import pytest

from phonopy.interface.calculator import read_crystal_structure
from phonopy.interface.octopus import (
    get_born_octopus,
    get_cell_from_octopus_lines,
    get_octopus_structure_lines,
    is_octopus_geometry,
    parse_octopus_born_charges,
    parse_octopus_epsilon,
    read_octopus,
    write_octopus,
)
from phonopy.interface.vasp import write_vasp
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms

# Real Octopus em_resp output for rock-salt NaCl (static response).
_EPSILON_FILE = """\
# Real part of dielectric constant
            2.588021            0.000000            0.000000
            0.000000            2.588021            0.000000
            0.000000            0.000000            2.588021
Isotropic average            2.588021

# Imaginary part of dielectric constant
            0.000000            0.000000            0.000000
            0.000000            0.000000            0.000000
            0.000000            0.000000            0.000000
Isotropic average            0.000000
"""

_BORN_FILE = """\
# (Frequency-dependent) Born effective charge tensors
Index:     1   Label:    Na   Ionic charge:     1.0000
            1.141799            0.000000           -0.000000
            0.000000            1.141799            0.000000
            0.000000            0.000000            1.141799
Isotropic average            1.141799

Index:     2   Label:    Cl   Ionic charge:     7.0000
           -1.141799            0.000000            0.000000
            0.000000           -1.141799           -0.000000
            0.000000           -0.000000           -1.141799
Isotropic average           -1.141799

# Discrepancy of Born effective charges from acoustic sum rule
           -0.025775            0.000000           -0.000000
            0.000000           -0.025775            0.000000
            0.000000            0.000000           -0.025775
Isotropic average           -0.025775
"""


def _nacl_cell() -> PhonopyAtoms:
    a = 5.64
    return PhonopyAtoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=(a / 2) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype="double"),
    )


def test_octopus_structure_lines_roundtrip():
    """get_cell_from_octopus_lines inverts get_octopus_structure_lines."""
    cell = _nacl_cell()
    result = get_cell_from_octopus_lines(get_octopus_structure_lines(cell))

    assert list(result.symbols) == list(cell.symbols)
    np.testing.assert_allclose(result.cell, cell.cell, atol=1e-6)
    np.testing.assert_allclose(
        result.scaled_positions, cell.scaled_positions, atol=1e-8
    )


def test_get_cell_from_octopus_lines_fixed_reference():
    """Parse a known 2x2x2 Si supercell block embedded among ignored lines.

    Non-block lines (e.g. ``CalculationMode = gs``) and comments must be
    ignored, and lattice vector i must be scaled by lattice parameter i.
    """
    lines = [
        "CalculationMode = gs   # ignored",
        "",
        "%LatticeParameters",
        " 14.511546484 | 14.511546484 | 14.511546484",
        "%",
        "# a comment",
        "%LatticeVectors",
        "  0.000000000 | 0.707106781 | 0.707106781",
        "  0.707106781 | 0.000000000 | 0.707106781",
        "  0.707106781 | 0.707106781 | 0.000000000",
        "%",
        "%ReducedCoordinates",
        '  "Si" | 0.000689106 | 0.000000000 | 0.000000000',
        '  "Si" | 0.500000000 | 0.000000000 | 0.000000000',
        "%",
    ]
    cell = get_cell_from_octopus_lines(lines)

    off = 14.511546484 * 0.707106781
    expected_cell = np.array([[0, off, off], [off, 0, off], [off, off, 0]])
    assert list(cell.symbols) == ["Si", "Si"]
    np.testing.assert_allclose(cell.cell, expected_cell, atol=1e-6)
    np.testing.assert_allclose(
        cell.scaled_positions,
        [[0.000689106, 0.0, 0.0], [0.5, 0.0, 0.0]],
        atol=1e-9,
    )


def test_get_cell_from_octopus_lines_rejects_non_numeric():
    """A geometry with unresolved variables (not the canonical format) fails."""
    lines = [
        "%LatticeParameters",
        " a | a | a",
        "%",
        "%LatticeVectors",
        " 0 | 0.5 | 0.5",
        " 0.5 | 0 | 0.5",
        " 0.5 | 0.5 | 0",
        "%",
        "%ReducedCoordinates",
        ' "Na" | 0 | 0 | 0',
        "%",
    ]
    with pytest.raises(ValueError):
        get_cell_from_octopus_lines(lines)


def test_read_octopus_file_roundtrip(tmp_path):
    """read_octopus reads back what write_octopus wrote."""
    cell = _nacl_cell()
    filename = tmp_path / "geometry-unitcell"
    write_octopus(filename, cell)
    result = read_octopus(filename)

    assert list(result.symbols) == list(cell.symbols)
    np.testing.assert_allclose(result.cell, cell.cell, atol=1e-6)
    np.testing.assert_allclose(
        result.scaled_positions, cell.scaled_positions, atol=1e-8
    )


def test_parse_octopus_epsilon(tmp_path):
    """parse_octopus_epsilon reads the real part of the dielectric tensor."""
    filename = tmp_path / "epsilon"
    filename.write_text(_EPSILON_FILE)
    epsilon = parse_octopus_epsilon(filename)

    assert epsilon.shape == (3, 3)
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-8)


def test_parse_octopus_born_charges(tmp_path):
    """parse_octopus_born_charges reads Z* and ignores the ASR discrepancy block."""
    filename = tmp_path / "born_charges"
    filename.write_text(_BORN_FILE)
    borns, labels = parse_octopus_born_charges(filename)

    assert borns.shape == (2, 3, 3)  # discrepancy block must not be parsed as an atom
    assert labels == ["Na", "Cl"]
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-8)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-8)


def test_get_born_octopus_end_to_end(tmp_path):
    """get_born_octopus ties the cell, epsilon and born_charges together."""
    write_octopus(tmp_path / "geometry-unitcell", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE)

    borns, epsilon, atom_indices = get_born_octopus(
        tmp_path / "geometry-unitcell",
        tmp_path / "epsilon",
        tmp_path / "born_charges",
    )

    # NaCl: Na and Cl are symmetry-independent, so both survive.
    assert atom_indices.tolist() == [0, 1]
    assert borns.shape == (2, 3, 3)
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-5)


def test_is_octopus_geometry(tmp_path):
    """A geometry include file is detected; a POSCAR is not."""
    cell = _nacl_cell()
    geom = tmp_path / "geometry-unitcell"
    poscar = tmp_path / "POSCAR"
    write_octopus(geom, cell)
    write_vasp(poscar, cell)

    assert is_octopus_geometry(geom) is True
    assert is_octopus_geometry(poscar) is False


def test_read_crystal_structure_octopus_accepts_geometry(tmp_path):
    """--octopus reads an Octopus geometry file (atomic units; no conversion)."""
    cell = _nacl_cell()
    geom = tmp_path / "geometry-unitcell"
    write_octopus(geom, cell)
    unitcell, _ = read_crystal_structure(geom, interface_mode="octopus")

    assert list(unitcell.symbols) == ["Na", "Cl"]
    np.testing.assert_allclose(unitcell.cell, cell.cell, atol=1e-6)


def test_read_crystal_structure_octopus_accepts_poscar(tmp_path):
    """--octopus still reads a POSCAR, converting the lattice Angstrom -> Bohr."""
    cell = _nacl_cell()  # lattice numbers treated as Angstrom in a POSCAR
    poscar = tmp_path / "POSCAR"
    write_vasp(poscar, cell)
    unitcell, _ = read_crystal_structure(poscar, interface_mode="octopus")

    bohr_angstrom = 1.0 / get_physical_units().Bohr
    assert list(unitcell.symbols) == ["Na", "Cl"]
    np.testing.assert_allclose(unitcell.cell, cell.cell * bohr_angstrom, atol=1e-6)


def test_get_born_octopus_accepts_poscar(tmp_path):
    """get_born_octopus also accepts the unit cell as a POSCAR."""
    write_vasp(tmp_path / "POSCAR", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE)

    borns, epsilon, atom_indices = get_born_octopus(
        tmp_path / "POSCAR",
        tmp_path / "epsilon",
        tmp_path / "born_charges",
    )

    assert atom_indices.tolist() == [0, 1]
    np.testing.assert_allclose(epsilon, 2.588021 * np.eye(3), atol=1e-6)
    np.testing.assert_allclose(borns[0], 1.141799 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(borns[1], -1.141799 * np.eye(3), atol=1e-5)
