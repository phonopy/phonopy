# SPDX-License-Identifier: BSD-3-Clause
"""Tests of Octopus calculator interface."""

import sys

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


def test_octopus_structure_lines_roundtrip_triclinic():
    """The writer must be exact for cells with unequal, non-orthogonal vectors.

    Octopus reconstructs lattice vector i as LatticeParameters[i] * row i of
    %LatticeVectors, so each row must be normalized by its own length.
    """
    cell = PhonopyAtoms(
        symbols=["Na", "Cl"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.array([[3.0, 0.0, 0.0], [1.0, 4.0, 0.0], [1.0, 2.0, 5.0]]),
    )
    result = get_cell_from_octopus_lines(get_octopus_structure_lines(cell))

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


# Complex (frequency-dependent) Born charges as printed by Octopus when
# write_real is false: "Real:"/"Imaginary:" sub-blocks after each "Index:".
_BORN_FILE_COMPLEX = """\
# (Frequency-dependent) Born effective charge tensors
Index:     1   Label:    Na   Ionic charge:     1.0000
Real:
            1.141799            0.000000           -0.000000
            0.000000            1.141799            0.000000
            0.000000            0.000000            1.141799
Imaginary:
            0.010000            0.000000            0.000000
            0.000000            0.010000            0.000000
            0.000000            0.000000            0.010000
"""

# Born charges violating the cubic site symmetry of rock-salt NaCl by far
# more than the 0.1 threshold of symmetrize_borns_and_epsilon.
_BORN_FILE_BROKEN_SYMMETRY = _BORN_FILE.replace(
    "            1.141799            0.000000           -0.000000",
    "            2.141799            0.000000           -0.000000",
    1,
)


def test_parse_octopus_born_charges_complex_rejected(tmp_path):
    """Frequency-dependent (complex) Born charges give a clear error."""
    filename = tmp_path / "born_charges"
    filename.write_text(_BORN_FILE_COMPLEX)
    with pytest.raises(ValueError, match="complex"):
        parse_octopus_born_charges(filename)


def test_get_born_octopus_warns_on_broken_symmetry(tmp_path):
    """Symmetry-broken Born charges emit a UserWarning."""
    write_octopus(tmp_path / "geometry-unitcell", _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE_BROKEN_SYMMETRY)

    with pytest.warns(UserWarning):
        get_born_octopus(
            tmp_path / "geometry-unitcell",
            tmp_path / "epsilon",
            tmp_path / "born_charges",
        )


def test_phonopy_octopus_born_script_symmetry_broken(tmp_path, monkeypatch, capsys):
    """The CLI reports '# Symmetry broken' instead of printing a BORN file."""
    from phonopy.scripts.phonopy_octopus_born import run

    geom = tmp_path / "geometry-unitcell"
    write_octopus(geom, _nacl_cell())
    (tmp_path / "epsilon").write_text(_EPSILON_FILE)
    (tmp_path / "born_charges").write_text(_BORN_FILE_BROKEN_SYMMETRY)

    monkeypatch.setattr(sys, "argv", ["phonopy-octopus-born", str(geom), str(tmp_path)])
    with pytest.raises(SystemExit):
        run()
    assert "# Symmetry broken" in capsys.readouterr().out


def _parse_phonon_modes_file(filename):
    """Parse a phonon modes file following the Octopus phonon_modes.F90 reads.

    Mimics the Fortran parser (format 1.0): header of ``Version:``,
    ``Nmodes:``, ``Natoms:``, ``Np:`` and a ``Masses:`` block, then per mode
    a ``frequency:`` line, one line of 3 floats per supercell atom, and an
    ``alpha:`` line. ``#`` comment lines between blocks are skipped.
    """
    with open(filename) as f:
        lines = [
            line.rstrip("\n")
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    assert lines[0].split() == ["Version:", "1.0"]
    assert lines[1].split()[0] == "Nmodes:"
    num_modes = int(lines[1].split()[1])
    assert lines[2].split()[0] == "Natoms:"
    num_atoms = int(lines[2].split()[1])
    assert lines[3].split()[0] == "Np:"
    num_super = int(lines[3].split()[1])
    assert lines[4].split() == ["Masses:"]
    i = 5
    masses = []
    while len(masses) < num_atoms:
        masses.extend(float(x) for x in lines[i].split())
        i += 1
    assert len(masses) == num_atoms

    modes = []
    while i < len(lines):
        assert lines[i].split()[0] == "frequency:", f"line: {lines[i]!r}"
        freq = float(lines[i].split()[1])
        i += 1
        vec = []
        while not lines[i].startswith("alpha:"):
            vals = [float(x) for x in lines[i].split()]
            assert len(vals) == 3
            vec.append(vals)
            i += 1
        assert len(vec) == num_atoms
        alpha = float(lines[i].split()[1])
        i += 1
        modes.append({"frequency": freq, "vec": np.array(vec), "alpha": alpha})

    return num_modes, num_atoms, num_super, np.array(masses), modes
