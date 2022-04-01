"""Tests for PhonopyAtoms."""
from io import StringIO

import numpy as np
import yaml

from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict

symbols_SiO2 = ["Si"] * 2 + ["O"] * 4
symbols_AcO2 = ["Ac"] * 2 + ["O"] * 4
_lattice = [[4.65, 0, 0], [0, 4.75, 0], [0, 0, 3.25]]
_points = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.3, 0.3, 0.0],
    [0.7, 0.7, 0.0],
    [0.2, 0.8, 0.5],
    [0.8, 0.2, 0.5],
]
cell_SiO2 = PhonopyAtoms(cell=_lattice, scaled_positions=_points, symbols=symbols_SiO2)
cell_AcO2 = PhonopyAtoms(cell=_lattice, scaled_positions=_points, symbols=symbols_AcO2)


def test_SiO2():
    """Test of attributes by SiO2."""
    _test_cell(cell_SiO2, _lattice, _points, symbols_SiO2)


def test_SiO2_copy(helper_methods):
    """Test of PhonopyAtoms.copy() by SiO2."""
    helper_methods.compare_cells(cell_SiO2, cell_SiO2.copy())
    helper_methods.compare_cells_with_order(cell_SiO2, cell_SiO2.copy())


def test_AcO2(helper_methods):
    """Test of attributes by AcO2."""
    _test_cell(cell_AcO2, _lattice, _points, symbols_AcO2)
    helper_methods.compare_cells(cell_AcO2, cell_AcO2.copy())
    helper_methods.compare_cells_with_order(cell_AcO2, cell_AcO2.copy())


def test_AcO2_copy():
    """Test of PhonopyAtoms.copy() by AcO2."""


def _test_cell(cell: PhonopyAtoms, lattice, points, symbols):
    np.testing.assert_allclose(cell.cell, lattice, atol=1e-8)
    for s1, s2 in zip(cell.symbols, symbols):
        assert s1 == s2
    diff = cell.scaled_positions - points
    diff -= np.rint(diff)
    dist = np.linalg.norm(np.dot(diff, cell.cell), axis=1)
    np.testing.assert_allclose(dist, np.zeros(len(dist)), atol=1e-8)


def test_phonopy_atoms_SiO2():
    """Test of PhonopyAtoms __str__ by SiO2."""
    _test_phonopy_atoms(cell_SiO2)


def test_phonopy_atoms_AcO2():
    """Test of PhonopyAtoms __str__ by AcO2."""
    _test_phonopy_atoms(cell_AcO2)


def test_Cr_magnetic_moments(convcell_cr: PhonopyAtoms):
    """Test by Cr with [1, -1] magnetic moments."""
    convcell_cr.magnetic_moments = [1, -1]
    _test_phonopy_atoms(convcell_cr)
    convcell_cr.magnetic_moments = None


def test_Cr_copy_magnetic_moments(convcell_cr: PhonopyAtoms, helper_methods):
    """Test by Cr with [1, -1] magnetic moments."""
    convcell_cr.magnetic_moments = [1, -1]
    helper_methods.compare_cells(convcell_cr, convcell_cr.copy())
    helper_methods.compare_cells_with_order(convcell_cr, convcell_cr.copy())
    convcell_cr.magnetic_moments = None


def test_parse_cell_dict(helper_methods):
    """Test parse_cell_dict."""
    cell = cell_SiO2
    points = []
    for coord, mass, symbol in zip(cell.scaled_positions, cell.masses, cell.symbols):
        points.append({"symbol": symbol, "coordinates": coord, "mass": mass})
    cell_dict = {"lattice": cell_SiO2.cell, "points": points}
    _cell = parse_cell_dict(cell_dict)
    helper_methods.compare_cells_with_order(cell, _cell)


def _test_phonopy_atoms(cell: PhonopyAtoms):
    with StringIO(str(PhonopyAtoms(atoms=cell))) as f:
        data = yaml.safe_load(f)
        np.testing.assert_allclose(cell.cell, data["lattice"], atol=1e-8)
        positions = []
        magmoms = []
        for atom, symbol in zip(data["points"], cell.symbols):
            positions.append(atom["coordinates"])
            if "magnetic_moment" in atom:
                magmoms.append(atom["magnetic_moment"])
            assert atom["symbol"] == symbol

        diff = cell.scaled_positions - positions
        diff -= np.rint(diff)
        dist = np.linalg.norm(np.dot(diff, cell.cell), axis=1)
        np.testing.assert_allclose(dist, np.zeros(len(dist)), atol=1e-8)

        if magmoms:
            np.testing.assert_allclose(cell.magnetic_moments, magmoms, atol=1e-8)
