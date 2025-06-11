"""Tests for PhonopyAtoms."""

from collections.abc import Callable
from io import StringIO

import numpy as np
import pytest
import yaml

from phonopy import Phonopy
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


def test_SiO2_copy(helper_methods: Callable):
    """Test of PhonopyAtoms.copy() by SiO2."""
    helper_methods.compare_cells(cell_SiO2, cell_SiO2.copy())
    helper_methods.compare_cells_with_order(cell_SiO2, cell_SiO2.copy())


def test_AcO2(helper_methods: Callable):
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


@pytest.mark.parametrize(
    "is_ncl,is_flat", [(False, False), (False, True), (True, False), (True, True)]
)
def test_Cr_magnetic_moments(convcell_cr: PhonopyAtoms, is_ncl: bool, is_flat: bool):
    """Test by Cr with magnetic moments."""
    print(len(convcell_cr))
    if is_ncl:
        if is_flat:
            convcell_cr.magnetic_moments = [0, 0, 1, 0, 0, -1]
        else:
            convcell_cr.magnetic_moments = [[0, 0, 1], [0, 0, -1]]
        np.testing.assert_allclose(
            convcell_cr.magnetic_moments, [[0, 0, 1], [0, 0, -1]]
        )
    else:
        if is_flat:
            convcell_cr.magnetic_moments = [1, -1]
        else:
            convcell_cr.magnetic_moments = [[1], [-1]]
        np.testing.assert_allclose(convcell_cr.magnetic_moments, [1, -1])
    _test_phonopy_atoms(convcell_cr)
    convcell_cr.magnetic_moments = None


@pytest.mark.parametrize("is_ncl", [False, True])
def test_Cr_copy_magnetic_moments(
    convcell_cr: PhonopyAtoms, is_ncl: bool, helper_methods
):
    """Test by Cr with magnetic moments."""
    if is_ncl:
        convcell_cr.magnetic_moments = [[0, 0, 1], [0, 0, -1]]
    else:
        convcell_cr.magnetic_moments = [1, -1]
    helper_methods.compare_cells(convcell_cr, convcell_cr.copy())
    helper_methods.compare_cells_with_order(convcell_cr, convcell_cr.copy())
    convcell_cr.magnetic_moments = None


def test_parse_cell_dict(helper_methods: Callable):
    """Test parse_cell_dict."""
    cell = cell_SiO2
    points = []
    for coord, mass, symbol in zip(cell.scaled_positions, cell.masses, cell.symbols):
        points.append({"symbol": symbol, "coordinates": coord, "mass": mass})
    cell_dict = {"lattice": cell_SiO2.cell, "points": points}
    _cell = parse_cell_dict(cell_dict)
    helper_methods.compare_cells_with_order(cell, _cell)


def test_PhonopyAtoms_with_Xn_symbol(ph_nacl: Phonopy):
    """Test of PhonopyAtoms with Xn symbol."""
    symbols = ph_nacl.unitcell.symbols
    symbols[-1] = "Cl1"
    masses = ph_nacl.unitcell.masses
    masses[-1] = 70.0
    numbers = ph_nacl.unitcell.numbers
    numbers[-1] = numbers[-1] + PhonopyAtoms._MOD_DIVISOR

    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
    )
    assert symbols == cell.symbols
    np.testing.assert_allclose(cell.masses, ph_nacl.unitcell.masses)
    np.testing.assert_equal(cell.numbers_with_shifts, numbers)
    np.testing.assert_equal(cell.numbers, ph_nacl.unitcell.numbers)

    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
        masses=masses,
    )
    assert symbols == cell.symbols
    np.testing.assert_allclose(cell.masses, masses)

    with pytest.raises(RuntimeError) as e:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            numbers=numbers,
        )
    assert str(e.value) == "Atomic numbers cannot be larger than 118."

    symbols[-1] = "Cl_1"
    with pytest.raises(RuntimeError) as e:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            symbols=symbols,
        )
    assert str(e.value) == "Invalid symbol: Cl_1."

    symbols[-1] = "Cl_0"
    with pytest.raises(RuntimeError) as e:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            symbols=symbols,
        )
    assert str(e.value) == "Invalid symbol: Cl_0."


def _test_phonopy_atoms(cell: PhonopyAtoms):
    with StringIO(str(cell.copy())) as f:
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
            assert cell.magnetic_moments is not None
            np.testing.assert_allclose(cell.magnetic_moments, magmoms, atol=1e-8)


def test_formula():
    """Test of PhonopyAtoms formula property."""
    # Test SiO2
    assert cell_SiO2.formula == "O4Si2"

    # Test AcO2
    assert cell_AcO2.formula == "Ac2O4"

    # Test with indexed symbols
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Fe", "Fe1"],
        masses=[55.845, 55.845],  # Add masses for Fe
    )
    assert cell.formula == "Fe2"

    # Test single atom
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0]],
        symbols=["H"],
    )
    assert cell.formula == "H"

    # Test edge cases
    # Empty cell
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[],
        symbols=[],
    )
    assert cell.formula == ""

    # Multiple digit indices
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0]],
        symbols=["Fe11", "Fe2"],
        masses=[55.845, 55.845],
    )
    assert cell.formula == "Fe2"


def test_formula_complex():
    """Test formula property with more complex structures."""
    # Test structure with multiple elements
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        symbols=["Fe", "O", "O", "Ti"],
    )
    assert cell.formula == "FeO2Ti"

    # Test structure with indexed symbols
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
        symbols=["Fe1", "Fe2", "O"],
        masses=[55.845, 55.845, 15.999],  # Add masses for Fe and O
    )
    assert cell.formula == "Fe2O"


def test_reduced_formula():
    """Test reduced_formula property."""
    # Test reduction of larger numbers
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        symbols=["Fe", "Fe", "O", "O"],
    )
    assert cell.formula == "Fe2O2"
    assert cell.reduced_formula == "FeO"

    # Test no reduction needed
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
        symbols=["Fe", "O", "O"],
    )
    assert cell.formula == "FeO2"
    assert cell.reduced_formula == "FeO2"

    # Test with indexed symbols
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0]],
        symbols=["Fe1", "Fe2", "O1", "O2"],
        masses=[55.845, 55.845, 15.999, 15.999],  # Add masses for Fe and O
    )
    assert cell.formula == "Fe2O2"
    assert cell.reduced_formula == "FeO"

    # Test single atom
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0]],
        symbols=["H"],
    )
    assert cell.formula == "H"
    assert cell.reduced_formula == "H"

    # Test edge cases
    # Empty cell
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[],
        symbols=[],
    )
    assert cell.reduced_formula == ""

    # Large numbers that reduce
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0]] * 12,
        symbols=["Fe"] * 6 + ["O"] * 6,
    )
    assert cell.formula == "Fe6O6"
    assert cell.reduced_formula == "FeO"

    # Prime numbers that don't reduce
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0]] * 5,
        symbols=["Fe"] * 2 + ["O"] * 3,
    )
    assert cell.formula == "Fe2O3"
    assert cell.reduced_formula == "Fe2O3"


@pytest.mark.parametrize(
    "symbols,expected_formula,expected_normalized",
    [
        (["Fe", "Fe", "O", "O"], "Fe2O2", "Fe0.5O0.5"),
        (["Fe", "O", "O"], "FeO2", "Fe0.333O0.667"),
        (["Fe", "Fe", "O", "O", "O"], "Fe2O3", "Fe0.4O0.6"),
        (["H"], "H", "H1.0"),
        ([], "", ""),
        (["Fe"] * 6 + ["O"] * 6, "Fe6O6", "Fe0.5O0.5"),
        # Test with indexed symbols
        (["Fe1", "Fe2", "O"], "Fe2O", "Fe0.667O0.333"),
    ],
)
def test_formulas(symbols, expected_formula, expected_normalized):
    """Test all formula properties."""
    # Create cell with appropriate masses for indexed symbols
    masses = None
    if any(s.rstrip("0123456789") != s for s in symbols):
        masses = []
        for s in symbols:
            base = s.rstrip("0123456789")
            if base == "Fe":
                masses.append(55.845)
            elif base == "O":
                masses.append(15.999)
            elif base == "H":
                masses.append(1.008)

    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0]] * len(symbols),
        symbols=symbols,
        masses=masses,
    )

    assert cell.formula == expected_formula
    assert cell.normalized_formula == expected_normalized
