"""Tests for PhonopyAtoms."""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO

import numpy as np
import pytest
import yaml

from phonopy import Phonopy
from phonopy.structure.atoms import (
    PhonopyAtoms,
    _Species,
    build_species_table_from_mixtures,
    build_species_table_from_symbols,
    parse_cell_dict,
)

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
    for s1, s2 in zip(cell.symbols, symbols, strict=True):
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
    for coord, mass, symbol in zip(
        cell.scaled_positions, cell.masses, cell.symbols, strict=True
    ):
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

    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
    )
    assert symbols == cell.symbols
    np.testing.assert_allclose(cell.masses, ph_nacl.unitcell.masses)
    # "Cl" and "Cl1" must get distinct species ids even though their atomic
    # number is the same.
    assert len(set(cell.species_ids.tolist())) == len(set(symbols))
    np.testing.assert_equal(cell.numbers, ph_nacl.unitcell.numbers)

    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
        masses=masses,
    )
    assert symbols == cell.symbols
    np.testing.assert_allclose(cell.masses, masses)

    with pytest.raises(ValueError) as e:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            numbers=[200] * len(symbols),
        )
    assert str(e.value) == "Atomic numbers must be in 1..118."

    symbols[-1] = "Cl_1"
    with pytest.raises(RuntimeError) as e2:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            symbols=symbols,
        )
    assert str(e2.value) == "Invalid symbol: Cl_1."

    symbols[-1] = "Cl_0"
    with pytest.raises(RuntimeError) as e2:
        _ = PhonopyAtoms(
            cell=ph_nacl.unitcell.cell,
            scaled_positions=ph_nacl.unitcell.scaled_positions,
            symbols=symbols,
        )
    assert str(e2.value) == "Invalid symbol: Cl_0."


def _test_phonopy_atoms(cell: PhonopyAtoms):
    with StringIO(str(cell.copy())) as f:
        data = yaml.safe_load(f)
        np.testing.assert_allclose(cell.cell, data["lattice"], atol=1e-8)
        positions = []
        magmoms = []
        for atom, symbol in zip(data["points"], cell.symbols, strict=True):
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


def test_PhonopyAtoms_mixture_construct_and_yaml_roundtrip():
    """PhonopyAtoms holds mixed-species sites and round-trips through YAML."""
    species, ids = build_species_table_from_mixtures(
        [[("Si", 1.0)], [("Ge", 0.5), ("Sn", 0.5)]]
    )
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        species_table=species,
        species_ids=ids,
    )
    assert cell.has_mixtures
    assert cell.symbols == ["Si", "GeSn"]
    np.testing.assert_allclose(cell.species_ids, [0, 1])
    # Mass of a mixed-species site is the weighted average of constituents.
    np.testing.assert_allclose(cell.masses[1], 0.5 * 72.64 + 0.5 * 118.710)
    # cell.numbers is undefined for cells with mixed-species sites.
    with pytest.raises(RuntimeError):
        _ = cell.numbers

    # YAML round-trip
    data = yaml.safe_load(str(cell))
    assert data["points"][0]["symbol"] == "Si"
    assert "mixture" not in data["points"][0]
    assert data["points"][1]["symbol"] == "GeSn"
    assert data["points"][1]["mixture"] == [["Ge", 0.5], ["Sn", 0.5]]

    cell2 = parse_cell_dict(data)
    assert cell2 is not None
    assert cell2.has_mixtures
    assert cell2.symbols == ["Si", "GeSn"]
    np.testing.assert_allclose(cell2.species_ids, [0, 1])
    np.testing.assert_allclose(cell2.masses, cell.masses)


def test_build_species_table_from_mixtures_weight_sum_error():
    """Weights of each mixture entry must sum to 1.0."""
    with pytest.raises(ValueError):
        build_species_table_from_mixtures([[("Ge", 0.3), ("Sn", 0.6)]])


def test_build_species_table_from_mixtures_sort_constituents_default():
    """By default, constituents are sorted so input order does not matter."""
    species_a, ids_a = build_species_table_from_mixtures([[("Ge", 0.5), ("Sn", 0.5)]])
    species_b, ids_b = build_species_table_from_mixtures([[("Sn", 0.5), ("Ge", 0.5)]])
    assert species_a == species_b
    np.testing.assert_array_equal(ids_a, ids_b)
    assert species_a[0].symbol == "GeSn"
    assert species_a[0].mixture == (("Ge", 0.5), ("Sn", 0.5))


def test_build_species_table_from_mixtures_sort_constituents_off():
    """Passing sort_constituents=False preserves caller order."""
    species, _ = build_species_table_from_mixtures(
        [[("Sn", 0.5), ("Ge", 0.5)]], sort_constituents=False
    )
    assert species[0].symbol == "SnGe"
    assert species[0].mixture == (("Sn", 0.5), ("Ge", 0.5))


def test_build_species_table_from_mixtures_distinct_weights_get_suffixes():
    """Same constituents, different weights produce distinct suffixed labels."""
    species, ids = build_species_table_from_mixtures(
        [
            [("Ge", 0.5), ("Sn", 0.5)],
            [("Ge", 0.25), ("Sn", 0.75)],
        ]
    )
    assert [sp.symbol for sp in species] == ["GeSn1", "GeSn2"]
    np.testing.assert_array_equal(ids, [0, 1])


def test_build_species_table_from_symbols_first_appearance_default():
    """Without order, the table keeps first-appearance order."""
    table, ids = build_species_table_from_symbols(["Au", "S", "Mo", "S", "Au"])
    assert [sp.symbol for sp in table] == ["Au", "S", "Mo"]
    np.testing.assert_array_equal(ids, [0, 1, 2, 1, 0])


def test_build_species_table_from_symbols_order_reorders_table():
    """A given order reorders the table while keeping per-atom identity."""
    symbols = ["Au", "S", "Mo", "S", "Au"]
    table, ids = build_species_table_from_symbols(symbols, order=["Au", "Mo", "S"])
    assert [sp.symbol for sp in table] == ["Au", "Mo", "S"]
    # Per-atom symbols are unchanged by the reordering.
    assert [table[i].symbol for i in ids] == symbols


def test_build_species_table_from_symbols_order_drops_unused():
    """Symbols listed in order but absent from symbols are dropped."""
    table, ids = build_species_table_from_symbols(["Au", "S"], order=["Au", "Mo", "S"])
    assert [sp.symbol for sp in table] == ["Au", "S"]
    np.testing.assert_array_equal(ids, [0, 1])


def test_build_species_table_from_symbols_order_missing_symbol_raises():
    """A symbol present in symbols but missing from order is an error."""
    with pytest.raises(ValueError):
        build_species_table_from_symbols(["Au", "S", "Mo"], order=["Au", "Mo"])


def test_PhonopyAtoms_input_mutual_exclusion():
    """Reject simultaneous symbols / numbers / species_table."""
    species, ids = build_species_table_from_mixtures([[("Si", 1.0)]])
    with pytest.raises(ValueError):
        PhonopyAtoms(
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            scaled_positions=[[0, 0, 0]],
            symbols=["Si"],
            species_table=species,
            species_ids=ids,
        )
    with pytest.raises(ValueError):
        # species_table without species_ids is rejected.
        PhonopyAtoms(
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            scaled_positions=[[0, 0, 0]],
            species_table=species,
        )


def test_Species_weight_validation():
    """Weighted species accept (0, 1) and reject everything else."""
    sp = _Species(symbol="Ge", atomic_number=32, weight=0.5)
    assert sp.weight == 0.5
    for bad in (0.0, 1.0, 1.5, -0.5):
        with pytest.raises(ValueError):
            _Species(symbol="Ge", atomic_number=32, weight=bad)
    # A merged mixture species cannot carry a concentration weight.
    with pytest.raises(ValueError):
        _Species(
            symbol="GeSn",
            atomic_number=None,
            mixture=(("Ge", 0.5), ("Sn", 0.5)),
            weight=0.5,
        )


def _make_weighted_GeSn_cell() -> PhonopyAtoms:
    """Return a non-merge species-resolved cell: Ge/Sn co-located on both sites."""
    species = [
        _Species(symbol="Ge", atomic_number=32, weight=0.5),
        _Species(symbol="Sn", atomic_number=50, weight=0.5),
    ]
    a = 5.789
    return PhonopyAtoms(
        cell=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        species_table=species,
        species_ids=[0, 1, 0, 1],
    )


def test_PhonopyAtoms_weighted_species_properties():
    """Weighted real species keep real elements and expose mixture_weights."""
    cell = _make_weighted_GeSn_cell()
    assert cell.has_weighted_species
    assert not cell.has_mixtures
    assert cell.symbols == ["Ge", "Sn", "Ge", "Sn"]
    # Real atomic numbers and masses, unlike merged mixture sites.
    np.testing.assert_array_equal(cell.numbers, [32, 50, 32, 50])
    np.testing.assert_allclose(cell.masses, [72.64, 118.71, 72.64, 118.71])
    np.testing.assert_allclose(cell.mixture_weights, [0.5, 0.5, 0.5, 0.5])
    # spglib types discriminate species (and thereby weights).
    np.testing.assert_array_equal(cell.totuple()[2], [0, 1, 0, 1])


def test_PhonopyAtoms_weighted_species_copy():
    """copy() preserves weighted species through the species table."""
    cell = _make_weighted_GeSn_cell()
    cell2 = cell.copy()
    assert cell2.has_weighted_species
    np.testing.assert_allclose(cell2.mixture_weights, cell.mixture_weights)
    assert cell2.species_table == cell.species_table


def test_PhonopyAtoms_mixture_weights_none_without_weights():
    """mixture_weights is None for ordinary and merge-style mixture cells."""
    assert cell_SiO2.mixture_weights is None
    assert not cell_SiO2.has_weighted_species
    species, ids = build_species_table_from_mixtures(
        [[("Si", 1.0)], [("Ge", 0.5), ("Sn", 0.5)]]
    )
    merge_cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        species_table=species,
        species_ids=ids,
    )
    assert merge_cell.mixture_weights is None
    assert not merge_cell.has_weighted_species


def test_PhonopyAtoms_mixture_weights_unity_for_unweighted_atom():
    """Atoms without a species weight contribute 1.0 to mixture_weights."""
    species = [
        _Species(symbol="Ge", atomic_number=32, weight=0.5),
        _Species(symbol="Sn", atomic_number=50, weight=0.5),
        _Species(symbol="Si", atomic_number=14),
    ]
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        species_table=species,
        species_ids=[0, 1, 2],
    )
    np.testing.assert_allclose(cell.mixture_weights, [0.5, 0.5, 1.0])


def test_PhonopyAtoms_weighted_species_yaml_roundtrip():
    """Weighted species round-trip through YAML; pure sites show weight 1.0.

    In a non-merge mixture cell the yaml is self-describing: every atom
    carries an explicit weight, with a pure site (weight=None in the model)
    written as 1.0 and normalized back to None on read.

    """
    species = [
        _Species(symbol="Ge", atomic_number=32, weight=0.5),
        _Species(symbol="Sn", atomic_number=50, weight=0.5),
        _Species(symbol="Si", atomic_number=14),
    ]
    cell = PhonopyAtoms(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        species_table=species,
        species_ids=[0, 1, 2],
    )

    # Every atom carries an explicit weight; the pure Si site shows 1.0.
    data = yaml.safe_load(str(cell))
    assert data["points"][0]["weight"] == 0.5
    assert data["points"][1]["weight"] == 0.5
    assert data["points"][2]["weight"] == 1.0

    cell2 = parse_cell_dict(data)
    assert cell2 is not None
    assert cell2.has_weighted_species
    assert cell2.symbols == ["Ge", "Sn", "Si"]
    np.testing.assert_allclose(cell2.mixture_weights, [0.5, 0.5, 1.0])
    # The pure site is normalized back to a plain species (weight=None).
    assert cell2.species_table[cell2.species_ids[2]].weight is None


def test_parse_cell_dict_all_unity_weights_is_ordinary_cell():
    """A hand-written cell with every weight 1.0 reads as an ordinary cell."""
    data = {
        "lattice": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "points": [
            {"symbol": "Si", "coordinates": [0, 0, 0], "weight": 1.0},
            {"symbol": "Si", "coordinates": [0.5, 0.5, 0.5], "weight": 1.0},
        ],
    }
    cell = parse_cell_dict(data)
    assert cell is not None
    assert not cell.has_weighted_species
    assert cell.mixture_weights is None
    assert cell.symbols == ["Si", "Si"]


def test_PhonopyAtoms_merge_and_weighted_cannot_coexist():
    """Merged mixture sites and weighted species are mutually exclusive."""
    merged, _ = build_species_table_from_mixtures([[("Ge", 0.5), ("Sn", 0.5)]])
    species = [merged[0], _Species(symbol="Si", atomic_number=14, weight=0.5)]
    with pytest.raises(RuntimeError):
        PhonopyAtoms(
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            species_table=species,
            species_ids=[0, 1],
        )


def test_import_deprecated_atom_data():
    """Test import of deprecated atom_data."""
    with pytest.warns(DeprecationWarning):
        from phonopy.structure.atoms import atom_data  # noqa: F401


def test_import_deprecated_symbol_map():
    """Test import of deprecated symbol_map."""
    with pytest.warns(DeprecationWarning):
        from phonopy.structure.atoms import symbol_map  # noqa: F401


def test_import_deprecated_isotope_data():
    """Test import of deprecated isotope_data."""
    with pytest.warns(DeprecationWarning):
        from phonopy.structure.atoms import isotope_data  # noqa: F401
