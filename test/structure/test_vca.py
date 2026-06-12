"""Tests for the non-merge VCA scheme (apply_vca and weighted species)."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy.structure.atoms import PhonopyAtoms, build_species_table_from_mixtures
from phonopy.structure.cells import apply_vca
from phonopy.structure.symmetry import Symmetry

_a = 5.789
_zincblende_lattice = [[0, _a / 2, _a / 2], [_a / 2, 0, _a / 2], [_a / 2, _a / 2, 0]]


def _make_GeSn_co_located_cell() -> PhonopyAtoms:
    """Return a zincblende cell with Ge and Sn co-located on both sites."""
    return PhonopyAtoms(
        symbols=["Ge", "Sn", "Ge", "Sn"],
        scaled_positions=[
            [0, 0, 0],
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25],
        ],
        cell=_zincblende_lattice,
    )


def test_apply_vca_basic():
    """apply_vca attaches weights without merging atoms."""
    cell = _make_GeSn_co_located_cell()
    vca = apply_vca(cell, weights=[0.5, 0.5, 0.5, 0.5])
    assert len(vca) == len(cell)
    assert vca.symbols == cell.symbols
    np.testing.assert_array_equal(vca.numbers, cell.numbers)
    np.testing.assert_allclose(vca.masses, cell.masses)
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5, 0.5, 0.5])
    assert vca.has_weighted_species
    assert not vca.has_mixtures
    # Both Ge atoms share one weighted species; likewise both Sn atoms.
    assert len(vca.species_table) == 2
    np.testing.assert_array_equal(vca.species_ids, [0, 1, 0, 1])
    # The input cell is not modified.
    assert not cell.has_weighted_species


def test_apply_vca_isolated_atom_keeps_species():
    """An isolated atom with weight 1.0 keeps its unweighted species."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn", "Si"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        cell=_zincblende_lattice,
    )
    vca = apply_vca(cell, weights=[0.5, 0.5, 1.0])
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5, 1.0])
    si = vca.species_table[int(vca.species_ids[2])]
    assert si.symbol == "Si"
    assert si.weight is None


def test_apply_vca_all_unity_weights_on_normal_cell():
    """A cell without overlaps and all-1.0 weights stays an ordinary cell."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
        cell=_zincblende_lattice,
    )
    vca = apply_vca(cell, weights=[1.0, 1.0])
    assert not vca.has_weighted_species
    assert vca.mixture_weights is None
    assert vca.species_table == cell.species_table


def test_apply_vca_degenerate_same_species_allowed():
    """Co-located atoms of the same (element, weight) are accepted."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Ge"],
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        cell=_zincblende_lattice,
    )
    vca = apply_vca(cell, weights=[0.5, 0.5])
    assert len(vca.species_table) == 1
    np.testing.assert_array_equal(vca.species_ids, [0, 0])
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5])


def test_apply_vca_length_mismatch():
    """Length of weights must match the number of atoms."""
    cell = _make_GeSn_co_located_cell()
    with pytest.raises(ValueError):
        apply_vca(cell, weights=[0.5, 0.5])


def test_apply_vca_isolated_atom_weight_error():
    """An isolated atom must carry weight 1.0."""
    cell = PhonopyAtoms(
        symbols=["Ge"],
        scaled_positions=[[0, 0, 0]],
        cell=_zincblende_lattice,
    )
    with pytest.raises(ValueError):
        apply_vca(cell, weights=[0.5])


def test_apply_vca_group_sum_error():
    """Weights of a co-located group must sum to 1.0."""
    cell = _make_GeSn_co_located_cell()
    with pytest.raises(ValueError):
        apply_vca(cell, weights=[0.6, 0.6, 0.5, 0.5])


def test_apply_vca_rejects_weighted_cell():
    """apply_vca cannot be applied twice."""
    vca = apply_vca(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        apply_vca(vca, weights=[0.5, 0.5, 0.5, 0.5])


def test_apply_vca_rejects_merge_cell():
    """apply_vca cannot be applied to a merge-style VCA cell."""
    species, ids = build_species_table_from_mixtures([[("Ge", 0.5), ("Sn", 0.5)]])
    merge_cell = PhonopyAtoms(
        cell=_zincblende_lattice,
        scaled_positions=[[0, 0, 0]],
        species_table=species,
        species_ids=ids,
    )
    with pytest.raises(ValueError):
        apply_vca(merge_cell, weights=[1.0])


def test_apply_vca_rejects_magnetic_cell():
    """apply_vca does not support cells carrying magnetic moments."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [0, 0, 0]],
        cell=_zincblende_lattice,
        magnetic_moments=[1.0, -1.0],
    )
    with pytest.raises(ValueError):
        apply_vca(cell, weights=[0.5, 0.5])


def test_apply_vca_symprec_controls_grouping():
    """Overlap detection follows the symprec tolerance."""
    cell = PhonopyAtoms(
        symbols=["Ge", "Sn"],
        scaled_positions=[[0, 0, 0], [1e-4, 0, 0]],
        cell=_zincblende_lattice,
    )
    # Within a loose tolerance the two atoms form one group.
    vca = apply_vca(cell, weights=[0.5, 0.5], symprec=1e-3)
    np.testing.assert_allclose(vca.mixture_weights, [0.5, 0.5])
    # With the default tolerance they are isolated atoms, whose weights
    # must be 1.0.
    with pytest.raises(ValueError):
        apply_vca(cell, weights=[0.5, 0.5])


def test_symmetry_GeSn_50_50_co_located():
    """Diamond symmetry of the 50/50 co-located cell is found by spglib."""
    vca = apply_vca(_make_GeSn_co_located_cell(), weights=[0.5, 0.5, 0.5, 0.5])
    symmetry = Symmetry(vca)
    assert symmetry.dataset.number == 227  # Fd-3m
    assert len(symmetry.symmetry_operations["rotations"]) == 48
    # Ge@A is equivalent to Ge@B; likewise for Sn. One independent atom
    # per species.
    np.testing.assert_array_equal(symmetry.get_map_atoms(), [0, 1, 0, 1])
    np.testing.assert_array_equal(symmetry.get_independent_atoms(), [0, 1])


def test_symmetry_distinct_concentrations_lower_symmetry():
    """Sites with different concentrations are not symmetry-equivalent."""
    vca = apply_vca(_make_GeSn_co_located_cell(), weights=[0.9, 0.1, 0.5, 0.5])
    symmetry = Symmetry(vca)
    assert symmetry.dataset.number == 216  # F-43m, no A<->B swap
    assert len(vca.species_table) == 4
