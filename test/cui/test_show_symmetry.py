"""Tests for phonopy.cui.show_symmetry helpers."""

import pytest
import spglib

from phonopy.cui.show_symmetry import _rebuild_bravais_cell
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture, build_mixture_cell

_rocksalt_lattice = [[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]]


def _refine(cell: PhonopyAtoms, symprec: float = 1e-5) -> PhonopyAtoms:
    spglib_cell = spglib.refine_cell(cell.totuple(), symprec)
    assert spglib_cell is not None
    return _rebuild_bravais_cell(cell, *spglib_cell)


def test_rebuild_bravais_cell_weighted_preserves_symbols_and_weights():
    """Non-merge weighted cells keep real symbols and weights (regression).

    Before the fix the rebuild only guarded ``has_mixtures``, so a weighted
    (non-merge) cell fell through to the atomic-number branch and misread the
    species ids that spglib returns for a site-mixture cell as atomic numbers.

    """
    cell = PhonopyAtoms(
        cell=_rocksalt_lattice,
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Ge", "Sn", "Te"],
    )
    vca = apply_site_mixture(cell, weights=[0.5, 0.5, 1.0])
    assert vca.has_weighted_species

    bravais = _refine(vca)

    # Real species (Ge/Sn/Te), not dummy elements from a species-id misread.
    assert set(bravais.symbols) == {"Ge", "Sn", "Te"}
    assert bravais.mixture_weights is not None
    weight_by_symbol = dict(zip(bravais.symbols, bravais.mixture_weights, strict=True))
    assert weight_by_symbol["Ge"] == 0.5
    assert weight_by_symbol["Sn"] == 0.5
    assert weight_by_symbol["Te"] == 1.0


def test_rebuild_bravais_cell_merge_keeps_mixture():
    """Merge-style mixture sites survive the Bravais rebuild."""
    cell = PhonopyAtoms(
        cell=_rocksalt_lattice,
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Ge", "Sn", "Te"],
    )
    merged = build_mixture_cell(cell, [0.5, 0.5, 1.0])
    assert merged.has_mixtures

    bravais = _refine(merged)

    assert bravais.has_mixtures
    assert "Te" in bravais.symbols


def test_rebuild_bravais_cell_ordinary():
    """Ordinary cells map atomic numbers directly (regression guard)."""
    cell = PhonopyAtoms(
        cell=_rocksalt_lattice,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Na", "Cl"],
    )

    bravais = _refine(cell)

    assert set(bravais.symbols) == {"Na", "Cl"}
    assert bravais.mixture_weights is None


def test_rebuild_bravais_cell_suffixed_symbols_raise():
    """Suffixed symbols sharing an atomic number are refused, not silently lost."""
    cell = PhonopyAtoms(
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Cl", "Cl1"],
    )
    assert not cell.is_site_mixture

    with pytest.raises(ValueError, match="suffixed symbols"):
        _refine(cell)
