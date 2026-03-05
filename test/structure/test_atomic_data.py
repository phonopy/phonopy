"""Tests for atomic_data."""

from __future__ import annotations

from phonopy.structure.atomic_data import get_atomic_data


def test_isotope_data_values_are_tuple_of_isotope_tuples_or_none():
    """Test isotope_data value structure.

    Each value must be either None or a tuple of isotope tuples:
    (mass_number: int, isotope_mass: float, abundance: float).

    """
    isotope_data = get_atomic_data().isotope_data

    for symbol, isotopes in isotope_data.items():
        if isotopes is None:
            continue

        assert isinstance(isotopes, tuple), f"{symbol}: value must be tuple or None"
        for isotope in isotopes:
            assert isinstance(isotope, tuple), (
                f"{symbol}: each isotope entry must be tuple, got {type(isotope)!r}"
            )
            assert len(isotope) == 3, (
                f"{symbol}: isotope entry must have 3 items, got {len(isotope)}"
            )
            mass_number, isotope_mass, abundance = isotope
            assert isinstance(mass_number, int), (
                f"{symbol}: mass number must be int, got {type(mass_number)!r}"
            )
            assert isinstance(isotope_mass, float), (
                f"{symbol}: isotope mass must be float, got {type(isotope_mass)!r}"
            )
            assert isinstance(abundance, float), (
                f"{symbol}: abundance must be float, got {type(abundance)!r}"
            )
