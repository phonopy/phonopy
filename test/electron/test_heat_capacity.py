# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the electronic heat capacity and the thermal-properties container."""

from __future__ import annotations

import numpy as np
import pytest
from test_electron import WEIGHTS_AL, _al_eigenvalues
from test_electron_free_energy_from_dos import (
    FERMI,
    WINDOW,
    _flat_dos,
    _half_filled_band,
    _n_electrons,
)

from phonopy.electron.kpoint_sum import (
    compute_free_energy_by_kpoint_sum,
    compute_thermal_properties_by_kpoint_sum,
)
from phonopy.electron.states import ElectronicStates, fermi_dirac_occupation
from phonopy.electron.tetrahedron import (
    compute_free_energy_by_tetrahedron,
    compute_thermal_properties_by_tetrahedron,
    free_energy_from_dos,
    thermal_properties_from_dos,
)
from phonopy.physical_units import get_physical_units
from phonopy.qha.thermal import compute_electronic_contributions_from_states


def _al_states() -> ElectronicStates:
    """Return the Al states of test_electron, which carry no sampling grid."""
    return ElectronicStates(
        eigenvalues=_al_eigenvalues(), weights=WEIGHTS_AL, n_electrons=3.0
    )


def _central_t_dsdt(entropy: np.ndarray, temperatures: np.ndarray) -> float:
    """Return T dS/dT at the middle of three equally spaced temperatures."""
    d_t = temperatures[2] - temperatures[0]
    return float(temperatures[1] * (entropy[2] - entropy[0]) / d_t)


def test_heat_capacity_matches_the_sommerfeld_limit():
    """Test the heat capacity against (pi^2/3) k^2 T g(E_F).

    0 K is left out of the temperatures on purpose: it is computed anyway.

    """
    g0 = 2.0
    energies, dos = _flat_dos(g0)
    temperatures = np.array([100.0, 300.0])
    properties = thermal_properties_from_dos(
        energies, dos, _n_electrons(energies, dos), temperatures, FERMI
    )

    kb = get_physical_units().KB
    expected = (np.pi**2 / 3.0) * kb**2 * temperatures * g0
    np.testing.assert_array_equal(properties.temperatures, temperatures)
    np.testing.assert_allclose(properties.heat_capacity, expected, rtol=2e-4)


def test_heat_capacity_follows_the_chemical_potential():
    """Test C_V = T dS/dT where mu moves with T.

    On a density of states that rises through the Fermi level, mu moves with
    T and the A_1^2 / A_0 term of C_V is about a tenth of it at 1000 K. The
    heat capacity at fixed mu is computed here too, to show that the
    comparison with T dS/dT would fail without that term.

    """
    energies = np.linspace(FERMI - WINDOW, FERMI + WINDOW, 8001)
    dos = 2.0 * np.exp(2.0 * (energies - FERMI))
    temperatures = np.array([999.0, 1000.0, 1001.0])
    properties = thermal_properties_from_dos(
        energies, dos, _n_electrons(energies, dos), temperatures, FERMI
    )

    heat_capacity = properties.heat_capacity[1]
    assert heat_capacity == pytest.approx(
        _central_t_dsdt(properties.entropy, temperatures), rel=1e-5
    )

    kb = get_physical_units().KB
    mu = properties.chemical_potential[1]
    occupation = fermi_dirac_occupation(energies, mu, kb * 1000.0)
    spread = dos * occupation * (1 - occupation)
    at_fixed_mu = np.trapezoid(spread * (energies - mu) ** 2, energies) / (
        kb * 1000.0**2
    )
    assert abs(at_fixed_mu - heat_capacity) > 0.05 * heat_capacity


def test_heat_capacity_by_tetrahedron_matches_t_dsdt():
    """Test C_V = T dS/dT through the tetrahedron method."""
    temperatures = np.array([299.0, 300.0, 301.0])
    properties = compute_thermal_properties_by_tetrahedron(
        _half_filled_band([8, 8, 8]), temperatures
    )

    assert properties.heat_capacity[1] > 0
    assert properties.heat_capacity[1] == pytest.approx(
        _central_t_dsdt(properties.entropy, temperatures), rel=1e-5
    )


def test_heat_capacity_by_kpoint_sum_matches_t_dsdt():
    """Test C_V = T dS/dT through the k-point sum, on the Al states."""
    temperatures = np.array([999.0, 1000.0, 1001.0])
    properties = compute_thermal_properties_by_kpoint_sum(_al_states(), temperatures)

    assert properties.heat_capacity[1] > 0
    assert properties.heat_capacity[1] == pytest.approx(
        _central_t_dsdt(properties.entropy, temperatures), rel=1e-5
    )


def test_heat_capacity_is_zero_at_zero_temperature():
    """Test that both routes report C_V = 0 at 0 K."""
    temperatures = np.array([0.0, 300.0])
    by_tetrahedron = compute_thermal_properties_by_tetrahedron(
        _half_filled_band([8, 8, 8]), temperatures
    )
    by_sum = compute_thermal_properties_by_kpoint_sum(_al_states(), temperatures)

    assert by_tetrahedron.heat_capacity[0] == 0.0
    assert by_sum.heat_capacity[0] == 0.0


def test_temperatures_without_zero_kelvin_give_the_same_values():
    """Test that leaving 0 K out changes nothing but the length."""
    states = _half_filled_band([8, 8, 8])
    with_zero = compute_thermal_properties_by_tetrahedron(states, [0.0, 300.0])
    without_zero = compute_thermal_properties_by_tetrahedron(states, [300.0])

    for field in ("free_energy", "entropy", "heat_capacity", "chemical_potential"):
        np.testing.assert_array_equal(
            getattr(without_zero, field), getattr(with_zero, field)[1:]
        )


def test_deprecated_tuple_functions_return_the_same_values():
    """Test that the deprecated tuple functions warn and agree with the new ones."""
    temperatures = np.array([0.0, 300.0, 1000.0])

    energies, dos = _flat_dos(2.0)
    n_electrons = _n_electrons(energies, dos)
    with pytest.warns(DeprecationWarning, match="free_energy_from_dos"):
        free_energy, entropy, mu = free_energy_from_dos(
            energies, dos, n_electrons, temperatures, FERMI
        )
    properties = thermal_properties_from_dos(
        energies, dos, n_electrons, temperatures, FERMI
    )
    np.testing.assert_array_equal(free_energy, properties.free_energy)
    np.testing.assert_array_equal(entropy, properties.entropy)
    np.testing.assert_array_equal(mu, properties.chemical_potential)

    states = _half_filled_band([8, 8, 8])
    with pytest.warns(DeprecationWarning, match="compute_free_energy_by_tetrahedron"):
        free_energy, entropy = compute_free_energy_by_tetrahedron(states, temperatures)
    properties = compute_thermal_properties_by_tetrahedron(states, temperatures)
    np.testing.assert_array_equal(free_energy, properties.free_energy)
    np.testing.assert_array_equal(entropy, properties.entropy)

    with pytest.warns(
        DeprecationWarning, match="compute_electronic_contributions_from_states"
    ):
        free_energy, entropy = compute_electronic_contributions_from_states(
            [states], temperatures, primitive_volumes=None
        )
    np.testing.assert_array_equal(free_energy[:, 0], properties.free_energy)
    np.testing.assert_array_equal(entropy[:, 0], properties.entropy)

    # The tuple form of the k-point sum returns the band sum itself, not
    # F(T) - F(0).
    with pytest.warns(DeprecationWarning, match="compute_free_energy_by_kpoint_sum"):
        free_energy, entropy = compute_free_energy_by_kpoint_sum(
            _al_states(), temperatures
        )
    properties = compute_thermal_properties_by_kpoint_sum(_al_states(), temperatures)
    np.testing.assert_array_equal(free_energy - free_energy[0], properties.free_energy)
    np.testing.assert_array_equal(entropy, properties.entropy)
