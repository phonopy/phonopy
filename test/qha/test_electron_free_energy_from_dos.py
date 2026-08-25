# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the electronic free energy computed from a density of states."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from phonopy.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.physical_units import get_physical_units
from phonopy.qha.electron import (
    ElectronFreeEnergy,
    ElectronicStates,
    compute_free_energy_and_entropy,
    compute_free_energy_by_tetrahedron,
    free_energy_from_dos,
)
from phonopy.qha.thermal import compute_electronic_contributions_from_states
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

FERMI = 5.0
WINDOW = 2.0


def _flat_dos(g0: float, n_points: int = 8001):
    """Return a constant density of states and the energies it sits on.

    A constant density of states is the one case with an analytic answer,
    F_el = -(pi^2/6) (k T)^2 g(E_F), which is what makes it worth testing
    against: it checks the whole chain of electron count, band energy and
    entropy rather than any one of them.

    """
    energies = np.linspace(FERMI - WINDOW, FERMI + WINDOW, n_points)
    return energies, np.full(n_points, g0, dtype="double")


def _n_electrons(energies, dos):
    """Return the electron count that makes the Fermi level consistent."""
    return float(np.trapezoid(np.where(energies <= FERMI, dos, 0.0), energies))


def test_sommerfeld_limit():
    """Test the free energy of a constant density of states.

    The Sommerfeld expansion is exact in the limit of a density of states
    that does not vary over the thermal window, so this is a closed-form
    check of the integration itself.

    """
    g0 = 2.0
    energies, dos = _flat_dos(g0)
    temperatures = np.array([0.0, 100.0, 200.0, 300.0])
    free_energy, _, _ = free_energy_from_dos(
        energies, dos, _n_electrons(energies, dos), temperatures, FERMI
    )

    kb = get_physical_units().KB
    expected = -(np.pi**2 / 6.0) * (kb * temperatures) ** 2 * g0
    np.testing.assert_allclose(free_energy, expected, rtol=2e-4, atol=1e-12)


def test_entropy_matches_the_sommerfeld_limit():
    """Test the entropy against (pi^2/3) k^2 T g(E_F)."""
    g0 = 2.0
    energies, dos = _flat_dos(g0)
    temperatures = np.array([0.0, 100.0, 300.0])
    _, entropy, _ = free_energy_from_dos(
        energies, dos, _n_electrons(energies, dos), temperatures, FERMI
    )

    kb = get_physical_units().KB
    expected = (np.pi**2 / 3.0) * kb**2 * temperatures * g0
    np.testing.assert_allclose(entropy, expected, rtol=2e-4, atol=1e-14)


def test_free_energy_is_zero_at_zero_temperature():
    """Test that the free energy is reported relative to T = 0."""
    energies, dos = _flat_dos(2.0)
    free_energy, _, _ = free_energy_from_dos(
        energies, dos, _n_electrons(energies, dos), [0.0, 300.0], FERMI
    )
    assert free_energy[0] == 0.0


def test_chemical_potential_starts_at_the_fermi_level():
    """Test that mu(T = 0) is the Fermi level exactly.

    Without a state count to solve against, the count below the window is
    defined at the Fermi level and mu(0) is that level by construction. It
    used to be solved for at T = 0 instead, where the occupation is a step
    and the count a staircase in mu, which quantized mu(0) to the energy
    grid and moved F(T) - F(0) by microelectronvolts.

    """
    energies, dos = _flat_dos(2.0)
    _, _, mu = free_energy_from_dos(
        energies, dos, _n_electrons(energies, dos), [0.0, 300.0], FERMI
    )
    assert mu[0] == FERMI


def test_energy_grid_halving_moves_the_free_energy_by_microelectronvolts():
    """Test the check that caught two earlier failures of this integration.

    Counting the states below the window by a quadrature made the answer grid
    dependent, and halving the energy grid then moved the free energy by
    milli-electronvolts. Anchoring the count to the Fermi level instead
    leaves it at the micro-electronvolt level, so this comparison is the one
    that tells the two apart.

    """
    # A sloped density of states, so that the grid actually has something to
    # resolve; a constant one would be integrated exactly at any spacing.
    fine_energies = np.linspace(FERMI - WINDOW, FERMI + WINDOW, 16001)
    fine_dos = 2.0 + 0.5 * (fine_energies - FERMI)
    coarse_energies = fine_energies[::2]
    coarse_dos = fine_dos[::2]

    temperatures = np.array([0.0, 300.0])
    fine, _, _ = free_energy_from_dos(
        fine_energies,
        fine_dos,
        _n_electrons(fine_energies, fine_dos),
        temperatures,
        FERMI,
    )
    coarse, _, _ = free_energy_from_dos(
        coarse_energies,
        coarse_dos,
        _n_electrons(coarse_energies, coarse_dos),
        temperatures,
        FERMI,
    )
    assert abs(fine[-1] - coarse[-1]) < 1e-6


def _half_filled_band(mesh) -> ElectronicStates:
    """Return a metal: one wide free-electron band, half filled.

    The Fermi level then sits inside the band, which is what makes the two
    integrators comparable at all. A set of narrow separated bands would put
    it in a gap and both would return zero.

    """
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.05, symbols=["Al"], scaled_positions=[[0.0, 0.0, 0.0]]
    )
    bz_grid = BZGrid(mesh, lattice=cell.cell, symmetry_dataset=Symmetry(cell).dataset)
    ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
    addresses = bz_grid.addresses[bz_grid.grg2bzg[ir_grid_points]]
    kpoints = addresses @ bz_grid.QDinv.T
    cartesian = kpoints @ np.linalg.inv(cell.cell).T
    band = 100.0 * (cartesian**2).sum(axis=1)

    states = ElectronicStates(
        eigenvalues=band[None, :, None],
        weights=ir_grid_weights / ir_grid_weights.sum(),
        n_electrons=1.0,
        kpoints=kpoints,
        mesh=np.asarray(mesh),
        cell=cell,
    )
    fermi = ElectronFreeEnergy(states.eigenvalues, states.weights, 1.0)
    fermi.run(1.0)
    return dataclasses.replace(states, fermi_energy=fermi.mu)


def test_tetrahedron_agrees_with_the_kpoint_sum_at_a_converged_mesh():
    """Test the two integrators against each other where both have converged.

    They share no machinery: one builds a density of states by the linear
    tetrahedron method and integrates it, the other sums Fermi-Dirac
    occupations over irreducible k-points. Agreement is therefore a real
    check rather than a restatement.

    The mesh has to be dense for the k-point sum, which is the whole reason
    the tetrahedron method is worth having: at 8x8x8 the tetrahedron is
    already within 5 per cent of its converged value while the k-point sum is
    20 per cent out.

    """
    temperatures = np.array([0.0, 300.0])
    states = _half_filled_band([48, 48, 48])

    tetrahedron, _ = compute_free_energy_by_tetrahedron(states, temperatures)
    k_sum, _ = compute_free_energy_and_entropy(states, temperatures)

    assert tetrahedron[-1] == pytest.approx(k_sum[-1] - k_sum[0], abs=2e-5)


def test_tetrahedron_converges_faster_than_the_kpoint_sum():
    """Test that the tetrahedron is close to its answer on a coarse mesh."""
    temperatures = np.array([0.0, 300.0])
    converged, _ = compute_free_energy_by_tetrahedron(
        _half_filled_band([48, 48, 48]), temperatures
    )
    coarse, _ = compute_free_energy_by_tetrahedron(
        _half_filled_band([8, 8, 8]), temperatures
    )
    coarse_k_sum, _ = compute_free_energy_and_entropy(
        _half_filled_band([8, 8, 8]), temperatures
    )

    reference = converged[-1]
    assert abs(coarse[-1] - reference) < 0.06 * abs(reference)
    assert abs((coarse_k_sum[-1] - coarse_k_sum[0]) - reference) > 0.15 * abs(reference)


def test_qha_integrates_by_tetrahedron_when_the_states_carry_the_grid():
    """Test that the QHA drivers use the grid when the states carry it.

    The two integrators are far apart on a coarse mesh, so which one ran is
    visible in the result rather than inferred.

    """
    temperatures = np.array([300.0])
    states = _half_filled_band([8, 8, 8])

    fe_el_rel, _ = compute_electronic_contributions_from_states([states], temperatures)
    tetrahedron, _ = compute_free_energy_by_tetrahedron(states, np.array([0.0, 300.0]))

    assert fe_el_rel[0, 0] == pytest.approx(tetrahedron[-1])


def test_qha_falls_back_to_the_kpoint_sum_without_the_grid():
    """Test that states without the grid are summed over k points."""
    temperatures = np.array([300.0])
    states = dataclasses.replace(
        _half_filled_band([8, 8, 8]), kpoints=None, mesh=None, cell=None
    )

    fe_el_rel, _ = compute_electronic_contributions_from_states([states], temperatures)
    k_sum, _ = compute_free_energy_and_entropy(states, np.array([0.0, 300.0]))

    assert fe_el_rel[0, 0] == pytest.approx(k_sum[-1] - k_sum[0])


def test_the_integration_method_is_reported(capsys):
    """Test that every run says how the electronic free energy was integrated.

    The states decide the method and nothing in the command line shows it, so
    a silent fall-back to the k-point sum would be invisible.

    """
    temperatures = np.array([300.0])
    with_grid = _half_filled_band([8, 8, 8])
    without_grid = dataclasses.replace(with_grid, kpoints=None, mesh=None, cell=None)

    compute_electronic_contributions_from_states([with_grid], temperatures)
    out = capsys.readouterr().out
    # The window and the spacing are reported too: they set what the tetrahedron
    # integrated, and a file of free energies keeps no record of them.
    assert "linear tetrahedron method (1 point, +-0.50 eV at 0.50 meV)" in out

    compute_electronic_contributions_from_states([without_grid], temperatures)
    assert "k-point sum (1 point)" in capsys.readouterr().out

    compute_electronic_contributions_from_states(
        [with_grid, without_grid], temperatures
    )
    out = capsys.readouterr().out
    assert "tetrahedron method (1 point," in out
    assert "and by the k-point sum (1 point)" in out


def test_window_must_contain_samples():
    """Test that a window with no density of states in it raises."""
    energies = np.linspace(0.0, 1.0, 101)
    dos = np.ones_like(energies)
    with pytest.raises(ValueError, match="contains no density-of-states"):
        free_energy_from_dos(energies, dos, 1.0, [300.0], FERMI, window=0.1)
