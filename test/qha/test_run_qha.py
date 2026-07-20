# SPDX-License-Identifier: BSD-3-Clause
"""Tests for run_qha in phonopy.qha.qha."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from qha_utils import (
    MESH,
    TEMPERATURES,
    internal_energies,
    scaled_phonopy,
    thermal_properties,
)

from phonopy import Phonopy, PhonopyQHA, run_qha
from phonopy.qha.electron import ElectronicStates, compute_free_energy_and_entropy
from phonopy.qha.lattice import LatticeParametersFit
from phonopy.qha.qha import QHAResult
from phonopy.structure.atoms import PhonopyAtoms


def _electronic_structures(volumes: NDArray[np.double]) -> list[ElectronicStates]:
    """Return synthetic metallic states with volume-dependent bandwidth."""
    rng = np.random.default_rng(12345)
    base = np.sort(rng.uniform(-5.0, 15.0, size=(1, 24, 40)), axis=2)
    states = []
    for v in volumes:
        scale = (v / volumes.mean()) ** (-2.0 / 3.0)  # free-electron-like
        states.append(
            ElectronicStates(
                eigenvalues=base * scale,
                weights=np.ones(24),
                n_electrons=8.0,
                volume=v,
            )
        )
    return states


@pytest.mark.parametrize("pressure", [None, 5.0])
def test_run_qha_matches_phonopy_qha(
    nacl_qha_phonopys: list[Phonopy], pressure: float | None
) -> None:
    """All quantities agree with the legacy PhonopyQHA on identical inputs."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    el_energies = internal_energies(volumes)

    result = run_qha(
        nacl_qha_phonopys,
        TEMPERATURES,
        internal_energies=el_energies,
        mesh=MESH,
        pressure=pressure,
    )

    fe_phonon, entropy, cv = thermal_properties(nacl_qha_phonopys)
    ref = PhonopyQHA(
        volumes=volumes,
        electronic_energies=el_energies,
        eos="vinet",
        temperatures=TEMPERATURES,
        free_energy=fe_phonon,
        cv=cv,
        entropy=entropy,
        pressure=pressure,
    )

    np.testing.assert_allclose(
        result.equilibrium_volumes, ref.volume_temperature, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.gibbs_free_energies, ref.gibbs_temperature, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.bulk_moduli, ref.bulk_modulus_temperature, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.thermal_expansion, ref.thermal_expansion, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.gruneisen_parameters, ref.gruneisen_temperature, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.helmholtz_volume, ref.helmholtz_volume, rtol=0, atol=1e-12
    )
    assert result.heat_capacity_P is not None
    np.testing.assert_allclose(
        result.heat_capacity_P.heat_capacities,
        ref.heat_capacity_P_polyfit,
        rtol=0,
        atol=1e-12,
    )


def test_lattice_data(
    qha_result_nacl: QHAResult, nacl_qha_phonopys: list[Phonopy]
) -> None:
    """Lattice output is consistent with a direct LatticeParametersFit."""
    result = qha_result_nacl
    assert result.lattice is not None

    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    lattice_lengths = np.array(
        [np.linalg.norm(ph.unitcell.cell, axis=1) for ph in nacl_qha_phonopys]
    )
    fit = LatticeParametersFit(volumes, lattice_lengths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = fit.evaluate(result.equilibrium_volumes)

    np.testing.assert_allclose(
        result.lattice.lattice_parameters, expected, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(result.lattice.k, fit.k, rtol=1e-13)

    # Exact volume consistency: k * a * b * c == V_0(T)
    np.testing.assert_allclose(
        result.lattice.k * result.lattice.lattice_parameters.prod(axis=1),
        result.equilibrium_volumes,
        rtol=1e-13,
    )


def test_axial_expansion_invariant(qha_result_nacl: QHAResult) -> None:
    """alpha_a + alpha_b + alpha_c matches the volumetric beta."""
    result = qha_result_nacl
    assert result.lattice is not None

    alpha_sum = result.lattice.axial_thermal_expansions.sum(axis=1)
    np.testing.assert_allclose(alpha_sum[0], 0.0, atol=1e-30)
    # The identity is exact in the continuum; central differences of a and
    # of V commute only to O(dT^2), so allow for that discretization error.
    np.testing.assert_allclose(alpha_sum[1:], result.thermal_expansion[1:], rtol=1e-2)


def test_monoclinic_disables_lattice(ph_nacl: Phonopy) -> None:
    """Triclinic and monoclinic crystals skip the lattice fit with a warning.

    A constant shear keeps the cell angles (and hence k) constant over the
    volume series, so only the symmetry-based gate can catch this case.

    """
    phonopys = []
    for s in np.linspace(0.98, 1.04, 5):
        ph_new = scaled_phonopy(ph_nacl, np.array([s, s, s]))
        cell = ph_new.unitcell
        lattice = cell.cell.copy()
        lattice[2] += 0.05 * lattice[0]  # constant shear -> monoclinic
        sheared_cell = PhonopyAtoms(
            symbols=cell.symbols,
            cell=lattice,
            scaled_positions=cell.scaled_positions,
            masses=cell.masses,
        )
        ph_sheared = Phonopy(
            sheared_cell,
            supercell_matrix=ph_new.supercell_matrix,
            primitive_matrix=ph_new.primitive_matrix,
            log_level=0,
        )
        ph_sheared.force_constants = ph_new.force_constants
        phonopys.append(ph_sheared)

    assert all(ph.symmetry.dataset.number <= 15 for ph in phonopys)

    volumes = np.array([ph.primitive.volume for ph in phonopys])
    with pytest.warns(UserWarning, match="Lattice parameter fitting was skipped"):
        result = run_qha(
            phonopys,
            TEMPERATURES,
            internal_energies=internal_energies(volumes),
            mesh=MESH,
        )

    assert result.lattice is None
    assert result.equilibrium_volumes.shape == (len(TEMPERATURES) - 1,)


def test_validation_errors(nacl_qha_phonopys: list[Phonopy], ph_nacl: Phonopy) -> None:
    """Malformed inputs raise before any phonon calculation."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    el_energies = internal_energies(volumes)

    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys[:4], TEMPERATURES, internal_energies=el_energies[:4])
    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys, TEMPERATURES[::-1], internal_energies=el_energies)
    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys, TEMPERATURES, internal_energies=el_energies[:-1])
    with pytest.raises(ValueError):
        run_qha(
            nacl_qha_phonopys,
            TEMPERATURES,
            internal_energies=el_energies,
            lattice_fit_degree=10,
        )
    with pytest.raises(ValueError):
        run_qha(
            nacl_qha_phonopys, TEMPERATURES, internal_energies=el_energies, eos="bogus"
        )

    ph_bare = Phonopy(
        ph_nacl.unitcell,
        supercell_matrix=ph_nacl.supercell_matrix,
        primitive_matrix=ph_nacl.primitive_matrix,
        log_level=0,
    )
    with pytest.raises(RuntimeError):
        run_qha([ph_bare] * 5, TEMPERATURES, internal_energies=el_energies[:5])


def test_run_qha_electronic_structures(nacl_qha_phonopys: list[Phonopy]) -> None:
    """electronic_structures input matches the legacy PhonopyQHA reference.

    The legacy PhonopyQHA is fed with F_el(T, V) built manually from the
    same electronic states, so the EOS fitting inputs are identical and
    the fitted quantities agree bitwise. C_P is checked against the
    legacy numerical -T d2G/dT2, which contains all contributions by
    construction; the two methods differ where C_V changes steeply
    (low T) and must agree tightly at high T.

    """
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    el_static = internal_energies(volumes)
    states = _electronic_structures(volumes)

    result = run_qha(
        nacl_qha_phonopys,
        TEMPERATURES,
        internal_energies=el_static,
        electronic_structures=states,
        mesh=MESH,
    )

    temps_with_anchor = np.concatenate([[0.0], TEMPERATURES])
    el2d = np.zeros((len(TEMPERATURES), len(volumes)))
    for i, electronic_states in enumerate(states):
        fe, _ = compute_free_energy_and_entropy(electronic_states, temps_with_anchor)
        # Parenthesized to match the operation order of run_qha bitwise.
        el2d[:, i] = el_static[i] + (fe[1:] - fe[0])

    fe_phonon, entropy, cv = thermal_properties(nacl_qha_phonopys)
    ref = PhonopyQHA(
        volumes=volumes,
        electronic_energies=el2d,
        eos="vinet",
        temperatures=TEMPERATURES,
        free_energy=fe_phonon,
        cv=cv,
        entropy=entropy,
    )

    np.testing.assert_allclose(
        result.equilibrium_volumes, ref.volume_temperature, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.thermal_expansion, ref.thermal_expansion, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.gibbs_free_energies, ref.gibbs_temperature, rtol=0, atol=1e-12
    )

    cp_new = result.heat_capacity_P.heat_capacities
    cp_ref = np.array(ref.heat_capacity_P_numerical)
    np.testing.assert_allclose(cp_new[2:], cp_ref[2:], atol=1.5)
    np.testing.assert_allclose(cp_new[6:], cp_ref[6:], atol=0.3)


def test_run_qha_electronic_structures_validation(
    nacl_qha_phonopys: list[Phonopy],
) -> None:
    """Inconsistent electronic inputs raise ValueError."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    el_static = internal_energies(volumes)
    states = _electronic_structures(volumes)
    el2d = np.zeros((len(TEMPERATURES), len(volumes)))

    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys, TEMPERATURES, internal_energies=el2d)
    with pytest.raises(ValueError):
        run_qha(
            nacl_qha_phonopys,
            TEMPERATURES,
            internal_energies=el2d,
            electronic_structures=states,
        )
    with pytest.raises(ValueError):
        run_qha(
            nacl_qha_phonopys,
            TEMPERATURES,
            internal_energies=el_static,
            electronic_structures=states[:-1],
        )
    # Volumes carried by ElectronicStates must match the unit cells.
    with pytest.raises(ValueError):
        run_qha(
            nacl_qha_phonopys,
            TEMPERATURES,
            internal_energies=el_static,
            electronic_structures=states[::-1],
        )
    # internal_energies=None requires electronic_structures with energies.
    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys, TEMPERATURES)
    with pytest.raises(ValueError):
        run_qha(nacl_qha_phonopys, TEMPERATURES, electronic_structures=states)


def test_run_qha_internal_energies_from_states(
    nacl_qha_phonopys: list[Phonopy],
) -> None:
    """internal_energies=None takes the energies from electronic_structures."""
    volumes = np.array([ph.primitive.volume for ph in nacl_qha_phonopys])
    el_static = internal_energies(volumes)
    states = [
        dataclasses.replace(electronic_states, internal_energy=energy)
        for electronic_states, energy in zip(
            _electronic_structures(volumes), el_static, strict=True
        )
    ]

    result = run_qha(
        nacl_qha_phonopys, TEMPERATURES, electronic_structures=states, mesh=MESH
    )
    ref = run_qha(
        nacl_qha_phonopys,
        TEMPERATURES,
        internal_energies=el_static,
        electronic_structures=states,
        mesh=MESH,
    )

    np.testing.assert_allclose(
        result.equilibrium_volumes, ref.equilibrium_volumes, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.gibbs_free_energies, ref.gibbs_free_energies, rtol=0, atol=0
    )


def test_result_immutability(qha_result_nacl: QHAResult) -> None:
    """QHAResult and its arrays are read-only."""
    result = qha_result_nacl

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.eos_name = "murnaghan"  # type: ignore[misc]
    assert not result.equilibrium_volumes.flags.writeable
    with pytest.raises(ValueError):
        result.equilibrium_volumes[0] = 0.0
    assert result.lattice is not None
    assert not result.lattice.lattice_parameters.flags.writeable
