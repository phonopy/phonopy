# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for QHA tests (not collected by pytest)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

MESH = [11, 11, 11]
TEMPERATURES = np.arange(0.0, 1501.0, 100.0)


def scaled_phonopy(ph: Phonopy, scales: NDArray[np.double]) -> Phonopy:
    """Return a Phonopy with lattice vectors scaled by the given factors.

    Row scaling preserves all cell angles. Force constants are softened
    with volume as fc / (V / V_ref)^2 to mimic quasi-harmonic behavior.

    """
    cell = ph.unitcell
    scaled_cell = PhonopyAtoms(
        symbols=cell.symbols,
        cell=cell.cell * np.asarray(scales)[:, None],
        scaled_positions=cell.scaled_positions,
        masses=cell.masses,
    )
    ph_new = Phonopy(
        scaled_cell,
        supercell_matrix=ph.supercell_matrix,
        primitive_matrix=ph.primitive_matrix,
        log_level=0,
    )
    v_ratio = ph_new.unitcell.volume / ph.unitcell.volume
    assert ph.force_constants is not None
    ph_new.force_constants = ph.force_constants / v_ratio**2
    return ph_new


def internal_energies(volumes: NDArray[np.double]) -> NDArray[np.double]:
    """Return a parabolic E(V) with its minimum inside the volume range.

    The curvature is scaled with the mean volume so that the implied bulk
    modulus (B = V d2E/dV2 ~ 0.5 eV/angstrom^3) is volume-scale
    independent.

    """
    v0 = volumes.mean() * 1.01
    return 0.25 / volumes.mean() * (volumes - v0) ** 2 - 40.0


def thermal_properties(
    phonopys: list[Phonopy],
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Compute thermal properties arrays as run_qha does internally."""
    shape = (len(TEMPERATURES), len(phonopys))
    fe_phonon = np.zeros(shape, dtype="double")
    entropy = np.zeros(shape, dtype="double")
    cv = np.zeros(shape, dtype="double")
    for i, ph in enumerate(phonopys):
        ph.run_mesh(MESH)
        tp = ph.run_thermal_properties(temperatures=TEMPERATURES)
        fe_phonon[:, i] = tp.free_energy
        entropy[:, i] = tp.entropy
        cv[:, i] = tp.heat_capacity
    return fe_phonon, entropy, cv
