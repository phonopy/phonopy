# SPDX-License-Identifier: BSD-3-Clause
"""Building blocks shared by the QHA drivers.

These helpers are used by both the volume-path driver (phonopy.qha.qha
run_qha) and the anisotropic driver (phonopy.qha.anisotropic
run_anisotropic_qha): phonon thermal-property sampling over a set of
Phonopy instances, the relative electronic free energy and entropy from
electronic states, and the read-only freezing of ndarray fields of the
immutable result dataclasses.

"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from phonopy.qha.electron import compute_free_energy_and_entropy

if TYPE_CHECKING:
    from phonopy.api_phonopy import Phonopy
    from phonopy.qha.electron import ElectronicStates


def freeze_ndarray_fields(obj: Any) -> None:
    """Replace ndarray fields of a frozen dataclass with read-only copies."""
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, np.ndarray):
            copied = value.copy()
            copied.flags.writeable = False
            object.__setattr__(obj, field.name, copied)


def compute_thermal_properties(
    phonopys: Sequence[Phonopy],
    temperatures: NDArray[np.double],
    mesh: float | Sequence[int] | NDArray[np.int64],
    verbose: bool = False,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Compute phonon thermal properties at each volume point.

    Returns (free_energy (kJ/mol), entropy (J/K/mol), cv (J/K/mol)), each
    with shape (temperatures, volumes).

    """
    nvol = len(phonopys)
    shape = (len(temperatures), nvol)
    fe_phonon = np.zeros(shape, dtype="double")
    entropy = np.zeros(shape, dtype="double")
    cv = np.zeros(shape, dtype="double")
    if verbose:
        print("# Phonon thermal properties")
    for i, ph in enumerate(phonopys):
        if verbose:
            print(
                "Computing phonon thermal properties "
                f"(volume {i + 1}/{nvol}, V = {ph.primitive.volume:.4f} A^3)"
            )
        ph.run_mesh(mesh)
        tp = ph.run_thermal_properties(temperatures=temperatures)
        fe_phonon[:, i] = tp.free_energy
        entropy[:, i] = tp.entropy
        cv[:, i] = tp.heat_capacity
    return fe_phonon, entropy, cv


def compute_electronic_contributions_from_states(
    electronic_structures: Sequence[ElectronicStates],
    temperatures: NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Compute relative band free energies and entropies at temperatures.

    Returns (fe_el_rel, s_el) with shape (temperatures, volumes) in eV and
    eV/K, respectively. fe_el_rel = fe(T) - fe(0) is anchored at T = 0,
    which is evaluated explicitly so that the temperature grid does not
    need to start at 0 K.

    """
    shape = (len(temperatures), len(electronic_structures))
    fe_el_rel = np.zeros(shape, dtype="double")
    s_el = np.zeros(shape, dtype="double")
    temps_with_anchor = np.concatenate([[0.0], temperatures])
    for i, electronic_states in enumerate(electronic_structures):
        fe, s = compute_free_energy_and_entropy(electronic_states, temps_with_anchor)
        fe_el_rel[:, i] = fe[1:] - fe[0]
        s_el[:, i] = s[1:]
    return fe_el_rel, s_el
