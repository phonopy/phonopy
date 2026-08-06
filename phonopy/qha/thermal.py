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
    is_gamma_center: bool = False,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Compute phonon thermal properties at each volume point.

    Returns (free_energy (kJ/mol), entropy (J/K/mol), cv (J/K/mol)), each
    with shape (temperatures, volumes).

    Parameters
    ----------
    phonopys : Sequence[Phonopy]
        One instance per volume point, each with force constants.
    temperatures : ndarray
        Temperatures in K.
    mesh : float or array_like
        Sampling mesh, as a length measure or as explicit numbers of
        divisions. A length is resolved against each instance's own
        reciprocal lattice, so cells of different shape can receive
        different numbers of divisions; explicit numbers sample every
        instance identically. The latter matters when the results are
        differentiated with respect to the lattice, as in an anisotropic
        quasi-harmonic calculation, where a change of divisions between
        neighbouring cells is a step in the quantity being differentiated.
    verbose : bool, optional
        Print progress. Default is False.
    is_gamma_center : bool, optional
        Generate a Gamma-centred mesh instead of the Monkhorst-Pack one.
        Ignored when mesh is a length, for which phonopy enforces a
        Gamma-centred mesh. Pass True alongside explicit numbers of
        divisions to reproduce what the corresponding length would have
        sampled; the default False keeps phonopy's own default and gives
        a grid shifted by half a division. Default is False.

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
        ph.run_mesh(mesh, is_gamma_center=is_gamma_center)
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
