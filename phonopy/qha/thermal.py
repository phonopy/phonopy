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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phonopy.api_phonopy import Phonopy
from phonopy.qha.electron import (
    compute_free_energy_by_tetrahedron,
    resolve_energy_window,
)
from phonopy.qha.electron_kpoint_sum import compute_free_energy_by_kpoint_sum
from phonopy.qha.electron_states import ElectronicStates


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


def _states_cell_volume(electronic_states: ElectronicStates) -> float | None:
    """Return the volume of the cell the states were computed on.

    The states record that cell in one of two ways, depending on where they
    came from, and neither is always there. A VASP reader sets ``cell``, and
    the anisotropic dataset restores it from the grid-point lattice; the
    electronic_states.hdf5 of the volume-path workflow keeps ``volume``
    instead. None means the states say nothing, and the caller then has to
    take them as already normalized.

    """
    if electronic_states.cell is not None:
        return float(electronic_states.cell.volume)
    if electronic_states.volume is not None:
        return float(electronic_states.volume)
    return None


def primitive_cell_fractions(
    electronic_structures: Sequence[ElectronicStates],
    primitive_volumes: Sequence[float] | NDArray[np.double] | None,
) -> NDArray[np.double]:
    """Return the factor putting each states quantity per primitive cell.

    Phonon thermal properties are per primitive cell, while a calculator
    reports quantities for the cell it was run on. The two cells coincide for
    a primitive lattice and differ by the centring multiplicity otherwise.
    The factor is the ratio of the two volumes, so it is read off the data
    rather than assumed: states computed on the primitive cell already give
    1, and nothing is scaled twice.

    It is 1 as well when primitive_volumes is None, or when the states record
    no cell of their own.

    Not every driver needs this. run_qha requires the states to sit on the
    same cell as the phonons and checks their volume, so whatever cell
    ph.primitive is -- the unit cell included -- one normalization already
    runs through its inputs. The anisotropic driver cannot ask for that: its
    reference is the conventional cell, from which the free lattice DOF are
    read, while primitive_matrix picks the primitive cell out of it, so a
    centred lattice puts the calculator's cell and the phonon normalization
    apart by construction.

    """
    fractions = np.ones(len(electronic_structures), dtype="double")
    if primitive_volumes is None:
        return fractions
    volumes = np.asarray(primitive_volumes, dtype="double")
    for i, electronic_states in enumerate(electronic_structures):
        states_volume = _states_cell_volume(electronic_states)
        if states_volume is not None and states_volume > 0.0:
            fractions[i] = volumes[i] / states_volume
    return fractions


def _report_primitive_cell_scaling(fractions: NDArray[np.double]) -> None:
    """Print the conversion to the primitive cell when there is one."""
    if np.allclose(fractions, 1.0):
        return
    ratios = sorted({round(1.0 / f, 6) for f in fractions})
    named = ", ".join(f"{r:g}" for r in ratios)
    print(
        f"The states were computed on a cell {named} times the primitive "
        f"cell, so F_el is scaled to the primitive cell."
    )


def compute_electronic_contributions_from_states(
    electronic_structures: Sequence[ElectronicStates],
    temperatures: NDArray[np.double],
    *,
    primitive_volumes: Sequence[float] | NDArray[np.double] | None,
    window: float | None = None,
    energy_spacing: float = 0.0005,
    require_tetrahedron: bool = False,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Compute relative band free energies and entropies at temperatures.

    Returns (fe_el_rel, s_el) with shape (temperatures, volumes) in eV and
    eV/K, respectively. fe_el_rel = fe(T) - fe(0) is anchored at T = 0,
    which is evaluated explicitly so that the temperature grid does not
    need to start at 0 K.

    The states are integrated over the cell they were computed on. Given
    primitive_volumes, the result is scaled to that normalization by
    primitive_cell_fractions. primitive_volumes is keyword-only and has no
    default, so that a caller adding phonon quantities to these has to say
    which cell it normalizes to; pass None when the caller has already
    established that the states and the phonons share one cell.

    States carrying the k-point grid they were computed on are integrated by
    the linear tetrahedron method, the rest by the k-point sum, which
    converges far more slowly. Which one ran is reported, since the states
    decide it and nothing in the command line shows it.

    """
    shape = (len(temperatures), len(electronic_structures))
    fe_el_rel = np.zeros(shape, dtype="double")
    s_el = np.zeros(shape, dtype="double")
    temps_with_anchor = np.concatenate([[0.0], temperatures])
    without_grid = [
        i
        for i, states in enumerate(electronic_structures)
        if not _has_tetrahedron_grid(states)
    ]
    if without_grid and require_tetrahedron:
        named = ", ".join(str(i + 1) for i in without_grid)
        raise ValueError(
            f"Grid point(s) {named} carry no k-point grid, so the linear "
            "tetrahedron method cannot run. The k-point sum that is left "
            "converges far too slowly for this term: it needs many more "
            "irreducible k points, and on a mesh chosen for the total energy "
            "the thermal expansion it gives can be off by a large factor. "
            "Recompute the static grid with a regular mesh, or give the free "
            "energies ready-made."
        )
    _report_electronic_integration(
        len(electronic_structures) - len(without_grid),
        len(without_grid),
        resolve_energy_window(window, temps_with_anchor),
        energy_spacing,
    )
    for i, electronic_states in enumerate(electronic_structures):
        if i in set(without_grid):
            fe, s = compute_free_energy_by_kpoint_sum(
                electronic_states, temps_with_anchor
            )
        else:
            fe, s = compute_free_energy_by_tetrahedron(
                electronic_states,
                temps_with_anchor,
                window=window,
                energy_spacing=energy_spacing,
            )
        # The k-point sum returns the whole band sum and the tetrahedron
        # returns it against 0 K, where fe[0] is zero; subtracting the anchor
        # covers both.
        fe_el_rel[:, i] = fe[1:] - fe[0]
        s_el[:, i] = s[1:]

    fractions = primitive_cell_fractions(electronic_structures, primitive_volumes)
    _report_primitive_cell_scaling(fractions)
    return fe_el_rel * fractions, s_el * fractions


def _report_electronic_integration(
    n_tetrahedron: int, n_sum: int, window: float, energy_spacing: float
) -> None:
    """Print how the electronic free energy is integrated at each point.

    The window and the spacing are named too: they set what the tetrahedron
    integrated, and a file of free energies keeps no record of them.

    """
    tetrahedron = (
        f"the linear tetrahedron method ({_points(n_tetrahedron)}, "
        f"+-{window:.2f} eV at {energy_spacing * 1e3:.2f} meV)"
    )
    k_sum = f"the k-point sum ({_points(n_sum)})"
    if n_sum == 0:
        print(f"Electronic free energy by {tetrahedron}.")
    elif n_tetrahedron == 0:
        print(
            f"Electronic free energy by {k_sum}: the states carry no k-point "
            "grid, so the tetrahedron method is unavailable."
        )
    else:
        print(f"Electronic free energy by {tetrahedron} and by {k_sum}.")


def _points(n: int) -> str:
    """Return a grid-point count with its noun."""
    return f"{n} point" if n == 1 else f"{n} points"


def _has_tetrahedron_grid(electronic_states: ElectronicStates) -> bool:
    """Return whether the states carry everything the tetrahedron needs.

    kpoints, mesh and cell are taken together or not at all. A Fermi energy
    the calculation reported is used when it is there and counted over the
    k points when it is not, so it is not part of this.

    """
    return (
        electronic_states.kpoints is not None
        and electronic_states.mesh is not None
        and electronic_states.cell is not None
    )
