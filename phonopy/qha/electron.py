# SPDX-License-Identifier: BSD-3-Clause
"""Electronic free energy by the linear tetrahedron method.

The electronic states themselves and their file format are in
electron_states, and the sum over irreducible k-points, which is what runs
where there is no sampling grid, in electron_kpoint_sum. Both are
re-exported here, where they used to live.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.phonon.grid import (
    BZGrid,
    get_grid_shift_from_kpoints,
    get_ir_grid_points,
    get_ir_kpoint_map,
)
from phonopy.phonon.spectrum import TetrahedronDOSAccumulator
from phonopy.physical_units import get_physical_units
from phonopy.qha.electron_kpoint_sum import (  # noqa: F401
    ElectronFreeEnergy,
    compute_free_energy_and_entropy,
    compute_free_energy_by_kpoint_sum,
    get_free_energy_at_T,
)
from phonopy.qha.electron_states import (  # noqa: F401
    ElectronicStates,
    entropy_terms,
    fermi_dirac_occupation,
    read_electronic_states_hdf5,
    resolve_spin_degeneracy,
    write_electronic_states_hdf5,
)
from phonopy.structure.symmetry import Symmetry

# Temperature in K standing in for 0 K, where k_B T is a division by zero.
_ZERO_TEMPERATURE = 1e-10


class _TetrahedronSampler:
    """The tetrahedron setup for one set of electronic states, kept for reuse.

    Building the BZ grid and analysing the symmetry costs more than sampling a
    handful of energies, so a caller that samples more than once -- solving for
    the chemical potential, then building a density of states -- holds one of
    these rather than starting over.

    """

    def __init__(self, electronic_states: ElectronicStates) -> None:
        """Init method."""
        states = electronic_states
        if states.kpoints is None or states.mesh is None or states.cell is None:
            raise ValueError(
                "The tetrahedron method needs kpoints, mesh and cell on the "
                "ElectronicStates; without them only the k-point sum is "
                "available."
            )
        symmetry_dataset = Symmetry(states.cell).dataset
        self._bz_grid = BZGrid(
            states.mesh, lattice=states.cell.cell, symmetry_dataset=symmetry_dataset
        )
        # The states carry where their k-points are but not how the mesh was
        # placed, so the shift is read off the k-points and the grid rebuilt
        # with it. A Gamma-centred mesh stops here.
        shift = get_grid_shift_from_kpoints(states.kpoints, self._bz_grid)
        if shift.any():
            self._bz_grid = BZGrid(
                states.mesh,
                lattice=states.cell.cell,
                symmetry_dataset=symmetry_dataset,
                is_shift=shift,
            )
        (
            self._ir_grid_points,
            self._ir_grid_weights,
            self._ir_grid_map,
        ) = get_ir_grid_points(self._bz_grid)
        self._id_map = get_ir_kpoint_map(states.kpoints, states.weights, self._bz_grid)
        self._eigenvalues = np.asarray(states.eigenvalues, dtype="double")
        self._degeneracy = resolve_spin_degeneracy(states)

    def sample(
        self,
        energies: Sequence[float] | NDArray[np.double],
        max_bytes: float = 2.0e8,
    ) -> tuple[NDArray[np.double], NDArray[np.double]]:
        """Return the density of states and the number of states below each energy.

        The density is in states/eV per cell, summed over spin channels, on the
        given energies. The count is the tetrahedron method's own integral of
        the same states, counted from -infinity and continuous in energy, which
        is what makes it the thing to solve a chemical potential against.

        Summing the spin channels here is what lets a spin-polarized calculation
        be treated no differently afterwards. The occupation depends on the
        energy and the chemical potential alone, so the electron count, the band
        energy and the entropy are all integrals of the summed density of states
        against one chemical potential.

        The energies are processed in blocks. The tetrahedron integration
        weights are built as one (ir points, sampling points, bands) array,
        which a dense mesh and a fine energy grid make far too large to hold,
        twice over. Sampling points are independent of one another, so
        splitting them changes nothing but the peak allocation.

        Parameters
        ----------
        energies : array_like
            Energies to sample the density of states at, in eV.
        max_bytes : float, optional
            Rough ceiling on the integration-weight array of one block, in bytes.
            Default is 2e8, i.e. 200 MB against the 400 MB that two of them take.
            Raise it to trade memory for fewer passes over the eigenvalues.

        Returns
        -------
        tuple of ndarray
            The density of states and the count, each shape=(len(energies),),
            dtype='double'

        """
        sampling_points = np.asarray(energies, dtype="double")
        # A band that stays outside the sampled energies has a zero integration
        # weight everywhere, so it is left out rather than integrated to zero.
        low = sampling_points.min()
        high = sampling_points.max()
        eigenvalues_all = self._eigenvalues
        reaches = np.logical_and(
            eigenvalues_all.max(axis=(0, 1)) >= low,
            eigenvalues_all.min(axis=(0, 1)) <= high,
        )
        n_ir = len(self._ir_grid_points)
        n_band = int(np.count_nonzero(reaches))
        # The bands left out are full or empty over the whole mesh, and a full
        # one holds exactly one state per spin channel.
        below = np.logical_and(~reaches, eigenvalues_all.max(axis=(0, 1)) < low)
        outside = float(np.count_nonzero(below)) * eigenvalues_all.shape[0]

        dos = np.zeros(len(sampling_points), dtype="double")
        count = np.full(len(sampling_points), outside, dtype="double")
        if n_band == 0:
            return dos, count * self._degeneracy

        affordable = max_bytes / (8.0 * n_ir * n_band)
        block = len(sampling_points)
        if affordable < block:
            block = max(1, int(affordable))
        for eigenvalues in eigenvalues_all[:, :, reaches]:
            mapped = np.ascontiguousarray(eigenvalues[self._id_map])
            for start in range(0, len(sampling_points), block):
                chunk = sampling_points[start : start + block]
                result = TetrahedronDOSAccumulator(
                    mapped,
                    self._bz_grid,
                    ir_grid_points=self._ir_grid_points,
                    ir_grid_weights=self._ir_grid_weights,
                    ir_grid_map=self._ir_grid_map,
                    sampling_points=chunk,
                ).result
                dos[start : start + len(chunk)] += result.density[0, :, 0]
                count[start : start + len(chunk)] += result.cumulative[0, :, 0]
        return dos * self._degeneracy, count * self._degeneracy


def _solve_chemical_potential(
    sampler: _TetrahedronSampler,
    n_electrons: float,
    centre: float,
    window: float,
) -> float:
    """Return the energy at which the tetrahedron count reaches n_electrons.

    The count is continuous in energy and the tetrahedron evaluates it at any
    energy asked for, so this is a root and not a table lookup: it owes
    nothing to the grid the density of states is later sampled on.

    """
    from scipy.optimize import brentq

    def count(mu: float) -> float:
        return float(sampler.sample(np.array([mu]))[1][0])

    low, high = centre - window, centre + window
    n_low, n_high = count(low), count(high)
    if not n_low <= n_electrons <= n_high:
        raise ValueError(
            f"{n_electrons} electrons are outside the {n_low} to {n_high} "
            f"states between {low} and {high} eV; widen the window."
        )
    return float(brentq(lambda mu: count(mu) - n_electrons, low, high, xtol=1e-10))


def resolve_energy_window(
    window: float | None, temperatures: Sequence[float] | NDArray[np.double]
) -> float:
    """Return the half-width of the energy window around the Fermi level in eV.

    None takes 12 k_B T of the highest temperature and at least 0.5 eV. Both
    numbers are empirical. Beyond 12 k_B T the Fermi factor is within 1e-5 of
    0 or 1; the floor is what acts at low temperatures, where 12 k_B T is a
    few tens of meV.

    """
    if window is not None:
        return float(window)
    t_max = float(np.max(np.asarray(temperatures, dtype="double")))
    return max(0.5, 12.0 * get_physical_units().KB * t_max)


def compute_free_energy_by_tetrahedron(
    electronic_states: ElectronicStates,
    temperatures: Sequence[float] | NDArray[np.double],
    window: float | None = None,
    energy_spacing: float = 0.0005,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return F(T) - F(0) and the entropy through the tetrahedron method.

    The counterpart of compute_free_energy_by_kpoint_sum, which sums
    Fermi-Dirac occupations over irreducible k-points. Which one runs is the
    caller's choice rather than an inference from the data: this one needs
    kpoints, mesh and cell on the states and raises without them.

    Parameters
    ----------
    electronic_states : ElectronicStates
        States carrying kpoints, mesh, cell and fermi_energy.
    temperatures : array_like
        Temperatures in K, the first of them 0.
    window : float, optional
        Half-width of the energy window around the Fermi level in eV. None,
        the default, takes 12 k_B T of the highest temperature and at least
        0.5 eV; see resolve_energy_window. A wider window costs time without
        moving the answer, since nothing outside it depends on temperature.
    energy_spacing : float, optional
        Spacing of the energy grid inside the window in eV. Default is
        0.0005, which is fine enough that halving it leaves the free energy
        where it was.

    Returns
    -------
    tuple of ndarray
        (F(T) - F(0) in eV, S(T) in eV/K), each of shape
        (len(temperatures),).

    """
    fermi = electronic_states.fermi_energy
    if fermi is None:
        fermi = _fermi_level_by_counting(electronic_states)
    window = resolve_energy_window(window, temperatures)
    sampler = _TetrahedronSampler(electronic_states)
    mu_0 = _solve_chemical_potential(
        sampler, electronic_states.n_electrons, fermi, window
    )
    n_points = int(round(2 * window / energy_spacing)) + 1
    energies = np.linspace(fermi - window, fermi + window, n_points)
    dos, _ = sampler.sample(energies)
    free_energy, entropy, _ = free_energy_from_dos(
        energies,
        dos,
        electronic_states.n_electrons,
        temperatures,
        fermi,
        mu_0=mu_0,
    )
    return free_energy, entropy


def free_energy_from_dos(
    energies: NDArray[np.double],
    dos: NDArray[np.double],
    n_electrons: float,
    temperatures: Sequence[float] | NDArray[np.double],
    fermi_energy: float,
    window: float | None = None,
    mu_0: float | None = None,
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    """Return F(T) - F(0), the entropy and mu(T) from a density of states.

        N(mu, T) = n_below + int_window g(E) f(E) dE = n_electrons
        E(T)     = int_window g(E) E f(E) dE
        T S(T)   = -k T int_window g(E) [f ln f + (1-f) ln(1-f)] dE
        F(T)     = E(T) - T S(T)

    **Every integral is restricted to a window around the Fermi level, and
    that is a matter of correctness rather than speed.** Taking the difference
    of two whole band sums would lose the free energy in their roundoff.
    Outside the window nothing depends on temperature: states below are
    occupied to better than 1e-17 at 40 k_B T and contribute a constant that
    cancels in F(T) - F(0), states above are empty.

    **The states below the window are never integrated.** The count below it
    is defined by anchoring to mu_0, the chemical potential at T = 0,

        n_below := n_electrons - int_window g(E) theta(mu_0 - E) dE

    so that the T -> 0 limit of mu is mu_0 itself and every remaining integral
    runs over a smooth stretch of the density of states. Anchoring to the
    Fermi level the calculation reports is not enough on its own; see mu_0
    below.

    Parameters
    ----------
    energies : ndarray
        Energies the density of states is sampled at, in eV.
    dos : ndarray
        Density of states in states/eV per cell.
    n_electrons : float
        Number of electrons in the cell.
    temperatures : array_like
        Temperatures in K. The first has to be 0, since it is the reference
        the free energies are reported against and the one temperature at
        which mu is mu_0 rather than solved for.
    fermi_energy : float
        Fermi energy in eV, used as the anchor described above.
    window : float, optional
        Half-width of the window around the Fermi level in eV. None, the
        default, takes every energy given, which is what a caller that built
        the grid inside its own window wants. A number narrows that grid
        further.
    mu_0 : float, optional
        Chemical potential at T = 0 in eV. Defaults to the Fermi energy,
        which the calculation reports from its own integration scheme and
        which therefore need not reproduce n_electrons on this density of
        states. Solving for it here instead would mean solving a step
        function on this grid, which quantizes mu(0) to the spacing and moves
        F(T) - F(0) by of order a ueV; the caller is expected to have it from
        the tetrahedron count, which is continuous in energy.

    Returns
    -------
    tuple of ndarray
        (F(T) - F(0) in eV, S(T) in eV/K, mu(T) in eV), each of shape
        (len(temperatures),).

    """
    from scipy.optimize import brentq

    kb = get_physical_units().KB
    temps = np.asarray(temperatures, dtype="double")
    if len(temps) == 0 or temps[0] != 0.0:
        given = temps[0] if len(temps) else "an empty list"
        raise ValueError(
            "The first temperature has to be 0 K, since the free energies are "
            f"reported against it, but {given} was given."
        )
    if window is None:
        e_win = np.asarray(energies, dtype="double")
        g_win = np.asarray(dos, dtype="double")
    else:
        inside = (energies >= fermi_energy - window) & (
            energies <= fermi_energy + window
        )
        if not inside.any():
            raise ValueError(
                f"The window {fermi_energy - window} to {fermi_energy + window} eV "
                "contains no density-of-states samples."
            )
        e_win = energies[inside]
        g_win = dos[inside]
    low, high = float(e_win.min()), float(e_win.max())
    if mu_0 is None:
        mu_0 = fermi_energy
    # The count below the window is anchored with the occupation of the
    # temperature loop's own T = 0, so that its T -> 0 limit is mu_0 itself
    # rather than a nearby grid point.
    kt_zero = kb * _ZERO_TEMPERATURE
    n_below = n_electrons - float(
        np.trapezoid(g_win * fermi_dirac_occupation(e_win, mu_0, kt_zero), e_win)
    )

    def excess_electrons(mu: float, kt: float) -> float:
        """Return the electron count at (mu, kt) less the target count."""
        counted = n_below + float(
            np.trapezoid(g_win * fermi_dirac_occupation(e_win, mu, kt), e_win)
        )
        return counted - n_electrons

    free_energies = np.zeros(len(temps), dtype="double")
    entropies = np.zeros(len(temps), dtype="double")
    mus = np.zeros(len(temps), dtype="double")
    for i, temperature in enumerate(temps):
        kt = max(kb * float(temperature), kt_zero)
        if temperature == 0.0:
            mu = mu_0
        else:
            mu = brentq(excess_electrons, low, high, args=(kt,), xtol=1e-12)
        occupation = fermi_dirac_occupation(e_win, mu, kt)
        band_energy = float(np.trapezoid(g_win * e_win * occupation, e_win))
        if temperature == 0.0:
            # Exactly zero by the third law. The step at mu_0 leaves one
            # half-occupied sample when mu_0 falls on the grid, and its
            # -k [f ln f + (1-f) ln(1-f)] would otherwise survive.
            entropies[i] = 0.0
        else:
            entropies[i] = -kb * float(
                np.trapezoid(g_win * entropy_terms(occupation), e_win)
            )
        free_energies[i] = band_energy - float(temperature) * entropies[i]
        mus[i] = mu

    return free_energies - free_energies[0], entropies, mus


def _fermi_level_by_counting(electronic_states: ElectronicStates) -> float:
    """Return the energy the electrons fill up to, counted over the k points.

    A stand-in for a Fermi energy the calculation did not report. It only has
    to place the energy window, which is a fraction of an eV wide, and the
    chemical potential the window is then used to solve for is what the free
    energy is built on.

    """
    states = electronic_states
    eigenvalues = np.asarray(states.eigenvalues, dtype="double")
    weights = np.asarray(states.weights, dtype="double")
    weights = weights / weights.sum()
    per_state = np.tile(
        np.repeat(weights, eigenvalues.shape[2]), eigenvalues.shape[0]
    ) * resolve_spin_degeneracy(states)
    flat = np.concatenate([spin.ravel() for spin in eigenvalues])
    order = np.argsort(flat)
    filled = int(np.searchsorted(np.cumsum(per_state[order]), states.n_electrons))
    return float(flat[order][min(filled, len(flat) - 1)])
