# SPDX-License-Identifier: BSD-3-Clause
"""Calculation of free energy of one-electronic states."""

from __future__ import annotations

import dataclasses
import os
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
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

# Temperature in K standing in for 0 K, where k_B T is a division by zero.
_ZERO_TEMPERATURE = 1e-10
# k_B T in eV standing in for 0 K in the k-point sum, which is k_B times a
# temperature five orders larger than the above. The two are not
# interchangeable: sharpening this step changes which states at the Fermi
# level are partially occupied, and with it the free energy at 0 K by a meV.
_ZERO_KT = 1e-10


@dataclasses.dataclass(frozen=True)
class ElectronicStates:
    """Electronic states at a volume point.

    Input container for computing electronic free energies with
    ElectronFreeEnergy.

    Attributes
    ----------
    eigenvalues : ndarray
        Eigenvalues in eV. shape=(spin, kpoints, bands). The spin axis has
        length 1 for non-spin-polarized and 2 for spin-polarized systems.
    weights : ndarray
        Relative k-point weights (e.g., number of arms of the k-star).
        shape=(kpoints,)
    n_electrons : float
        Number of electrons in the unit cell.
    volume : float, optional
        Unit cell volume in angstrom^3. Used only for consistency checks
        against the unit cells the states belong to.
    internal_energy : float, optional
        Static internal energy of the unit cell in eV, e.g., the
        energy (sigma->0) of the calculation the eigenvalues come from.
    spin_degeneracy : int, optional
        Number of electrons one eigenvalue can hold. With None (default) it
        is inferred from the length of the spin axis of eigenvalues. The
        inference is wrong for non-collinear calculations, where the spin
        axis has length 1 but each spinor state holds one electron, so 1
        must be given explicitly. See ElectronFreeEnergy.
    fermi_energy : float, optional
        Fermi energy in eV as reported by the calculation the eigenvalues
        come from. Used to anchor the electron count when the free energy is
        computed from a density of states; see free_energy_from_dos.
    kpoints : ndarray, optional
        The irreducible k-points the eigenvalues sit on, in fractional
        coordinates of the reciprocal basis vectors.
        shape=(kpoints, 3), dtype='double'
    mesh : ndarray, optional
        Numbers of mesh divisions, or the grid generating matrix of a
        generalized regular grid. shape=(3,) or (3, 3), dtype='int64'
    cell : PhonopyAtoms, optional
        The cell the eigenvalues were computed for, needed for the symmetry
        search behind the grid.

    kpoints, mesh and cell are what the tetrahedron method needs and are
    given together or not at all. Without them only the k-point sum is
    available, which is what every caller before them did.

    """

    eigenvalues: NDArray[np.double]
    weights: NDArray[np.int64] | NDArray[np.double]
    n_electrons: float
    volume: float | None = None
    internal_energy: float | None = None
    spin_degeneracy: int | None = None
    fermi_energy: float | None = None
    kpoints: NDArray[np.double] | None = None
    mesh: NDArray[np.int64] | None = None
    cell: PhonopyAtoms | None = None

    def __post_init__(self) -> None:
        """Convert the array fields to ndarray and validate their shapes."""
        object.__setattr__(
            self, "eigenvalues", np.asarray(self.eigenvalues, dtype="double")
        )
        object.__setattr__(self, "weights", np.asarray(self.weights))
        if self.kpoints is not None:
            object.__setattr__(
                self, "kpoints", np.asarray(self.kpoints, dtype="double")
            )
        if self.mesh is not None:
            object.__setattr__(self, "mesh", np.asarray(self.mesh))
        if self.eigenvalues.ndim != 3:
            raise ValueError(
                "eigenvalues must have shape (spin, kpoints, bands), not "
                f"{self.eigenvalues.shape}."
            )
        if self.eigenvalues.shape[0] not in (1, 2):
            raise ValueError(
                "The spin axis of eigenvalues must have length 1 or 2, not "
                f"{self.eigenvalues.shape[0]}."
            )
        if self.weights.ndim != 1 or len(self.weights) != self.eigenvalues.shape[1]:
            raise ValueError("weights must have one value per k-point of eigenvalues.")
        if self.spin_degeneracy not in (None, 1, 2):
            raise ValueError(
                f"spin_degeneracy must be 1 or 2, not {self.spin_degeneracy}."
            )
        grid = (self.kpoints, self.mesh, self.cell)
        if any(item is not None for item in grid) and any(
            item is None for item in grid
        ):
            raise ValueError(
                "kpoints, mesh and cell describe the sampling grid together; "
                "give all three or none."
            )
        if self.kpoints is not None and self.mesh is not None:
            if self.kpoints.shape != (self.eigenvalues.shape[1], 3):
                raise ValueError(
                    "kpoints must have shape (kpoints, 3) matching "
                    f"eigenvalues, not {self.kpoints.shape}."
                )
            if self.mesh.shape not in ((3,), (3, 3)):
                raise ValueError(
                    f"mesh must have shape (3,) or (3, 3), not {self.mesh.shape}."
                )


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
        self._degeneracy = _spin_degeneracy(states)

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

        The k-point sum that ElectronFreeEnergy performs integrates a
        delta-function density of states and converges far too slowly for the
        electronic free energy: reaching the same answer needs many more
        irreducible k-points. Building the density of states by the
        tetrahedron method instead converges at the mesh a static calculation
        would use anyway.

        Summing the spin channels here is what lets a spin-polarized calculation
        be treated no differently afterwards. The occupation depends on the
        energy and the chemical potential alone, so the electron count, the band
        energy and the entropy are all integrals of the summed density of states
        against one chemical potential.

        The energies are processed in blocks. The tetrahedron integration weights
        are built as one (ir points, sampling points, bands) array, which for a
        dense mesh and the fine energy grid this needs is enormous: 7980
        irreducible k-points, 8001 energies and 40 bands is 20 GB, and it is
        allocated twice. Sampling points are independent of one another, so
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
    numbers are empirical. The Fermi factor is within 1e-5 of 0 or 1 at
    12 k_B T, and on HCP Ti at 400 K, where the 0.5 eV floor is what acts,
    widening the window to 2.0 eV moved the free energy by 0.005 ueV at every
    one of 25 grid points.

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

    The counterpart of compute_free_energy_and_entropy, which sums
    Fermi-Dirac occupations over irreducible k-points. Which one runs is the
    caller's choice rather than an inference from the data: this one needs
    kpoints, mesh and cell on the states and raises without them.

    **The two differ in what they return.** This one gives the free energy
    against its own value at 0 K: the states outside the window are never
    integrated, so the band sum itself would depend on the window and only
    the difference means anything. compute_free_energy_and_entropy sums every
    occupied state and returns that sum. Subtracting the value at the first
    temperature, as compute_electronic_contributions_from_states does, leaves
    the same quantity either way.

    **A shifted mesh loses a little of the density.** Where all four corners
    of a tetrahedron fall in one symmetry orbit, every band is flat on it and
    its weight is a delta function the sampled density cannot carry. The
    number of such tetrahedra does not grow with the mesh, so the loss falls
    away as the mesh is refined -- a per cent on a coarse mesh, well under a
    tenth of that on one a static calculation would use. A Gamma-centred mesh
    has none of them.

    Parameters
    ----------
    electronic_states : ElectronicStates
        States carrying kpoints, mesh, cell and fermi_energy.
    temperatures : array_like
        Temperatures in K, the first of them 0.
    window : float, optional
        Half-width of the energy window around the Fermi level in eV. None,
        the default, takes 12 k_B T of the highest temperature and at least
        0.5 eV. Beyond that the Fermi factor is within 1e-5 of 0 or 1, and on
        HCP Ti at 400 K the whole 0.5 to 2.0 eV range agrees to 0.005 ueV at
        every grid point while 0.5 eV costs a seventh of the time.
    energy_spacing : float, optional
        Spacing of the energy grid inside the window in eV. Default is
        0.0005, at which halving the grid moves the free energy by less than
        a ueV; at 0.001 it still moves by about 1 ueV.

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

        N(mu, T) = int g(E) f(E) dE = n_electrons
        E(T)     = int g(E) E f(E) dE
        T S(T)   = -k T int g(E) [f ln f + (1-f) ln(1-f)] dE
        F(T)     = E(T) - T S(T)

    **Every integral is restricted to a window around the Fermi level, and
    that is a matter of correctness rather than speed.** The band energy over
    all occupied states is of order -200 eV while the electronic free energy
    is of order 10 ueV, so a difference of totals loses fourteen digits.
    Outside the window nothing depends on temperature: states below are
    occupied to better than 1e-17 at 40 k_B T and contribute a constant that
    cancels in F(T) - F(0), states above are empty.

    **The states below the window are never integrated.** They include deep
    semicore states, which are near-delta peaks that no quadrature on a meV
    grid integrates to better than a few hundredths of an electron. Whatever
    absorbs that error is the chemical potential, and 0.03 electrons is a
    16 meV shift in mu, which is k_B T. Instead the count below the window is
    defined by anchoring to mu_0, the chemical potential at T = 0,

        n_below := n_electrons - int_window g(E) f(E; mu_0, T -> 0) dE

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
        np.trapezoid(g_win * _occupation(e_win, mu_0, kt_zero), e_win)
    )

    def excess_electrons(mu: float, kt: float) -> float:
        """Return the electron count at (mu, kt) less the target count."""
        counted = n_below + float(
            np.trapezoid(g_win * _occupation(e_win, mu, kt), e_win)
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
        occupation = _occupation(e_win, mu, kt)
        band_energy = float(np.trapezoid(g_win * e_win * occupation, e_win))
        if temperature == 0.0:
            # Exactly zero by the third law. The step at mu_0 leaves one
            # half-occupied sample when mu_0 falls on the grid, and its
            # -k [f ln f + (1-f) ln(1-f)] would otherwise survive.
            entropies[i] = 0.0
        else:
            entropies[i] = -kb * float(
                np.trapezoid(g_win * _entropy_terms(occupation), e_win)
            )
        free_energies[i] = band_energy - float(temperature) * entropies[i]
        mus[i] = mu

    return free_energies - free_energies[0], entropies, mus


def _occupation(
    energies: NDArray[np.double], mu: float, kt: float
) -> NDArray[np.double]:
    """Return the Fermi-Dirac occupation, guarded against overflow."""
    return 1.0 / (1.0 + np.exp(np.clip((energies - mu) / kt, -100.0, 100.0)))


def _entropy_terms(occupations: NDArray[np.double]) -> NDArray[np.double]:
    """Return f ln f + (1 - f) ln(1 - f), of the same shape as occupations.

    The terms vanish wherever the occupation saturates; masking keeps log(0)
    out of them rather than relying on cancellation.

    """
    mask = (occupations > 1e-12) & (occupations < 1.0 - 1e-12)
    safe = np.where(mask, occupations, 0.5)
    return np.where(mask, safe * np.log(safe) + (1.0 - safe) * np.log1p(-safe), 0.0)


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
    ) * _spin_degeneracy(states)
    flat = np.concatenate([spin.ravel() for spin in eigenvalues])
    order = np.argsort(flat)
    filled = int(np.searchsorted(np.cumsum(per_state[order]), states.n_electrons))
    return float(flat[order][min(filled, len(flat) - 1)])


def _spin_degeneracy(electronic_states: ElectronicStates) -> int:
    """Return the number of electrons one eigenvalue holds.

    Inferred from the spin axis as ElectronFreeEnergy does, unless given
    explicitly. The inference is wrong for a non-collinear calculation, whose
    spin axis has length 1 while each spinor holds one electron.

    """
    if electronic_states.spin_degeneracy is not None:
        return electronic_states.spin_degeneracy
    return 2 if electronic_states.eigenvalues.shape[0] == 1 else 1


def compute_free_energy_and_entropy(
    electronic_states: ElectronicStates,
    temperatures: Sequence[float] | NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return band free energies and entropies at temperatures.

    Each value is the whole band sum at that temperature, E - TS over the
    occupied states, on the energy zero the eigenvalues carry. It is not a
    difference from 0 K, which is what compute_free_energy_by_tetrahedron
    returns instead.

    Parameters
    ----------
    electronic_states : ElectronicStates
        Electronic states at a volume point.
    temperatures : array_like
        Temperatures in K. shape=(temperatures,)

    Returns
    -------
    tuple of ndarray
        Band free energies in eV and entropies S_el in eV/K at the given
        temperatures. shape=(temperatures,) each.

    """
    efe = ElectronFreeEnergy(
        electronic_states.eigenvalues,
        electronic_states.weights,
        electronic_states.n_electrons,
        spin_degeneracy=electronic_states.spin_degeneracy,
    )
    free_energies = []
    entropies = []
    for temp in np.array(temperatures, dtype="double"):
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
        # ElectronFreeEnergy.entropy returns T * S in eV.
        if temp > _ZERO_TEMPERATURE:
            entropies.append(efe.entropy / temp)
        else:
            entropies.append(0.0)
    return (
        np.array(free_energies, dtype="double"),
        np.array(entropies, dtype="double"),
    )


def write_electronic_states_hdf5(
    electronic_structures: Sequence[ElectronicStates],
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> None:
    """Write electronic states in hdf5.

    All ElectronicStates must carry volume and internal_energy. The file
    contains one group "volume-XXX" per volume point with the datasets
    eigenvalues ((spin, kpoints, bands), eV), weights ((kpoints,)),
    n_electrons, volume (angstrom^3), and energy (eV, static internal
    energy). spin_degeneracy and fermi_energy are written only when they are
    set. The number of volume points is stored in the root attribute
    "n_volumes".

    """
    import h5py

    with h5py.File(filename, "w") as w:
        w.attrs["creator"] = "phonopy"
        w.attrs["n_volumes"] = len(electronic_structures)
        for i, electronic_states in enumerate(electronic_structures):
            if (
                electronic_states.volume is None
                or electronic_states.internal_energy is None
            ):
                raise ValueError(
                    f"electronic_structures[{i}] must carry volume and internal_energy."
                )
            group = w.create_group(f"volume-{i:03d}")
            group.create_dataset(
                "eigenvalues",
                data=electronic_states.eigenvalues,
                compression="gzip",
            )
            group.create_dataset("weights", data=electronic_states.weights)
            group.create_dataset(
                "n_electrons", data=float(electronic_states.n_electrons)
            )
            group.create_dataset("volume", data=float(electronic_states.volume))
            group.create_dataset(
                "energy", data=float(electronic_states.internal_energy)
            )
            if electronic_states.spin_degeneracy is not None:
                group.create_dataset(
                    "spin_degeneracy", data=int(electronic_states.spin_degeneracy)
                )
            if electronic_states.fermi_energy is not None:
                group.create_dataset(
                    "fermi_energy", data=float(electronic_states.fermi_energy)
                )


def read_electronic_states_hdf5(
    filename: str | os.PathLike = "electronic_states.hdf5",
) -> list[ElectronicStates]:
    """Read electronic states from hdf5.

    Returns a list of ElectronicStates in the file order, each carrying
    volume and internal_energy. The list is the electronic_structures
    parameter of run_qha; internal_energies can then be given as None.

    """
    import h5py

    electronic_structures = []
    with h5py.File(filename, "r") as f:
        n_volumes = int(f.attrs["n_volumes"])
        for i in range(n_volumes):
            group = f[f"volume-{i:03d}"]
            electronic_structures.append(
                ElectronicStates(
                    eigenvalues=group["eigenvalues"][:],
                    weights=group["weights"][:],
                    n_electrons=float(group["n_electrons"][()]),
                    volume=float(group["volume"][()]),
                    internal_energy=float(group["energy"][()]),
                    spin_degeneracy=(
                        int(group["spin_degeneracy"][()])
                        if "spin_degeneracy" in group
                        else None
                    ),
                    fermi_energy=(
                        float(group["fermi_energy"][()])
                        if "fermi_energy" in group
                        else None
                    ),
                )
            )
    return electronic_structures


def get_free_energy_at_T(
    tmin: float,
    tmax: float,
    tstep: float,
    eigenvalues: NDArray[np.double],
    weights: NDArray[np.int64] | NDArray[np.double],
    n_electrons: float,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return free energies at given temperatures."""
    free_energies = []
    efe = ElectronFreeEnergy(eigenvalues, weights, n_electrons)
    temperatures = np.arange(tmin, tmax + 1e-8, tstep, dtype="double")
    for temp in temperatures:
        efe.run(float(temp))
        free_energies.append(efe.free_energy)
    return temperatures, np.array(free_energies, dtype="double")


class ElectronFreeEnergy:
    r"""Class to calculate free energy of one-electronic states.

    Fixed density-of-states approximation for energy and entropy of electrons.

    This is supposed to be used for metals, i.e., chemical potential is not
    in band gap.

    Entropy
    -------

    .. math::

       S_\text{el}(V) = -gk_{\mathrm{B}}\Sigma_i \{ f_i(V) \ln f_i(V) +
       [1-f_i(V)]\ln [1-f_i(V)] \}

    .. math::

       f_i(V) = \left\{ 1 + \exp\left[\frac{\epsilon_i(V) - \mu(V)}{T}\right]
       \right\}^{-1}

    where :math:`g` is 2 for non-spin polarized systems and 1 for spin
    polarized systems.

    Energy
    ------

    .. math::

       E_\text{el}(V) = g\sum_i f_i(V) \epsilon_i(V)

    Checking mu
    -----------

    :math:`\mu` is the root of the electron-count equation, solved by
    _chemical_potential():

    .. math::

       N = \frac{g}{\sum_k w_k} \sum_k w_k \sum_i f_{ki}(V)

    where :math:`N` is the number of electrons in the unit cell and
    :math:`w_k` the k-point weights. Only the eigenvalues, the weights,
    :math:`N` and :math:`g` enter, so comparing mu against the Fermi energy
    reported by the electronic structure code that produced them checks all
    four.

    The comparison is meaningful only between quantities at the same
    temperature, so pair them by how that code broadened the occupations:

    - Schemes that reproduce the zero-broadening answer -- tetrahedron methods,
      Methfessel-Paxton, Gaussian -- against mu at 0 K. Their width is a
      numerical device rather than a temperature; treating it as one makes the
      agreement worse, not better.
    - Fermi-Dirac (thermal) smearing of width sigma against mu at
      T = sigma / k_B.

    Attributes
    ----------
    entropy: float
        Entropy in eV (T * S).
    energy: float
        Energy in eV.
    free_energy: float
        energy - entropy in eV.
    mu: float
        Chemical potential in eV.

    """

    def __init__(
        self,
        eigenvalues: NDArray[np.double],
        weights: NDArray[np.int64] | NDArray[np.double],
        n_electrons: float,
        spin_degeneracy: int | None = None,
    ) -> None:
        """Init method.

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues in eV.
            dtype='double'
            shape=(spin, kpoints, bands)
        weights: ndarray
            Relative k-point weights, e.g., geometric k-point weights
            (number of arms of k-star in BZ) or normalized weights.
            shape=(irreducible_kpoints,)
        n_electrons: float
            Number of electrons in unit cell.
        spin_degeneracy: int, optional
            Number of electrons one eigenvalue can hold, i.e., g above. With
            None (default) it is inferred from the length of the spin axis of
            eigenvalues: 2 for length 1 and 1 for length 2. This inference is
            wrong for non-collinear calculations, whose spin axis has length
            1 while each spinor state holds one electron, so g = 1 must be
            given explicitly there.

        """
        # shape=(kpoints, spin, bands)
        self._eigenvalues = np.array(
            eigenvalues.swapaxes(0, 1), dtype="double", order="C"
        )
        self._weights = weights
        self._n_electrons = n_electrons

        if spin_degeneracy is None:
            if self._eigenvalues.shape[1] == 1:
                self._g = 2
            elif self._eigenvalues.shape[1] == 2:
                self._g = 1
            else:
                raise RuntimeError(
                    "The spin axis of eigenvalues must have length 1 or 2, not "
                    f"{self._eigenvalues.shape[1]}."
                )
        elif spin_degeneracy in (1, 2):
            self._g = spin_degeneracy
        else:
            raise ValueError(f"spin_degeneracy must be 1 or 2, not {spin_degeneracy}.")

        self._T: float
        self._f: NDArray[np.double]  # occupation numbers, shape=(kpoints, spin, bands)
        self._mu: float | None = None
        self._entropy: float | None = None
        self._energy: float | None = None

    def run(self, temp: float) -> None:
        """Calculate free energies.

        Parameters
        ----------
        temp: float
            Temperature in K

        """
        if temp < _ZERO_TEMPERATURE:
            self._T = _ZERO_KT
        else:
            self._T = temp * get_physical_units().KB
        mu = self._chemical_potential()
        self._mu = mu
        self._f = self._occupation_number(self._eigenvalues, mu)
        self._entropy = self._get_entropy()
        self._energy = self._get_energy()

    @property
    def free_energy(self) -> float:
        """Return free energies."""
        return self.energy - self.entropy

    @property
    def energy(self) -> float:
        """Return energies."""
        if self._energy is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._energy

    @property
    def entropy(self) -> float:
        """Return entropies."""
        if self._entropy is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._entropy

    @property
    def mu(self) -> float:
        """Return chemical potential."""
        if self._mu is None:
            raise RuntimeError("Run method has not been called yet.")
        return self._mu

    def _get_entropy(self) -> float:
        # f: shape=(kpoints, spin*bands), row i holds all (spin, band)
        # occupation numbers at the i-th irreducible k-point.
        f = self._f.reshape(len(self._weights), -1)
        entropy = -(_entropy_terms(f).sum(axis=1) * self._weights).sum()
        return float(entropy * self._g * self._T / self._weights.sum())

    def _get_energy(self) -> float:
        # occ_eigvals: shape=(kpoints, spin, bands), same as self._eigenvalues.
        occ_eigvals = self._f * self._eigenvalues
        # reshape to (kpoints, spin*bands), sum over spin*bands leaves
        # shape=(kpoints,), one value per irreducible k-point, matching
        # self._weights for the np.dot below.
        return float(
            np.dot(
                occ_eigvals.reshape(len(self._weights), -1).sum(axis=1), self._weights
            )
            * self._g
            / self._weights.sum()
        )

    def _chemical_potential(self) -> float:
        try:
            from scipy.optimize import brentq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        emin = np.min(self._eigenvalues)
        emax = np.max(self._eigenvalues)
        # brentq's default xtol (2e-12) is too loose here: near T -> 0 the
        # occupation number is a near step function, so n(mu) can change by
        # O(1e-3) for an O(1e-10) change in mu. A tight xtol is needed to
        # match the occupation numbers (and hence energy/entropy) to the
        # precision expected by callers.
        mu = brentq(
            lambda mu: self._number_of_electrons(mu) - self._n_electrons,
            emin,
            emax,
            xtol=1e-14,
        )
        return float(mu)

    def _number_of_electrons(self, mu: float) -> float:
        # eigvals: shape=(kpoints, spin*bands); occupation_number keeps the
        # same shape, and summing over spin*bands leaves shape=(kpoints,),
        # matching self._weights for the np.dot below.
        eigvals = self._eigenvalues.reshape(len(self._weights), -1)
        n = (
            np.dot(self._occupation_number(eigvals, mu).sum(axis=1), self._weights)
            * self._g
            / self._weights.sum()
        )
        return float(n)

    def _occupation_number(
        self, e: NDArray[np.double], mu: float
    ) -> NDArray[np.double]:
        """Return occupation numbers, same shape as `e`."""
        return _occupation(e, mu, self._T)
