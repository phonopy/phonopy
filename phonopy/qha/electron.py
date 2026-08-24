# SPDX-License-Identifier: BSD-3-Clause
"""Calculation of free energy of one-electronic states."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms


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
        """Validate shapes."""
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
        if self.kpoints is not None:
            if self.kpoints.shape != (self.eigenvalues.shape[1], 3):
                raise ValueError(
                    "kpoints must have shape (kpoints, 3) matching "
                    f"eigenvalues, not {self.kpoints.shape}."
                )
            if np.asarray(self.mesh).shape not in ((3,), (3, 3)):
                raise ValueError(
                    "mesh must have shape (3,) or (3, 3), not "
                    f"{np.asarray(self.mesh).shape}."
                )


def compute_tetrahedron_dos(
    electronic_states: ElectronicStates,
    energies: Sequence[float] | NDArray[np.double],
    max_bytes: float = 2.0e8,
) -> NDArray[np.double]:
    """Return the electronic density of states by the linear tetrahedron method.

    In states/eV per cell, summed over spin channels, on the given energies.

    The k-point sum that ElectronFreeEnergy performs integrates a
    delta-function density of states and converges far too slowly for the
    electronic free energy: reaching the same answer needs of order twenty
    times the irreducible k-points. Building the density of states by the
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
    electronic_states : ElectronicStates
        States carrying kpoints, mesh and cell.
    energies : array_like
        Energies to sample the density of states at, in eV.
    max_bytes : float, optional
        Rough ceiling on the integration-weight array of one block, in bytes.
        Default is 2e8, i.e. 200 MB against the 400 MB that two of them take.
        Raise it to trade memory for fewer passes over the eigenvalues.

    Returns
    -------
    ndarray
        shape=(len(energies),), dtype='double'

    """
    from phonopy.phonon.grid import BZGrid, get_ir_grid_points, get_ir_kpoint_map
    from phonopy.phonon.spectrum import TetrahedronDOSAccumulator
    from phonopy.structure.symmetry import Symmetry

    states = electronic_states
    if states.kpoints is None or states.mesh is None or states.cell is None:
        raise ValueError(
            "The tetrahedron method needs kpoints, mesh and cell on the "
            "ElectronicStates; without them only the k-point sum is "
            "available."
        )

    bz_grid = BZGrid(
        states.mesh,
        lattice=states.cell.cell,
        symmetry_dataset=Symmetry(states.cell).dataset,
    )
    ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bz_grid)
    id_map = get_ir_kpoint_map(states.kpoints, states.weights, bz_grid)

    sampling_points = np.asarray(energies, dtype="double")
    n_ir, n_band = states.eigenvalues.shape[1:]
    affordable = max_bytes / (8.0 * n_ir * n_band)
    block = len(sampling_points)
    if affordable < block:
        block = max(1, int(affordable))

    dos = np.zeros(len(sampling_points), dtype="double")
    for eigenvalues in states.eigenvalues:
        mapped = np.ascontiguousarray(eigenvalues[id_map])
        for start in range(0, len(sampling_points), block):
            chunk = sampling_points[start : start + block]
            result = TetrahedronDOSAccumulator(
                mapped,
                bz_grid,
                ir_grid_points=ir_grid_points,
                ir_grid_weights=ir_grid_weights,
                ir_grid_map=ir_grid_map,
                sampling_points=chunk,
            ).result
            dos[start : start + len(chunk)] += result.density[0, :, 0]
    return dos * _spin_degeneracy(states)


def compute_free_energy_by_tetrahedron(
    electronic_states: ElectronicStates,
    temperatures: Sequence[float] | NDArray[np.double],
    window: float = 2.0,
    energy_spacing: float = 0.0005,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Return F(T) - F(0) and the entropy through the tetrahedron method.

    The counterpart of compute_free_energy_and_entropy, which sums
    Fermi-Dirac occupations over irreducible k-points. Which one runs is the
    caller's choice rather than an inference from the data: this one needs
    kpoints, mesh and cell on the states and raises without them.

    Parameters
    ----------
    electronic_states : ElectronicStates
        States carrying kpoints, mesh, cell and fermi_energy.
    temperatures : array_like
        Temperatures in K.
    window : float, optional
        Half-width of the energy window around the Fermi level in eV.
        Default is 2.0.
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
    if electronic_states.fermi_energy is None:
        raise ValueError(
            "The tetrahedron method anchors the electron count to the "
            "calculation's Fermi energy, which these ElectronicStates do not "
            "carry."
        )
    fermi = electronic_states.fermi_energy
    n_points = int(round(2 * window / energy_spacing)) + 1
    energies = np.linspace(fermi - window, fermi + window, n_points)
    dos = compute_tetrahedron_dos(electronic_states, energies)
    free_energy, entropy, _ = free_energy_from_dos(
        energies,
        dos,
        electronic_states.n_electrons,
        temperatures,
        fermi,
        window=window,
    )
    return free_energy, entropy


def free_energy_from_dos(
    energies: NDArray[np.double],
    dos: NDArray[np.double],
    n_electrons: float,
    temperatures: Sequence[float] | NDArray[np.double],
    fermi_energy: float,
    window: float = 2.0,
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
    defined by anchoring to the Fermi level the calculation reports,

        n_below := n_electrons - int_window g(E) theta(E_F - E) dE

    so that mu(T = 0) = E_F by construction and every remaining integral runs
    over a smooth stretch of the density of states.

    Parameters
    ----------
    energies : ndarray
        Energies the density of states is sampled at, in eV.
    dos : ndarray
        Density of states in states/eV per cell, as
        compute_tetrahedron_dos returns it.
    n_electrons : float
        Number of electrons in the cell.
    temperatures : array_like
        Temperatures in K.
    fermi_energy : float
        Fermi energy in eV, used as the anchor described above.
    window : float, optional
        Half-width of the window around the Fermi level in eV. Default is
        2.0, which is 58 k_B T at 400 K.

    Returns
    -------
    tuple of ndarray
        (F(T) - F(0) in eV, S(T) in eV/K, mu(T) in eV), each of shape
        (len(temperatures),).

    """
    from scipy.optimize import brentq

    kb = get_physical_units().KB
    temps = np.asarray(temperatures, dtype="double")
    low, high = fermi_energy - window, fermi_energy + window
    inside = (energies >= low) & (energies <= high)
    if not inside.any():
        raise ValueError(
            f"The window {low} to {high} eV contains no density-of-states samples."
        )
    e_win = energies[inside]
    g_win = dos[inside]
    n_below = n_electrons - float(
        np.trapezoid(np.where(e_win <= fermi_energy, g_win, 0.0), e_win)
    )

    free_energies = np.zeros(len(temps), dtype="double")
    entropies = np.zeros(len(temps), dtype="double")
    mus = np.zeros(len(temps), dtype="double")
    for i, temperature in enumerate(temps):
        kt = kb * max(float(temperature), 1e-10)

        def count(mu: float, kt: float = kt) -> float:
            return n_below + float(
                np.trapezoid(g_win * _occupation(e_win, mu, kt), e_win)
            )

        mu = brentq(lambda mu: count(mu) - n_electrons, low, high, xtol=1e-12)
        occupation = _occupation(e_win, mu, kt)
        band_energy = float(np.trapezoid(g_win * e_win * occupation, e_win))
        # The integrand vanishes wherever the occupation saturates; masking
        # keeps log(0) out of it rather than relying on cancellation.
        mask = (occupation > 1e-12) & (occupation < 1.0 - 1e-12)
        safe = np.where(mask, occupation, 0.5)
        terms = np.where(
            mask, safe * np.log(safe) + (1.0 - safe) * np.log1p(-safe), 0.0
        )
        entropies[i] = -kb * float(np.trapezoid(g_win * terms, e_win))
        free_energies[i] = band_energy - float(temperature) * entropies[i]
        mus[i] = mu

    return free_energies - free_energies[0], entropies, mus


def _occupation(
    energies: NDArray[np.double], mu: float, kt: float
) -> NDArray[np.double]:
    """Return the Fermi-Dirac occupation, guarded against overflow."""
    return 1.0 / (1.0 + np.exp(np.clip((energies - mu) / kt, -100.0, 100.0)))


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
        if temp > 1e-10:
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
        self._mu = None
        self._entropy = None
        self._energy = None

    def run(self, temp: float) -> None:
        """Calculate free energies.

        Parameters
        ----------
        temp: float
            Temperature in K

        """
        if temp < 1e-10:
            self._T = 1e-10
        else:
            self._T = temp * get_physical_units().KB
        self._mu = self._chemical_potential()
        self._f = self._occupation_number(self._eigenvalues, self._mu)
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
        mask = (f > 1e-12) & (f < 1 - 1e-12)
        f_safe = np.where(mask, f, 0.5)  # avoid log(0); masked out below anyway
        terms = np.where(
            mask, f_safe * np.log(f_safe) + (1 - f_safe) * np.log(1 - f_safe), 0.0
        )
        entropy = -(terms.sum(axis=1) * self._weights).sum()
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
        de = (e - mu) / self._T
        de = np.where(de < 100, de, 100.0)  # To avoid overflow
        de = np.where(de > -100, de, -100.0)  # To avoid underflow
        return 1.0 / (1 + np.exp(de))
