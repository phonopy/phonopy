# SPDX-License-Identifier: BSD-3-Clause
"""Quasi-harmonic approximation driver returning immutable results.

The public entry point is run_qha(), which takes one Phonopy instance per
volume point, computes thermal properties internally, fits the total free
energy to an equation of state at each temperature, and returns a frozen
QHAResult dataclass. File output and plotting live in separate modules.

"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import spglib
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units
from phonopy.qha.calc import (
    compute_gruneisen_parameters,
    compute_heat_capacity_p_polyfit,
    compute_volumetric_thermal_expansion,
)
from phonopy.qha.electron import ElectronicStates
from phonopy.qha.eos import fit_to_eos, get_eos
from phonopy.qha.lattice import LatticeParametersFit, compute_axial_thermal_expansion
from phonopy.qha.thermal import (
    compute_electronic_contributions_from_states,
    compute_thermal_properties,
    freeze_ndarray_fields,
)
from phonopy.structure.symmetry import NosymDataset

if TYPE_CHECKING:
    from phonopy.api_phonopy import Phonopy


@dataclasses.dataclass(frozen=True)
class QHACpPolyfitData:
    """C_P computed via polynomial fits of Cv(V) and S(V).

    All temperature-indexed arrays are aligned with QHAResult.temperatures
    (length N). Attributes with N - 1 entries correspond to
    temperatures[1:]. When electronic structures are supplied to run_qha,
    Cv and S include the electronic contributions.

    Attributes
    ----------
    heat_capacities : ndarray
        Heat capacities at constant pressure in J/K/mol with a leading 0.0
        element. shape=(N,)
    dsdv : ndarray
        dS/dV at temperatures in J/K/mol/angstrom^3 with a leading 0.0
        element. shape=(N,)
    volume_cv_parameters : ndarray
        Degree-4 polynomial coefficients of Cv(V) fits. shape=(N - 1, 5)
    volume_entropy_parameters : ndarray
        Degree-4 polynomial coefficients of S(V) fits. shape=(N - 1, 5)
    volume_cv : ndarray
        (V, Cv) data pairs used for the fits. shape=(N - 1, volumes, 2)
    volume_entropy : ndarray
        (V, S) data pairs used for the fits. shape=(N - 1, volumes, 2)

    """

    heat_capacities: NDArray[np.double]
    dsdv: NDArray[np.double]
    volume_cv_parameters: NDArray[np.double]
    volume_entropy_parameters: NDArray[np.double]
    volume_cv: NDArray[np.double]
    volume_entropy: NDArray[np.double]

    def __post_init__(self) -> None:
        """Make ndarray fields read-only."""
        freeze_ndarray_fields(self)


@dataclasses.dataclass(frozen=True)
class QHALatticeData:
    """Lattice parameters and axial thermal expansions at temperatures.

    Attributes
    ----------
    lattice_parameters : ndarray
        Lattice-vector lengths (a, b, c) at temperatures in angstrom.
        shape=(N, 3)
    axial_thermal_expansions : ndarray
        Linear thermal expansion coefficients (alpha_a, alpha_b, alpha_c)
        at temperatures in 1/K with a leading row of zeros. shape=(N, 3)
    k : float
        Geometric constant k = V / (a b c) determined from the input
        cells, where V is the primitive cell volume and a, b, c are the
        lattice-vector lengths of the unit cells; k therefore also
        absorbs the unit-cell to primitive-cell volume ratio.
    ratio_coefficients : ndarray
        Polynomial coefficients of the axial ratios b/a and c/a vs V in
        np.polyfit order. shape=(2, degree + 1)

    """

    lattice_parameters: NDArray[np.double]
    axial_thermal_expansions: NDArray[np.double]
    k: float
    ratio_coefficients: NDArray[np.double]

    def __post_init__(self) -> None:
        """Make ndarray fields read-only."""
        freeze_ndarray_fields(self)


@dataclasses.dataclass(frozen=True)
class QHAResult:
    """Immutable results of a quasi-harmonic approximation calculation.

    All temperature-indexed arrays have the same length N, which is one
    less than the number of input temperature points at which the EOS
    fitting succeeded (the extra point is consumed by finite differences).
    Derived quantities computed by central differences (thermal_expansion,
    gruneisen_parameters) carry a leading 0.0 element. All energies and
    volumes refer to the primitive cell, to which the phonon thermal
    properties are normalized.

    Attributes
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(N,)
    volumes : ndarray
        Primitive cell volumes of the input volume grid in angstrom^3.
        shape=(volumes,)
    helmholtz_volume : ndarray
        Total free energies (electronic + phonon [+ pV]) at temperatures
        and input volumes in eV. shape=(N, volumes)
    eos_name : str
        Name of the equation of state used for fitting.
    eos_parameters : ndarray
        Fitted EOS parameters [E_0 (eV), B_0 (eV/angstrom^3), B'_0,
        V_0 (angstrom^3)] at temperatures. shape=(N, 4)
    equilibrium_volumes : ndarray
        Equilibrium volumes V_0 at temperatures in angstrom^3. shape=(N,)
    gibbs_free_energies : ndarray
        Gibbs free energies at temperatures in eV. shape=(N,)
    bulk_moduli : ndarray
        Bulk moduli at temperatures in GPa. shape=(N,)
    thermal_expansion : ndarray
        Volumetric thermal expansion coefficients beta at temperatures in
        1/K. shape=(N,)
    gruneisen_parameters : ndarray
        Thermodynamic Gruneisen parameters at temperatures. shape=(N,)
    heat_capacity_P : QHACpPolyfitData
        C_P computed via polynomial fits of Cv(V) and S(V).
    lattice : QHALatticeData or None
        Lattice parameters at temperatures. None for triclinic and
        monoclinic crystals and when the input cells are not consistent
        with volume-independent cell angles.

    """

    temperatures: NDArray[np.double]
    volumes: NDArray[np.double]
    helmholtz_volume: NDArray[np.double]
    eos_name: str
    eos_parameters: NDArray[np.double]
    equilibrium_volumes: NDArray[np.double]
    gibbs_free_energies: NDArray[np.double]
    bulk_moduli: NDArray[np.double]
    thermal_expansion: NDArray[np.double]
    gruneisen_parameters: NDArray[np.double]
    heat_capacity_P: QHACpPolyfitData
    lattice: QHALatticeData | None

    def __post_init__(self) -> None:
        """Make ndarray fields read-only."""
        freeze_ndarray_fields(self)


def run_qha(
    phonopys: Sequence[Phonopy],
    temperatures: Sequence[float] | NDArray[np.double],
    internal_energies: Sequence[float] | NDArray[np.double] | None = None,
    electronic_structures: Sequence[ElectronicStates] | None = None,
    mesh: float | Sequence[int] | NDArray[np.int64] = 100.0,
    pressure: float | None = None,
    eos: str = "vinet",
    lattice_fit_degree: int = 2,
    verbose: bool = False,
) -> QHAResult:
    """Run a quasi-harmonic approximation calculation.

    For each Phonopy instance (one per volume point, with force constants
    set), mesh sampling and thermal properties are computed internally on
    the given temperature grid. The total free energy F(V; T) is fitted to
    an equation of state at each temperature and thermodynamic quantities
    are derived from the fits. Lattice parameters a(T), b(T), c(T) are
    obtained by fitting the axial ratios vs volume and evaluating them at
    the equilibrium volumes; this is skipped with a warning for triclinic
    and monoclinic crystals and when the cell angles of the input cells
    depend on volume.

    Note that finite differences consume one temperature point: supply one
    more point than the temperature range of interest.

    Parameters
    ----------
    phonopys : Sequence[Phonopy]
        One Phonopy instance per volume point with force constants set.
        At least 5 instances are needed.
    temperatures : array_like
        Temperatures in K in strictly ascending order. shape=(temperatures,)
    internal_energies : array_like, optional
        Static internal energies U(V) other than the phonon free energy,
        e.g., electronic total energies from first-principles
        calculations or potential energies from machine learning
        potentials, in eV per primitive cell (consistently with the
        normalization of the phonon thermal properties).
        shape=(volumes,). When None, the internal energies carried by
        electronic_structures are used (e.g., loaded from
        electronic_states.hdf5); an explicitly given array takes
        precedence. At least one of internal_energies and
        electronic_structures is therefore required. Temperature
        dependence of the electronic system is supported through
        electronic_structures.
    electronic_structures : Sequence[ElectronicStates], optional
        Electronic states (eigenvalues, k-point weights, number of
        electrons) at each volume point, computed for the primitive
        cell. When given, the electronic free energies
        F_el(T, V) = internal_energies + fe(T) - fe(0) are
        computed internally within the fixed density-of-states (Mermin)
        approximation, which is intended for metals (see
        phonopy.qha.electron.ElectronFreeEnergy). The electronic
        entropies are obtained analytically and the heat capacities by a
        single numerical differentiation; both enter C_P and the
        Gruneisen parameters.
    mesh : float or array_like, optional
        Mesh numbers passed to Phonopy.run_mesh.
    pressure : float, optional
        Pressure in GPa added to the free energy as a pV term.
    eos : str, optional
        Equation of state used for fitting: 'vinet', 'murnaghan' or
        'birch_murnaghan'.
    lattice_fit_degree : int, optional
        Degree of the polynomials fitted to the axial ratios vs V.
    verbose : bool, optional
        Print fitted EOS parameters at each temperature.

    Returns
    -------
    QHAResult

    """
    temps_in, el = _validate_inputs(
        phonopys,
        internal_energies,
        temperatures,
        electronic_structures,
        eos,
        lattice_fit_degree,
    )
    # Phonon thermal properties are normalized per primitive cell, so the
    # volumes (and the input internal energies) refer to the primitive cell.
    volumes = np.array([ph.primitive.volume for ph in phonopys], dtype="double")
    lattice_lengths = np.array(
        [np.linalg.norm(ph.unitcell.cell, axis=1) for ph in phonopys], dtype="double"
    )

    fe_phonon, entropy, cv = compute_thermal_properties(
        phonopys, temps_in, mesh, verbose
    )

    units = get_physical_units()
    fe_phonon_ev = fe_phonon / units.EvTokJmol
    if electronic_structures is not None:
        fe_el_rel, s_el = compute_electronic_contributions_from_states(
            electronic_structures, temps_in
        )
        el = el + fe_el_rel
        cv_el = temps_in[:, None] * np.gradient(s_el, temps_in, axis=0, edge_order=2)
        to_j_mol = units.EvTokJmol * 1000
        entropy = entropy + s_el * to_j_mol
        cv = cv + cv_el * to_j_mol
    if pressure is not None:
        el = el + volumes * pressure / units.EVAngstromToGPa

    kept, eos_parameters, helmholtz_volume = _fit_eos_at_temperatures(
        volumes, fe_phonon_ev, el, temps_in, eos, verbose
    )

    temps = temps_in[kept]
    cv_kept = cv[kept]
    entropy_kept = entropy[kept]
    equilibrium_volumes = np.array(eos_parameters[:, 3])
    gibbs_free_energies = np.array(eos_parameters[:, 0])
    bulk_moduli = np.array(eos_parameters[:, 1] * units.EVAngstromToGPa)

    thermal_expansion = compute_volumetric_thermal_expansion(temps, equilibrium_volumes)
    gruneisen_parameters = compute_gruneisen_parameters(
        volumes, equilibrium_volumes, bulk_moduli, thermal_expansion, cv_kept
    )
    heat_capacity_P = _make_heat_capacity_data(
        temps, volumes, equilibrium_volumes, cv_kept, entropy_kept
    )

    n = len(temps) - 1
    lattice: QHALatticeData | None = None
    if _is_triclinic_or_monoclinic(phonopys):
        warnings.warn(
            "Lattice parameter fitting was skipped: cell angles of "
            "triclinic and monoclinic crystals may depend on volume.",
            UserWarning,
            stacklevel=2,
        )
    else:
        lattice = _make_lattice_data(
            volumes, lattice_lengths, lattice_fit_degree, temps, equilibrium_volumes, n
        )

    return QHAResult(
        temperatures=temps[:n],
        volumes=volumes,
        helmholtz_volume=helmholtz_volume[:n],
        eos_name=eos,
        eos_parameters=eos_parameters[:n],
        equilibrium_volumes=equilibrium_volumes[:n],
        gibbs_free_energies=gibbs_free_energies[:n],
        bulk_moduli=bulk_moduli[:n],
        thermal_expansion=thermal_expansion,
        gruneisen_parameters=gruneisen_parameters,
        heat_capacity_P=heat_capacity_P,
        lattice=lattice,
    )


def _validate_inputs(
    phonopys: Sequence[Phonopy],
    internal_energies: Sequence[float] | NDArray[np.double] | None,
    temperatures: Sequence[float] | NDArray[np.double],
    electronic_structures: Sequence[ElectronicStates] | None,
    eos: str,
    lattice_fit_degree: int,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Validate run_qha inputs and return them as arrays.

    Returns (temperatures, internal_energies).

    """
    temps_in = np.array(temperatures, dtype="double")
    if temps_in.ndim != 1 or len(temps_in) < 3:
        raise ValueError("temperatures must be a 1D array with at least 3 points.")
    if not (np.diff(temps_in) > 0).all():
        raise ValueError("temperatures must be in strictly ascending order.")
    nvol = len(phonopys)
    if nvol < 5:
        raise ValueError(
            "At least 5 volume points (Phonopy instances) are needed for QHA."
        )
    if nvol < lattice_fit_degree + 1:
        raise ValueError(
            f"At least {lattice_fit_degree + 1} volume points are needed for "
            f"lattice parameter fitting with polynomials of degree "
            f"{lattice_fit_degree}."
        )
    if eos not in ("vinet", "murnaghan", "birch_murnaghan"):
        raise ValueError(
            f"eos must be 'vinet', 'murnaghan' or 'birch_murnaghan', not {eos!r}."
        )
    if internal_energies is None:
        if electronic_structures is None:
            raise ValueError(
                "internal_energies can be omitted only when "
                "electronic_structures are given."
            )
        if any(
            electronic_states.internal_energy is None
            for electronic_states in electronic_structures
        ):
            raise ValueError(
                "internal_energies can be omitted only when all "
                "electronic_structures carry internal_energy."
            )
        el = np.array(
            [
                electronic_states.internal_energy
                for electronic_states in electronic_structures
            ],
            dtype="double",
        )
    else:
        el = np.array(internal_energies, dtype="double")
    if el.ndim != 1 or len(el) != nvol:
        raise ValueError(
            "internal_energies must be a 1D array of static energies "
            "with one value per Phonopy instance."
        )
    if electronic_structures is not None:
        if len(electronic_structures) != nvol:
            raise ValueError(
                "electronic_structures must have one entry per Phonopy instance."
            )
        for i, (ph, electronic_states) in enumerate(
            zip(phonopys, electronic_structures, strict=True)
        ):
            if electronic_states.volume is None:
                continue
            volume = ph.primitive.volume
            if abs(electronic_states.volume - volume) > 1e-4 * volume:
                raise ValueError(
                    f"Volume of electronic_structures[{i}] "
                    f"({electronic_states.volume}) does not match the "
                    f"primitive cell volume of phonopys[{i}] ({volume})."
                )
    for i, ph in enumerate(phonopys):
        if ph.force_constants is None:
            raise RuntimeError(f"Force constants are not set in phonopys[{i}].")
    return temps_in, el


def _is_triclinic_or_monoclinic(phonopys: Sequence[Phonopy]) -> bool:
    """Return True when any input crystal is triclinic or monoclinic.

    Space-group numbers 1-15 are triclinic and monoclinic, where the cell
    angles are free parameters that may depend on volume. The space-group
    number is obtained via the hall number, which both the ordinary and
    the magnetic spglib datasets provide. When symmetry search is
    disabled (no hall number available), False is returned and the
    k-constancy check of LatticeParametersFit remains the only guard.

    """
    for ph in phonopys:
        dataset = ph.symmetry.dataset
        if isinstance(dataset, NosymDataset):
            continue
        spg_type = spglib.get_spacegroup_type(dataset.hall_number)
        if spg_type is None:
            raise RuntimeError(
                "Space group type could not be determined from hall_number."
            )
        if spg_type.number <= 15:
            return True
    return False


def _fit_eos_at_temperatures(
    volumes: NDArray[np.double],
    fe_phonon_ev: NDArray[np.double],
    el: NDArray[np.double],
    temperatures: NDArray[np.double],
    eos: str,
    verbose: bool,
) -> tuple[list[int], NDArray[np.double], NDArray[np.double]]:
    """Fit the total free energy to the EOS at each temperature.

    Temperature points where the fitting fails are skipped with a warning.
    Returns (kept temperature indices, EOS parameters with shape
    (num_kept, 4), total free energies with shape (num_kept, volumes)).

    """
    units = get_physical_units()
    eos_func = get_eos(eos)

    if verbose:
        print("# EOS fitting")
        print(("#%11s" + "%14s" * 4) % ("T", "E_0", "B_0", "B'_0", "V_0"))

    kept: list[int] = []
    parameter_list: list[NDArray[np.double]] = []
    free_energy_list: list[NDArray[np.double]] = []
    for i in range(len(temperatures)):
        el_i = el if el.ndim == 1 else el[i]
        fe = fe_phonon_ev[i] + el_i
        try:
            ep = fit_to_eos(volumes, fe, eos_func)
        except RuntimeError:
            warnings.warn(
                f"EOS fitting failed at T={temperatures[i]:.1f} K; this "
                "temperature point is skipped.",
                UserWarning,
                stacklevel=2,
            )
            continue
        kept.append(i)
        parameter_list.append(ep)
        free_energy_list.append(fe)
        if verbose:
            print(
                ("%14.6f" * 5)
                % (
                    temperatures[i],
                    ep[0],
                    ep[1] * units.EVAngstromToGPa,
                    ep[2],
                    ep[3],
                )
            )

    if len(kept) < 3:
        raise RuntimeError("EOS fitting succeeded at fewer than 3 temperature points.")

    return (
        kept,
        np.array(parameter_list, dtype="double"),
        np.array(free_energy_list, dtype="double"),
    )


def _make_heat_capacity_data(
    temperatures: NDArray[np.double],
    volumes: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    cv: NDArray[np.double],
    entropy: NDArray[np.double],
) -> QHACpPolyfitData:
    """Compute C_P by polynomial fits and pack it into QHACpPolyfitData."""
    polyfit_arrays = compute_heat_capacity_p_polyfit(
        temperatures, volumes, equilibrium_volumes, cv, entropy
    )
    return QHACpPolyfitData(
        heat_capacities=polyfit_arrays.cp,
        dsdv=polyfit_arrays.dsdv,
        volume_cv_parameters=np.array(polyfit_arrays.volume_cv_parameters),
        volume_entropy_parameters=np.array(polyfit_arrays.volume_entropy_parameters),
        volume_cv=np.array(
            [np.array([volumes, cv[j]]).T for j in range(1, len(temperatures) - 1)]
        ),
        volume_entropy=np.array(
            [np.array([volumes, entropy[j]]).T for j in range(1, len(temperatures) - 1)]
        ),
    )


def _make_lattice_data(
    volumes: NDArray[np.double],
    lattice_lengths: NDArray[np.double],
    degree: int,
    temperatures: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    n: int,
) -> QHALatticeData | None:
    """Fit lattice parameters vs volume and evaluate at V_0(T).

    Returns None with a warning when the input cells are not consistent
    with volume-independent cell angles.

    """
    try:
        lattice_fit = LatticeParametersFit(volumes, lattice_lengths, degree=degree)
    except RuntimeError as exc:
        warnings.warn(
            f"Lattice parameter fitting was skipped: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None
    lattice_parameters = lattice_fit.evaluate(equilibrium_volumes)
    axial_thermal_expansions = compute_axial_thermal_expansion(
        temperatures, lattice_parameters
    )
    return QHALatticeData(
        lattice_parameters=lattice_parameters[:n],
        axial_thermal_expansions=axial_thermal_expansions,
        k=lattice_fit.k,
        ratio_coefficients=lattice_fit.ratio_coefficients,
    )
