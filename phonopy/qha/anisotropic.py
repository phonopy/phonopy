"""Anisotropic quasi-harmonic approximation over lattice degrees of freedom.

Unlike the volume-path QHA in phonopy.qha.qha, which fits a 1D equation of
state F(V), the anisotropic method fits the Helmholtz free energy directly
over the independent lattice-vector lengths and minimizes it per
temperature, giving axis-resolved thermal expansion. The free lattice
degrees of freedom are 1 (cubic), 2 (hexagonal, tetragonal, rhombohedral;
a and c) or 3 (orthorhombic; a, b and c). Cell angles are held fixed;
monoclinic and triclinic crystals, whose angles are additional degrees of
freedom, are out of scope for now.

"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from phonopy.physical_units import get_physical_units
from phonopy.qha.calc import (
    compute_volumetric_thermal_expansion,
    generate_total_degree_exponents,
    polynomial_design_matrix,
)
from phonopy.qha.lattice import compute_axial_thermal_expansion
from phonopy.qha.thermal import (
    compute_electronic_contributions_from_states,
    compute_thermal_properties,
    freeze_ndarray_fields,
)

if TYPE_CHECKING:
    from phonopy.api_phonopy import Phonopy
    from phonopy.qha.electron import ElectronicStates


class FreeEnergySurfaceFit:
    """Least-squares polynomial fit of a free energy over lattice DOF.

    The free energy F is fitted as a total-degree multivariate polynomial
    of the free lattice-vector lengths x (one component per independent
    lattice degree of freedom, 1 to 3). The fit variables are
    non-dimensionalized as u = (x - center) / scale, where center is the
    mean and scale the half-range of the sample points along each
    dimension, so the least-squares design matrix is well conditioned. The
    polynomial minimum, taken as the equilibrium lattice parameters at a
    given temperature, is located with scipy from the sample centroid.

    A total degree of 2 already carries the cross terms (e.g. a * c) that
    encode the anisotropic coupling; higher degrees capture the anharmonic
    curvature of the surface.

    """

    def __init__(
        self,
        points: Sequence[Sequence[float]] | NDArray[np.double],
        values: Sequence[float] | NDArray[np.double],
        degree: int = 3,
    ) -> None:
        """Init method.

        Parameters
        ----------
        points : array_like
            Free lattice-vector lengths at each sample point in angstrom.
            shape=(n_points, ndim)
        values : array_like
            Free energy at each sample point in eV. shape=(n_points,)
        degree : int, optional
            Total degree of the fitted polynomial.

        """
        self._points = np.array(points, dtype="double")
        self._values = np.array(values, dtype="double")
        if self._points.ndim != 2:
            raise ValueError("points must have shape (n_points, ndim).")
        if self._values.shape != (self._points.shape[0],):
            raise ValueError("values must have shape (n_points,).")
        self._ndim = self._points.shape[1]
        if not 1 <= self._ndim <= 3:
            raise ValueError("The number of lattice DOF (ndim) must be 1, 2 or 3.")
        self._degree = degree

        self._center = self._points.mean(axis=0)
        half_range = 0.5 * (self._points.max(axis=0) - self._points.min(axis=0))
        if not (half_range > 0).all():
            raise ValueError(
                "Sample points must span a non-zero range along every "
                "lattice degree of freedom."
            )
        self._scale = half_range

        self._exponents = generate_total_degree_exponents(self._ndim, degree)
        n_terms = self._exponents.shape[0]
        if self._points.shape[0] < n_terms:
            raise RuntimeError(
                f"At least {n_terms} sample points are needed to fit a "
                f"total-degree {degree} polynomial in {self._ndim} variables, "
                f"but {self._points.shape[0]} were given."
            )

        design = polynomial_design_matrix(self._scaled(self._points), self._exponents)
        coefficients, _, rank, _ = np.linalg.lstsq(design, self._values, rcond=None)
        self._coefficients = coefficients
        self._n_terms = n_terms
        self._rank = int(rank)
        self._rms_residual = float(
            np.sqrt(np.mean((design @ coefficients - self._values) ** 2))
        )
        # Set by minimize(); None until it has run.
        self._minimize_converged: bool | None = None
        self._minimum_extrapolated: bool | None = None

    def _scaled(self, points: NDArray[np.double]) -> NDArray[np.double]:
        """Non-dimensionalize points as (x - center) / scale."""
        return (points - self._center) / self._scale

    @property
    def ndim(self) -> int:
        """Return the number of lattice degrees of freedom."""
        return self._ndim

    @property
    def degree(self) -> int:
        """Return the total degree of the fitted polynomial."""
        return self._degree

    @property
    def coefficients(self) -> NDArray[np.double]:
        """Return the fitted polynomial coefficients in scaled variables.

        Aligned with the rows of `exponents`. shape=(n_terms,)

        """
        return self._coefficients

    @property
    def exponents(self) -> NDArray[np.int64]:
        """Return the monomial exponent tuples, shape (n_terms, ndim)."""
        return self._exponents

    @property
    def n_terms(self) -> int:
        """Return the number of polynomial terms C(ndim + degree, degree)."""
        return self._n_terms

    @property
    def rank(self) -> int:
        """Return the rank of the least-squares design matrix.

        Equal to n_terms for a well-posed fit. A smaller value means the
        sample points do not constrain every polynomial term (rank
        deficient), so the fitted surface is under-determined.

        """
        return self._rank

    @property
    def is_rank_deficient(self) -> bool:
        """Return whether the design matrix rank is below n_terms."""
        return self._rank < self._n_terms

    @property
    def rms_residual(self) -> float:
        """Return the RMS residual of the fit over the sample points in eV."""
        return self._rms_residual

    @property
    def minimize_converged(self) -> bool | None:
        """Return whether the last minimize() converged.

        None until minimize() has been called.

        """
        return self._minimize_converged

    @property
    def minimum_extrapolated(self) -> bool | None:
        """Return whether the last minimize() left the sampled box.

        None until minimize() has been called.

        """
        return self._minimum_extrapolated

    def evaluate(
        self, points: Sequence[Sequence[float]] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the fitted free energy at points in eV.

        Parameters
        ----------
        points : array_like
            Lattice-vector lengths in angstrom. shape=(n_points, ndim)

        Returns
        -------
        ndarray
            Fitted free energy in eV. shape=(n_points,)

        """
        pts = np.array(points, dtype="double")
        design = polynomial_design_matrix(self._scaled(pts), self._exponents)
        return design @ self._coefficients

    def gradient(
        self, points: Sequence[Sequence[float]] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the gradient dF/dx at points in eV/angstrom.

        The derivative of each monomial is obtained by lowering its
        exponent along the differentiated dimension and multiplying by the
        original exponent, then rescaling by 1 / scale for the chain rule
        from the non-dimensionalized variables.

        Parameters
        ----------
        points : array_like
            Lattice-vector lengths in angstrom. shape=(n_points, ndim)

        Returns
        -------
        ndarray
            Gradient of the fitted free energy. shape=(n_points, ndim)

        """
        pts = np.array(points, dtype="double")
        u = self._scaled(pts)
        gradient = np.zeros((pts.shape[0], self._ndim), dtype="double")
        for j in range(self._ndim):
            factor = self._exponents[:, j].astype("double")
            reduced = self._exponents.copy()
            reduced[:, j] = np.maximum(self._exponents[:, j] - 1, 0)
            design_dj = polynomial_design_matrix(u, reduced) * factor[None, :]
            gradient[:, j] = (design_dj @ self._coefficients) / self._scale[j]
        return gradient

    def minimize(
        self, x0: Sequence[float] | NDArray[np.double] | None = None
    ) -> NDArray[np.double]:
        """Return the lattice DOF x that minimize the fitted free energy.

        The minimization starts from x0 (the sample centroid by default)
        and uses the analytic gradient. A warning is issued when the
        located minimum lies outside the sampled box, i.e. the result is
        an extrapolation of the fit.

        Parameters
        ----------
        x0 : array_like, optional
            Initial guess for the lattice DOF in angstrom. shape=(ndim,)

        Returns
        -------
        ndarray
            Minimizing lattice DOF in angstrom. shape=(ndim,)

        """
        try:
            from scipy.optimize import minimize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("You need to install python-scipy.") from exc

        start = self._center if x0 is None else np.array(x0, dtype="double")

        def fun(x: NDArray[np.double]) -> float:
            return float(self.evaluate(x[None, :])[0])

        def jac(x: NDArray[np.double]) -> NDArray[np.double]:
            return self.gradient(x[None, :])[0]

        result = minimize(fun, start, jac=jac, method="BFGS")
        self._minimize_converged = bool(result.success)
        if not result.success:
            warnings.warn(
                f"Free energy surface minimization did not converge: {result.message}",
                UserWarning,
                stacklevel=2,
            )
        x_min = np.array(result.x, dtype="double")
        low = self._points.min(axis=0)
        high = self._points.max(axis=0)
        extrapolated = bool((x_min < low).any() or (x_min > high).any())
        self._minimum_extrapolated = extrapolated
        if extrapolated:
            warnings.warn(
                "The free energy minimum lies outside the sampled lattice "
                "range; the equilibrium lattice parameters are extrapolated.",
                UserWarning,
                stacklevel=2,
            )
        return x_min


@dataclasses.dataclass(frozen=True)
class AnisotropicQHAResult:
    """Immutable results of an anisotropic quasi-harmonic calculation.

    Temperature-indexed arrays have the same length N, which is one less
    than the number of input temperature points (the extra point is
    consumed by the finite differences of the thermal expansions).
    Quantities computed by central differences (thermal_expansion,
    axial_thermal_expansions) carry a leading zero. Energies and volumes
    refer to the primitive cell, consistently with the phonon thermal
    properties, while the lattice parameters are the unit-cell
    lattice-vector lengths.

    Attributes
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(N,)
    lattice_lengths : ndarray
        Unit-cell lattice-vector lengths (a, b, c) of the input sample
        cells in angstrom. shape=(n_points, 3)
    free_lattice_indices : ndarray
        Column indices of lattice_lengths that are the independent free
        lattice degrees of freedom (e.g. [0, 2] for hexagonal a and c).
        shape=(ndim,)
    surface_degree : int
        Total degree of the polynomial fitted to F over the free lattice
        DOF at each temperature.
    helmholtz_lattice : ndarray
        Total free energies (electronic + phonon [+ pV]) at temperatures
        and input sample cells in eV. shape=(N, n_points)
    equilibrium_lattice_parameters : ndarray
        Equilibrium lattice-vector lengths (a, b, c) at temperatures in
        angstrom, from the per-temperature surface minima. shape=(N, 3)
    equilibrium_volumes : ndarray
        Primitive cell volumes at the equilibrium lattice parameters in
        angstrom^3. shape=(N,)
    gibbs_free_energies : ndarray
        Minimized total free energies at temperatures in eV (Helmholtz, or
        Gibbs when a pressure is given). shape=(N,)
    thermal_expansion : ndarray
        Volumetric thermal expansion coefficients beta at temperatures in
        1/K with a leading zero. shape=(N,)
    axial_thermal_expansions : ndarray
        Linear thermal expansion coefficients (alpha_a, alpha_b, alpha_c)
        at temperatures in 1/K with a leading row of zeros. shape=(N, 3)
    surface_fit_rms : ndarray
        RMS residual of the free-energy surface polynomial fit at each
        temperature in eV; a fit-quality diagnostic. shape=(N,)
    surface_fit_rank : int
        Rank of the least-squares design matrix. Constant across
        temperatures, because the sample points and the monomial basis do
        not depend on temperature. Equal to surface_n_terms for a
        well-posed fit; a smaller value flags a rank-deficient
        (under-determined) fit.
    surface_n_terms : int
        Number of polynomial terms, C(ndim + surface_degree,
        surface_degree). The fit is rank deficient when surface_fit_rank
        is below this value.
    minimum_extrapolated : ndarray
        Per-temperature boolean flag, True when the located free-energy
        minimum lies outside the sampled lattice box, i.e. the equilibrium
        lattice parameters are extrapolated. shape=(N,)

    """

    temperatures: NDArray[np.double]
    lattice_lengths: NDArray[np.double]
    free_lattice_indices: NDArray[np.int64]
    surface_degree: int
    helmholtz_lattice: NDArray[np.double]
    equilibrium_lattice_parameters: NDArray[np.double]
    equilibrium_volumes: NDArray[np.double]
    gibbs_free_energies: NDArray[np.double]
    thermal_expansion: NDArray[np.double]
    axial_thermal_expansions: NDArray[np.double]
    surface_fit_rms: NDArray[np.double]
    surface_fit_rank: int
    surface_n_terms: int
    minimum_extrapolated: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Make ndarray fields read-only."""
        freeze_ndarray_fields(self)


def run_anisotropic_qha(
    phonopys: Sequence[Phonopy],
    temperatures: Sequence[float] | NDArray[np.double],
    internal_energies: Sequence[float] | NDArray[np.double] | None = None,
    electronic_structures: Sequence[ElectronicStates] | None = None,
    mesh: float | Sequence[int] | NDArray[np.int64] = 100.0,
    pressure: float | None = None,
    surface_degree: int = 3,
    verbose: bool = False,
) -> AnisotropicQHAResult:
    """Run an anisotropic quasi-harmonic approximation calculation.

    For each Phonopy instance (one per lattice grid point, with force
    constants set), mesh sampling and thermal properties are computed
    internally on the given temperature grid. The independent free lattice
    degrees of freedom (1 to 3) are detected from which lattice-vector
    lengths vary across the input cells, with symmetry-tied lengths (e.g.
    a and b for hexagonal cells) counted once. At each temperature the
    total free energy F(x; T) over the free lattice DOF x is fitted to a
    total-degree polynomial and minimized, giving the equilibrium lattice
    parameters a(T), b(T), c(T) and, by central differences, the axial
    thermal expansions.

    Note that finite differences consume one temperature point: supply one
    more point than the temperature range of interest.

    Parameters
    ----------
    phonopys : Sequence[Phonopy]
        One Phonopy instance per lattice grid point with force constants
        set. Enough points are needed to fit the surface polynomial
        (C(ndim + surface_degree, surface_degree) at least). The grid need
        not be regular; scattered sample cells are accepted.
    temperatures : array_like
        Temperatures in K in strictly ascending order. shape=(temperatures,)
    internal_energies : array_like, optional
        Static internal energies U per lattice grid point in eV per
        primitive cell (electronic total energies or machine-learning
        potential energies), consistently with the phonon normalization.
        shape=(n_points,). When None, the internal energies carried by
        electronic_structures are used.
    electronic_structures : Sequence[ElectronicStates], optional
        Electronic states at each lattice grid point; when given the
        electronic free energies and entropies are added to the phonon
        contributions, as in run_qha.
    mesh : float or array_like, optional
        Mesh numbers passed to Phonopy.run_mesh.
    pressure : float, optional
        Pressure in GPa added to the free energy as a pV term, turning the
        minimized free energy into a Gibbs free energy.
    surface_degree : int, optional
        Total degree of the polynomial fitted to F over the free lattice
        DOF.
    verbose : bool, optional
        Print the equilibrium lattice parameters at each temperature.

    Returns
    -------
    AnisotropicQHAResult

    """
    temps_in, el = _validate_anisotropic_inputs(
        phonopys, internal_energies, temperatures, electronic_structures
    )
    lattice_lengths = np.array(
        [np.linalg.norm(ph.unitcell.cell, axis=1) for ph in phonopys], dtype="double"
    )
    # Phonon thermal properties are normalized per primitive cell, so the
    # volumes (and the input internal energies) refer to the primitive cell.
    volumes = np.array([ph.primitive.volume for ph in phonopys], dtype="double")

    free_indices, column_map, fixed_values = _detect_lattice_dof(lattice_lengths)
    ndim = len(free_indices)
    n_terms = generate_total_degree_exponents(ndim, surface_degree).shape[0]
    if len(phonopys) < n_terms:
        raise ValueError(
            f"At least {n_terms} lattice grid points are needed to fit a "
            f"total-degree {surface_degree} polynomial in {ndim} free lattice "
            f"DOF, but {len(phonopys)} were given."
        )
    free_points = lattice_lengths[:, free_indices]

    fe_phonon, _, _ = compute_thermal_properties(phonopys, temps_in, mesh, verbose)
    units = get_physical_units()
    el = _add_static_contributions(
        el,
        fe_phonon / units.EvTokJmol,
        electronic_structures,
        temps_in,
        volumes,
        pressure,
    )

    m = len(temps_in)
    n_points = len(phonopys)
    helmholtz_lattice = np.zeros((m, n_points), dtype="double")
    equilibrium_lattice_parameters = np.zeros((m, 3), dtype="double")
    gibbs_free_energies = np.zeros(m, dtype="double")
    surface_fit_rms = np.zeros(m, dtype="double")
    minimum_extrapolated = np.zeros(m, dtype=bool)
    surface_fit_rank = n_terms

    axis_labels = ("a", "b", "c")
    if verbose:
        print("# Anisotropic free energy surface fitting")
        free_axes = ", ".join(axis_labels[col] for col in free_indices)
        print(f"Free lattice DOF: {ndim} ({free_axes})")
        for col in range(3):
            if column_map[col] < 0:
                print(f"Fixed length {axis_labels[col]} = {fixed_values[col]:.6f} A")
        print(
            f"Sample cells: {n_points}, polynomial terms: {n_terms} "
            f"(total degree {surface_degree})"
        )
        for pos, col in enumerate(free_indices):
            lo = free_points[:, pos].min()
            hi = free_points[:, pos].max()
            print(f"Sampled range {axis_labels[col]}: [{lo:.6f}, {hi:.6f}] A")

    for i in range(m):
        fe = el[i]
        helmholtz_lattice[i] = fe
        fit = FreeEnergySurfaceFit(free_points, fe, degree=surface_degree)
        if i == 0:
            # The design matrix rank is temperature independent (only the
            # fitted values change), so it is inspected once.
            surface_fit_rank = fit.rank
            if fit.is_rank_deficient:
                warnings.warn(
                    f"The free energy surface fit is rank deficient "
                    f"(rank {fit.rank} < {fit.n_terms} terms): the sampled "
                    f"lattice cells do not constrain every polynomial term. "
                    f"Add or better spread the sample cells, or lower "
                    f"surface_degree.",
                    UserWarning,
                    stacklevel=2,
                )
            if verbose:
                status = "rank deficient" if fit.is_rank_deficient else "full rank"
                print(f"Design matrix rank: {fit.rank} / {fit.n_terms} ({status})")
        x_min = fit.minimize()
        surface_fit_rms[i] = fit.rms_residual
        minimum_extrapolated[i] = bool(fit.minimum_extrapolated)
        gibbs_free_energies[i] = float(fit.evaluate(x_min[None, :])[0])
        equilibrium_lattice_parameters[i] = _reconstruct_lattice_parameters(
            x_min, column_map, fixed_values
        )
        if verbose:
            a, b, c = equilibrium_lattice_parameters[i]
            flag = "  [extrapolated]" if minimum_extrapolated[i] else ""
            print(
                f"T = {temps_in[i]:8.2f} K  a = {a:.6f}  b = {b:.6f}  "
                f"c = {c:.6f} A  fit RMS = {fit.rms_residual:.3e} eV{flag}"
            )

    k = float((volumes / lattice_lengths.prod(axis=1)).mean())
    equilibrium_volumes = k * equilibrium_lattice_parameters.prod(axis=1)
    thermal_expansion = compute_volumetric_thermal_expansion(
        temps_in, equilibrium_volumes
    )
    axial_thermal_expansions = compute_axial_thermal_expansion(
        temps_in, equilibrium_lattice_parameters
    )

    n = m - 1
    return AnisotropicQHAResult(
        temperatures=temps_in[:n],
        lattice_lengths=lattice_lengths,
        free_lattice_indices=np.array(free_indices, dtype="int64"),
        surface_degree=surface_degree,
        helmholtz_lattice=helmholtz_lattice[:n],
        equilibrium_lattice_parameters=equilibrium_lattice_parameters[:n],
        equilibrium_volumes=equilibrium_volumes[:n],
        gibbs_free_energies=gibbs_free_energies[:n],
        thermal_expansion=thermal_expansion,
        axial_thermal_expansions=axial_thermal_expansions,
        surface_fit_rms=surface_fit_rms[:n],
        surface_fit_rank=surface_fit_rank,
        surface_n_terms=n_terms,
        minimum_extrapolated=minimum_extrapolated[:n],
    )


def _validate_anisotropic_inputs(
    phonopys: Sequence[Phonopy],
    internal_energies: Sequence[float] | NDArray[np.double] | None,
    temperatures: Sequence[float] | NDArray[np.double],
    electronic_structures: Sequence[ElectronicStates] | None,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Validate run_anisotropic_qha inputs and return them as arrays.

    Returns (temperatures, internal_energies).

    """
    temps_in = np.array(temperatures, dtype="double")
    if temps_in.ndim != 1 or len(temps_in) < 3:
        raise ValueError("temperatures must be a 1D array with at least 3 points.")
    if not (np.diff(temps_in) > 0).all():
        raise ValueError("temperatures must be in strictly ascending order.")
    n_points = len(phonopys)
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
    if el.ndim != 1 or len(el) != n_points:
        raise ValueError(
            "internal_energies must be a 1D array with one value per Phonopy instance."
        )
    if electronic_structures is not None and len(electronic_structures) != n_points:
        raise ValueError(
            "electronic_structures must have one entry per Phonopy instance."
        )
    for i, ph in enumerate(phonopys):
        if ph.force_constants is None:
            raise RuntimeError(f"Force constants are not set in phonopys[{i}].")
    return temps_in, el


def _add_static_contributions(
    el: NDArray[np.double],
    fe_phonon_ev: NDArray[np.double],
    electronic_structures: Sequence[ElectronicStates] | None,
    temperatures: NDArray[np.double],
    volumes: NDArray[np.double],
    pressure: float | None,
) -> NDArray[np.double]:
    """Assemble the total free energy F(T) at each sample cell in eV.

    Adds the phonon free energy, the relative electronic free energy (when
    electronic_structures are given) and the pV term (when a pressure is
    given) to the static internal energies. Returns an array of shape
    (temperatures, n_points).

    """
    total = fe_phonon_ev + el
    if electronic_structures is not None:
        fe_el_rel, _ = compute_electronic_contributions_from_states(
            electronic_structures, temperatures
        )
        total = total + fe_el_rel
    if pressure is not None:
        total = total + volumes * pressure / get_physical_units().EVAngstromToGPa
    return total


def _detect_lattice_dof(
    lattice_lengths: NDArray[np.double], tol: float = 1e-6
) -> tuple[list[int], NDArray[np.int64], NDArray[np.double]]:
    """Determine the independent free lattice-length DOF from the samples.

    Columns of lattice_lengths that are equal across all samples (e.g. a
    and b for hexagonal or tetragonal cells) are tied by symmetry and
    count as a single degree of freedom; a column that does not vary is a
    fixed dimension.

    Returns the representative column index of each varying group (the
    free DOF), a per-column map to the position of its free DOF in that
    list (or -1 when the column is fixed), and the mean value of each
    column (used to fill fixed columns when rebuilding a, b, c).

    """
    n_col = lattice_lengths.shape[1]
    group_of = np.full(n_col, -1, dtype="int64")
    n_groups = 0
    for col in range(n_col):
        if group_of[col] >= 0:
            continue
        group_of[col] = n_groups
        for other in range(col + 1, n_col):
            if group_of[other] < 0 and np.allclose(
                lattice_lengths[:, col], lattice_lengths[:, other], rtol=tol, atol=0.0
            ):
                group_of[other] = n_groups
        n_groups += 1

    free_indices: list[int] = []
    group_free_pos = np.full(n_groups, -1, dtype="int64")
    for group in range(n_groups):
        col = int(np.argmax(group_of == group))
        column = lattice_lengths[:, col]
        if column.max() - column.min() > tol * abs(column.mean()):
            group_free_pos[group] = len(free_indices)
            free_indices.append(col)

    if not free_indices:
        raise ValueError(
            "No lattice degree of freedom varies across the input cells; "
            "the anisotropic QHA needs cells sampled over the lattice "
            "parameters."
        )

    column_map = np.array(
        [group_free_pos[group_of[col]] for col in range(n_col)], dtype="int64"
    )
    fixed_values = lattice_lengths.mean(axis=0)
    return free_indices, column_map, fixed_values


def _reconstruct_lattice_parameters(
    x_min: NDArray[np.double],
    column_map: NDArray[np.int64],
    fixed_values: NDArray[np.double],
) -> NDArray[np.double]:
    """Rebuild (a, b, c) from the free DOF minimum and the fixed columns."""
    abc = np.array(fixed_values, dtype="double")
    for col in range(len(column_map)):
        if column_map[col] >= 0:
            abc[col] = x_min[column_map[col]]
    return abc
