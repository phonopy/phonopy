# SPDX-License-Identifier: BSD-3-Clause
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
from phonopy.qha.anisotropic_dataset import check_cells_are_one_crystal
from phonopy.qha.calc import (
    compute_volumetric_thermal_expansion,
    generate_total_degree_exponents,
    polynomial_design_matrix,
)
from phonopy.qha.lattice import LatticeGrid, compute_axial_thermal_expansion
from phonopy.qha.lattice_smoothing import (
    SMOOTHING_METHODS,
    LatticeSmoothingFit,
    SmoothingMethod,
    fit_lattice_parameter,
)
from phonopy.qha.thermal import (
    compute_electronic_contributions_from_states,
    compute_thermal_properties,
    freeze_ndarray_fields,
    primitive_cell_fractions,
)

if TYPE_CHECKING:
    from phonopy.api_phonopy import Phonopy
    from phonopy.qha.electron_states import ElectronicStates


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
            Free lattice-vector lengths of the conventional unit cell at
            each sample point in angstrom. shape=(n_points, ndim)
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
            Lattice-vector lengths of the conventional unit cell in
            angstrom. shape=(n_points, ndim)

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
            Lattice-vector lengths of the conventional unit cell in
            angstrom. shape=(n_points, ndim)

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
        self,
        x0: Sequence[float] | NDArray[np.double] | None = None,
        gtol: float = 1e-9,
        convergence_gtol: float = 1e-6,
    ) -> NDArray[np.double]:
        """Return the lattice DOF x that minimize the fitted free energy.

        The minimization starts from x0 (the sample centroid by default)
        and uses the analytic gradient. A warning is issued when the
        located minimum lies outside the sampled box, i.e. the result is
        an extrapolation of the fit.

        The default gtol is much tighter than the scipy default of 1e-5.
        The position error of the located minimum scales as
        gtol / curvature, so a soft lattice DOF stops far from the true
        minimum under a loose tolerance. Minima located independently at
        neighbouring temperatures then scatter by that amount, and the
        finite differences taken to obtain thermal expansion coefficients
        amplify the scatter. The fit is a polynomial with an analytic
        gradient, so a tight tolerance costs little.

        Convergence is judged on the gradient actually reached rather than
        on the scipy success flag. Under a tight gtol the line search
        routinely stops with "precision loss" after descending far below
        the requested tolerance, which is not a failure.

        Parameters
        ----------
        x0 : array_like, optional
            Initial guess for the lattice DOF in angstrom. shape=(ndim,)
        gtol : float, optional
            Gradient inf-norm at which BFGS terminates, in eV/angstrom.
        convergence_gtol : float, optional
            Gradient inf-norm in eV/angstrom below which the located point
            counts as a minimum regardless of the scipy success flag.

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

        result = minimize(fun, start, jac=jac, method="BFGS", options={"gtol": gtol})
        x_min = np.array(result.x, dtype="double")
        gradient_norm = float(np.abs(jac(x_min)).max())
        converged = bool(result.success) or gradient_norm < convergence_gtol
        self._minimize_converged = converged
        if not converged:
            warnings.warn(
                f"Free energy surface minimization did not converge: "
                f"{result.message} (gradient inf-norm {gradient_norm:.3e} eV/A)",
                UserWarning,
                stacklevel=2,
            )
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

    Temperature-indexed arrays have the same length N. With
    lattice_smoothing="none" the thermal expansions are central
    differences, which leave the highest input temperature without a
    value, and N is one less than the number of input temperature points;
    with a smoothing method the expansions are the analytic slope of the
    fitted model and N is the number of input points. Quantities computed
    by central differences (thermal_expansion,
    axial_thermal_expansions) carry a leading zero. Energies and volumes
    refer to the primitive cell, consistently with the phonon thermal
    properties, while the lattice parameters are the lattice-vector
    lengths of the conventional unit cell.

    Attributes
    ----------
    temperatures : ndarray
        Temperatures in K. shape=(N,)
    lattice_grid : LatticeGrid
        The input sample cells: their lattice-vector lengths and the
        primitive volume of each, from which the volume of any cell of the
        same shape follows.
        Which lengths are the free lattice DOF is the grid's own
        knowledge, in lattice_grid.free_axis_indices.
    polynomial_degree : int
        Total degree of the polynomial fitted to F over the free lattice
        DOF at each temperature.
    helmholtz_lattice : ndarray
        Total free energies (electronic + phonon [+ pV]) at temperatures
        and input sample cells in eV. shape=(N, n_points)
    equilibrium_lattice_parameters : ndarray
        Equilibrium lattice-vector lengths (a, b, c) of the conventional
        unit cell at temperatures in angstrom, from the per-temperature
        surface minima. shape=(N, 3)
    unsmoothed_lattice_parameters : ndarray
        The same lengths as the surface minima gave them, before any
        smoothing, in angstrom. Equal to equilibrium_lattice_parameters
        when nothing was smoothed, and carried so that the smoothing can be
        seen against what it was fitted to. shape=(N, 3)
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
        not depend on temperature. Equal to polynomial_n_terms for a
        well-posed fit; a smaller value flags a rank-deficient
        (under-determined) fit.
    polynomial_n_terms : int
        Number of polynomial terms, C(ndim + polynomial_degree,
        polynomial_degree). The fit is rank deficient when surface_fit_rank
        is below this value.
    minimum_extrapolated : ndarray
        Per-temperature boolean flag, True when the located free-energy
        minimum lies outside the sampled lattice box, i.e. the equilibrium
        lattice parameters are extrapolated. shape=(N,)
    mesh : float or array_like, optional
        Mesh numbers used for the phonon sampling, recorded as they were
        given. The axial thermal expansions are sensitive to this setting,
        so it is carried with the result and written into the output
        headers. None when the result was built without recording it.
    lattice_smoothing_fit : LatticeSmoothingFit, optional
        The fitted models of a(T), b(T), c(T) themselves, and None when the
        lattice parameters were not smoothed. It is the one record of the
        smoothing: whether there was any, how many Einstein terms it used,
        and what the *_at methods below evaluate between the temperatures of
        this result.
    with_electronic : bool
        Whether the electronic free energy F_el was included. Recorded for
        the same reason as mesh: it shifts the axial split substantially
        while leaving the volumetric expansion nearly unchanged.
    pressure : float, optional
        Pressure in GPa added as the pV term, or None when the minimized
        free energy is the Helmholtz free energy.

    """

    temperatures: NDArray[np.double]
    lattice_grid: LatticeGrid
    polynomial_degree: int
    helmholtz_lattice: NDArray[np.double]
    equilibrium_lattice_parameters: NDArray[np.double]
    unsmoothed_lattice_parameters: NDArray[np.double]
    equilibrium_volumes: NDArray[np.double]
    gibbs_free_energies: NDArray[np.double]
    thermal_expansion: NDArray[np.double]
    axial_thermal_expansions: NDArray[np.double]
    surface_fit_rms: NDArray[np.double]
    surface_fit_rank: int
    polynomial_n_terms: int
    minimum_extrapolated: NDArray[np.bool_]
    mesh: float | Sequence[int] | NDArray[np.int64] | None = None
    lattice_smoothing_fit: LatticeSmoothingFit | None = None
    with_electronic: bool = False
    pressure: float | None = None

    def __post_init__(self) -> None:
        """Make ndarray fields read-only."""
        freeze_ndarray_fields(self)

    @property
    def lattice_smoothing(self) -> SmoothingMethod:
        """Return the smoothing that was applied along temperature.

        "none" when the lattice parameters are the surface minima
        themselves, which is what carrying no fit means.

        """
        if self.lattice_smoothing_fit is None:
            return "none"
        return self.lattice_smoothing_fit.method

    def _require_smoothing_fit(self) -> LatticeSmoothingFit:
        """Return the lattice smoothing fit, or say why there is none."""
        if self.lattice_smoothing_fit is None:
            raise ValueError(
                "The lattice parameters were not smoothed, so this result has "
                "no model of them along temperature: they exist at "
                "result.temperatures and nowhere else. Run with "
                "lattice_smoothing='einstein' to obtain one."
            )
        return self.lattice_smoothing_fit

    def lattice_parameters_at(
        self, temperatures: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the conventional unit cell's (a, b, c) at temperatures in K.

        shape=(temperatures, 3), in angstrom. The temperatures need not be
        this result's, but must lie within their range: the fitted model
        interpolates and does not extrapolate.

        """
        fit = self._require_smoothing_fit()
        return self.lattice_grid.spread(fit.equilibrium_free_axis_lengths(temperatures))

    def axial_thermal_expansions_at(
        self, temperatures: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return (alpha_a, alpha_b, alpha_c) at the given temperatures in K.

        shape=(temperatures, 3), in 1/K, as the analytic slope of the fitted
        model over the model itself. Takes the same temperatures
        lattice_parameters_at does.

        """
        fit = self._require_smoothing_fit()
        slopes = self.lattice_grid.spread(
            fit.equilibrium_free_axis_slopes(temperatures)
        )
        return slopes / self.lattice_parameters_at(temperatures)

    def thermal_expansion_at(
        self, temperatures: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the volumetric thermal expansion beta at temperatures in K.

        shape=(temperatures,), in 1/K. V is the product of the three lengths,
        so beta is the sum of the axial terms.

        """
        return self.axial_thermal_expansions_at(temperatures).sum(axis=1)

    def equilibrium_volumes_at(
        self, temperatures: Sequence[float] | NDArray[np.double]
    ) -> NDArray[np.double]:
        """Return the primitive cell volume at the given temperatures in K.

        shape=(temperatures,), in angstrom^3. The volume is recomputed from
        the interpolated lengths rather than interpolated on its own, so that
        it stays the volume of the cell those lengths describe.

        """
        return self.lattice_grid.abc_to_primitive_volume(
            self.lattice_parameters_at(temperatures)
        )


def run_anisotropic_qha(
    phonopys: Sequence[Phonopy],
    temperatures: Sequence[float] | NDArray[np.double],
    internal_energies: Sequence[float] | NDArray[np.double] | None = None,
    electronic_structures: Sequence[ElectronicStates] | None = None,
    electronic_free_energies: (
        Sequence[Sequence[float]] | NDArray[np.double] | None
    ) = None,
    phonon_free_energies: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    mesh: float | Sequence[int] | NDArray[np.int64] = 200.0,
    pressure: float | None = None,
    polynomial_degree: int = 3,
    lattice_smoothing: SmoothingMethod | None = None,
    smoothing_terms: int = 2,
    verbose: bool = False,
    is_gamma_center: bool = False,
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
    parameters a(T), b(T), c(T). The axial thermal expansions are the
    temperature derivative of those, taken as central differences with
    lattice_smoothing="none" and as the analytic slope of the smoothed
    model otherwise.

    Note that with lattice_smoothing="none" the central differences consume
    one temperature point: supply one more point than the temperature range
    of interest.

    Parameters
    ----------
    phonopys : Sequence[Phonopy]
        One Phonopy instance per lattice grid point with force constants
        set. Enough points are needed to fit the surface polynomial, at
        least C(d + polynomial_degree, polynomial_degree) of them with d the
        number of free lattice DOF. The grid need not be regular;
        scattered sample cells are accepted.
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
    electronic_free_energies : array_like, optional
        Electronic free energies F_el(T) - F_el(0) in eV per primitive cell,
        already computed outside, with shape (temperatures, n_points). The
        counterpart of phonon_free_energies for the electronic term, and
        mutually exclusive with electronic_structures.

        The integration is what makes this worth having: on a dense mesh the
        linear tetrahedron method costs a minute or more per grid point, and
        computing it once outside lets it be parallelized over the grid,
        reused across runs, or replaced by another method entirely. The
        values must be anchored at T = 0 and normalized per primitive cell,
        consistently with internal_energies.
    phonon_free_energies : array_like, optional
        Vibrational free energies in eV per primitive cell, already computed
        outside, with shape (temperatures, n_points). Given these, the mesh
        sampling is skipped and ``mesh`` is unused; the Phonopy instances then
        supply only the cells and volumes, and their force constants are
        neither required nor read.

        This is the way in for methods whose force constants depend on
        temperature, such as SSCHA or TDEP: one force-constant set per grid
        point cannot represent them, so their free energy has to be computed
        per temperature and handed over. The values must be normalized per
        primitive cell, consistently with internal_energies.
    mesh : float or array_like, optional
        Mesh passed to Phonopy.run_mesh, 200 by default. This is denser
        than the 100 of run_qha, deliberately: the axial split is a
        difference of large Grueneisen components and needs the denser mesh,
        while the volumetric expansion, being their average, is converged at
        100. Unused when phonon_free_energies is given.

        A length measure is resolved against each grid point's own
        reciprocal lattice, so cells that differ enough in a lattice length
        receive different numbers of divisions. That is a step in F_phonon
        across the lattice grid, i.e. in the very quantity this function
        differentiates, and it falls on whichever pair of neighbouring grid
        points happens to straddle the rounding. Explicit numbers of
        divisions avoid it by sampling every grid point identically; pass
        is_gamma_center=True with them to keep the Gamma-centred grid a
        length would have given.
    is_gamma_center : bool, optional
        Generate a Gamma-centred mesh instead of the Monkhorst-Pack one.
        Ignored when mesh is a length, for which phonopy enforces a
        Gamma-centred mesh, so this only takes effect together with
        explicit numbers of divisions. Default is False, phonopy's own
        default, which shifts the grid by half a division.
    pressure : float, optional
        Pressure in GPa added to the free energy as a pV term, turning the
        minimized free energy into a Gibbs free energy.
    polynomial_degree : int, optional
        Total degree of the polynomial fitted to F over the free lattice
        DOF.
    lattice_smoothing : Literal["none", "einstein"] or None, optional
        Smooth the equilibrium lattice parameters along temperature before
        differentiating them. See
        phonopy.qha.lattice_smoothing.fit_lattice_parameter. None, the
        default, takes "einstein" when phonon_free_energies is given and
        "none" otherwise. A smoothed result carries the fits themselves in
        lattice_smoothing_fit, so a(T), b(T), c(T) and the expansions can be
        evaluated between the temperatures given here.

        The thermal expansions are a temperature derivative of a(T),
        b(T), c(T), so a scatter in those reaches them amplified: with
        "none" it is central differences of the minima themselves, and
        with a smoothing method it is the analytic slope of the fitted
        model. Free energies from a sampled method -- SSCHA, or any
        other route whose free energy is a Monte Carlo average -- carry
        such a scatter, and it falls on each temperature independently,
        since each is minimized on its own. Free energies from force
        constants do not, and "none" is right for them.
    smoothing_terms : int, optional
        Number of Einstein terms the smoothing fits, at least 2. Default
        is 2. Unused with lattice_smoothing="none".
    verbose : bool, optional
        Print the equilibrium lattice parameters at each temperature.

    Returns
    -------
    AnisotropicQHAResult

    """
    if lattice_smoothing is not None and lattice_smoothing not in SMOOTHING_METHODS:
        raise ValueError(
            f"lattice_smoothing must be one of {SMOOTHING_METHODS} or None, "
            f"not {lattice_smoothing!r}."
        )
    temps_in, static_energies = _validate_anisotropic_inputs(
        phonopys,
        internal_energies,
        temperatures,
        electronic_structures,
        phonon_free_energies,
        electronic_free_energies,
    )
    # Phonon thermal properties are normalized per primitive cell, so the
    # volumes the grid derives (and the input internal energies) refer to it.
    lattice_grid = LatticeGrid(
        np.array([ph.unitcell.cell for ph in phonopys], dtype="double"),
        phonopys[0].primitive_matrix,
    )
    volumes = lattice_grid.primitive_volumes
    free_axis_lengths = lattice_grid.free_axis_lengths
    n_terms = _n_polynomial_terms(
        free_axis_lengths.shape[1], polynomial_degree, len(phonopys)
    )

    total_free_energies = _total_free_energies(
        phonopys,
        temps_in,
        static_energies,
        volumes,
        phonon_free_energies,
        electronic_structures,
        electronic_free_energies,
        mesh,
        pressure,
        is_gamma_center,
        verbose,
    )

    if verbose:
        _print_polynomial_fit_setup(lattice_grid, polynomial_degree, n_terms)

    minima = _minimize_free_energy_surfaces(
        total_free_energies, lattice_grid, temps_in, polynomial_degree, verbose
    )
    if lattice_smoothing is None:
        # phonon_free_energies usually comes from a sampled method, and the
        # thermal expansions amplify the scatter it leaves.
        lattice_smoothing = "einstein" if phonon_free_energies is not None else "none"

    if lattice_smoothing == "none":
        equilibrium_lattice_parameters = minima.equilibrium_lattice_parameters
        axial_slopes = None
        smoothing_fit = None
    else:
        smoothing_fit = _fit_lattice_smoothing(
            lattice_smoothing,
            lattice_grid,
            temps_in,
            minima.equilibrium_lattice_parameters,
            smoothing_terms,
        )
        equilibrium_lattice_parameters = lattice_grid.spread(
            smoothing_fit.equilibrium_free_axis_lengths(temps_in)
        )
        axial_slopes = lattice_grid.spread(
            smoothing_fit.equilibrium_free_axis_slopes(temps_in)
        )

    equilibrium_volumes = lattice_grid.abc_to_primitive_volume(
        equilibrium_lattice_parameters
    )
    thermal_expansion, axial_thermal_expansions, n_returned = _thermal_expansions(
        temps_in, equilibrium_lattice_parameters, equilibrium_volumes, axial_slopes
    )
    return AnisotropicQHAResult(
        temperatures=temps_in[:n_returned],
        lattice_grid=lattice_grid,
        polynomial_degree=polynomial_degree,
        helmholtz_lattice=minima.helmholtz_lattice[:n_returned],
        equilibrium_lattice_parameters=equilibrium_lattice_parameters[:n_returned],
        unsmoothed_lattice_parameters=minima.equilibrium_lattice_parameters[
            :n_returned
        ],
        equilibrium_volumes=equilibrium_volumes[:n_returned],
        gibbs_free_energies=minima.gibbs_free_energies[:n_returned],
        thermal_expansion=thermal_expansion,
        axial_thermal_expansions=axial_thermal_expansions,
        surface_fit_rms=minima.surface_fit_rms[:n_returned],
        surface_fit_rank=minima.surface_fit_rank,
        polynomial_n_terms=n_terms,
        minimum_extrapolated=minima.minimum_extrapolated[:n_returned],
        mesh=mesh,
        lattice_smoothing_fit=smoothing_fit,
        with_electronic=(
            electronic_structures is not None or electronic_free_energies is not None
        ),
        pressure=pressure,
    )


def _n_polynomial_terms(n_free_dof: int, polynomial_degree: int, n_points: int) -> int:
    """Return the number of surface polynomial terms over the free lattice DOF.

    Raises ValueError when the input cells cannot determine that many terms.

    """
    n_terms = generate_total_degree_exponents(n_free_dof, polynomial_degree).shape[0]
    if n_points < n_terms:
        raise ValueError(
            f"At least {n_terms} lattice grid points are needed to fit a "
            f"total-degree {polynomial_degree} polynomial in {n_free_dof} free lattice "
            f"DOF, but {n_points} were given."
        )
    return n_terms


def _total_free_energies(
    phonopys: Sequence[Phonopy],
    temperatures: NDArray[np.double],
    static_energies: NDArray[np.double],
    volumes: NDArray[np.double],
    phonon_free_energies: Sequence[Sequence[float]] | NDArray[np.double] | None,
    electronic_structures: Sequence[ElectronicStates] | None,
    electronic_free_energies: Sequence[Sequence[float]] | NDArray[np.double] | None,
    mesh: float | Sequence[int] | NDArray[np.int64],
    pressure: float | None,
    is_gamma_center: bool,
    verbose: bool,
) -> NDArray[np.double]:
    """Return the total free energy of each sample cell at each temperature.

    The phonon term is sampled on the mesh unless phonon_free_energies is
    given, and the electronic term is integrated over the electronic states
    unless electronic_free_energies is given. Those two computations are
    what this step costs; the tetrahedron integration of a dense mesh takes
    a minute or more per grid point. The static energies and, when a
    pressure is given, the pV term are added to them.

    Returns an array of shape (temperatures, n_points) in eV per primitive
    cell.

    """
    if phonon_free_energies is None:
        fe_phonon, _, _ = compute_thermal_properties(
            phonopys, temperatures, mesh, verbose, is_gamma_center=is_gamma_center
        )
        total = fe_phonon / get_physical_units().EvTokJmol
    else:
        total = np.array(phonon_free_energies, dtype="double")
    total = total + static_energies

    if electronic_free_energies is not None:
        total = total + np.array(electronic_free_energies, dtype="double")
    elif electronic_structures is not None:
        fe_electronic, _ = compute_electronic_contributions_from_states(
            electronic_structures, temperatures, primitive_volumes=volumes
        )
        total = total + fe_electronic

    if pressure is not None:
        total = total + volumes * pressure / get_physical_units().EVAngstromToGPa
    return total


def _print_polynomial_fit_setup(
    lattice_grid: LatticeGrid,
    polynomial_degree: int,
    n_terms: int,
) -> None:
    """Print the free lattice DOF and the sampled range of each.

    Parameters
    ----------
    lattice_grid : LatticeGrid
        The input sample cells.
    polynomial_degree : int
        Total degree of the polynomial fitted to F over the free DOF.
    n_terms : int
        Number of terms that polynomial has.

    """
    axis_labels = ("a", "b", "c")
    free_axis_indices = lattice_grid.free_axis_indices
    free_axis_lengths = lattice_grid.free_axis_lengths
    print("# Anisotropic free energy surface fitting")
    free_axes = ", ".join(axis_labels[col] for col in free_axis_indices)
    print(f"Free lattice DOF: {len(free_axis_indices)} ({free_axes})")
    print(
        f"Sample cells: {lattice_grid.n_points}, polynomial terms: {n_terms} "
        f"(total degree {polynomial_degree})"
    )
    for pos, col in enumerate(free_axis_indices):
        lo = free_axis_lengths[:, pos].min()
        hi = free_axis_lengths[:, pos].max()
        print(f"Sampled range {axis_labels[col]}: [{lo:.6f}, {hi:.6f}] A")


def _fit_lattice_smoothing(
    method: SmoothingMethod,
    lattice_grid: LatticeGrid,
    temperatures: NDArray[np.double],
    lattice_parameters: NDArray[np.double],
    n_terms: int,
) -> LatticeSmoothingFit:
    """Fit one lattice length per free lattice DOF along temperature.

    Only the representative length of each free DOF is fitted, so a length
    tied to another reads that fit through LatticeGrid.spread rather than a
    second fit of the same numbers.

    Parameters
    ----------
    method : Literal["einstein"]
        The smoothing to apply. A method that is named but not fitted here
        raises, rather than being answered with an Einstein fit.
    lattice_grid : LatticeGrid
        The input sample cells, which say which lengths are free.
    temperatures : ndarray
        Temperatures in K. shape=(temperatures,)
    lattice_parameters : ndarray
        The conventional unit cell's lengths (a, b, c) to fit, in
        angstrom. shape=(temperatures, 3)
    n_terms : int
        Number of Einstein terms in each fit.

    """
    if method != "einstein":
        raise ValueError(f"Lattice smoothing {method!r} is not implemented.")
    fits = tuple(
        fit_lattice_parameter(
            temperatures, lattice_parameters[:, column], n_terms=n_terms
        )
        for column in lattice_grid.free_axis_indices
    )
    return LatticeSmoothingFit(free_axis_fits=fits, method=method)


@dataclasses.dataclass(frozen=True)
class _FreeEnergySurfaceMinima:
    """The per-temperature free-energy surfaces and where they are minimized.

    Attributes
    ----------
    helmholtz_lattice : ndarray
        The fitted free energies at the input cells in eV.
        shape=(temperatures, n_points)
    equilibrium_lattice_parameters : ndarray
        Lattice-vector lengths (a, b, c) of the conventional unit cell at
        the surface minimum of each temperature in angstrom.
        shape=(temperatures, 3)
    gibbs_free_energies : ndarray
        The free energy at each minimum in eV. shape=(temperatures,)
    surface_fit_rms : ndarray
        RMS residual of the surface fit at each temperature in eV.
        shape=(temperatures,)
    minimum_extrapolated : ndarray
        True where the minimum lies outside the sampled lattice box.
        shape=(temperatures,)
    surface_fit_rank : int
        Rank of the least-squares design matrix, which does not depend on
        temperature.

    """

    helmholtz_lattice: NDArray[np.double]
    equilibrium_lattice_parameters: NDArray[np.double]
    gibbs_free_energies: NDArray[np.double]
    surface_fit_rms: NDArray[np.double]
    minimum_extrapolated: NDArray[np.bool_]
    surface_fit_rank: int


def _minimize_free_energy_surfaces(
    free_energies: NDArray[np.double],
    lattice_grid: LatticeGrid,
    temperatures: NDArray[np.double],
    polynomial_degree: int,
    verbose: bool,
) -> _FreeEnergySurfaceMinima:
    """Fit and minimize the free energy surface at each temperature.

    Parameters
    ----------
    free_energies : ndarray
        Total free energy of every input cell at every temperature in eV
        per primitive cell. shape=(temperatures, n_points)
    lattice_grid : LatticeGrid
        The input sample cells: the surface is fitted over their free
        lattice DOF, and its minimum is spread back onto a, b and c.
    temperatures : ndarray
        Temperatures in K. shape=(temperatures,)
    polynomial_degree : int
        Total degree of the polynomial fitted to F over the free DOF.
    verbose : bool
        Print the fit rank and the minimum found at each temperature.

    """
    free_axis_lengths = lattice_grid.free_axis_lengths
    n_temperatures, n_points = free_energies.shape
    helmholtz_lattice = np.zeros((n_temperatures, n_points), dtype="double")
    equilibrium_lattice_parameters = np.zeros((n_temperatures, 3), dtype="double")
    gibbs_free_energies = np.zeros(n_temperatures, dtype="double")
    surface_fit_rms = np.zeros(n_temperatures, dtype="double")
    minimum_extrapolated = np.zeros(n_temperatures, dtype=bool)
    surface_fit_rank = 0

    for i in range(n_temperatures):
        fe = free_energies[i]
        helmholtz_lattice[i] = fe
        fit = FreeEnergySurfaceFit(free_axis_lengths, fe, degree=polynomial_degree)
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
                    f"polynomial_degree.",
                    UserWarning,
                    stacklevel=3,
                )
            if verbose:
                status = "rank deficient" if fit.is_rank_deficient else "full rank"
                print(f"Design matrix rank: {fit.rank} / {fit.n_terms} ({status})")
        x_min = fit.minimize()
        surface_fit_rms[i] = fit.rms_residual
        minimum_extrapolated[i] = bool(fit.minimum_extrapolated)
        gibbs_free_energies[i] = float(fit.evaluate(x_min[None, :])[0])
        equilibrium_lattice_parameters[i] = lattice_grid.spread(x_min)
        if verbose:
            a, b, c = equilibrium_lattice_parameters[i]
            flag = "  [extrapolated]" if minimum_extrapolated[i] else ""
            print(
                f"T = {temperatures[i]:8.2f} K  a = {a:.6f}  b = {b:.6f}  "
                f"c = {c:.6f} A  fit RMS = {fit.rms_residual:.3e} eV{flag}"
            )

    return _FreeEnergySurfaceMinima(
        helmholtz_lattice=helmholtz_lattice,
        equilibrium_lattice_parameters=equilibrium_lattice_parameters,
        gibbs_free_energies=gibbs_free_energies,
        surface_fit_rms=surface_fit_rms,
        minimum_extrapolated=minimum_extrapolated,
        surface_fit_rank=surface_fit_rank,
    )


def _thermal_expansions(
    temperatures: NDArray[np.double],
    equilibrium_lattice_parameters: NDArray[np.double],
    equilibrium_volumes: NDArray[np.double],
    axial_slopes: NDArray[np.double] | None,
) -> tuple[NDArray[np.double], NDArray[np.double], int]:
    """Return beta, the axial expansions, and how many temperatures they cover.

    axial_slopes is None when the lattice parameters were not smoothed.

    """
    n_temperatures = len(temperatures)
    if axial_slopes is None:
        # The central differences leave the highest temperature without a
        # value, so it is not returned.
        thermal_expansion = compute_volumetric_thermal_expansion(
            temperatures, equilibrium_volumes
        )
        axial_thermal_expansions = compute_axial_thermal_expansion(
            temperatures, equilibrium_lattice_parameters
        )
        return thermal_expansion, axial_thermal_expansions, n_temperatures - 1

    # The smoothed lattice parameters come from a model that is
    # differentiable in closed form, so its slope is taken directly rather
    # than approximated by differences of it. V is the product of the three
    # lengths, so beta is the sum of the axial terms.
    axial_thermal_expansions = axial_slopes / equilibrium_lattice_parameters
    thermal_expansion = axial_thermal_expansions.sum(axis=1)
    return thermal_expansion, axial_thermal_expansions, n_temperatures


def _validate_anisotropic_inputs(
    phonopys: Sequence[Phonopy],
    internal_energies: Sequence[float] | NDArray[np.double] | None,
    temperatures: Sequence[float] | NDArray[np.double],
    electronic_structures: Sequence[ElectronicStates] | None,
    phonon_free_energies: Sequence[Sequence[float]] | NDArray[np.double] | None = None,
    electronic_free_energies: (
        Sequence[Sequence[float]] | NDArray[np.double] | None
    ) = None,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Validate run_anisotropic_qha inputs and return them as arrays.

    Returns (temperatures, internal_energies).

    """
    temps_in = np.array(temperatures, dtype="double")
    if temps_in.ndim != 1 or len(temps_in) < 3:
        raise ValueError("temperatures must be a 1D array with at least 3 points.")
    if not (np.diff(temps_in) > 0).all():
        raise ValueError("temperatures must be in strictly ascending order.")
    check_cells_are_one_crystal(
        [ph.primitive_matrix for ph in phonopys],
        [ph.supercell_matrix for ph in phonopys],
        [ph.unitcell.symbols for ph in phonopys],
    )
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
        static_energies = np.array(
            [
                electronic_states.internal_energy
                for electronic_states in electronic_structures
            ],
            dtype="double",
        ) * primitive_cell_fractions(
            electronic_structures,
            [ph.primitive.volume for ph in phonopys],
        )
    else:
        static_energies = np.array(internal_energies, dtype="double")
    if static_energies.ndim != 1 or len(static_energies) != n_points:
        raise ValueError(
            "internal_energies must be a 1D array with one value per Phonopy instance."
        )
    if electronic_structures is not None and len(electronic_structures) != n_points:
        raise ValueError(
            "electronic_structures must have one entry per Phonopy instance."
        )
    if electronic_structures is not None and electronic_free_energies is not None:
        raise ValueError(
            "electronic_structures and electronic_free_energies are two ways "
            "of giving the same term; give one or the other."
        )
    if electronic_free_energies is not None:
        fe_el = np.array(electronic_free_energies, dtype="double")
        if fe_el.shape != (len(temps_in), n_points):
            raise ValueError(
                f"electronic_free_energies must have shape "
                f"{(len(temps_in), n_points)} (temperatures, Phonopy "
                f"instances), but has {fe_el.shape}."
            )
    if phonon_free_energies is None:
        for i, ph in enumerate(phonopys):
            if ph.force_constants is None:
                raise RuntimeError(f"Force constants are not set in phonopys[{i}].")
    else:
        # The free energies replace the mesh sampling entirely, so the force
        # constants are not consulted and need not be set.
        fe = np.array(phonon_free_energies, dtype="double")
        if fe.shape != (len(temps_in), n_points):
            raise ValueError(
                f"phonon_free_energies must have shape "
                f"{(len(temps_in), n_points)} (temperatures, Phonopy "
                f"instances), but has {fe.shape}."
            )
    return temps_in, static_energies
