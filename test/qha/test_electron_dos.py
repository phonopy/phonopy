# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the tetrahedron electronic density of states."""

from __future__ import annotations

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.grid import BZGrid, get_ir_grid_points
from phonopy.qha.electron import ElectronicStates, compute_tetrahedron_dos
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry


def _states_on_grid(
    cell: PhonopyAtoms,
    mesh,
    n_bands: int = 4,
    n_spin: int = 1,
    spin_degeneracy: int | None = None,
    use_grg: bool = False,
) -> ElectronicStates:
    """Return states on a grid, with eigenvalues of a free-electron shape.

    The normalization of the density of states holds whatever the eigenvalues
    are, so they can be invented; what is under test is the mapping, the
    multiplicities and the tetrahedron weights.

    """
    bz_grid = BZGrid(
        mesh,
        lattice=cell.cell,
        symmetry_dataset=Symmetry(cell).dataset,
        use_grg=use_grg,
    )
    ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
    addresses = bz_grid.addresses[bz_grid.grg2bzg[ir_grid_points]]
    kpoints = addresses @ bz_grid.QDinv.T

    reciprocal = np.linalg.inv(cell.cell)
    cartesian = kpoints @ reciprocal.T
    squared = (cartesian**2).sum(axis=1)
    eigenvalues = np.empty((n_spin, len(kpoints), n_bands), dtype="double")
    for spin in range(n_spin):
        for band in range(n_bands):
            eigenvalues[spin, :, band] = squared * (band + 1) + band + 0.1 * spin

    return ElectronicStates(
        eigenvalues=eigenvalues,
        weights=ir_grid_weights / ir_grid_weights.sum(),
        n_electrons=float(n_bands),
        spin_degeneracy=spin_degeneracy,
        kpoints=kpoints,
        mesh=np.asarray(bz_grid.grid_matrix if use_grg else mesh),
        cell=cell,
    )


# The trapezoidal rule over a density of states does not converge smoothly:
# the derivative is discontinuous at every band edge, so refining the energy
# grid moves the residual around at the 1e-3 level rather than shrinking it.
# Measured here over a sixteenfold refinement, and consistent with the
# 32.017-32.024 against 32 that the real VASP grids give.
NORMALIZATION_TOLERANCE = 2e-3


def _integral(states: ElectronicStates, n_points: int = 4001) -> float:
    """Return the density of states integrated over the whole spectrum."""
    low = float(states.eigenvalues.min()) - 1.0
    high = float(states.eigenvalues.max()) + 1.0
    energies = np.linspace(low, high, n_points)
    return float(np.trapezoid(compute_tetrahedron_dos(states, energies), energies))


def test_dos_normalization(aln_cell: PhonopyAtoms):
    """Test that the density of states integrates to bands x spin degeneracy.

    This is the check that the k-point mapping, the multiplicities and the
    tetrahedron weights are all consistent with each other; a wrong mapping
    shows up here immediately.

    """
    states = _states_on_grid(aln_cell, [5, 5, 4], n_bands=4)
    assert _integral(states) == pytest.approx(8.0, rel=NORMALIZATION_TOLERANCE)


def test_dos_normalization_non_collinear(aln_cell: PhonopyAtoms):
    """Test the normalization when each state holds one electron.

    A non-collinear calculation has a spin axis of length 1, like a
    non-spin-polarized one, but its spinors hold one electron each.

    """
    states = _states_on_grid(aln_cell, [5, 5, 4], n_bands=4, spin_degeneracy=1)
    assert _integral(states) == pytest.approx(4.0, rel=NORMALIZATION_TOLERANCE)


def test_dos_normalization_spin_polarized(aln_cell: PhonopyAtoms):
    """Test that the two spin channels are summed.

    Their density of states is added rather than kept apart because the
    occupation depends on the energy and the chemical potential alone, so
    every integral that follows is against the sum.

    """
    states = _states_on_grid(aln_cell, [5, 5, 4], n_bands=4, n_spin=2)
    assert _integral(states) == pytest.approx(8.0, rel=NORMALIZATION_TOLERANCE)


def test_dos_normalization_generalized_regular_grid(ph_tio2: Phonopy):
    """Test the normalization on a grid with a non-diagonal generating matrix."""
    states = _states_on_grid(ph_tio2.primitive, 30.0, n_bands=3, use_grg=True)
    assert _integral(states) == pytest.approx(6.0, rel=NORMALIZATION_TOLERANCE)


def test_dos_is_non_negative(aln_cell: PhonopyAtoms):
    """Test that the density of states never comes out negative."""
    states = _states_on_grid(aln_cell, [5, 5, 4], n_bands=4)
    energies = np.linspace(
        float(states.eigenvalues.min()), float(states.eigenvalues.max()), 501
    )
    assert (compute_tetrahedron_dos(states, energies) >= 0.0).all()


def test_dos_does_not_depend_on_the_block_size(aln_cell: PhonopyAtoms):
    """Test that splitting the energies changes nothing but the memory.

    The integration weights are built as one (ir points, sampling points,
    bands) array, which reaches tens of gigabytes on a dense mesh with the
    fine energy grid the free energy needs. Sampling points are independent,
    so they are processed in blocks; this is the check that the split is
    exact rather than approximate.

    """
    states = _states_on_grid(aln_cell, [5, 5, 4], n_bands=4)
    energies = np.linspace(
        float(states.eigenvalues.min()), float(states.eigenvalues.max()), 401
    )

    whole = compute_tetrahedron_dos(states, energies, max_bytes=np.inf)
    for max_bytes in (1.0, 1e4, 1e5):
        np.testing.assert_allclose(
            compute_tetrahedron_dos(states, energies, max_bytes=max_bytes),
            whole,
            rtol=0.0,
            atol=0.0,
        )


def test_tetrahedron_dos_needs_the_grid(aln_cell: PhonopyAtoms):
    """Test that states without the grid fields refuse the tetrahedron."""
    states = _states_on_grid(aln_cell, [5, 5, 4])
    without = ElectronicStates(
        eigenvalues=states.eigenvalues,
        weights=states.weights,
        n_electrons=states.n_electrons,
    )
    with pytest.raises(ValueError, match="needs kpoints, mesh and cell"):
        compute_tetrahedron_dos(without, np.linspace(0.0, 1.0, 11))


def test_electronic_states_grid_fields_go_together(aln_cell: PhonopyAtoms):
    """Test that kpoints, mesh and cell are given together or not at all."""
    states = _states_on_grid(aln_cell, [5, 5, 4])
    with pytest.raises(ValueError, match="all three or none"):
        ElectronicStates(
            eigenvalues=states.eigenvalues,
            weights=states.weights,
            n_electrons=states.n_electrons,
            kpoints=states.kpoints,
        )
