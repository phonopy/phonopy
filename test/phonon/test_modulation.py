# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Modulation."""

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixNAC
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.structure.atoms import PhonopyAtoms

lattice_NaCl222 = [
    [0.000000000000000, 5.690301476175671, 5.690301476175671],
    [5.690301476175671, 0.000000000000000, 5.690301476175671],
    [5.690301476175671, 5.690301476175671, 0.000000000000000],
]
positions_NaCl222 = [
    [0.99273744, 0.00726256, 0.00726256],
    [0.49273744, 0.00726256, 0.00726256],
    [0.00726256, 0.49273744, 0.99273744],
    [0.50726256, 0.49273744, 0.99273744],
    [0.00726256, 0.99273744, 0.49273744],
    [0.50726256, 0.99273744, 0.49273744],
    [0.99273744, 0.50726256, 0.50726256],
    [0.49273744, 0.50726256, 0.50726256],
    [0.24550088, 0.25449912, 0.25449912],
    [0.74550088, 0.25449912, 0.25449912],
    [0.25449912, 0.74550088, 0.24550088],
    [0.75449912, 0.74550088, 0.24550088],
    [0.25449912, 0.24550088, 0.74550088],
    [0.75449912, 0.24550088, 0.74550088],
    [0.24550088, 0.75449912, 0.75449912],
    [0.74550088, 0.75449912, 0.75449912],
]
symbols_NaCl222 = ["Na"] * 8 + ["Cl"] * 8
cell_NaCl222 = PhonopyAtoms(
    cell=lattice_NaCl222, scaled_positions=positions_NaCl222, symbols=symbols_NaCl222
)


def test_modulation(ph_nacl: Phonopy, helper_methods):
    """Test to calculate modulation by NaCl."""
    ph = ph_nacl
    mod = ph.run_modulations([2, 2, 2], [[[0, 0.5, 0.5], 2, 2, 0]])
    cells = mod.modulated_supercells
    # _show(cells)
    helper_methods.compare_cells_with_order(cells[0], cell_NaCl222)


def test_modulation_properties(ph_nacl: Phonopy, helper_methods):
    """Test direct attribute access on the Modulation result object."""
    ph = ph_nacl
    ph.run_modulations([2, 2, 2], [[[0, 0.5, 0.5], 2, 2, 0], [[0, 0.5, 0.5], 3, 2, 0]])
    mod = ph.modulation
    assert mod is not None

    # Properties agree with the legacy getters.
    cells = mod.get_modulated_supercells()
    cells_prop = mod.modulated_supercells
    assert len(cells_prop) == 2
    for cell, cell_prop in zip(cells, cells_prop, strict=True):
        helper_methods.compare_cells_with_order(cell_prop, cell)
    u, supercell = mod.get_modulations_and_supercell()
    np.testing.assert_array_equal(mod.modulations, u)
    helper_methods.compare_cells_with_order(mod.supercell, supercell)
    assert mod.modulations.shape == (2, len(supercell), 3)

    # Combined cell carries the displacements of all modes summed.
    combined = mod.modulated_supercell
    positions = supercell.positions + mod.modulations.sum(axis=0).real
    scaled_positions = np.dot(positions, np.linalg.inv(supercell.cell))
    scaled_positions -= np.floor(scaled_positions)
    np.testing.assert_allclose(combined.scaled_positions, scaled_positions, atol=1e-12)

    # Per-mode data previously only available in the yaml output.
    assert mod.frequencies.shape == (2,)
    assert (mod.frequencies > 0).all()
    assert mod.eigenvectors.shape == (2, len(ph.primitive) * 3)


# Eigenvectors of Si at W, q=(1/4, 3/4, 1/2), rotated by delta_q=[1, 2, 3].
# Each row is one band, with the phase of each band fixed arbitrarily.
freqs_Si_W = [5.7905215, 5.7905215, 11.10314312, 11.10314312, 13.79304175, 13.79304175]
eigvecs_Si_W = [
    [
        0.14617063 - 0.29234125j,
        -0.29234125 - 0.14617063j,
        0.53510885,
        -0.32684746,
        -0.32684746j,
        -0.23930795 + 0.4786159j,
    ],
    [
        -0.14617063 + 0.29234125j,
        0.29234125 + 0.14617063j,
        0.53510885,
        -0.32684746,
        -0.32684746j,
        0.23930795 - 0.4786159j,
    ],
    [0.5, 0.5j, 0, -0.2236068 - 0.4472136j, -0.4472136 + 0.2236068j, 0],
    [0.5, 0.5j, 0, 0.2236068 + 0.4472136j, 0.4472136 - 0.2236068j, 0],
    [
        0.16921627 - 0.33843255j,
        -0.33843255 - 0.16921627j,
        0.46223211,
        0.37837909,
        0.37837909j,
        0.20671648 - 0.41343297j,
    ],
    [
        -0.16921627 + 0.33843255j,
        0.33843255 + 0.16921627j,
        0.46223211,
        0.37837909,
        0.37837909j,
        -0.20671648 + 0.41343297j,
    ],
]


def test_modulation_lifted_degeneracy(ph_si: Phonopy):
    """Test eigenvectors of degenerate modes rotated by delta_q.

    At W of Si, every pair of degenerate bands is split at first order
    along delta_q=[1, 2, 3], so the rotated eigenvectors are unique up to
    a phase factor of each band and do not depend on the eigensolver.

    """
    ph = ph_si
    q = [0.25, 0.75, 0.5]
    modes = [[q, band, 1, 0] for band in range(6)]
    mod = ph.run_modulations([1, 1, 1], modes, delta_q=[1, 2, 3])
    np.testing.assert_allclose(mod.frequencies, freqs_Si_W, atol=1e-6)
    for vec, vec_ref in zip(mod.eigenvectors, eigvecs_Si_W, strict=True):
        phase = np.vdot(vec_ref, vec)
        phase /= abs(phase)
        np.testing.assert_allclose(vec / phase, vec_ref, atol=1e-6)


@pytest.mark.parametrize("q", [[0.25, 0.75, 0.5], [0, 0.5, 0.5]])
def test_modulation_eigenvectors(ph_si: Phonopy, q: list[float]):
    """Test eigenvectors of Si without delta_q by rotation-invariant properties."""
    _assert_modulation_eigenvectors(ph_si, q)


def test_modulation_eigenvectors_nac_q_direction(ph_nacl: Phonopy):
    """Test eigenvectors of NaCl at Gamma with nac_q_direction."""
    _assert_modulation_eigenvectors(ph_nacl, [0, 0, 0], nac_q_direction=[1, 0, 0])


def _assert_modulation_eigenvectors(
    ph: Phonopy, q: list[float], nac_q_direction: list[float] | None = None
):
    """Assert eigenvectors of all bands at q by rotation-invariant properties.

    The eigenvectors of degenerate bands are determined only up to a
    unitary rotation within the degenerate subspace.  Each eigenvector
    has to satisfy D v = w v with unit norm, and the projector onto each
    degenerate subspace, sum_b v_b v_b^dagger, has to agree with that
    from numpy.linalg.eigh.

    """
    num_band = len(ph.primitive) * 3
    modes = [[q, band, 1, 0] for band in range(num_band)]
    mod = ph.run_modulations([1, 1, 1], modes, nac_q_direction=nac_q_direction)

    dm = ph.dynamical_matrix
    assert dm is not None
    if isinstance(dm, DynamicalMatrixNAC):
        dm.run(q, q_direction=nac_q_direction)
    else:
        dm.run(q)
    dynmat = dm.dynamical_matrix
    assert dynmat is not None
    eigvals_ref, eigvecs_ref = np.linalg.eigh(dynmat)
    freqs_ref = (
        np.sqrt(np.abs(eigvals_ref)) * np.sign(eigvals_ref) * ph.unit_conversion_factor
    )
    np.testing.assert_allclose(mod.frequencies, freqs_ref, atol=1e-6)

    vecs = mod.eigenvectors.T  # columns are eigenvectors as numpy.linalg.eigh.
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=0), 1, atol=1e-10)
    np.testing.assert_allclose(dynmat @ vecs, vecs * eigvals_ref, atol=1e-8)
    for deg in degenerate_sets(eigvals_ref, cutoff=1e-6):
        proj = vecs[:, deg] @ vecs[:, deg].conj().T
        proj_ref = eigvecs_ref[:, deg] @ eigvecs_ref[:, deg].conj().T
        np.testing.assert_allclose(proj, proj_ref, atol=1e-8)


def _show(cells):
    for cell in cells:
        for p in cell.scaled_positions:
            print("[%.8f, %.8f, %.8f]," % tuple(p))
