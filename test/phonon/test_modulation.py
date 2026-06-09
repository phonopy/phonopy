"""Tests for Modulation."""

import numpy as np

from phonopy import Phonopy
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


def _show(cells):
    for cell in cells:
        for p in cell.scaled_positions:
            print("[%.8f, %.8f, %.8f]," % tuple(p))
