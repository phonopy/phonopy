"""Tests for Modulation."""

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
    ph.run_modulations([2, 2, 2], [[[0, 0.5, 0.5], 2, 2, 0]])
    cells = ph.get_modulated_supercells()
    # _show(cells)
    helper_methods.compare_cells_with_order(cells[0], cell_NaCl222)


def _show(cells):
    for cell in cells:
        for p in cell.scaled_positions:
            print("[%.8f, %.8f, %.8f]," % tuple(p))
