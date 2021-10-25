"""Tests for Modulation."""
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

lattice_NaCl222 = [
    [0.000000000000000, 5.690301476175671, 5.690301476175671],
    [5.690301476175671, 0.000000000000000, 5.690301476175671],
    [5.690301476175671, 5.690301476175671, 0.000000000000000],
]
positions_NaCl222 = [
    [0.00608478, 0.99337469, 0.00662531],
    [0.50608478, 0.99337469, 0.00662531],
    [0.99391522, 0.50662531, 0.99337469],
    [0.49391522, 0.50662531, 0.99337469],
    [0.99391522, 0.00662531, 0.49337469],
    [0.49391522, 0.00662531, 0.49337469],
    [0.00608478, 0.49337469, 0.50662531],
    [0.50608478, 0.49337469, 0.50662531],
    [0.24492438, 0.25552650, 0.24447350],
    [0.74492438, 0.25552650, 0.24447350],
    [0.25507562, 0.74447350, 0.25552650],
    [0.75507562, 0.74447350, 0.25552650],
    [0.25507562, 0.24447350, 0.75552650],
    [0.75507562, 0.24447350, 0.75552650],
    [0.24492438, 0.75552650, 0.74447350],
    [0.74492438, 0.75552650, 0.74447350],
]
symbols_NaCl222 = ["Na"] * 8 + ["Cl"] * 8
cell_NaCl222 = PhonopyAtoms(
    cell=lattice_NaCl222, scaled_positions=positions_NaCl222, symbols=symbols_NaCl222
)


def test_modulation(ph_nacl: Phonopy, helper_methods):
    """Test to calculate modulation by NaCl."""
    ph = ph_nacl
    ph.set_modulations([2, 2, 2], [[[0, 0.5, 0.5], 1, 2, 0]])
    cells = ph.get_modulated_supercells()
    helper_methods.compare_cells_with_order(cells[0], cell_NaCl222)


def _show(cells):
    for cell in cells:
        for p in cell.scaled_positions:
            print("[%.8f, %.8f, %.8f]," % tuple(p))
