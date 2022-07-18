"""Tests for ABACUS calculater interface."""
import os

import numpy as np

from phonopy.interface.abacus import read_abacus
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_abacus():
    """Test of read_ABACUS."""
    cell, pps, orbitals = read_abacus(
        os.path.join(data_dir, "NaCl-abacus.stru"), elements=["Na", "Cl"]
    )
    filename = os.path.join(data_dir, "NaCl-abinit-pwscf.yaml")
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
