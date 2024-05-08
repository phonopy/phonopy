"""Tests for CASTEP calculator interface."""

import os

import numpy as np

from phonopy.interface.castep import read_castep
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_castep():
    """Test read CASTEP file."""
    cell = read_castep(os.path.join(data_dir, "NaCl-castep.cell"))
    filename = os.path.join(data_dir, "NaCl-castep.yaml")
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
