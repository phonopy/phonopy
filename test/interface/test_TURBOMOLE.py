"""Tests for TURBOMOLE cauclator interface."""

import os

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.turbomole import read_turbomole

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_turbomole():
    """Test read_turbomole."""
    cell = read_turbomole(os.path.join(data_dir, "Si-TURBOMOLE-control"))
    filename = os.path.join(data_dir, "Si-TURBOMOLE.yaml")
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
