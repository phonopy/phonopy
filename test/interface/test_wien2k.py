"""Tests for wien2k interface."""

import os

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.wien2k import parse_wien2k_struct

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_parse_wien2k_struct():
    """Test structure parsing."""
    filename_BaGa2 = os.path.join(data_dir, "BaGa2.struct")
    cell, _, _, _ = parse_wien2k_struct(filename_BaGa2)
    filename = os.path.join(data_dir, "BaGa2-wien2k.yaml")
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
