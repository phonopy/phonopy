"""Tests for CP2K calculator interface."""

import os

import numpy as np
import pytest

from phonopy.interface.cp2k import read_cp2k
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))

CP2K_INPUT_TOOLS_AVAILABLE = True

try:
    import cp2k_input_tools  # noqa F401
except ImportError:
    CP2K_INPUT_TOOLS_AVAILABLE = False


@pytest.mark.skipif(
    not CP2K_INPUT_TOOLS_AVAILABLE, reason="not found cp2k-input-tools package"
)
def test_read_cp2k():
    """Test read_cp2k."""
    cell, _ = read_cp2k(os.path.join(data_dir, "Si-CP2K.inp"))
    cell_ref = read_cell_yaml(os.path.join(data_dir, "Si-CP2K.yaml"))
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
