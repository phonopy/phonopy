"""Tests for PWmat calculater interface."""

import os

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.pwmat import read_atom_config
from phonopy.structure import cells

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_pwmat():
    """Test of read_PWmat."""
    cell = read_atom_config(os.path.join(data_dir, "Si-pwmat.config"))
    filename = os.path.join(data_dir, "Si-pwmat.yaml")
    cell_ref = read_cell_yaml(filename)
    assert cells.isclose(cell, cell_ref)
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r


def test_magmom():
    """Test of read_PWmat_magmom."""
    cell = read_atom_config(os.path.join(data_dir, "Si-pwmat.config"))
    filename = os.path.join(data_dir, "Si-pwmat.yaml")
    cell_ref = read_cell_yaml(filename)
    assert cells.isclose(cell, cell_ref)
    diff_mag = cell_ref.magnetic_moments - np.array([1] * 8)
    assert (np.abs(diff_mag) < 1e-5).all()
