"""Tests of Abinit calculator interface."""

import io
import pathlib

import numpy as np

from phonopy.interface.abinit import get_abinit_structure, read_abinit
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


def test_read_abinit():
    """Test of read_abinit."""
    cell = read_abinit(cwd / "NaCl-abinit.in")
    filename = cwd / "NaCl-abinit-pwscf.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r


def test_get_abinit_structure():
    """Test get_abinit_structure."""
    cell_ref = read_abinit(cwd / "NaCl-abinit.in")
    cell = read_abinit(io.StringIO(get_abinit_structure(cell_ref)))
    isclose(cell_ref, cell)
