"""Tests for QE calculater interface."""

import pathlib

import numpy as np

from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.qe import read_pwscf
from phonopy.structure.symmetry import Symmetry

cwd = pathlib.Path(__file__).parent


def test_read_pwscf():
    """Test of read_pwscf with default scaled positions."""
    _test_read_pwscf("NaCl-pwscf.in")


def test_read_pwscf_angstrom():
    """Test of read_pwscf with angstrom coordinates."""
    _test_read_pwscf("NaCl-pwscf-angstrom.in")


def test_read_pwscf_bohr():
    """Test of read_pwscf with bohr coordinates."""
    _test_read_pwscf("NaCl-pwscf-bohr.in")


def test_read_pwscf_NaCl_Xn():
    """Test of read_pwscf."""
    cell, pp_filenames = read_pwscf(cwd / "NaCl-pwscf-Xn.in")
    print(cell)
    symnums = pp_filenames.keys()
    assert set(symnums) == {"Na", "Cl", "Cl1"}
    np.testing.assert_allclose(
        cell.masses,
        [
            22.98976928,
            22.98976928,
            22.98976928,
            22.98976928,
            35.453,
            35.453,
            70.0,
            70.0,
        ],
    )
    assert ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl1", "Cl1"] == cell.symbols

    cell_ref, pp_filenames = read_pwscf(cwd / "NaCl-pwscf.in")
    symops = Symmetry(cell).symmetry_operations
    symops_ref = Symmetry(cell_ref).symmetry_operations
    np.testing.assert_allclose(symops["translations"], symops_ref["translations"])
    np.testing.assert_array_equal(symops["rotations"], symops_ref["rotations"])


def _test_read_pwscf(filename):
    """Test of read_pwscf."""
    cell, pp_filenames = read_pwscf(cwd / filename)
    filename = cwd / "NaCl-abinit-pwscf.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r
