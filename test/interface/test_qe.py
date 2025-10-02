"""Tests for QE calculater interface."""

from __future__ import annotations

import pathlib

import numpy as np

from phonopy.file_IO import get_io_module_to_decompress
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.qe import PH_Q2R, read_pwscf
from phonopy.structure.symmetry import Symmetry

cwd = pathlib.Path(__file__).parent


def test_read_pwscf():
    """Test of read_pwscf with default scaled positions.

    Keywords appear in the following order:

    ATOMIC_SPECIES
    ATOMIC_POSITIONS
    CELL_PARAMETERS
    K_POINTS

    """
    _test_read_pwscf("NaCl-pwscf.in")


def test_read_pwscf_2():
    """Test of read_pwscf with default scaled positions.

    Keywords appear in different order from test_read_pwscf.

    ATOMIC_SPECIES
    ATOMIC_POSITIONS
    K_POINTS
    CELL_PARAMETERS

    """
    _test_read_pwscf("NaCl-pwscf-2.in")


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


def test_make_fc_q2r():
    """Test make_fc_q2r."""
    fc_0_10 = [
        [6.06001648e-05, -3.48358667e-05, -8.14194922e-05],
        [-3.48358667e-05, 1.21530469e-05, -1.65827117e-04],
        [-8.14194922e-05, -1.65827117e-04, -1.14989696e-04],
    ]
    fc_1_10 = [
        [-4.05313258e-04, 1.92325415e-10, -9.48168187e-11],
        [5.67066085e-11, -3.30626094e-04, 1.06319726e-03],
        [-7.34782548e-11, 3.68433780e-04, 8.99705485e-04],
    ]

    fc_filename = cwd / "NaCl-q2r.fc.xz"
    myio = get_io_module_to_decompress(fc_filename)
    with myio.open(fc_filename) as f:
        primcell_filename = cwd / "NaCl-q2r.in"
        cell, _ = read_pwscf(primcell_filename)
        q2r = PH_Q2R(f)
        q2r.run(cell)

    assert q2r.fc is not None
    np.testing.assert_allclose(fc_0_10, q2r.fc[0, 10], atol=1e-8)
    np.testing.assert_allclose(fc_1_10, q2r.fc[1, 10], atol=1e-8)


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
