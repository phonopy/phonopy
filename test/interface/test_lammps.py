"""Tests for lammps calculater interface."""

import io
import pathlib
from pathlib import Path

import numpy as np
import pytest

from phonopy.interface.lammps import (
    LammpsForcesLoader,
    LammpsStructureDumper,
    LammpsStructureLoader,
    parse_set_of_forces,
    read_lammps,
    rotate_lammps_forces,
    write_lammps,
    write_supercells_with_displacements,
)
from phonopy.interface.phonopy_yaml import read_phonopy_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_cell_matrix_from_lattice, isclose
from phonopy.structure.symmetry import Symmetry

cwd = Path(__file__).parent

phonopy_atoms = {
    symbol: f"""lattice:
- [     2.923479689273095,     0.000000000000000,     0.000000000000000 ] # a
- [    -1.461739844636547,     2.531807678358337,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     4.624022835916574 ] # c
points:
- symbol: {symbol}  # 1
  coordinates: [  0.333333333333334,  0.666666666666667,  0.750000000000000 ]
- symbol: {symbol}  # 2
  coordinates: [  0.666666666666667,  0.333333333333333,  0.250000000000000 ]
"""
    for symbol in ["H", "Ti"]
}


@pytest.mark.parametrize("symbol", ["H", "Ti"])
def test_LammpsStructure(helper_methods, symbol):
    """Test of LammpsStructureLoader.load(stream)."""
    with open(cwd / f"lammps_structure_{symbol}") as fp:
        cell = LammpsStructureLoader().load(fp).cell
    _assert_LammpsStructure(cell, symbol, helper_methods)


@pytest.mark.parametrize("symbol", ["H", "Ti"])
def test_LammpsStructure_from_file(helper_methods, symbol):
    """Test of LammpsStructureLoader.load(filename)."""
    cell = LammpsStructureLoader().load(cwd / f"lammps_structure_{symbol}").cell
    _assert_LammpsStructure(cell, symbol, helper_methods)


def _assert_LammpsStructure(cell: PhonopyAtoms, symbol: str, helper_methods):
    phyml = read_phonopy_yaml(io.StringIO(phonopy_atoms[symbol]))
    helper_methods.compare_cells_with_order(cell, phyml.unitcell)
    symmetry = Symmetry(phyml.unitcell)
    assert symmetry.dataset.number == 194


def test_LammpsStructureDumper(primcell_nacl: PhonopyAtoms, helper_methods):
    """Test of LammpsStructureDumper."""
    lmpsd = LammpsStructureDumper(primcell_nacl)
    cell_stream = io.StringIO("\n".join(lmpsd.get_lines()))
    lmpsd_cell = LammpsStructureLoader().load(cell_stream).cell
    pcell_rot = primcell_nacl.copy()
    pcell_rot.cell = get_cell_matrix_from_lattice(pcell_rot.cell)
    helper_methods.compare_cells_with_order(pcell_rot, lmpsd_cell)


@pytest.mark.parametrize("symbol", ["H", "Ti", "Ti_id"])
def test_LammpsStructureDumper_Ti(symbol, helper_methods):
    """Test of LammpsStructureDumper with Ti (with and without Atom Type Labels)."""
    cell = LammpsStructureLoader().load(cwd / f"lammps_structure_{symbol}").cell
    lmpsd = LammpsStructureDumper(cell)
    cell_stream = io.StringIO("\n".join(lmpsd.get_lines()))
    lmpsd_cell = LammpsStructureLoader().load(cell_stream).cell
    helper_methods.compare_cells_with_order(cell, lmpsd_cell)


def test_LammpsForcesLoader():
    """Test of LammpsForcesLoader with HCP Ti.

    This is forces of 4x4x3 supercell of lammps_structure_Ti with a displacement
    [ 0.0064452834123435,  0.0000000000000000,  0.0076458041914876 ].

    """
    forces = LammpsForcesLoader().load(cwd / "lammps_forces_Ti.0").forces
    # print("%15.10f %15.10f %15.10f" % tuple(forces[0]))
    # print("%15.10f %15.10f %15.10f" % tuple(forces[-1]))
    np.testing.assert_allclose(
        forces[0], [-0.0337045900, -0.0000210300, -0.0399063800], atol=1e-8
    )
    np.testing.assert_allclose(
        forces[-1], [-0.0000369000, -0.0000003300, 0.0001166000], atol=1e-8
    )


# ---------------------------------------------------------------------------
# read_lammps / write_lammps
# ---------------------------------------------------------------------------


def test_read_lammps_natoms() -> None:
    """lammps_structure_Ti has 2 Ti atoms."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    assert len(cell) == 2


def test_read_lammps_symbols() -> None:
    """All atoms should be Ti."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    assert all(s == "Ti" for s in cell.symbols)


def test_read_lammps_cell_shape() -> None:
    """Lattice must be a 3×3 array."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    assert cell.cell.shape == (3, 3)


def test_read_lammps_lattice_diagonal() -> None:
    """Diagonal elements must match xlo_xhi, ylo_yhi, zlo_zhi ranges."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    np.testing.assert_allclose(cell.cell[0, 0], 2.923479689273095, atol=1e-10)
    np.testing.assert_allclose(cell.cell[1, 1], 2.531807678358337, atol=1e-10)
    np.testing.assert_allclose(cell.cell[2, 2], 4.624022835916574, atol=1e-10)


def test_read_lammps_lattice_off_diagonal() -> None:
    """Xy tilt must be parsed from the xy xz yz line."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    np.testing.assert_allclose(cell.cell[1, 0], -1.461739844636547, atol=1e-10)
    np.testing.assert_allclose(cell.cell[2, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(cell.cell[2, 1], 0.0, atol=1e-10)


def test_write_lammps_roundtrip(tmp_path: pathlib.Path) -> None:
    """write_lammps then read_lammps must reproduce the original cell."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    fpath = tmp_path / "Ti.lammps"
    write_lammps(str(fpath), cell)
    cell2 = read_lammps(str(fpath))

    assert isclose(cell, cell2)
    np.testing.assert_allclose(cell.cell, cell2.cell, atol=1e-8)
    diff = cell.scaled_positions - cell2.scaled_positions
    diff -= np.rint(diff)
    assert np.abs(diff).max() < 1e-8
    assert list(cell.symbols) == list(cell2.symbols)


# ---------------------------------------------------------------------------
# parse_set_of_forces
# ---------------------------------------------------------------------------


def test_parse_set_of_forces_length() -> None:
    """parse_set_of_forces returns one entry per file."""
    result = parse_set_of_forces(96, [cwd / "lammps_forces_Ti.0"], verbose=False)
    assert len(result) == 1


def test_parse_set_of_forces_shape() -> None:
    """Each force array must have shape (num_atoms, 3)."""
    result = parse_set_of_forces(96, [cwd / "lammps_forces_Ti.0"], verbose=False)
    assert result[0].shape == (96, 3)


def test_parse_set_of_forces_drift_removed() -> None:
    """After drift removal the sum of forces must be near zero."""
    result = parse_set_of_forces(96, [cwd / "lammps_forces_Ti.0"], verbose=False)
    np.testing.assert_allclose(result[0].sum(axis=0), [0.0, 0.0, 0.0], atol=1e-10)


def test_parse_set_of_forces_wrong_natoms_returns_empty() -> None:
    """Wrong num_atoms causes parse_set_of_forces to return []."""
    result = parse_set_of_forces(1, [cwd / "lammps_forces_Ti.0"], verbose=False)
    assert result == []


# ---------------------------------------------------------------------------
# rotate_lammps_forces
# ---------------------------------------------------------------------------


def test_rotate_lammps_forces_modifies_in_place() -> None:
    """rotate_lammps_forces must modify force_sets elements in place."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    original = np.ones((2, 3))
    force_sets = [original]
    rotate_lammps_forces(force_sets, cell.cell, verbose=False)
    assert force_sets[0] is original


def test_rotate_lammps_forces_shape_preserved() -> None:
    """Force array shape must be unchanged after rotation."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    forces = np.random.default_rng(0).random((2, 3))
    force_sets = [forces.copy()]
    rotate_lammps_forces(force_sets, cell.cell, verbose=False)
    assert force_sets[0].shape == (2, 3)


# ---------------------------------------------------------------------------
# write_supercells_with_displacements
# ---------------------------------------------------------------------------


def test_write_supercells_filenames(tmp_path: pathlib.Path) -> None:
    """Supercell and displacement files (no extension) must be created."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    ids = np.array([1, 2])
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(cell, [cell, cell], ids, pre_filename=pre)
    assert (tmp_path / "supercell").exists()
    assert (tmp_path / "supercell-001").exists()
    assert (tmp_path / "supercell-002").exists()


def test_write_supercells_custom_prefix(tmp_path: pathlib.Path) -> None:
    """Custom pre_filename is used for all output files."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    pre = str(tmp_path / "disp")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    assert (tmp_path / "disp").exists()
    assert (tmp_path / "disp-001").exists()


def test_write_supercells_content_readable(tmp_path: pathlib.Path) -> None:
    """Written supercell file must be parseable by read_lammps."""
    cell = read_lammps(cwd / "lammps_structure_Ti")
    pre = str(tmp_path / "supercell")
    write_supercells_with_displacements(cell, [cell], np.array([1]), pre_filename=pre)
    cell2 = read_lammps(str(tmp_path / "supercell"))
    assert isclose(cell, cell2)
