"""Tests of VASP calculator interface."""

import tarfile
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.vasp import (
    Vasprun,
    VasprunxmlExpat,
    get_vasp_structure_lines,
    parse_set_of_forces,
    read_vasp,
    read_vasp_from_strings,
    read_XDATCAR,
    write_XDATCAR,
)

cwd = Path(__file__).parent


def test_read_vasp():
    """Test read_vasp."""
    cell = read_vasp(cwd / ".." / "POSCAR_NaCl")
    filename = cwd / "NaCl-vasp.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r


@pytest.mark.parametrize(
    "first_line_str, is_vasp4",
    [(None, True), (None, False), ("my_comment", True), ("my_comment", False)],
)
def test_get_vasp_structure_lines(helper_methods, is_vasp4, first_line_str):
    """Test get_vasp_structure_lines (almost write_vasp)."""
    filename = cwd / "NaCl-vasp.yaml"
    cell_ref = read_cell_yaml(filename)
    lines = get_vasp_structure_lines(
        cell_ref, direct=True, is_vasp4=is_vasp4, first_line_str=first_line_str
    )
    cell = read_vasp_from_strings("\n".join(lines))
    helper_methods.compare_cells_with_order(cell, cell_ref)
    if is_vasp4:
        # is_vasp4 is True, first_line_str is ignored.
        assert lines[0] == "Na Cl"
    elif first_line_str:
        assert lines[0] == first_line_str


def test_get_vasp_structure_lines_shuffled_positions(helper_methods):
    """Test get_vasp_structure_lines with a cell having shuffled positions.

    Order of atoms is sorted by chemical symbols. Therefore,
    helper_methods.compare_cells_with_order fails.

    """
    poscar_yaml = """lattice:
- [     5.690301476175672,     0.000000000000000,     0.000000000000000 ] # a
- [     0.000000000000000,     5.690301476175672,     0.000000000000000 ] # b
- [     0.000000000000000,     0.000000000000000,     5.690301476175672 ] # c
points:
- symbol: Na # 1
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 22.989769
- symbol: Cl # 2
  coordinates: [  0.500000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 35.453000
- symbol: Na # 3
  coordinates: [  0.000000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 22.989769
- symbol: Cl # 4
  coordinates: [  0.500000000000000,  0.500000000000000,  0.500000000000000 ]
  mass: 35.453000
- symbol: Na # 5
  coordinates: [  0.500000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 22.989769
- symbol: Cl # 6
  coordinates: [  0.000000000000000,  0.000000000000000,  0.500000000000000 ]
  mass: 35.453000
- symbol: Na # 7
  coordinates: [  0.500000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 22.989769
- symbol: Cl # 8
  coordinates: [  0.000000000000000,  0.500000000000000,  0.000000000000000 ]
  mass: 35.453000
  """
    cell_ref = read_cell_yaml(StringIO(poscar_yaml))
    lines = get_vasp_structure_lines(cell_ref, direct=True)
    cell = read_vasp_from_strings("\n".join(lines))
    with pytest.raises(AssertionError):
        helper_methods.compare_cells_with_order(cell, cell_ref)
    helper_methods.compare_cells(cell, cell_ref)
    perm = []
    for p_i in cell.scaled_positions:
        diff = cell_ref.scaled_positions - p_i
        diff -= np.rint(diff)
        dists = np.linalg.norm(diff @ cell.cell, axis=1)
        perm.append(np.where(dists < 1e-8)[0][0])
    np.testing.assert_array_equal(perm, [0, 2, 4, 6, 1, 3, 5, 7])
    np.testing.assert_array_equal(cell.numbers, [11, 11, 11, 11, 17, 17, 17, 17])


def test_parse_vasprun_xml():
    """Test parsing vasprun.xml with expat."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    filename = cwd / ".." / "FORCE_SETS_NaCl"
    dataset = parse_FORCE_SETS(filename=filename)
    energy_ref = [-216.82820693, -216.82817843]
    for i, member in enumerate(_tar.getmembers()):
        vr = Vasprun(_tar.extractfile(member), use_expat=True)
        # for force in vr.read_forces():
        #     print("% 15.8f % 15.8f % 15.8f" % tuple(force))
        # print("")
        ref = dataset["first_atoms"][i]["forces"]
        np.testing.assert_allclose(ref, vr.read_forces(), atol=1e-8)
        np.testing.assert_allclose(energy_ref[i], vr.read_energy(), atol=1e-8)


def test_VasprunxmlExpat():
    """Test VasprunxmlExpat."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    for member in _tar.getmembers():
        vasprun = VasprunxmlExpat(_tar.extractfile(member))
        vasprun.parse()
        np.testing.assert_equal(vasprun.fft_grid, [64, 64, 64])
        np.testing.assert_equal(vasprun.fft_fine_grid, [128, 128, 128])
        assert vasprun.efermi is None
        assert vasprun.symbols == ["Na"] * 32 + ["Cl"] * 32
        np.testing.assert_almost_equal(vasprun.NELECT, 448)
        np.testing.assert_almost_equal(vasprun.volume, 1473.99433936)
        break


def test_parse_set_of_forces():
    """Test parse_set_of_forces."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    fps = [_tar.extractfile(member) for member in _tar.getmembers()]
    calc_dataset = parse_set_of_forces(64, fps)
    filename = cwd / ".." / "FORCE_SETS_NaCl"
    dataset = parse_FORCE_SETS(filename=filename)
    force_sets = [dataset["first_atoms"][i]["forces"] for i in (0, 1)]
    energy_ref = [-216.82820693, -216.82817843]
    np.testing.assert_allclose(
        calc_dataset["points"][0][0], [0.00087869, 0.0, 0.0], atol=1e-5
    )
    np.testing.assert_allclose(
        calc_dataset["points"][1][32], [0.25087869, 0.25, 0.25], atol=1e-5
    )
    np.testing.assert_allclose(force_sets, calc_dataset["forces"], atol=1e-8)
    np.testing.assert_allclose(
        energy_ref, calc_dataset["supercell_energies"], atol=1e-8
    )


def test_read_XDATCAR():
    """Test read_XDATCAR."""
    filename_xdatcar = cwd / "XDATCAR-NaCl"
    lattice, positions = read_XDATCAR(filename_xdatcar)

    np.testing.assert_allclose(lattice, np.eye(3) * 22.562240, atol=1e-8)
    np.testing.assert_allclose(
        positions[0, 0], [0.99854664, 0.00265936, 0.00701660], atol=1e-8
    )
    np.testing.assert_allclose(
        positions[-1, -1], [0.75034458, 0.74972945, 0.87262656], atol=1e-8
    )


def test_write_XDATCAR():
    """Test write_XDATCAR."""
    filename_vasprun = cwd / "vasprun.xml.tar.bz2"
    _tar = tarfile.open(filename_vasprun)
    for _, member in enumerate(_tar.getmembers()):
        vasprun = VasprunxmlExpat(_tar.extractfile(member))
        vasprun.parse()
        break

    with tempfile.TemporaryFile() as fp:
        write_XDATCAR(vasprunxml_expat=vasprun, fileptr=fp)
        fp.seek(0)
        lattice, positions = read_XDATCAR(fileptr=fp)

    np.testing.assert_allclose(lattice, np.eye(3) * 11.38060295, atol=1e-8)
    np.testing.assert_allclose(positions[0, 0], [0.00087869, 0, 0], atol=1e-8)
    np.testing.assert_allclose(positions[-1, -1], [0.5, 0.5, 0.75], atol=1e-8)
