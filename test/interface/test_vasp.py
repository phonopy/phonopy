"""Tests of VASP calculator interface."""
import os
import tarfile
import tempfile

import numpy as np

from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.vasp import (
    Vasprun,
    VasprunxmlExpat,
    get_vasp_structure_lines,
    read_vasp,
    read_vasp_from_strings,
    read_XDATCAR,
    write_XDATCAR,
)

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_read_vasp():
    """Test read_vasp."""
    cell = read_vasp(os.path.join(data_dir, "..", "POSCAR_NaCl"))
    filename = os.path.join(data_dir, "NaCl-vasp.yaml")
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols):
        assert s == s_r


def test_get_vasp_structure_lines(helper_methods):
    """Test get_vasp_structure_lines (almost write_vasp)."""
    filename = os.path.join(data_dir, "NaCl-vasp.yaml")
    cell_ref = read_cell_yaml(filename)
    lines = get_vasp_structure_lines(cell_ref, direct=True)
    cell = read_vasp_from_strings("\n".join(lines))
    helper_methods.compare_cells_with_order(cell, cell_ref)


def test_parse_vasprun_xml():
    """Test parsing vasprun.xml with expat."""
    filename_vasprun = os.path.join(data_dir, "vasprun.xml.tar.bz2")
    _tar = tarfile.open(filename_vasprun)
    filename = os.path.join(data_dir, "../FORCE_SETS_NaCl")
    dataset = parse_FORCE_SETS(filename=filename)
    for i, member in enumerate(_tar.getmembers()):
        vr = Vasprun(_tar.extractfile(member), use_expat=True)
        # for force in vr.read_forces():
        #     print("% 15.8f % 15.8f % 15.8f" % tuple(force))
        # print("")
        ref = dataset["first_atoms"][i]["forces"]
        np.testing.assert_allclose(ref, vr.read_forces(), atol=1e-8)


def test_VasprunxmlExpat():
    """Test VasprunxmlExpat."""
    filename_vasprun = os.path.join(data_dir, "vasprun.xml.tar.bz2")
    _tar = tarfile.open(filename_vasprun)
    for i, member in enumerate(_tar.getmembers()):
        vasprun = VasprunxmlExpat(_tar.extractfile(member))
        vasprun.parse()
        np.testing.assert_equal(vasprun.fft_grid, [64, 64, 64])
        np.testing.assert_equal(vasprun.fft_fine_grid, [128, 128, 128])
        assert vasprun.efermi is None
        assert vasprun.symbols == ["Na"] * 32 + ["Cl"] * 32
        np.testing.assert_almost_equal(vasprun.NELECT, 448)
        np.testing.assert_almost_equal(vasprun.volume, 1473.99433936)
        break


def test_read_XDATCAR():
    """Test read_XDATCAR."""
    filename_xdatcar = os.path.join(data_dir, "XDATCAR-NaCl")
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
    filename_vasprun = os.path.join(data_dir, "vasprun.xml.tar.bz2")
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
