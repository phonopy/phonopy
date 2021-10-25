"""Tests of VASP calculator interface."""
import os
import tarfile

import numpy as np

from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.interface.vasp import Vasprun, read_vasp

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
