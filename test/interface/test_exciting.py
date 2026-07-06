"""Tests of exciting calculator interface."""

import io
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from phonopy.interface.exciting import (
    get_exciting_structure,
    read_exciting,
)
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.cells import isclose

cwd = pathlib.Path(__file__).parent


def test_read_exciting() -> None:
    """Test of read_exciting."""
    cell = read_exciting(cwd / "exciting/input.xml")
    filename = cwd / "exciting/PbTiO3.yaml"
    cell_ref = read_cell_yaml(filename)
    assert (np.abs(cell.cell - cell_ref.cell) < 1e-5).all()
    diff_pos = cell.scaled_positions - cell_ref.scaled_positions
    diff_pos -= np.rint(diff_pos)
    assert (np.abs(diff_pos) < 1e-5).all()
    for s, s_r in zip(cell.symbols, cell_ref.symbols, strict=True):
        assert s == s_r


def test_get_exciting_structure() -> None:
    """Round-trip: read → get_exciting_structure → read back preserves cell."""
    cell_ref = read_exciting(cwd / "exciting/input.xml")

    tree = get_exciting_structure(cell_ref)
    ET.indent(tree, space="  ", level=0)
    xml_str = ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
    cell = read_exciting(io.StringIO(xml_str))
    assert isclose(cell_ref, cell)
