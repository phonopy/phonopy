"""Tests for QE calculater interface."""
import io
from pathlib import Path

import pytest

from phonopy.interface.lammps import LammpsStructureDumper, LammpsStructureLoader
from phonopy.interface.phonopy_yaml import read_phonopy_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_cell_matrix_from_lattice
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
    assert symmetry.dataset["number"] == 194


def test_LammpsStructureDumper(primcell_nacl: PhonopyAtoms, helper_methods):
    """Test of LammpsStructureDumper."""
    lmpsd = LammpsStructureDumper(primcell_nacl)
    cell_stream = io.StringIO("\n".join(lmpsd.get_lines()))
    lmpsd_cell = LammpsStructureLoader().load(cell_stream).cell
    pcell_rot = primcell_nacl.copy()
    pcell_rot.cell = get_cell_matrix_from_lattice(pcell_rot.cell)
    helper_methods.compare_cells_with_order(pcell_rot, lmpsd_cell)


@pytest.mark.parametrize("symbol", ["H", "Ti"])
def test_LammpsStructureDumper_Ti(symbol, helper_methods):
    """Test of LammpsStructureDumper with Ti (with and without Atom Type Labels)."""
    cell = LammpsStructureLoader().load(cwd / f"lammps_structure_{symbol}").cell
    lmpsd = LammpsStructureDumper(cell)
    cell_stream = io.StringIO("\n".join(lmpsd.get_lines()))
    lmpsd_cell = LammpsStructureLoader().load(cell_stream).cell
    helper_methods.compare_cells_with_order(cell, lmpsd_cell)
