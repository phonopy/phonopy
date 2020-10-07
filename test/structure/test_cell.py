import os
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell, get_primitive, TrimmedCell
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_supercell_convcell_si(convcell_si):
    smat = np.diag([1, 2, 3])
    fname = "SiO2-123.yaml"
    scell = get_supercell(convcell_si, smat)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    _compare_cells(scell, cell_ref)


def test_get_supercell_primcell_si(primcell_si):
    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    fname = "Si-conv.yaml"
    scell = get_supercell(primcell_si, smat)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    _compare_cells(scell, cell_ref)


def test_get_primitive_convcell_nacl(convcell_nacl, primcell_nacl):
    pcell = get_primitive(convcell_nacl, [[0, 0.5, 0.5],
                                          [0.5, 0, 0.5],
                                          [0.5, 0.5, 0]])
    _compare_cells(pcell, primcell_nacl)


def test_TrimmedCell(convcell_nacl):
    pmat = [[0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]]
    smat2 = np.eye(3, dtype='intc') * 2
    pmat2 = np.dot(np.linalg.inv(smat2), pmat)
    smat3 = np.eye(3, dtype='intc') * 3
    pmat3 = np.dot(np.linalg.inv(smat3), pmat)

    cell = convcell_nacl
    scell2 = get_supercell(cell, smat2)
    scell3 = get_supercell(cell, smat3)
    n = len(scell3) // 2
    # swap first and last half of atomic order
    indices = [i + n for i in range(n)] + list(range(n))
    scell3_swap = PhonopyAtoms(
        cell=scell3.cell,
        scaled_positions=scell3.scaled_positions[indices],
        numbers=scell3.numbers[indices])
    tcell2 = TrimmedCell(pmat2, scell2)
    tcell3 = TrimmedCell(pmat3, scell3_swap,
                         positions_to_reorder=tcell2.scaled_positions)
    _compare_cells(tcell2, tcell3)


def _compare_cells(cell, cell_ref):
    np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=1e-5)
    diff = cell.scaled_positions - cell_ref.scaled_positions
    diff -= np.rint(diff)
    dist = (np.dot(diff, cell.cell) ** 2).sum(axis=1)
    assert (dist < 1e-5).all()
    np.testing.assert_array_equal(cell.numbers, cell_ref.numbers)
    np.testing.assert_allclose(cell.masses, cell_ref.masses, atol=1e-5)
