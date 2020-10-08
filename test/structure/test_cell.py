import os
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell, get_primitive, TrimmedCell
from phonopy.interface.phonopy_yaml import read_cell_yaml

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_supercell_convcell_sio2(convcell_sio2):
    _test_get_supercell_convcell_sio2(convcell_sio2, is_old_style=True)


def test_get_supercell_convcell_sio2_snf(convcell_sio2):
    _test_get_supercell_convcell_sio2(convcell_sio2, is_old_style=False)


def test_get_supercell_primcell_si(primcell_si):
    _test_get_supercell_primcell_si(primcell_si, is_old_style=True)


def test_get_supercell_primcell_si_snf(primcell_si):
    _test_get_supercell_primcell_si(primcell_si, is_old_style=False)


def test_get_supercell_nacl_snf(convcell_nacl):
    cell = convcell_nacl
    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    scell = get_supercell(cell, smat, is_old_style=True)
    scell_snf = get_supercell(cell, smat, is_old_style=False)
    _compare_cells(scell, scell_snf)


def _test_get_supercell_convcell_sio2(convcell_sio2, is_old_style=True):
    smat = np.diag([1, 2, 3])
    fname = "SiO2-123.yaml"
    scell = get_supercell(convcell_sio2, smat, is_old_style=is_old_style)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    if is_old_style is True:
        _compare_cells_with_order(scell, cell_ref)
    else:
        _compare_cells(scell, cell_ref)


def _test_get_supercell_primcell_si(primcell_si, is_old_style=True):
    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    fname = "Si-conv.yaml"
    scell = get_supercell(primcell_si, smat, is_old_style=is_old_style)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    if is_old_style is True:
        _compare_cells_with_order(scell, cell_ref)
    else:
        _compare_cells(scell, cell_ref)


def test_get_primitive_convcell_nacl(convcell_nacl, primcell_nacl):
    pcell = get_primitive(convcell_nacl, [[0, 0.5, 0.5],
                                          [0.5, 0, 0.5],
                                          [0.5, 0.5, 0]])
    _compare_cells_with_order(pcell, primcell_nacl)


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
    _compare_cells_with_order(tcell2, tcell3)


def _compare_cells_with_order(cell, cell_ref):
    np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=1e-5)
    diff = cell.scaled_positions - cell_ref.scaled_positions
    diff -= np.rint(diff)
    dist = (np.dot(diff, cell.cell) ** 2).sum(axis=1)
    assert (dist < 1e-5).all()
    np.testing.assert_array_equal(cell.numbers, cell_ref.numbers)
    np.testing.assert_allclose(cell.masses, cell_ref.masses, atol=1e-5)


def _compare_cells(cell, cell_ref):
    np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=1e-5)

    indices = []
    for pos in cell.scaled_positions:
        diff = cell_ref.scaled_positions - pos
        diff -= np.rint(diff)
        dist = (np.dot(diff, cell.cell) ** 2).sum(axis=1)
        matches = np.where(dist < 1e-5)[0]
        assert len(matches) == 1
        indices.append(matches[0])

    np.testing.assert_array_equal(cell.numbers, cell_ref.numbers[indices])
    np.testing.assert_allclose(cell.masses, cell_ref.masses[indices],
                               atol=1e-5)
