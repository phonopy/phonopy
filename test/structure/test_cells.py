"""Tests of routines in cells.py."""

import os
from collections.abc import Callable

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import (
    Primitive,
    ShortestPairs,
    TrimmedCell,
    compute_all_sg_permutations,
    compute_permutation_for_rotation,
    convert_to_phonopy_primitive,
    dense_to_sparse_svecs,
    get_angles,
    get_cell_matrix_from_lattice,
    get_cell_parameters,
    get_primitive,
    get_primitive_matrix,
    get_supercell,
    isclose,
    sparse_to_dense_svecs,
)

data_dir = os.path.dirname(os.path.abspath(__file__))
primitive_matrix_nacl = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

multi_nacl_ref = [
    1,
    1,
    2,
    1,
    2,
    1,
    4,
    1,
    2,
    1,
    4,
    1,
    4,
    1,
    8,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    1,
    4,
    2,
    4,
    1,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    1,
    2,
    1,
    4,
    2,
    2,
    2,
    4,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    4,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    4,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    4,
    1,
    2,
    1,
    4,
    1,
    4,
    1,
    8,
    1,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    2,
    1,
    2,
    2,
    4,
    1,
    4,
    2,
    1,
    1,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    4,
    1,
    2,
    2,
    4,
    2,
    1,
    1,
    2,
    1,
    2,
    1,
    4,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    4,
    2,
]
svecs_nacl_ref10 = [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
svecs_nacl_ref30 = [
    [-0.5, -0.5, 0.0],
    [-0.5, 0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
]


def test_compute_permutation_sno2(ph_sno2: Phonopy):
    """Test of compute_permutation by SnO2."""
    _test_compute_permutation(ph_sno2)


def test_compute_permutation_tio2(ph_tio2: Phonopy):
    """Test of compute_permutation by TiO2."""
    _test_compute_permutation(ph_tio2)


def test_compute_permutation_nacl(ph_nacl: Phonopy):
    """Test of compute_permutation by NaCl."""
    _test_compute_permutation(ph_nacl)


def _test_compute_permutation(ph: Phonopy):
    symmetry = ph.primitive_symmetry
    ppos = ph.primitive.scaled_positions
    plat = ph.primitive.cell.T
    symprec = symmetry.tolerance
    rots = symmetry.symmetry_operations["rotations"]
    trans = symmetry.symmetry_operations["translations"]
    perms = compute_all_sg_permutations(ppos, rots, trans, plat, symprec)
    for i, (r, t) in enumerate(zip(rots, trans)):
        ppos_rot = np.dot(ppos, r.T) + t
        perm = compute_permutation_for_rotation(ppos, ppos_rot, plat, symprec)
        np.testing.assert_array_equal(perms[i], perm)
        diff = ppos[perm] - ppos_rot
        diff -= np.rint(diff)
        assert ((np.dot(diff, plat) ** 2).sum(axis=1) < symprec).all()


@pytest.mark.parametrize("nosnf", [True, False])
def test_get_supercell_convcell_sio2(
    convcell_sio2: PhonopyAtoms, nosnf, helper_methods
):
    """Test of get_supercell with/without SNF by SiO2."""
    _test_get_supercell_convcell_sio2(convcell_sio2, helper_methods, is_old_style=nosnf)


@pytest.mark.parametrize("nosnf", [True, False])
def test_get_supercell_primcell_si(
    primcell_si: PhonopyAtoms, nosnf, helper_methods: Callable
):
    """Test of get_supercell with/without SNF by Si."""
    _test_get_supercell_primcell_si(primcell_si, helper_methods, is_old_style=nosnf)


def test_get_supercell_nacl_snf(
    nacl_unitcell_order1: PhonopyAtoms, helper_methods: Callable
):
    """Test of get_supercell using SNF by NaCl."""
    cell = nacl_unitcell_order1
    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    scell = get_supercell(cell, smat, is_old_style=True)
    scell_snf = get_supercell(cell, smat, is_old_style=False)
    helper_methods.compare_cells(scell, scell_snf)


@pytest.mark.parametrize("is_ncl", [False, True])
def test_get_supercell_Cr_with_magmoms(
    convcell_cr: PhonopyAtoms, is_ncl: bool, helper_methods: Callable
):
    """Test of get_supercell using SNF by Cr with magnetic moments."""
    if is_ncl:
        convcell_cr.magnetic_moments = [[0, 0, 1], [0, 0, -1]]
        ref_magmoms = [[0, 0, 1]] * 4 + [[0, 0, -1]] * 4
    else:
        convcell_cr.magnetic_moments = [1, -1]
        ref_magmoms = [1.0] * 4 + [-1.0] * 4

    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    scell = get_supercell(convcell_cr, smat, is_old_style=True)

    np.testing.assert_allclose(
        scell.magnetic_moments,
        ref_magmoms,
        atol=1e-8,
    )
    scell_snf = get_supercell(convcell_cr, smat, is_old_style=False)
    helper_methods.compare_cells(scell, scell_snf)
    convcell_cr.magnetic_moments = None


def _test_get_supercell_convcell_sio2(
    convcell_sio2: PhonopyAtoms, helper_methods, is_old_style=True
):
    smat = np.diag([1, 2, 3])
    fname = "SiO2-123.yaml"
    scell = get_supercell(convcell_sio2, smat, is_old_style=is_old_style)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    if is_old_style is True:
        helper_methods.compare_cells_with_order(scell, cell_ref)
    else:
        helper_methods.compare_cells(scell, cell_ref)


def _test_get_supercell_primcell_si(
    primcell_si: PhonopyAtoms, helper_methods, is_old_style=True
):
    smat = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    fname = "Si-conv.yaml"
    scell = get_supercell(primcell_si, smat, is_old_style=is_old_style)
    cell_ref = read_cell_yaml(os.path.join(data_dir, fname))
    if is_old_style is True:
        helper_methods.compare_cells_with_order(scell, cell_ref)
    else:
        helper_methods.compare_cells(scell, cell_ref)


def test_get_primitive_convcell_nacl(
    nacl_unitcell_order1: PhonopyAtoms, primcell_nacl: PhonopyAtoms, helper_methods
):
    """Test get_primitive by NaCl."""
    pcell = get_primitive(nacl_unitcell_order1, primitive_matrix=primitive_matrix_nacl)
    helper_methods.compare_cells_with_order(pcell, primcell_nacl)


def test_get_primitive_convcell_nacl_with_cetring_symbol(
    nacl_unitcell_order1: PhonopyAtoms, primcell_nacl: PhonopyAtoms, helper_methods
):
    """Test get_primitive by NaCl."""
    pcell = get_primitive(nacl_unitcell_order1, primitive_matrix="F")
    helper_methods.compare_cells_with_order(pcell, primcell_nacl)


@pytest.mark.parametrize("is_ncl", [False, True])
def test_get_primitive_convcell_Cr_with_magmoms(
    convcell_cr: PhonopyAtoms, is_ncl: bool, helper_methods: Callable
):
    """Test get_primitive by Cr with magmoms."""
    if is_ncl:
        convcell_cr.magnetic_moments = [[0, 0, 1], [0, 0, -1]]
    else:
        convcell_cr.magnetic_moments = [1, -1]
    smat = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    scell = get_supercell(convcell_cr, smat, is_old_style=True)
    pmat = np.linalg.inv(smat)
    pcell = get_primitive(scell, primitive_matrix=pmat)
    helper_methods.compare_cells(convcell_cr, pcell)
    convcell_cr.magnetic_moments = None


@pytest.mark.parametrize("store_dense_svecs", [True, False])
def test_get_primitive_convcell_nacl_svecs(
    nacl_unitcell_order1: PhonopyAtoms, store_dense_svecs
):
    """Test shortest vectors by NaCl."""
    pcell = get_primitive(
        nacl_unitcell_order1,
        primitive_matrix=primitive_matrix_nacl,
        store_dense_svecs=store_dense_svecs,
    )
    svecs, multi = pcell.get_smallest_vectors()
    if store_dense_svecs:
        assert svecs.shape == (54, 3)
        assert multi.shape == (8, 2, 2)
        assert np.sum(multi[:, :, 0]) == 54
        assert np.sum(multi[-1:, -1, :]) == 54
    else:
        assert svecs.shape == (8, 2, 27, 3)
        assert multi.shape == (8, 2)


def test_TrimmedCell(nacl_unitcell_order1: PhonopyAtoms, helper_methods: Callable):
    """Test TrimmedCell by NaCl."""
    pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    smat2 = np.eye(3, dtype="intc") * 2
    pmat2 = np.dot(np.linalg.inv(smat2), pmat)
    smat3 = np.eye(3, dtype="intc") * 3
    pmat3 = np.dot(np.linalg.inv(smat3), pmat)

    cell = nacl_unitcell_order1
    scell2 = get_supercell(cell, smat2)
    scell3 = get_supercell(cell, smat3)
    n = len(scell3) // 2
    # swap first and last half of atomic order
    indices = [i + n for i in range(n)] + list(range(n))
    scell3_swap = PhonopyAtoms(
        cell=scell3.cell,
        scaled_positions=scell3.scaled_positions[indices],
        symbols=[scell3.symbols[i] for i in indices],
    )
    tcell2 = TrimmedCell(pmat2, scell2)
    tcell3 = TrimmedCell(
        pmat3, scell3_swap, positions_to_reorder=tcell2.scaled_positions
    )
    helper_methods.compare_cells_with_order(tcell2, tcell3)


def test_ShortestPairs_sparse_nacl(ph_nacl: Phonopy, helper_methods: Callable):
    """Test ShortestPairs (parse) by NaCl."""
    scell = ph_nacl.supercell
    pcell = ph_nacl.primitive
    pos = scell.scaled_positions
    spairs = ShortestPairs(scell.cell, pos, pos[pcell.p2s_map], store_dense_svecs=False)
    svecs = spairs.shortest_vectors
    multi = spairs.multiplicities
    np.testing.assert_array_equal(multi.ravel(), multi_nacl_ref)
    pos_from_svecs = svecs[:, 0, 0, :] + pos[0]
    np.testing.assert_allclose(svecs_nacl_ref10, svecs[1, 0, :2], atol=1e-8)
    np.testing.assert_allclose(svecs_nacl_ref30, svecs[3, 0, :4], atol=1e-8)
    helper_methods.compare_positions_with_order(pos_from_svecs, pos, scell.cell)


def test_ShortestPairs_dense_nacl(ph_nacl: Phonopy, helper_methods: Callable):
    """Test ShortestPairs (dense) by NaCl."""
    scell = ph_nacl.supercell
    pcell = ph_nacl.primitive
    pos = scell.scaled_positions
    spairs = ShortestPairs(scell.cell, pos, pos[pcell.p2s_map], store_dense_svecs=True)
    svecs = spairs.shortest_vectors
    multi = spairs.multiplicities
    assert multi[-1, -1, :].sum() == multi[:, :, 0].sum()
    np.testing.assert_array_equal(multi[:, :, 0].ravel(), multi_nacl_ref)
    np.testing.assert_allclose(
        svecs_nacl_ref10, svecs[multi[1, 0, 1] : multi[1, 0, :].sum()], atol=1e-8
    )
    np.testing.assert_allclose(
        svecs_nacl_ref30, svecs[multi[3, 0, 1] : multi[3, 0, :].sum()], atol=1e-8
    )
    pos_from_svecs = svecs[multi[:, 0, 1], :] + pos[0]
    helper_methods.compare_positions_with_order(pos_from_svecs, pos, scell.cell)


def test_sparse_to_dense_and_dense_to_sparse_nacl(ph_nacl: Phonopy):
    """Test for sparse_to_dense_svecs and dense_to_sparse_svecs by NaCl."""
    _test_sparse_to_dense_and_dense_to_sparse(ph_nacl.supercell, ph_nacl.primitive)


def test_sparse_to_dense_and_dense_to_sparse_tipn3(ph_tipn3: Phonopy):
    """Test for sparse_to_dense_svecs and dense_to_sparse_svecs by TiPN3."""
    _test_sparse_to_dense_and_dense_to_sparse(ph_tipn3.supercell, ph_tipn3.primitive)


def test_sparse_to_dense_and_dense_to_sparse_al2o3(convcell_al2o3: PhonopyAtoms):
    """Test for sparse_to_dense_svecs and dense_to_sparse_svecs by Al2O3."""
    smat = np.diag([3, 3, 2])
    scell = get_supercell(convcell_al2o3, smat)

    pmat = np.diag([1.0 / 3, 1.0 / 3, 1.0 / 2]) @ get_primitive_matrix("R")
    pcell = get_primitive(scell, primitive_matrix=pmat)
    _test_sparse_to_dense_and_dense_to_sparse(scell, pcell)

    pmat = np.diag([1.0 / 3, 1.0 / 3, 1.0 / 2])
    pcell = get_primitive(scell, primitive_matrix=pmat)
    _test_sparse_to_dense_and_dense_to_sparse(scell, pcell)


def _test_sparse_to_dense_and_dense_to_sparse(scell: PhonopyAtoms, pcell: Primitive):
    pos = scell.scaled_positions

    spairs = ShortestPairs(scell.cell, pos, pos[pcell.p2s_map], store_dense_svecs=False)
    svecs = spairs.shortest_vectors
    multi = spairs.multiplicities

    spairs = ShortestPairs(scell.cell, pos, pos[pcell.p2s_map], store_dense_svecs=True)
    dsvecs = spairs.shortest_vectors
    dmulti = spairs.multiplicities

    _dsvecs, _dmulti = sparse_to_dense_svecs(svecs, multi)

    np.testing.assert_array_equal(dmulti, _dmulti)
    np.testing.assert_allclose(dsvecs, _dsvecs, rtol=0, atol=1e-8)

    _ssvecs, _smulti = dense_to_sparse_svecs(dsvecs, dmulti)

    np.testing.assert_array_equal(multi, _smulti)
    np.testing.assert_allclose(svecs, _ssvecs, rtol=0, atol=1e-8)


def test_isclose(ph_nacl: Phonopy):
    """Test of isclose wit same order of atoms.."""
    scell = ph_nacl.supercell
    pcell = ph_nacl.primitive
    assert isclose(pcell, pcell)
    assert isclose(scell, scell)
    assert not isclose(scell, pcell)


def test_isclose_with_arbitrary_order(
    nacl_unitcell_order1: PhonopyAtoms, nacl_unitcell_order2: PhonopyAtoms
):
    """Test of isclose with different order."""
    cell1 = nacl_unitcell_order1
    cell2 = nacl_unitcell_order2
    assert not isclose(cell1, cell2)
    _isclose = isclose(cell1, cell2, with_arbitrary_order=True)
    assert isinstance(_isclose, bool)
    assert _isclose
    order = isclose(cell1, cell2, with_arbitrary_order=True, return_order=True)
    np.testing.assert_array_equal(order, [0, 4, 1, 5, 2, 6, 3, 7])


def test_convert_to_phonopy_primitive(ph_nacl: Phonopy):
    """Test for convert_to_phonopy_primitive."""
    scell = ph_nacl.supercell
    pcell = ph_nacl.primitive
    _pcell = convert_to_phonopy_primitive(scell, pcell)
    assert isclose(pcell, _pcell)

    # Changing order of atoms is not allowed.
    points = pcell.scaled_positions[[1, 0]]
    symbols = [pcell.symbols[i] for i in (1, 0)]
    cell = pcell.cell
    pcell_mode = PhonopyAtoms(cell=cell, scaled_positions=points, symbols=symbols)
    with pytest.raises(RuntimeError):
        _pcell = convert_to_phonopy_primitive(scell, pcell_mode)


def test_get_cell_matrix_from_lattice(primcell_nacl: PhonopyAtoms):
    """Test for test_get_cell_matrix_from_lattice."""
    pcell = primcell_nacl
    lattice = get_cell_matrix_from_lattice(pcell.cell)
    np.testing.assert_allclose(
        get_angles(lattice, is_radian=False),
        get_angles(pcell.cell, is_radian=False),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        get_angles(lattice, is_radian=True),
        get_angles(pcell.cell, is_radian=True),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        get_cell_parameters(lattice), get_cell_parameters(pcell.cell), atol=1e-8
    )
    np.testing.assert_allclose(
        [
            [4.02365076, 0.0, 0.0],
            [2.01182538, 3.48458377, 0.0],
            [2.01182538, 1.16152792, 3.28529709],
        ],
        lattice,
        atol=1e-7,
    )


def test_get_supercell_with_Xn_symbol(ph_nacl: Phonopy):
    """Test of get_supercell with Xn symbol."""
    symbols = ph_nacl.unitcell.symbols
    symbols[-1] = "Cl1"
    masses = ph_nacl.unitcell.masses
    masses[-1] = 70.0
    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
        masses=masses,
    )
    scell = get_supercell(cell, np.diag([2, 2, 2]))
    assert scell.symbols[-8:] == ["Cl1"] * 8
    np.testing.assert_allclose(scell.masses[-8:], [70.0] * 8)


def test_get_primitive_with_Xn_symbol(ph_nacl: Phonopy):
    """Test of get_primitive with Xn symbol.

    Symbols with index breaks symmetry to make primitive cell.

    Can not make primitive cell like:
    ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl1"] -> ["Na", "Cl"]

    """
    symbols = ph_nacl.unitcell.symbols
    symbols[-1] = "Cl1"
    masses = ph_nacl.unitcell.masses
    masses[-1] = 70.0
    cell = PhonopyAtoms(
        cell=ph_nacl.unitcell.cell,
        scaled_positions=ph_nacl.unitcell.scaled_positions,
        symbols=symbols,
        masses=masses,
    )
    with pytest.raises(RuntimeError) as e:
        get_primitive(cell, primitive_matrix="F")
    assert str(e.value).split("\n")[0] == "Atom symbol mapping failure."
