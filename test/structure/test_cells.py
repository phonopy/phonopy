"""Tests of routines in cells.py."""

import os
import warnings
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
    build_mixture_cell,
    compute_all_sg_permutations,
    compute_permutation_for_rotation,
    dense_to_sparse_svecs,
    get_angles,
    get_cell_matrix_from_lattice,
    get_cell_parameters,
    get_primitive,
    get_primitive_matrix,
    get_supercell,
    guess_primitive_matrix,
    isclose,
    sparse_to_dense_svecs,
)
from phonopy.structure.mixture import get_mixture_expansion

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
    for i, (r, t) in enumerate(zip(rots, trans, strict=True)):
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


def test_guess_primitive_matrix_Cr_type_iv(convcell_cr: PhonopyAtoms):
    """AFM bcc Cr is Type-IV; XSG primitive matrix is returned with warning."""
    convcell_cr.magnetic_moments = [1, -1]
    try:
        with pytest.warns(UserWarning, match="magnetic"):
            pmat = guess_primitive_matrix(convcell_cr)
    finally:
        convcell_cr.magnetic_moments = None
    # By XSG as Pm-3m, the magnetic ordering is preserved.
    np.testing.assert_allclose(abs(np.linalg.det(pmat)), 1.0, atol=1e-8)


def test_guess_primitive_matrix_Cr_fm_no_warning(convcell_cr: PhonopyAtoms):
    """FM bcc Cr is not Type-IV; no warning is emitted."""
    convcell_cr.magnetic_moments = [1, 1]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pmat = guess_primitive_matrix(convcell_cr)
    finally:
        convcell_cr.magnetic_moments = None
    np.testing.assert_allclose(abs(np.linalg.det(pmat)), 0.5, atol=1e-8)


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
    smat2 = np.eye(3, dtype="int64") * 2
    pmat2 = np.dot(np.linalg.inv(smat2), pmat)
    smat3 = np.eye(3, dtype="int64") * 3
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


def test_guess_primitive_matrix_distinguish_symbol_index():
    """Auto primitive matrix can respect suffixed-symbol species on demand.

    By default the suffix is a calculator label: spglib sees plain Cu,
    the guessed matrix reduces to one atom, and the species-aware
    primitive construction refuses it. With
    distinguish_symbol_index=True the guessed matrix preserves the
    Cu1/Cu2 grouping (CuAu-I like ordering: 2-atom primitive cell).

    """
    a = 3.6
    cell = PhonopyAtoms(
        symbols=["Cu1", "Cu1", "Cu2", "Cu2"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        cell=np.eye(3) * a,
    )

    pmat_default = guess_primitive_matrix(cell)
    with pytest.raises(RuntimeError) as e:
        get_primitive(cell, pmat_default)
    assert str(e.value).split("\n")[0] == "Atom symbol mapping failure."

    pmat = guess_primitive_matrix(cell, distinguish_symbol_index=True)
    primitive = get_primitive(cell, pmat)
    assert len(primitive) == 2
    assert primitive.symbols == ["Cu1", "Cu2"]

    ph = Phonopy(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix="auto",
        distinguish_symbol_index=True,
    )
    assert len(ph.primitive) == 2
    assert ph.symmetry.dataset.number == 123  # P4/mmm


def test_build_mixture_cell_GeSn():
    """Two overlapping Ge/Sn pairs collapse into two GeSn mixture sites."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])

    assert len(mixed_cell) == 2
    assert mixed_cell.has_mixtures
    assert mixed_cell.symbols == ["GeSn", "GeSn"]
    np.testing.assert_allclose(
        mixed_cell.scaled_positions, [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    )
    np.testing.assert_allclose(mixed_cell.species_ids, [0, 0])
    expected_mass = 0.5 * 72.64 + 0.5 * 118.71
    np.testing.assert_allclose(mixed_cell.masses, [expected_mass, expected_mass])


def test_build_mixture_cell_distinct_mixtures_get_suffixes():
    """When two distinct mixtures share a composite label, suffixes are added."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        symbols=["Ge", "Sn", "Ge", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.25, 0.75])

    assert mixed_cell.has_mixtures
    assert mixed_cell.symbols == ["GeSn1", "GeSn2"]


def test_build_mixture_cell_unique_composite_no_suffix():
    """A single GeSn mixture in the cell stays unsuffixed."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        symbols=["Ge", "Sn", "Si"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 1.0])

    assert mixed_cell.symbols == ["GeSn", "Si"]


def test_build_mixture_cell_weight_sum_error():
    """Group weights must sum to 1.0."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        symbols=["Ge", "Sn"],
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_mixture_cell(cell, [0.4, 0.4])


def test_build_mixture_cell_isolated_atom_weight_must_be_one():
    """A non-overlapping atom must carry weight 1.0."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        symbols=["Ge", "Sn"],
    )
    with pytest.raises(ValueError, match="must be 1.0"):
        build_mixture_cell(cell, [1.0, 0.5])


def test_build_mixture_cell_length_mismatch():
    """Weights length must match natoms."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0]],
        symbols=["Si"],
    )
    with pytest.raises(ValueError, match="must match number of atoms"):
        build_mixture_cell(cell, [1.0, 0.0])


def test_build_mixture_cell_rejects_already_mixed_cell():
    """Re-applying build_mixture_cell on a mixed cell raises."""
    a = 4.0
    cell = PhonopyAtoms(
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]],
        scaled_positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        symbols=["Ge", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5])
    with pytest.raises(ValueError, match="already contains"):
        build_mixture_cell(mixed_cell, [1.0])


def test_build_mixture_cell_supercell_through_phonopy(ph_nacl: Phonopy):
    """A mixed unitcell flows through Phonopy and produces a supercell with mixtures."""
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed_cell = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])
    ph = Phonopy(mixed_cell, supercell_matrix=np.diag([2, 2, 2]))
    assert ph.supercell.has_mixtures
    assert ph.primitive.has_mixtures
    assert len(ph.supercell) == 16


def test_GeSn_mixture_force_constants_e2e():
    """End-to-end: GeSn 50/50 supercell with raw expanded forces builds FC.

    Construct the canonical GeSn 50/50 zincblende cell, generate
    displacements through Phonopy, plug in synthetic *expanded* forces
    of shape (num_supercells, n_expanded, 3) (the shape that VASP would
    emit on a mixture-expanded SPOSCAR), and confirm produce_force_constants
    runs end-to-end. The resulting FC has per-site shape, demonstrating
    the FC-time reduction of raw forces.

    """
    a = 2.82173
    cell = PhonopyAtoms(
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        symbols=["Ge", "Ge", "Sn", "Sn"],
    )
    mixed = build_mixture_cell(cell, [0.5, 0.5, 0.5, 0.5])
    ph = Phonopy(mixed, supercell_matrix=np.diag([2, 2, 2]))
    ph.generate_displacements(distance=0.01)
    n_sites = len(ph.supercell)
    site_indices, _ = get_mixture_expansion(ph.supercell)
    n_expanded = int(site_indices.size)
    assert n_expanded == 2 * n_sites

    rng = np.random.default_rng(seed=42)
    dataset = ph.dataset
    assert dataset is not None
    for entry in dataset["first_atoms"]:
        forces = rng.standard_normal((n_expanded, 3)) * 1e-3
        forces -= forces.mean(axis=0)  # zero net force
        entry["forces"] = forces

    ph.dataset = dataset
    ph.produce_force_constants()

    fc = ph.force_constants
    assert fc is not None
    assert fc.shape == (n_sites, n_sites, 3, 3)
    assert ph.dataset["first_atoms"][0]["forces"].shape == (n_expanded, 3)
