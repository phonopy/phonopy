import numpy as np
from phonopy.structure.symmetry import (
    Symmetry, symmetrize_borns_and_epsilon, _get_mapping_between_cells,
    collect_unique_rotations)
from phonopy.structure.cells import get_supercell
import os

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_map_operations(convcell_nacl):
    symprec = 1e-5
    cell = convcell_nacl
    scell = get_supercell(cell, np.diag([2, 2, 2]), symprec=symprec)
    symmetry = Symmetry(scell, symprec=symprec)
    map_ops = symmetry.get_map_operations().copy()
    # start = time.time()
    # symmetry._set_map_operations()
    # end = time.time()
    # print(end - start)
    # map_ops_old = symmetry.get_map_operations().copy()
    # assert (map_ops == map_ops_old).all()
    map_atoms = symmetry.get_map_atoms()
    positions = scell.scaled_positions
    rotations = symmetry.symmetry_operations['rotations']
    translations = symmetry.symmetry_operations['translations']
    for i, (op_i, atom_i) in enumerate(zip(map_ops, map_atoms)):
        r_pos = np.dot(rotations[op_i], positions[i]) + translations[op_i]
        diff = positions[atom_i] - r_pos
        diff -= np.rint(diff)
        assert (diff < symprec).all()


def test_magmom(convcell_cr):
    symprec = 1e-5
    cell = convcell_cr
    symmetry_nonspin = Symmetry(cell, symprec=symprec)
    atom_map_nonspin = symmetry_nonspin.get_map_atoms()
    len_sym_nonspin = len(
        symmetry_nonspin.get_symmetry_operations()['rotations'])

    spin = [1, -1]
    cell_withspin = cell.copy()
    cell_withspin.set_magnetic_moments(spin)
    symmetry_withspin = Symmetry(cell_withspin, symprec=symprec)
    atom_map_withspin = symmetry_withspin.get_map_atoms()
    len_sym_withspin = len(
        symmetry_withspin.get_symmetry_operations()['rotations'])

    broken_spin = [1, -2]
    cell_brokenspin = cell.copy()
    cell_brokenspin = cell.copy()
    cell_brokenspin.set_magnetic_moments(broken_spin)
    symmetry_brokenspin = Symmetry(cell_brokenspin, symprec=symprec)
    atom_map_brokenspin = symmetry_brokenspin.get_map_atoms()
    len_sym_brokenspin = len(
        symmetry_brokenspin.get_symmetry_operations()['rotations'])

    assert (atom_map_nonspin == atom_map_withspin).all()
    assert (atom_map_nonspin != atom_map_brokenspin).any()
    assert len_sym_nonspin == len_sym_withspin
    assert len_sym_nonspin != len_sym_brokenspin


def test_symmetrize_borns_and_epsilon_nacl(ph_nacl):
    nac_params = ph_nacl.nac_params
    borns, epsilon = symmetrize_borns_and_epsilon(
        nac_params['born'], nac_params['dielectric'], ph_nacl.primitive)
    np.testing.assert_allclose(borns, nac_params['born'], atol=1e-8)
    np.testing.assert_allclose(epsilon, nac_params['dielectric'], atol=1e-8)


def test_symmetrize_borns_and_epsilon_tio2(ph_tio2):
    nac_params = ph_tio2.nac_params
    borns, epsilon = symmetrize_borns_and_epsilon(
        nac_params['born'], nac_params['dielectric'], ph_tio2.primitive)
    # np.testing.assert_allclose(borns, nac_params['born'], atol=1e-8)
    np.testing.assert_allclose(epsilon, nac_params['dielectric'], atol=1e-8)


def test_Symmetry_pointgroup(ph_tio2):
    assert ph_tio2.symmetry.pointgroup_symbol == r'4/mmm'


def test_with_pmat_and_smat(ph_nacl):
    pcell = ph_nacl.primitive
    scell = ph_nacl.supercell
    idx = [scell.u2u_map[i] for i in scell.s2u_map[pcell.p2s_map]]
    uborns, uepsilon = _get_nac_params_in_unitcell(ph_nacl)
    borns, epsilon = symmetrize_borns_and_epsilon(
        uborns,
        uepsilon,
        ph_nacl.unitcell,
        primitive_matrix=ph_nacl.primitive_matrix,
        supercell_matrix=ph_nacl.supercell_matrix)
    np.testing.assert_allclose(borns, uborns[idx], atol=1e-8)
    np.testing.assert_allclose(epsilon, uepsilon, atol=1e-8)


def test_with_pcell(ph_nacl):
    pcell = ph_nacl.primitive
    scell = ph_nacl.supercell
    idx = [scell.u2u_map[i] for i in scell.s2u_map[pcell.p2s_map]]
    idx2 = _get_mapping_between_cells(pcell, pcell)
    np.testing.assert_array_equal(idx2, np.arange(len(pcell)))

    uborns, uepsilon = _get_nac_params_in_unitcell(ph_nacl)
    borns, epsilon = symmetrize_borns_and_epsilon(
        uborns, uepsilon, ph_nacl.unitcell, primitive=pcell)
    np.testing.assert_allclose(borns, uborns[idx][idx2], atol=1e-8)
    np.testing.assert_allclose(epsilon, uepsilon, atol=1e-8)


def test_site_symmetry(ph_sno2):
    site_sym0 = ph_sno2.symmetry.get_site_symmetry(0)
    ref0 = [1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, -1, 0, -1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 1, 0, 0, 0, -1,
            0, -1, 0, -1, 0, 0, 0, 0, -1]
    site_sym36 = ph_sno2.symmetry.get_site_symmetry(36)
    ref36 = [1, 0, 0, 0, 1, 0, 0, 0, 1,
             1, 0, 0, 0, 1, 0, 0, 0, -1,
             0, 1, 0, 1, 0, 0, 0, 0, 1,
             0, 1, 0, 1, 0, 0, 0, 0, -1]
    np.testing.assert_array_equal(site_sym0.ravel(), ref0)
    np.testing.assert_array_equal(site_sym36.ravel(), ref36)


def test_collect__unique_rotations(ph_nacl):
    rotations = ph_nacl.symmetry.symmetry_operations['rotations']
    ptg = collect_unique_rotations(rotations)
    assert len(rotations) == 1536
    assert len(ptg) == 48
    assert len(ptg) == len(ph_nacl.symmetry.pointgroup_operations)


def test_reciprocal_operations(ph_zr3n4):
    """Zr3N4 is a non-centrosymmetric crystal"""
    ptg = ph_zr3n4.symmetry.pointgroup_operations
    rops = ph_zr3n4.symmetry.reciprocal_operations
    matches = []
    for r in ptg:
        for i, rec_r in enumerate(rops):
            if (r == rec_r).all():
                matches.append(i)
                break
    assert len(np.unique(matches)) == len(ptg)
    found_inv = False
    for rec_r in rops:
        if (rec_r == -np.eye(3, dtype=int)).all():
            found_inv = True
            break
    assert found_inv


def _get_nac_params_in_unitcell(ph):
    nac_params = ph.nac_params
    uepsilon = nac_params['dielectric']
    pborns = nac_params['born']
    s2p_map = ph.primitive.s2p_map
    p2p_map = ph.primitive.p2p_map
    s2pp_map = [p2p_map[i] for i in s2p_map]
    sborns = pborns[s2pp_map]
    uborns = np.array([sborns[i] for i in ph.supercell.u2s_map])

    return uborns, uepsilon
