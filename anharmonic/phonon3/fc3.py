import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, set_permutation_symmetry, set_translational_invariance_per_index, distribute_force_constants, solve_force_constants, get_rotated_displacement, get_positions_sent_by_rot_inv
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry
from anharmonic.file_IO import write_fc2_dat

def get_fc3(supercell,
            disp_dataset,
            fc2,
            symmetry,
            is_translational_symmetry=False,
            verbose=False):
    num_atom = supercell.get_number_of_atoms()
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype='double')


    _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry,
                         verbose)

    if verbose:
        print "----- Copying fc3 -----"

    first_disp_atoms = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()

    distribute_fc3(fc3,
                   first_disp_atoms,
                   lattice,
                   positions,
                   rotations,
                   translations,
                   symprec,
                   verbose)

    if is_translational_symmetry:
        set_translational_invariance_fc3_per_index(fc3)

    return fc3

def distribute_fc3(fc3,
                   first_disp_atoms,
                   lattice,
                   positions,
                   rotations,
                   translations,
                   symprec,
                   verbose):
    num_atom = len(positions)

    for i in range(num_atom):
        if i in first_disp_atoms:
            continue

        for atom_index_done in first_disp_atoms:
            rot_num = get_atom_mapping_by_symmetry(positions,
                                                   i,
                                                   atom_index_done,
                                                   rotations,
                                                   translations,
                                                   symprec)
            if rot_num > -1:
                i_rot = atom_index_done
                rot = rotations[rot_num]
                trans = translations[rot_num]
                break

        if rot_num < 0:
            print "Position or symmetry may be wrong."
            raise ValueError

        if verbose > 2:
            print "  [ %d, x, x ] to [ %d, x, x ]" % (i_rot + 1, i + 1)
            sys.stdout.flush()

        atom_mapping = np.zeros(num_atom, dtype='intc')
        for j in range(num_atom):
            atom_mapping[j] = get_atom_by_symmetry(positions,
                                                   rot,
                                                   trans,
                                                   j,
                                                   symprec)
        rot_cart_inv = np.double(
            similarity_transformation(lattice, rot).T.copy())

        try:
            import anharmonic._phono3py as phono3c
            phono3c.distribute_fc3(fc3,
                                   i,
                                   atom_mapping,
                                   rot_cart_inv)
        
        except ImportError:
            for j in range(num_atom):
                j_rot = atom_mapping[j]
                for k in range(num_atom):
                    k_rot = atom_mapping[k]
                    fc3[i, j, k] = third_rank_tensor_rotation(
                        rot_cart_inv, fc3[i_rot, j_rot, k_rot])

def symmetrize_fc3(fc3):
    num_atom = fc3.shape[0]
    fc3_sym = np.zeros(fc3.shape, dtype='double')
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                fc3_sym[i, j, k] = symmetrize_fc3_part(fc3, i, j, k)

    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                fc3[i, j, k] = fc3_sym[i, j, k]

def symmetrize_fc3_part(fc3, a, b, c):
    tensor3 = np.zeros((3,3,3), dtype='double')
    for i in range(3):
        for j in range(3):
            for k in range(3):
                tensor3[i, j, k] = (fc3[a, b, c, i, j, k] +
                                    fc3[c, a, b, k, i, j] +
                                    fc3[b, c, a, j, k, i] +
                                    fc3[a, c, b, i, k, j] +
                                    fc3[b, a, c, j, i, k] +
                                    fc3[c, b, a, k, j, i]) / 6
    return tensor3

def set_translational_invariance_fc3(fc3):
    for i in range(3):
        set_translational_invariance_fc3_per_index(fc3, index=i)

def set_translational_invariance_fc3_per_index(fc3, index=0):
    for i in range(fc3.shape[(1 + index) % 3]):
        for j in range(fc3.shape[(2 + index) % 3]):
            for k in range(fc3.shape[3]):
                for l in range(fc3.shape[4]):
                    for m in range(fc3.shape[5]):
                        if index == 0:
                            fc3[:, i, j, k, l, m] -= np.sum(
                                fc3[:, i, j, k, l, m]) / fc3.shape[0]
                        elif index == 1:
                            fc3[j, :, i, k, l, m] -= np.sum(
                                fc3[j, :, i, k, l, m]) / fc3.shape[1]
                        elif index == 2:
                            fc3[i, j, :, k, l, m] -= np.sum(
                                fc3[i, j, :, k, l, m]) / fc3.shape[2]
    
def third_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3,3,3), dtype='double')
    for i in (0,1,2):
        for j in (0,1,2):
            for k in (0,1,2):
                rot_tensor[i, j, k] = _third_rank_tensor_rotation_elem(
                    rot_cart, tensor, i, j, k)
    return rot_tensor

def _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry,
                         verbose):
    symprec = symmetry.get_symmetry_tolerance()
    unique_first_atom_nums = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    for first_atom_num in unique_first_atom_nums:
        _get_fc3_one_atom(fc3,
                          supercell,
                          disp_dataset,
                          fc2,
                          first_atom_num,
                          symmetry.get_site_symmetry(first_atom_num),
                          is_translational_symmetry,
                          symprec,
                          verbose)

def _get_fc3_one_atom(fc3,
                      supercell,
                      disp_dataset,
                      fc2,
                      first_atom_num,
                      site_symmetry,
                      is_translational_symmetry,
                      symprec,
                      verbose):
    displacements_first = []
    delta_fc2s = []
    for dataset_first_atom in disp_dataset['first_atoms']:
        if first_atom_num != dataset_first_atom['number']:
            continue
        
        displacements_first.append(dataset_first_atom['displacement'])
        if 'delta_fc2' in dataset_first_atom:
            delta_fc2s.append(dataset_first_atom['delta_fc2'])
        else:
            direction = np.dot(dataset_first_atom['displacement'],
                               np.linalg.inv(supercell.get_cell()))
            reduced_site_sym = get_reduced_site_symmetry(
                site_symmetry, direction, symprec)
            delta_fc2s.append(_get_delta_fc2(
                    dataset_first_atom,
                    fc2,
                    supercell,
                    reduced_site_sym,
                    is_translational_symmetry,
                    symprec))

    solve_fc3(fc3,
              first_atom_num,
              supercell,
              site_symmetry,
              displacements_first,
              delta_fc2s,
              symprec)

    if verbose:
        print "Displacements for fc3[ %d, x, x ]" % (first_atom_num + 1)
        for i, v in enumerate(displacements_first):
            print "  [%7.4f %7.4f %7.4f]" % tuple(v)
            sys.stdout.flush()
        if verbose > 2:
            print "Site symmetry:"
            for i, v in enumerate(site_symmetry):
                print "  [%2d %2d %2d] #%2d" % tuple(list(v[0])+[i+1])
                print "  [%2d %2d %2d]" % tuple(v[1])
                print "  [%2d %2d %2d]\n" % tuple(v[2])
                sys.stdout.flush()

def get_atom_by_symmetry(positions,
                         rotation,
                         trans,
                         atom_number,
                         symprec=1e-5):

    rot_pos = np.dot(positions[atom_number], rotation.T) + trans
    for i, pos in enumerate(positions):
        diff = pos - rot_pos
        if (abs(diff -diff.round()) < symprec).all():
            return i

    print 'Position or symmetry is wrong.'
    raise ValueError

def get_atom_mapping_by_symmetry(positions,
                                 atom_search, 
                                 atom_target,
                                 rotations,
                                 translations,
                                 symprec):
    map_sym = -1

    for i, (r, t) in enumerate(zip(rotations, translations)):
        rot_pos = np.dot(positions[atom_search], r.T) + t
        diff = rot_pos - positions[atom_target]
        if (abs(diff -diff.round()) < symprec).all():
            map_sym = i
            break

    return map_sym

def _get_delta_fc2(dataset_first_atom,
                   fc2,
                   supercell,
                   reduced_site_sym,
                   is_translational_symmetry,
                   symprec):
    disp_fc2 = _get_constrained_fc2(supercell,
                                    dataset_first_atom,
                                    reduced_site_sym,
                                    symprec)
    if is_translational_symmetry:
        set_translational_invariance_per_index(disp_fc2)
            
    return disp_fc2 - fc2

def _get_constrained_fc2(supercell,
                         displacements,
                         reduced_site_sym,
                         symprec):
    """
    displacements = {'number': 3,
                     'displacement': [0.01, 0.00, 0.01]
                     'second_atoms': [{'number': 7,
                                       'displacements': [[]],
                                       'delta_forces': []}]}

    number: Atomic index, starting with 0.
    displacement: displacement of 1st displaced atom in Cartesian.
    displacements: displacements of 2st displaced atom in Cartesian.
    delta_forces: diff. of 2 atomic disp. forces and 1 atomic disp. forces
    Number of elements of 'displacements' and 'delta_forces' are same.
    """
    num_atom = supercell.get_number_of_atoms()
    atom1 = displacements['number']
    fc2 = np.zeros((num_atom, num_atom, 3, 3), dtype='double')
    atom_list_done = []
    for disps_second in displacements['second_atoms']:
        disps2 = disps_second['displacements']
        atom2 = disps_second['number']
        sets_of_forces = disps_second['delta_forces']
        atom_list_done.append(atom2)
        bond_sym = get_bond_symmetry(
            reduced_site_sym,
            supercell.get_scaled_positions(),
            atom1,
            atom2,
            symprec)

        solve_force_constants(fc2,
                              atom2,
                              disps2,
                              sets_of_forces,
                              supercell,
                              bond_sym,
                              symprec)

    # Shift positions according to set atom1 is at origin
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[atom1].copy()
    positions -= pos_center
    atom_list = range(num_atom)
    distribute_force_constants(fc2,
                               atom_list,
                               atom_list_done,
                               lattice,
                               positions,
                               np.intc(reduced_site_sym).copy(),
                               np.zeros((len(reduced_site_sym), 3),
                                        dtype='double'),
                               symprec)
    return fc2
        

def solve_fc3(fc3,
              first_atom_num,
              supercell,
              site_symmetry,
              displacements_first,
              delta_fc2s,
              symprec):
    lattice = supercell.get_cell().T
    site_sym_cart = [similarity_transformation(lattice, sym)
                     for sym in site_symmetry]
    num_atom = supercell.get_number_of_atoms()
    positions = supercell.get_scaled_positions()
    pos_center = positions[first_atom_num].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                 site_symmetry,
                                                 symprec)
    
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)
    inv_U = np.linalg.pinv(rot_disps)
    for (i, j) in list(np.ndindex(num_atom, num_atom)):
        fc3[first_atom_num, i, j] = np.dot(inv_U, _get_rotated_fc2s(
                i, j, delta_fc2s, rot_map_syms, site_sym_cart)).reshape(3, 3, 3)

def show_drift_fc3(fc3, name="fc3"):
    num_atom = fc3.shape[0]
    maxval1 = 0
    maxval2 = 0
    maxval3 = 0
    for i, j, k, l, m in list(np.ndindex((num_atom, num_atom, 3, 3, 3))):
        val1 = fc3[:, i, j, k, l, m].sum()
        val2 = fc3[i, :, j, k, l, m].sum()
        val3 = fc3[i, j, :, k, l, m].sum()
        if abs(val1) > abs(maxval1):
            maxval1 = val1
        if abs(val2) > abs(maxval2):
            maxval2 = val2
        if abs(val3) > abs(maxval3):
            maxval3 = val3
    print ("max drift of %s:" % name), maxval1, maxval2, maxval3 
        
def _get_rotated_fc2s(i, j, fc2s, rot_map_syms, site_sym_cart):
    num_sym = len(site_sym_cart)
    rotated_fc2s = []
    for fc2 in fc2s:
        for sym, map_sym in zip(site_sym_cart, rot_map_syms):
            fc2_rot = fc2[map_sym[i], map_sym[j]]
            rotated_fc2s.append(similarity_transformation(sym, fc2_rot))
    return np.reshape(rotated_fc2s, (-1, 9))
            
def _third_rank_tensor_rotation_elem(rot, tensor, l, m, n):
    sum_elems = 0.
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                sum_elems += rot[l, i] * rot[m, j] * rot[n, k] * tensor[i, j, k]
    return sum_elems

