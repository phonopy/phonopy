import sys
import numpy as np
from anharmonic.fc2 import get_restricted_fc2
from phonopy.harmonic.force_constants import similarity_transformation, set_permutation_symmetry, set_translational_invariance
from anharmonic.displacement_fc3 import get_reduced_site_symmetry
from anharmonic.file_IO import write_fc2_dat

def get_fc3(supercell,
            disp_dataset,
            fc2,
            symmetry,
            is_translational_symmetry=False,
            verbose=False):
    num_atom = supercell.get_number_of_atoms()
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype=float)

    _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry,
                         verbose)

    first_disp_atoms = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    distribute_fc3(fc3,
                   first_disp_atoms,
                   supercell,
                   symmetry,
                   verbose=verbose)

    if is_translational_symmetry:
        set_translational_invariance_fc3(fc3)

    return fc3

def symmetrize_fc3(fc3):
    num_atom = fc3.shape[0]
    fc3_sym = np.zeros(fc3.shape, dtype=float)
    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                fc3_sym[i, j, k] = symmetrize_fc3_part(fc3, i, j, k)

    for i in range(num_atom):
        for j in range(num_atom):
            for k in range(num_atom):
                fc3[i, j, k] = fc3_sym[i, j, k]

def symmetrize_fc3_part(fc3, a, b, c):
    tensor3 = np.zeros((3,3,3), dtype=float)
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
    for i in range(fc3.shape[1]):
        for j in range(fc3.shape[2]):
            for k in range(fc3.shape[3]):
                for l in range(fc3.shape[4]):
                    for m in range(fc3.shape[5]):
                        fc3[:, i, j, k, l, m] -= np.sum(
                            fc3[:, i, j, k, l, m]) / fc3.shape[0]
    
def distribute_fc3(fc3,
                   first_disp_atoms,
                   supercell,
                   symmetry,
                   verbose=False):

    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()
    lattice = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    num_atoms = supercell.get_number_of_atoms()

    if verbose:
        print "----- Copying fc3 -----"

    for i in range(num_atoms):
        if i in first_disp_atoms:
            continue

        for atom_index_done in first_disp_atoms:
            rot_num = get_atom_mapping_by_symmetry(positions,
                                                   i,
                                                   atom_index_done,
                                                   symmetry)
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


        rot_cart_inv = similarity_transformation(lattice.T, rot).T

        try:
            import anharmonic._phono3py as phono3c
            phono3c.distribute_fc3(fc3,
                                   i,
                                   i_rot,
                                   positions,
                                   rot,
                                   rot_cart_inv.copy(),
                                   trans,
                                   symprec)
        
        except ImportError:
            for j in range(num_atoms):
                j_rot = get_atom_by_symmetry(positions,
                                             rot,
                                             trans,
                                             j,
                                             symprec)
                for k in range(num_atoms):
                    k_rot = get_atom_by_symmetry(positions,
                                                 rot,
                                                 trans,
                                                 k,
                                                 symprec)
                    fc3[i, j, k] = _third_rank_tensor_rotation(
                        rot_cart_inv, fc3[i_rot, j_rot, k_rot])

def get_fc3_one_atom(fc3,
                     supercell,
                     disp_dataset,
                     fc2,
                     first_atom_num,
                     site_symmetry,
                     is_translational_symmetry=False,
                     symprec=1e-5,
                     verbose=False):
    displacements_first = []
    delta_fc2s = []
    for dataset_first_atom in disp_dataset['first_atoms']:
        if first_atom_num == dataset_first_atom['number']:
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

    _solve_fc3(fc3,
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

    for i, v in enumerate(delta_fc2s):
        filename = "delta_fc2-%d-%d.dat" % (first_atom_num + 1, i + 1)
        write_fc2_dat(v, filename=filename)
        
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
                                 symmetry):
    map_sym = -1
    rotations = symmetry.get_symmetry_operations()['rotations']
    trans = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()

    for i, (r, t) in enumerate(zip(rotations, trans)):
        rot_pos = np.dot(positions[atom_search], r.T) + t
        diff = rot_pos - positions[atom_target]
        if (abs(diff -diff.round()) < symprec).all():
            map_sym = i
            break

    return map_sym

def _get_fc3_least_atoms(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         symmetry,
                         is_translational_symmetry=False,
                         verbose=False):
    symprec = symmetry.get_symmetry_tolerance()
    unique_first_atom_nums = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    for first_atom_num in unique_first_atom_nums:
        get_fc3_one_atom(fc3,
                         supercell,
                         disp_dataset,
                         fc2,
                         first_atom_num,
                         symmetry.get_site_symmetry(first_atom_num),
                         is_translational_symmetry,
                         symprec,
                         verbose)

def _get_delta_fc2(dataset_first_atom,
                   fc2,
                   supercell,
                   reduced_site_sym,
                   is_translational_symmetry,
                   symprec=1e-5):
    disp_fc2 = get_restricted_fc2(supercell,
                                  dataset_first_atom,
                                  reduced_site_sym,
                                  symprec)
    if is_translational_symmetry:
        set_translational_invariance(disp_fc2)
            
    return disp_fc2 - fc2

def _solve_fc3(fc3,
               first_atom_num,
               supercell,
               site_sym,
               displacements_first,
               delta_fc2s,
               symprec=1e-5):
    num_atoms = supercell.get_number_of_atoms()
    lattice = supercell.get_cell()
    positions = supercell.get_scaled_positions()
    rot_disps = np.zeros((0, 27), dtype=float)

    for i, disp_cart in enumerate(displacements_first):
        for j, rot in enumerate(site_sym):
            rot_cart = similarity_transformation(lattice.T, rot)
	    rot_disps = np.vstack(
                (rot_disps, np.dot(_expand_displacement_third(disp_cart),
                                   _get_symmetry_matrices_third(rot_cart))))

    inv_U = np.linalg.pinv(rot_disps)

    for i in range(num_atoms):
        for j in range(num_atoms):
            delta_fc2_rot = np.zeros(
                (len(delta_fc2s), len(site_sym), 3, 3), dtype=float)

            for k, delta_fc2 in enumerate(delta_fc2s):
                for l, rot in enumerate(site_sym):
                    rot_atom_i = _get_atom_by_site_symmetry(positions,
                                                            rot,
                                                            i,
                                                            first_atom_num,
                                                            symprec)
                    rot_atom_j = _get_atom_by_site_symmetry(positions,
                                                            rot,
                                                            j,
                                                            first_atom_num,
                                                            symprec)
                    delta_fc2_rot[k, l] = delta_fc2[rot_atom_i, rot_atom_j]

            fc3[first_atom_num, i, j] = np.dot(
                inv_U, delta_fc2_rot.reshape(-1,1)).reshape(3, 3, 3)

def _get_atom_by_site_symmetry(positions,
                               rotation,
                               atom_num,
                               atom_center,
                               symprec=1e-5):

    rot_pos = np.dot(positions[atom_num] - positions[atom_center],
                     rotation.transpose()) + positions[atom_center]

    for i, pos in enumerate(positions):
        diff = pos - rot_pos
        if (abs(diff - diff.round()) < symprec).all():
            return i

    print 'Position or site symmetry is wrong.'
    raise ValueError

def _expand_displacement_third(displacement):
    """
    [dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0  0  0  0  0  0]
    [ 0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0  0  0  0  0]
    [ 0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0  0  0  0]
    [ 0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0  0  0]
    [ 0  0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0  0]
    [ 0  0  0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0  0]
    [ 0  0  0  0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0  0]
    [ 0  0  0  0  0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz  0]
    [ 0  0  0  0  0  0  0  0 dx  0  0  0  0  0  0  0  0 dy  0  0  0  0  0  0  0  0 dz]
    """
    d = displacement
    return np.hstack((np.eye(9) * d[0], np.eye(9) * d[1], np.eye(9) * d[2]))

def _get_symmetry_matrices_third(rotation_cartesian):
    """ Set 27x27 symmetry matricies """
    r = rotation_cartesian
    mat = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                psi = []
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            psi.append(r[i, l] * r[j, m] * r[k, n])
                mat.append(psi)
    return np.array(mat)

def _third_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3,3,3), dtype=float)
    for i in (0,1,2):
        for j in (0,1,2):
            for k in (0,1,2):
                rot_tensor[i, j, k] = _third_rank_tensor_rotation_elem(
                    rot_cart, tensor, i, j, k)
    return rot_tensor

def _third_rank_tensor_rotation_elem(rot, tensor, l, m, n):
    sum = 0.
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for k in (0, 1, 2):
                sum += rot[l, i] * rot[m, j] * rot[n, k] * tensor[i, j, k]
    return sum

