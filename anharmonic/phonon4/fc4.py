import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv, get_rotated_displacement
from phonopy.harmonic.force_constants import show_drift_force_constants
from anharmonic.phonon3.fc3 import set_translational_invariance_fc3_per_index, solve_fc3, distribute_fc3, third_rank_tensor_rotation, get_atom_mapping_by_symmetry, get_atom_by_symmetry, show_drift_fc3, set_permutation_symmetry_fc3, get_delta_fc2, get_constrained_fc2
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry
from phonopy.structure.symmetry import Symmetry

def get_fc4(supercell,
            disp_dataset,
            fc3,
            symmetry,
            translational_symmetry_type=0,
            is_permutation_symmetry=False,
            verbose=False):

    num_atom = supercell.get_number_of_atoms()
    fc4 = np.zeros((num_atom, num_atom, num_atom, num_atom,
                    3, 3, 3, 3), dtype='double')

    if verbose:
        print "----- Calculating fc4 -----"

    _get_fc4_least_atoms(fc4,
                         supercell,
                         disp_dataset,
                         fc3,
                         symmetry,
                         translational_symmetry_type,
                         is_permutation_symmetry,
                         verbose)

    if verbose:
        print "(Copying fc4...)"

    first_disp_atoms = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    rotations = symmetry.get_symmetry_operations()['rotations']
    translations = symmetry.get_symmetry_operations()['translations']
    symprec = symmetry.get_symmetry_tolerance()
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    distribute_fc4(fc4,
                   first_disp_atoms,
                   lattice,
                   positions,
                   rotations,
                   translations,
                   symprec,
                   verbose)

    if translational_symmetry_type > 0:
        set_translational_invariance_fc4_per_index(fc4)

    if is_permutation_symmetry:
        set_permutation_symmetry_fc4(fc4)

    return fc4
    
def set_translational_invariance_fc4(fc4):
    try:
        import anharmonic._phono4py as phono4c
        for i in range(4):
            phono4c.translational_invariance_fc4(fc4, i)
    except ImportError:
        for i in range(4):
            set_translational_invariance_fc4_per_index(fc4, index=i)

def set_translational_invariance_fc4_per_index(fc4, index=0):
    try:
        import anharmonic._phono4py as phono4c
        phono4c.translational_invariance_fc4(fc4, index)
    except ImportError:
        set_translational_invariance_fc4_per_index_py(fc4, index)
    
def set_translational_invariance_fc4_per_index_py(fc4, index=0):
    for i, j, k, l, m, n, p in list(np.ndindex(
            (fc4.shape[(1 + index) % 3],
             fc4.shape[(2 + index) % 3],
             fc4.shape[(3 + index) % 3]) + fc4.shape[4:])):
        if index == 0:
            fc4[:, i, j, k, l, m, n, p] -= np.sum(
                fc4[:, i, j, k, l, m, n, p]) / fc4.shape[0]
        elif index == 1:
            fc4[k, :, i, j, l, m, n, p] -= np.sum(
                fc4[j, :, i, k, l, m, n, p]) / fc4.shape[1]
        elif index == 2:
            fc4[j, k, :, i, l, m, n, p] -= np.sum(
                fc4[j, k, :, i, l, m, n, p]) / fc4.shape[2]
        elif index == 3:
            fc4[i, j, k, :, l, m, n, p] -= np.sum(
                fc4[i, j, k, :, l, m, n, p]) / fc4.shape[3]

def set_permutation_symmetry_fc4(fc4):
    try:
        import anharmonic._phono4py as phono4c
        phono4c.permutation_symmetry_fc4(fc4)
    except ImportError:
        num_atom = fc4.shape[0]
        fc4_sym = np.zeros(fc4.shape, dtype='double')
        for (i, j, k, l) in list(
            np.ndindex(num_atom, num_atom, num_atom, num_atom)):
            fc4_sym[i, j, k, l] = set_permutation_symmetry_fc4_part(
                fc4, i, j, k, l)
    
        for (i, j, k, l) in list(
            np.ndindex(num_atom, num_atom, num_atom, num_atom)):
            fc4[i, j, k, l] = fc4_sym[i, j, k, l]

def set_permutation_symmetry_fc4_part(fc4, a, b, c, d):
    tensor4 = np.zeros((3, 3, 3, 3), dtype='double')
    for (i, j, k, l) in list(np.ndindex(3, 3, 3, 3)):
        tensor4[i, j, k, l] = (
            fc4[a, b, c, d, i, j, k, l] +
            fc4[a, b, d, c, i, j, l, k] +
            fc4[a, c, b, d, i, k, j, l] +
            fc4[a, c, d, b, i, k, l, j] +
            fc4[a, d, b, c, i, l, j, k] +
            fc4[a, d, c, b, i, l, k, j] +
            fc4[b, a, c, d, j, i, k, l] +
            fc4[b, a, d, c, j, i, l, k] +
            fc4[b, c, a, d, j, k, i, l] +
            fc4[b, c, d, a, j, k, l, i] +
            fc4[b, d, a, c, j, l, i, k] +
            fc4[b, d, c, a, j, l, k, i] +
            fc4[c, a, b, d, k, i, j, l] +
            fc4[c, a, d, b, k, i, l, j] +
            fc4[c, b, a, d, k, j, i, l] +
            fc4[c, b, d, a, k, j, l, i] +
            fc4[c, d, a, b, k, l, i, j] +
            fc4[c, d, b, a, k, l, j, i] +
            fc4[d, a, b, c, l, i, j, k] +
            fc4[d, a, c, b, l, i, k, j] +
            fc4[d, b, a, c, l, j, i, k] +
            fc4[d, b, c, a, l, j, k, i] +
            fc4[d, c, a, b, l, k, i, j] +
            fc4[d, c, b, a, l, k, j, i]) / 24

    return tensor4

def show_drift_fc4(fc4, name="fc4"):
    try:
        import anharmonic._phono4py as phono4c
        (maxval1,
         maxval2,
         maxval3,
         maxval4) = phono4c.drift_fc4(fc4)
    except ImportError:
        num_atom = fc4.shape[0]
        maxval1 = 0
        maxval2 = 0
        maxval3 = 0
        maxval4 = 0
        for i, j, k, l, m, n, p in list(
            np.ndindex((num_atom, num_atom, num_atom, 3, 3, 3, 3))):
            val1 = fc4[:, i, j, k, l, m, n, p].sum()
            val2 = fc4[k, :, i, j, l, m, n, p].sum()
            val3 = fc4[j, k, :, i, l, m, n, p].sum()
            val4 = fc4[i, j, k, :, l, m, n, p].sum()
            if abs(val1) > abs(maxval1):
                maxval1 = val1
            if abs(val2) > abs(maxval2):
                maxval2 = val2
            if abs(val3) > abs(maxval3):
                maxval3 = val3
            if abs(val4) > abs(maxval4):
                maxval4 = val4
    
    print ("max drift of %s:" % name), maxval1, maxval2, maxval3, maxval4

def distribute_fc4(fc4,
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

        if verbose > 1:
            print "  [ %d, x, x, x ] to [ %d, x, x, x ]" % (i_rot + 1, i + 1)
            sys.stdout.flush()

        atom_mapping = np.zeros(num_atom, dtype='intc')
        for j in range(num_atom):
            atom_mapping[j] = get_atom_by_symmetry(positions,
                                                   rot,
                                                   trans,
                                                   j,
                                                   symprec)
            
        rot_cart_inv = np.array(
            similarity_transformation(lattice, rot).T,
            order='C', dtype='double')

        try:
            import anharmonic._phono4py as phono4c
            phono4c.distribute_fc4(fc4,
                                   i,
                                   atom_mapping,
                                   rot_cart_inv)
        
        except ImportError:
            for j in range(num_atom):
                j_rot = atom_mapping[j]
                for k in range(num_atom):
                    k_rot = atom_mapping[k]
                    for l in range(num_atom):
                        l_rot = atom_mapping[l]
                        fc4[i, j, k, l] = _fourth_rank_tensor_rotation(
                            rot_cart_inv, fc4[i_rot, j_rot, k_rot, l_rot])

def _get_fc4_least_atoms(fc4,
                         supercell,
                         disp_dataset,
                         fc3,
                         symmetry,
                         translational_symmetry_type,
                         is_permutation_symmetry,
                         verbose):
    symprec = symmetry.get_symmetry_tolerance()
    unique_first_atom_nums = np.unique(
        [x['number'] for x in disp_dataset['first_atoms']])
    for first_atom_num in unique_first_atom_nums:
        _get_fc4_one_atom(fc4,
                          supercell,
                          disp_dataset,
                          fc3,
                          first_atom_num,
                          symmetry.get_site_symmetry(first_atom_num),
                          translational_symmetry_type,
                          is_permutation_symmetry,
                          symprec,
                          verbose)

def _get_fc4_one_atom(fc4,
                      supercell,
                      disp_dataset,
                      fc3,
                      first_atom_num,
                      site_symmetry,
                      translational_symmetry_type,
                      is_permutation_symmetry,
                      symprec,
                      verbose):
    
    displacements_first = []
    delta_fc3s = []
    for dataset_first_atom in disp_dataset['first_atoms']:
        if first_atom_num != dataset_first_atom['number']:
            continue

        displacements_first.append(dataset_first_atom['displacement'])
        direction = np.dot(dataset_first_atom['displacement'],
                           np.linalg.inv(supercell.get_cell()))
        reduced_site_sym = get_reduced_site_symmetry(
            site_symmetry, direction, symprec)

        if verbose:
            print ("First displacements for fc4[ %d, x, x, x ]" %
                   (first_atom_num + 1))
            for i, v in enumerate(displacements_first):
                print "  [%7.4f %7.4f %7.4f]" % tuple(v)
                sys.stdout.flush()

        delta_fc3s.append(_get_delta_fc3(
                dataset_first_atom,
                fc3,
                supercell,
                reduced_site_sym,
                translational_symmetry_type,
                is_permutation_symmetry,
                symprec,
                verbose))

    _solve_fc4(fc4,
               first_atom_num,
               supercell,
               site_symmetry,
               displacements_first,
               np.array(delta_fc3s, dtype='double', order='C'),
               symprec)

    if verbose > 2:
        print "Site symmetry:"
        for i, v in enumerate(site_symmetry):
            print "  [%2d %2d %2d] #%2d" % tuple(list(v[0])+[i+1])
            print "  [%2d %2d %2d]" % tuple(v[1])
            print "  [%2d %2d %2d]\n" % tuple(v[2])
            sys.stdout.flush()

def _get_delta_fc3(dataset_first_atom,
                   fc3,
                   supercell,
                   reduced_site_sym,
                   translational_symmetry_type,
                   is_permutation_symmetry,
                   symprec,
                   verbose):
    disp_fc3 = _get_constrained_fc3(supercell,
                                    dataset_first_atom,
                                    reduced_site_sym,
                                    translational_symmetry_type,
                                    is_permutation_symmetry,
                                    symprec,
                                    verbose)
    
    if verbose:
        show_drift_fc3(disp_fc3, name="fc3 with disp.")

    return disp_fc3 - fc3

def _get_constrained_fc3(supercell,
                         displacements,
                         reduced_site_sym,
                         translational_symmetry_type,
                         is_permutation_symmetry,
                         symprec,
                         verbose):
    """
    Two displacements and force constants calculation (e.g. DFPT)

        displacements = {'number': 3,
                         'displacement': [0.01, 0.00, 0.01]
                         'second_atoms': [{'number': 7,
                                           'displacement': [],
                                           'delta_fc2': }]}

    Three displacements and force calculation

        displacements = {'number': 3,
                         'displacement': [0.01, 0.00, 0.01]
                         'second_atoms': [{'number': 7,
                                           'displacement': [],
                                           'third_atoms': ... }]}
           third_atoms is like:
                         'third_atoms': [{'number': 56,
                                          'displacement': [],
                                          'delta_forces': ... }]}
    """
    num_atom = supercell.get_number_of_atoms()
    atom1 = displacements['number']
    disp1 = displacements['displacement']
    delta_fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3),
                         dtype='double')

    if 'delta_forces' in displacements['second_atoms'][0]:
        fc2_with_one_disp = get_constrained_fc2(supercell,
                                                displacements['second_atoms'],
                                                atom1,
                                                reduced_site_sym,
                                                translational_symmetry_type,
                                                is_permutation_symmetry,
                                                symprec)
    
    atom_list = np.unique([x['number'] for x in displacements['second_atoms']])
    for atom2 in atom_list:
        bond_sym = get_bond_symmetry(
            reduced_site_sym,
            supercell.get_scaled_positions(),
            atom1,
            atom2,
            symprec)
        disps2 = []
        delta_fc2s = []
        for disps_second in displacements['second_atoms']:
            if atom2 != disps_second['number']:
                continue
            disps2.append(disps_second['displacement'])

            if 'delta_fc2' in disps_second:
                delta_fc2s.append(disps_second['delta_fc2'])
            else:
                direction = np.dot(disps_second['displacement'],
                                   np.linalg.inv(supercell.get_cell()))
                reduced_bond_sym = get_reduced_site_symmetry(
                    bond_sym, direction, symprec)

                delta_fc2s.append(get_delta_fc2(
                    disps_second['third_atoms'],
                    atom2,
                    fc2_with_one_disp,
                    supercell,
                    reduced_bond_sym,
                    translational_symmetry_type,
                    is_permutation_symmetry,
                    symprec))
    
            if verbose > 1:
                print ("Second displacements for fc4[ %d, %d, x, x ]" %
                       (atom1 + 1, atom2 + 1))
                for i, v in enumerate(disps2):
                    print "  [%7.4f %7.4f %7.4f]" % tuple(v)

        solve_fc3(delta_fc3,
                  atom2,
                  supercell,
                  bond_sym,
                  disps2,
                  delta_fc2s,
                  symprec=symprec)

    # Shift positions according to set atom1 is at origin
    lattice = supercell.get_cell().T
    positions = supercell.get_scaled_positions()
    pos_center = positions[atom1].copy()
    positions -= pos_center

    if verbose:
        print "(Copying delta fc3...)"

    fc3 = distribute_fc3(delta_fc3,
                         atom_list,
                         lattice,
                         positions,
                         np.array(reduced_site_sym, dtype='intc', order='C'),
                         np.zeros((len(reduced_site_sym), 3), dtype='double'),
                         symprec,
                         verbose)

    if translational_symmetry_type > 0:
        set_translational_invariance_fc3_per_index(fc3)

    if is_permutation_symmetry:
        set_permutation_symmetry_fc3(fc3)

    return fc3

def _solve_fc4(fc4,
               first_atom_num,
               supercell,
               site_symmetry,
               displacements_first,
               delta_fc3s,
               symprec):
    lattice = supercell.get_cell().T
    site_sym_cart = np.array([similarity_transformation(lattice, sym)
                              for sym in site_symmetry],
                             dtype='double', order='C')
    num_atom = supercell.get_number_of_atoms()
    positions = supercell.get_scaled_positions()
    pos_center = positions[first_atom_num].copy()
    positions -= pos_center
    rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                 site_symmetry,
                                                 symprec)
    
    rot_disps = get_rotated_displacement(displacements_first, site_sym_cart)
    inv_U = np.linalg.pinv(rot_disps)

    for (i, j, k) in list(np.ndindex(num_atom, num_atom, num_atom)):
        fc4[first_atom_num, i, j, k] = np.dot(
            inv_U, _rotate_delta_fc3s(
                i, j, k, delta_fc3s, rot_map_syms, site_sym_cart)
            ).reshape(3, 3, 3, 3)
            
def _rotate_delta_fc3s(i, j, k, delta_fc3s, rot_map_syms, site_sym_cart):
    rotated_fc3s = np.zeros((len(delta_fc3s), len(site_sym_cart), 3, 3, 3),
                            dtype='double')
    try:
        import anharmonic._phono4py as phono4c
        phono4c.rotate_delta_fc3s_elem(rotated_fc3s,
                                       delta_fc3s,
                                       rot_map_syms,
                                       site_sym_cart,
                                       i,
                                       j,
                                       k)
        return np.reshape(rotated_fc3s, (-1, 27))
    except ImportError:
        print "Copying delta fc3s at (%d, %d, %d)" % (i + 1, j + 1, k + 1)
        for l, fc3 in enumerate(delta_fc3s):
            for m, (sym, map_sym) in enumerate(zip(site_sym_cart,
                                                   rot_map_syms)):
                fc3_rot = fc3[map_sym[i], map_sym[j], map_sym[k]]
                rotated_fc3s[l, m] = third_rank_tensor_rotation(sym, fc3_rot)
        return np.reshape(rotated_fc3s, (-1, 27))

def _fourth_rank_tensor_rotation(rot_cart, tensor):
    rot_tensor = np.zeros((3, 3, 3, 3), dtype='double')
    for i, j, k, l in list(np.ndindex(3, 3, 3, 3)):
        rot_tensor[i, j, k, l] = _fourth_rank_tensor_rotation_elem(
            rot_cart, tensor, i, j, k, l)
    return rot_tensor

def _fourth_rank_tensor_rotation_elem(rot, tensor, l, m, n, p):
    sum_elems = 0.
    for i, j, k, l in list(np.ndindex(3, 3, 3, 3)):
        sum_elems += (rot[l, i] * rot[m, j] * rot[n, k] * rot[p, l] *
                      tensor[i, j, k, l])
    return sum_elems
