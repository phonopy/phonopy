import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv, get_rotated_displacement
from anharmonic.phonon3.fc3 import set_translational_invariance_fc3_per_index, solve_fc3, distribute_fc3, third_rank_tensor_rotation, get_atom_mapping_by_symmetry, get_atom_by_symmetry
from anharmonic.file_IO import write_fc4_to_hdf5
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry

def get_fc4(supercell,
            disp_dataset,
            fc3,
            symmetry,
            is_translational_symmetry=False,
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
                         is_translational_symmetry,
                         verbose)

    if verbose:
        print "----- Copying fc4 -----"

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

    print "Wriging fc4.hdf5"
    write_fc4_to_hdf5(fc4)
    
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
            
        rot_cart_inv = np.double(
            similarity_transformation(lattice, rot).T.copy())

        try:
            import anharmonic._phono3py as phono3c
            phono3c.distribute_fc4(fc4,
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

def _get_fc4_least_atoms(fc4,
                         supercell,
                         disp_dataset,
                         fc3,
                         symmetry,
                         is_translational_symmetry,
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
                          is_translational_symmetry,
                          symprec,
                          verbose)

def _get_fc4_one_atom(fc4,
                      supercell,
                      disp_dataset,
                      fc3,
                      first_atom_num,
                      site_symmetry,
                      is_translational_symmetry,
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
                is_translational_symmetry,
                symprec,
                verbose))

    _solve_fc4(fc4,
               first_atom_num,
               supercell,
               site_symmetry,
               displacements_first,
               np.double(delta_fc3s),
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
                   is_translational_symmetry,
                   symprec,
                   verbose):
    disp_fc3 = _get_constrained_fc3(supercell,
                                    dataset_first_atom,
                                    reduced_site_sym,
                                    symprec,
                                    verbose)

    if is_translational_symmetry:
        set_translational_invariance_fc3_per_index(disp_fc3)
            
    return disp_fc3 - fc3

def _get_constrained_fc3(supercell,
                         displacements,
                         reduced_site_sym,
                         symprec,
                         verbose):
    """
    displacements = {'number': 3,
                     'displacement': [0.01, 0.00, 0.01]
                     'second_atoms': [{'number': 7,
                                       'displacements': [[]],
                                       'delta_fc2': []}]}

    number: Atomic index, starting with 0.
    displacement: displacement of 1st displaced atom in Cartesian.
    displacements: displacements of 2st displaced atom in Cartesian.
    delta_fc2: diff. of 2 atomic disp. fc2 and 1 atomic disp. fc2
    Number of elements of 'displacements' and 'delta_fc2' are same.
    """
    num_atom = supercell.get_number_of_atoms()
    atom1 = displacements['number']
    disp1 = displacements['displacement']
    fc3 = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype='double')
    atom_list_done = []

    for disps_second in displacements['second_atoms']:
        disps2 = disps_second['displacements']
        atom2 = disps_second['number']
        delta_fc2s = disps_second['delta_fc2']
        atom_list_done.append(atom2)
        bond_sym = get_bond_symmetry(
            reduced_site_sym,
            supercell.get_scaled_positions(),
            atom1,
            atom2,
            symprec)

        if verbose > 1:
            print ("Second displacements for fc4[ %d, %d, x, x ]" %
                   (atom1 + 1, atom2 + 1))
            for i, v in enumerate(disps2):
                print "  [%7.4f %7.4f %7.4f]" % tuple(v)
        solve_fc3(fc3,
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
        print "----- Copying delta fc3 -----"
    distribute_fc3(fc3,
                   atom_list_done,
                   lattice,
                   positions,
                   np.intc(reduced_site_sym).copy(),
                   np.zeros((len(reduced_site_sym), 3),
                            dtype='double'),
                   symprec,
                   verbose)
    return fc3

def _solve_fc4(fc4,
               first_atom_num,
               supercell,
               site_symmetry,
               displacements_first,
               delta_fc3s,
               symprec):
    lattice = supercell.get_cell().T
    site_sym_cart = np.double([similarity_transformation(lattice, sym)
                               for sym in site_symmetry])
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

def _rotate_delta_fc3s(i, j, k, fc3s, rot_map_syms, site_sym_cart):
    rotated_fc3s = np.zeros((len(fc3s), len(site_sym_cart), 3, 3, 3),
                            dtype='double')
    try:
        import anharmonic._phono3py as phono3c
        phono3c.rotate_delta_fc3s(rotated_fc3s,
                                  fc3s,
                                  rot_map_syms,
                                  site_sym_cart,
                                  i,
                                  j,
                                  k)
        return np.reshape(rotated_fc3s, (-1, 27))
    except ImportError:
        print "Copying delta fc3s at (%d, %d, %d)" % (i + 1, j + 1, k + 1)
        for l, fc3 in enumerate(fc3s):
            for m, (sym, map_sym) in enumerate(zip(site_sym_cart, rot_map_syms)):
                fc3_rot = fc3[map_sym[i], map_sym[j], map_sym[k]]
                rotated_fc3s[l, m] = third_rank_tensor_rotation(sym, fc3_rot)
        return np.reshape(rotated_fc3s, (-1, 27))
