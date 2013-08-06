import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_rotated_displacement, get_rotated_forces, get_positions_sent_by_rot_inv

class FC3Fit:
    def __init__(self,
                 supercell,
                 disp_dataset,
                 symmetry):

        self._scell = supercell
        self._lattice = supercell.get_cell().T
        self._positions = supercell.get_scaled_positions()
        self._num_atom = len(self._positions)
        self._dataset = disp_dataset
        self._symmetry = symmetry
        self._symprec = symmetry.get_symmetry_tolerance()
        
        self._fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3),
                             dtype='double')
        self._fc3 = np.zeros((self._num_atom, self._num_atom, self._num_atom,
                              3, 3, 3), dtype='double')

    def run(self):
        self._get_matrices()

    def _get_matrices(self):
        self._get_matrices_1st(self._dataset)

    def _get_matrices_1st(self, dataset):
        unique_first_atom_nums = np.unique(
            [x['number'] for x in dataset['first_atoms']])
        
        for first_atom_num in unique_first_atom_nums:
            datasets_1st = []
            for dataset_1st in dataset['first_atoms']:
                if first_atom_num != dataset_1st['number']:
                    continue
                datasets_1st.append(dataset_1st)
            self._get_matrices_2nd(datasets_1st)

    def _get_matrices_2nd(self, datasets_1st):
        second_atom_nums = []
        for dataset_1st in datasets_1st:
            second_atom_nums.append(
                [x['number'] for x in dataset_1st['second_atoms']])
        unique_second_atom_nums = np.unique(np.array(second_atom_nums))
        first_atom_num = datasets_1st[0]['number']

        for second_atom_num in unique_second_atom_nums:
            for dataset_1st in datasets_1st:
                disp1 = dataset_1st['displacement']
                disp_pairs = []
                sets_of_forces = []
                for dataset_2nd in dataset_1st['second_atoms']:
                    if second_atom_num != dataset_2nd['number']:
                        continue
                    disp_pairs.append([disp1, dataset_2nd['displacement']])
                    sets_of_forces.append(dataset_2nd['forces'])
    
            site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
            site_sym_cart = [similarity_transformation(self._lattice, sym)
                             for sym in site_symmetry]

            # rot_disps1 (Num-pairs x num-site-syms, 3)
            # rot_disps2 (Num-pairs x num-site-syms, 9)
            rot_disps1, rot_disps2 = self._rotate_displacements(disp_pairs,
                                                                site_sym_cart)
            rot_disps = np.hstack((rot_disps1, rot_disps2))
            rot_forces = self._create_force_matrix(sets_of_forces,
                                                   site_symmetry,
                                                   site_sym_cart,
                                                   first_atom_num)

            print rot_disps.shape, rot_forces.shape
            
            fc = self._solve(rot_disps, rot_forces).flatten()
            fc2 = fc[:self._num_atom * 3 * 3].reshape((self._num_atom, 3, 3))
            self._fc2[first_atom_num] += fc2
            fc3 = fc[self._num_atom * 3 * 3:].reshape((self._num_atom, 3, 3, 3))
            self._fc3[first_atom_num, second_atom_num] = fc3
            
        self._fc2[first_atom_num] /= len(unique_second_atom_nums)
        print self._fc3
        

    def _create_force_matrix(self,
                             sets_of_forces,
                             site_symmetry,
                             site_sym_cart,
                             disp_atom_num):
        positions = (self._positions.copy() -
                     self._positions[disp_atom_num])
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)
        force_matrix = []
        for i in range(self._num_atom):
            for forces in sets_of_forces:
                force_matrix.append(
                    get_rotated_forces(forces[rot_map_syms[:, i]],
                                       site_sym_cart))
        return np.reshape(force_matrix, (self._num_atom, -1, 3))
        
    def _solve(self, rot_disps, rot_forces):
        inv_disps = np.linalg.pinv(rot_disps)
        return np.array([-np.dot(inv_disps, f) for f in rot_forces])

    def _rotate_displacements(self, disp_pairs, site_sym_cart):
        rot_disps1 = []
        rot_disps2 = []
        for (u1, u2) in disp_pairs:
            rot_disps1.append([np.dot(sym, u1) for sym in site_sym_cart])
            rot_disps2.append([np.outer(np.dot(sym, u1), np.dot(sym, u2))
                               for sym in site_sym_cart])
        return (np.reshape(rot_disps1, (-1, 3)),
                np.reshape(rot_disps2, (-1, 9)))
    
            
class FC2Fit:
    def __init__(self,
                 supercell,
                 disp_dataset,
                 symmetry):

        self._scell = supercell
        self._lattice = supercell.get_cell().T
        self._positions = supercell.get_scaled_positions()
        self._num_atom = len(self._positions)
        self._dataset = disp_dataset
        self._symmetry = symmetry
        self._symprec = symmetry.get_symmetry_tolerance()
        
        self._fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3),
                             dtype='double')

    def run(self):
        self._get_matrices()
        print self._fc2

    def _get_matrices(self):
        unique_first_atom_nums = np.unique(
            [x['number'] for x in self._dataset['first_atoms']])
        
        for first_atom_num in unique_first_atom_nums:
            disps = []
            sets_of_forces = []
            for dataset_1st in self._dataset['first_atoms']:
                if first_atom_num != dataset_1st['number']:
                    continue
                disps.append(dataset_1st['displacement'])
                sets_of_forces.append(dataset_1st['forces'])

            site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
            site_sym_cart = [similarity_transformation(self._lattice, sym)
                             for sym in site_symmetry]
            rot_disps = get_rotated_displacement(disps, site_sym_cart)
            rot_forces = self._create_force_matrix(sets_of_forces,
                                                   site_symmetry,
                                                   site_sym_cart,
                                                   first_atom_num)
            self._fc2[first_atom_num, :] = self._solve(rot_disps, rot_forces)

    def _create_force_matrix(self,
                             sets_of_forces,
                             site_symmetry,
                             site_sym_cart,
                             disp_atom_num):
        positions = (self._positions.copy() -
                     self._positions[disp_atom_num])
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)
        force_matrix = []
        for i in range(self._num_atom):
            for forces in sets_of_forces:
                force_matrix.append(
                    get_rotated_forces(forces[rot_map_syms[:, i]],
                                       site_sym_cart))
        return np.reshape(force_matrix, (self._num_atom, -1, 3))
        
    def _solve(self, rot_disps, rot_forces):
        inv_disps = np.linalg.pinv(rot_disps)
        return [-np.dot(inv_disps, f) for f in rot_forces]
        
            
