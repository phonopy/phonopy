import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry
from anharmonic.phonon4.fc4 import distribute_fc4
from anharmonic.phonon3.fc3 import distribute_fc3

class FC4Fit:
    def __init__(self,
                 supercell,
                 disp_dataset,
                 symmetry,
                 verbose=False):

        self._scell = supercell
        self._lattice = supercell.get_cell().T
        self._positions = supercell.get_scaled_positions()
        self._num_atom = len(self._positions)
        self._dataset = disp_dataset
        self._symmetry = symmetry
        self._verbose = verbose
        
        self._symprec = symmetry.get_symmetry_tolerance()
        
        self._fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3),
                             dtype='double')
        self._fc3 = np.zeros((self._num_atom, self._num_atom, self._num_atom,
                              3, 3, 3), dtype='double')
        self._fc4 = np.zeros(
            (self._num_atom, self._num_atom, self._num_atom, self._num_atom,
             3, 3, 3, 3), dtype='double')

    def run(self):
        self._calculate()

    def get_fc3(self):
        return self._fc3
        
    def get_fc4(self):
        return self._fc4
        
    def _calculate(self):
        unique_first_atom_nums = np.unique(
            [x['number'] for x in self._dataset['first_atoms']])

        for first_atom_num in unique_first_atom_nums:
            disp_triplets = []
            sets_of_forces = []
            for dataset_1st in self._dataset['first_atoms']:
                if first_atom_num != dataset_1st['number']:
                    continue
                d1 = dataset_1st['displacement']
                d3, f = self._collect_forces_and_disps(dataset_1st)
                disp_triplets.append(d3)
                sets_of_forces.append(f)
                # disp_triplets.append(self._get_disp_triplets(d1, d2, d3))

            self._fit(first_atom_num, disp_triplets, sets_of_forces)

        rotations = self._symmetry.get_symmetry_operations()['rotations']
        translations = self._symmetry.get_symmetry_operations()['translations']

        print "ditributing fc4..."
        distribute_fc4(self._fc4,
                       unique_first_atom_nums,
                       self._lattice,
                       self._positions,
                       rotations,
                       translations,
                       self._symprec,
                       self._verbose)

        print "ditributing fc3..."
        distribute_fc3(self._fc3,
                       unique_first_atom_nums,
                       self._lattice,
                       self._positions,
                       rotations,
                       translations,
                       self._symprec,
                       self._verbose)

    def _fit(self, first_atom_num, disp_triplets, sets_of_forces):
        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)
        site_syms_cart = [similarity_transformation(self._lattice, sym)
                          for sym in site_symmetry]
        
        for second_atom_num in range(self._num_atom):
            rot_disps_set = []
            for third_atom_num in range(self._num_atom):
                rot_disps_set.append(self._create_displacement_matrix(
                        second_atom_num,
                        third_atom_num,
                        disp_triplets,
                        site_syms_cart,
                        rot_map_syms))
                print rot_disps_set[third_atom_num].shape
            row_nums = np.intc([x.shape[0] for x in rot_disps_set])
            max_row_num = max(row_nums)
            column_num = rot_disps_set[0].shape[1]
            rot_disps = np.zeros((self._num_atom, max_row_num * column_num),
                                 dtype='double')
            for i in range(self._num_atom):
                rot_disps[
                    i, :row_nums[i] * column_num] = rot_disps_set[i].flatten()
            inv_disps = np.zeros_like(rot_disps)
            import anharmonic._phono3py as phono3c
            info = phono3c.pinv_mt(rot_disps,
                                   inv_disps,
                                   row_nums,
                                   max_row_num,
                                   column_num,
                                   1e-13)
            
            for third_atom_num in range(self._num_atom):
                rot_forces = self._create_force_matrix(
                    second_atom_num,
                    third_atom_num,
                    sets_of_forces,
                    site_syms_cart,
                    rot_map_syms)

                fc = self._solve_mt(
                    inv_disps[third_atom_num,
                              :row_nums[third_atom_num] * column_num].reshape(
                        column_num, row_nums[third_atom_num]), rot_forces)

                # fc = self._solve(rot_disps_set[third_atom_num], rot_forces)
                fc2 = fc[:, 1:4, :].reshape((self._num_atom, 3, 3))
                # fc3 = fc[:, 19:28, :].reshape((self._num_atom, 3, 3, 3))
                # fc4 = fc[:, 226:253, :].reshape((self._num_atom, 3, 3, 3, 3))
                fc3 = fc[:, 19:28, :].reshape((self._num_atom, 3, 3, 3))
                fc4 = fc[:, 172:199, :].reshape((self._num_atom, 3, 3, 3, 3))

                # self._fc3[first_atom_num,
                #           second_atom_num] = fc3 * 2
                # self._fc4[first_atom_num,
                #           second_atom_num,
                #           third_atom_num] = fc4 * 6

                self._fc3[first_atom_num,
                          second_atom_num] = fc3
                self._fc4[first_atom_num,
                          second_atom_num,
                          third_atom_num] = fc4

            print second_atom_num + 1

    def _solve_mt(self, inv_disps, rot_forces):
        fc = []
        for i in range(self._num_atom):
            fc.append(-np.dot(inv_disps, rot_forces[i]))
        
        return np.array(fc)
        
            
    def _solve(self, rot_disps, rot_forces):
        cutoff = 1e-13
        import anharmonic._phono3py as phono3c
        (m, n) = rot_disps.shape
        inv_disps = np.zeros((n, m), dtype='double')
        info = phono3c.pinv(np.double(rot_disps), inv_disps, cutoff)
        if info > 0:
            print "SVD did not converge."
        elif info < 0:
            print "%d-th argument had an illegal value." % (-info)
        
        # inv_disps = np.linalg.pinv(rot_disps)
        for i in range(self._num_atom):
            fc.append(-np.dot(inv_disps, rot_forces[i]))
        
        return np.array(fc)

    def _create_force_matrix(self,
                             second_atom_num,
                             third_atom_num,
                             sets_of_forces,
                             site_syms_cart,
                             rot_map_syms):
        force_matrix = []

        for fourth_atom_num in range(self._num_atom):
            force_matrix_atom = []
            for forces in sets_of_forces:
                for rot_atom_map, sym in zip(rot_map_syms, site_syms_cart):
                    rot_num2 = rot_atom_map[second_atom_num]
                    rot_num3 = rot_atom_map[third_atom_num]
                    for f in forces[rot_num2][rot_num3]:
                        force_matrix_atom.append(
                            np.dot(sym, f[rot_atom_map[fourth_atom_num]]))

            force_matrix.append(force_matrix_atom)

        return np.double(force_matrix)

    def _create_displacement_matrix(self,
                                    second_atom_num,
                                    third_atom_num,
                                    disp_triplets,
                                    site_syms_cart,
                                    rot_map_syms):
        rot_disp1s = []
        rot_disp2s = []
        rot_disp3s = []
        rot_pairs = []
        rot_triplets = []

        for disps in disp_triplets:
            for rot_atom_map, sym in zip(rot_map_syms, site_syms_cart):
                rot_num2 = rot_atom_map[second_atom_num]
                rot_num3 = rot_atom_map[third_atom_num]
                for (u1, u2, u3) in disps[rot_num2][rot_num3]:
                    Su1 = np.dot(sym, u1)
                    Su2 = np.dot(sym, u2)
                    Su3 = np.dot(sym, u3)
                    rot_disp1s.append(Su1)
                    rot_disp2s.append(Su2)
                    rot_disp3s.append(Su3)
                    rot_pairs.append(self._get_pair_tensor(Su1, Su2, Su3))
                    rot_triplets.append(self._get_triplet_tensor(Su1, Su2, Su3))

        ones = -np.ones(len(rot_disp1s)).reshape((-1, 1))

        return np.hstack((ones, rot_disp1s, rot_disp2s, rot_disp3s,
                          rot_pairs, rot_triplets))

    def _get_pair_tensor(self, u1, u2, u3):
        # 0 (0, 0)
        # 1 (0, 1)
        # 2 (0, 2)
        # 3 (1, 0)
        # 4 (1, 1)
        # 5 (1, 2)
        # 6 (2, 0)
        # 7 (2, 1)
        # 8 (2, 2)

        u = [u1, u2, u3]
        tensor = []
        for (i, j) in ((0, 0),
                       (0, 1),
                       (0, 2),
                       (1, 1),
                       (1, 2),
                       (2, 2)):
        # for (i, j) in list(np.ndindex(3, 3)):
            for u1x in u[i]:
                for u2x in u[j]:
                    tensor.append(u1x * u2x)
        return tensor

    def _get_triplet_tensor(self, u1, u2, u3):
        u = [u1, u2, u3]
        tensor = []

        # 0 (0, 0, 0)
        # 1 (0, 0, 1)
        # 2 (0, 0, 2)
        # 3 (0, 1, 0)
        # 4 (0, 1, 1)
        # 5 (0, 1, 2)
        # 6 (0, 2, 0)
        # 7 (0, 2, 1)
        # 8 (0, 2, 2)
        # 9 (1, 0, 0)
        # 10 (1, 0, 1)
        # 11 (1, 0, 2)
        # 12 (1, 1, 0)
        # 13 (1, 1, 1)
        # 14 (1, 1, 2)
        # 15 (1, 2, 0)
        # 16 (1, 2, 1)
        # 17 (1, 2, 2)
        # 18 (2, 0, 0)
        # 19 (2, 0, 1)
        # 20 (2, 0, 2)
        # 21 (2, 1, 0)
        # 22 (2, 1, 1)
        # 23 (2, 1, 2)
        # 24 (2, 2, 0)
        # 25 (2, 2, 1)
        # 26 (2, 2, 2)

        for (i, j, k) in ((0, 0, 0),
                          (0, 0, 1),
                          (0, 0, 2),
                          (0, 1, 1),
                          (0, 1, 2),
                          (0, 2, 2),
                          (1, 1, 1),
                          (1, 1, 2),
                          (1, 2, 2),
                          (2, 2, 2)):
        # for (i, j, k) in list(np.ndindex(3, 3, 3)):
            for u1x in u[i]:
                for u2x in u[j]:
                    for u3x in u[k]:
                        tensor.append(u1x * u2x * u3x)
        return tensor
                        
    def _collect_forces_and_disps(self, dataset_1st):
        disp1 = dataset_1st['displacement']
        disps = [None] * self._num_atom
        forces = [None] * self._num_atom
        first_atom_num = dataset_1st['number']
        reduced_site_sym = self._get_reduced_site_sym(dataset_1st)

        for i in range(self._num_atom):
            for dataset_2nd in dataset_1st['second_atoms']:
                if dataset_2nd['number'] != i:
                    continue
                disp2 = dataset_2nd['displacement']
                disps_3 = [None] * self._num_atom
                forces_3 = [None] * self._num_atom
                for j in range(self._num_atom):
                    for dataset_3rd in dataset_2nd['third_atoms']:
                        if dataset_3rd['number'] != j:
                            continue
                        if disps_3[j] is None:
                            disps_3[j] = []
                            forces_3[j] = []
                        disps_3[j].append(
                            [disp1, disp2, dataset_3rd['displacement']])
                        forces_3[j].append(dataset_3rd['forces'])

                for j in range(self._num_atom):
                    if disps_3[j] is None:
                        reduced_bond_sym = self._get_reduced_bond_sym(
                            reduced_site_sym,
                            first_atom_num,
                            i,
                            disp2)
                        self._distribute_3(
                            disps_3,
                            forces_3,
                            i,
                            j,
                            reduced_bond_sym)
                
                if disps[i] is None:
                    disps[i] = []
                    forces[i] = []
                    for j in range(self._num_atom):
                        disps[i].append(disps_3[j])
                        forces[i].append(forces_3[j])
                else:
                    for j in range(self._num_atom):
                        disps[i][j] += disps_3[j]
                        forces[i][j] += forces_3[j]

        for i in range(self._num_atom):
            if disps[i] is None:
                self._distribute_2(
                    disps,
                    forces,
                    first_atom_num,
                    i,
                    reduced_site_sym)
                
        # for i in range(self._num_atom):
        #     if disps[i] is not None:
        #         for j in range(self._num_atom):
        #             print i + 1, j + 1,
        #             if disps[i][j] is not None:
        #                 print len(disps[i][j])
        #             else:
        #                 print "None"
        #     else:
        #         print i + 1, "None"
        # sys.exit(1)

        return disps, forces

    def _distribute_2(self,
                      disps,
                      forces,
                      first_atom_num,
                      second_atom_num,
                      reduced_site_sym):
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     reduced_site_sym,
                                                     self._symprec)

        sym_cart = None
        rot_atom_map = None
        for i, sym in enumerate(reduced_site_sym):
            if disps[rot_map_syms[i, second_atom_num]] is not None:
                sym_cart = similarity_transformation(self._lattice, sym)
                rot_atom_map = rot_map_syms[i, :]
                break

        assert sym_cart is not None, "Something is wrong."

        forces[second_atom_num] = []
        disps[second_atom_num] = []
        for i in range(self._num_atom):
            forces_2 = [
                np.dot(f[rot_atom_map], sym_cart.T)
                for f in forces[rot_atom_map[second_atom_num]][rot_atom_map[i]]]
            disps_2 = [
                [d3[0], np.dot(sym_cart, d3[1]), np.dot(sym_cart, d3[2])]
                for d3 in disps[rot_atom_map[second_atom_num]][rot_atom_map[i]]]

            forces[second_atom_num].append(forces_2)
            disps[second_atom_num].append(disps_2)

    def _distribute_3(self,
                      disps_3,
                      forces_3,
                      second_atom_num,
                      third_atom_num,
                      reduced_bond_sym):
        positions = self._positions.copy() - self._positions[second_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     reduced_bond_sym,
                                                     self._symprec)

        sym_cart = None
        rot_atom_map = None
        for i, sym in enumerate(reduced_bond_sym):
            if disps_3[rot_map_syms[i, third_atom_num]] is not None:
                sym_cart = similarity_transformation(self._lattice, sym)
                rot_atom_map = rot_map_syms[i, :]
                break

        assert sym_cart is not None, "Something is wrong."

        forces = [np.dot(f[rot_atom_map], sym_cart.T)
                  for f in forces_3[rot_atom_map[third_atom_num]]]
        disps = [[d3[0], d3[1], np.dot(sym_cart, d3[2])]
                 for d3 in disps_3[rot_atom_map[third_atom_num]]]

        disps_3[third_atom_num] = disps
        forces_3[third_atom_num] = forces

    def _get_reduced_site_sym(self, dataset_1st):
        disp1 = dataset_1st['displacement']
        first_atom_num = dataset_1st['number']
        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        direction = np.dot(np.linalg.inv(self._lattice), disp1)
        reduced_site_sym = get_reduced_site_symmetry(site_symmetry,
                                                     direction,
                                                     self._symprec)
        return reduced_site_sym

    def _get_reduced_bond_sym(self,
                              reduced_site_sym,
                              first_atom_num,
                              second_atom_num,
                              disp2):
        bond_sym = get_bond_symmetry(
            reduced_site_sym,
            self._positions,
            first_atom_num,
            second_atom_num,
            self._symprec)
        direction = np.dot(np.linalg.inv(self._lattice), disp2)
        reduced_bond_sym = get_reduced_site_symmetry(
            bond_sym, direction, self._symprec)

        return reduced_bond_sym

