import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry, get_bond_symmetry
from anharmonic.phonon4.fc4 import distribute_fc4

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
                d3, f, d2 = self._collect_forces_and_disps(dataset_1st)
                sets_of_forces.append(f)
                disp_triplets.append(self._get_disp_triplets(d1, d2, d3))

            self._fit(first_atom_num, disp_triplets, sets_of_forces)

        rotations = self._symmetry.get_symmetry_operations()['rotations']
        translations = self._symmetry.get_symmetry_operations()['translations']

        print "ditributing..."
        distribute_fc4(self._fc4,
                       unique_first_atom_nums,
                       self._lattice,
                       self._positions,
                       rotations,
                       translations,
                       self._symprec,
                       self._verbose)

    def _get_disp_triplets(self, disp1, disp2s, disp3s):
        disp_triplets = []
        for d2_dirs, d3_dirs in zip(disp2s, disp3s):
            disp_triplets_u2 = []
            for d2, d3_atoms in zip(d2_dirs, d3_dirs):
                disp_triplets_u3 = []
                for d3_dirs_atom in d3_atoms:
                    disp_triplets_u3.append(
                        [[disp1, d2, d3] for d3 in d3_dirs_atom])
                disp_triplets_u2.append(disp_triplets_u3)
            disp_triplets.append(disp_triplets_u2)
        return disp_triplets

    def _fit(self, first_atom_num, disp_triplets, sets_of_forces):
        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)
        site_syms_cart = [similarity_transformation(self._lattice, sym)
                          for sym in site_symmetry]

        for second_atom_num in range(self._num_atom):
            for third_atom_num in range(self._num_atom):
                rot_disps = self._create_displacement_matrix(second_atom_num,
                                                             third_atom_num,
                                                             disp_triplets,
                                                             site_syms_cart,
                                                             rot_map_syms)
                rot_forces = self._create_force_matrix(second_atom_num,
                                                       third_atom_num,
                                                       sets_of_forces,
                                                       site_syms_cart,
                                                       rot_map_syms)
                fc = self._solve(rot_disps, rot_forces)
                fc2_1 = fc[:, 1:4, :].reshape((self._num_atom, 3, 3))
                fc2_2 = fc[:, 4:7, :].reshape((self._num_atom, 3, 3))
                fc2_3 = fc[:, 7:10, :].reshape((self._num_atom, 3, 3))
                fc3_1 = fc[:, 10:19, :].reshape((self._num_atom, 3, 3, 3))
                fc3_2 = fc[:, 19:28, :].reshape((self._num_atom, 3, 3, 3))
                fc3_3 = fc[:, 28:37, :].reshape((self._num_atom, 3, 3, 3))
                fc4 = fc[:, 37:, :].reshape((self._num_atom, 3, 3, 3, 3))

                self._fc4[first_atom_num,
                          second_atom_num,
                          third_atom_num] = fc4

            print second_atom_num + 1

    def _solve(self, rot_disps, rot_forces):
        fc = []
        inv_disps = np.linalg.pinv(rot_disps)
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
            for forces_2nd_atoms in sets_of_forces:
                for map_sym, sym in zip(rot_map_syms, site_syms_cart):
                    for forces_3rd_atoms in forces_2nd_atoms[
                        map_sym[second_atom_num]]:
                        for forces in forces_3rd_atoms[map_sym[third_atom_num]]:
                            force_matrix_atom.append(
                                np.dot(sym, forces[map_sym[fourth_atom_num]]))

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
        rot_pair12s = []
        rot_pair23s = []
        rot_pair31s = []
        rot_triplets = []

        for triplets_2nd_atoms in disp_triplets:
            for rot_atom_map, ssym_c in zip(rot_map_syms, site_syms_cart):
                for triplets_3rd_atoms in triplets_2nd_atoms[
                    rot_atom_map[second_atom_num]]:
                    for (u1, u2, u3) in triplets_3rd_atoms[
                        rot_atom_map[third_atom_num]]:

                        Su1 = np.dot(ssym_c, u1)
                        Su2 = np.dot(ssym_c, u2)
                        Su3 = np.dot(ssym_c, u3)
                        rot_disp1s.append(Su1)
                        rot_disp2s.append(Su2)
                        rot_disp3s.append(Su3)
                        rot_pair12s.append(
                            np.outer(Su1, Su2).flatten())
                        rot_pair23s.append(
                            np.outer(Su2, Su3).flatten())
                        rot_pair31s.append(
                            np.outer(Su3, Su1).flatten())
                        rot_triplets.append(
                            self._get_triplet_tensor(Su1, Su2, Su3))

        ones = np.ones(len(rot_disp1s)).reshape((-1, 1))

        return np.hstack((ones, rot_disp1s, rot_disp2s, rot_disp3s,
                          rot_pair12s, rot_pair23s, rot_pair31s, rot_triplets))

    def _get_triplet_tensor(self, u1, u2, u3):
        tensor = []
        for u1x in u1:
            for u2x in u2:
                for u3x in u3:
                   tensor.append(u1x * u2x * u3x)
        return tensor
    
    def _collect_forces_and_disps(self, dataset_1st):
        first_atom_num = dataset_1st['number']
        second_atom_nums = [x['number'] for x in dataset_1st['second_atoms']]
        unique_second_atom_nums = np.unique(np.array(second_atom_nums))
        reduced_site_sym = self._get_reduced_site_sym(dataset_1st)

        set_of_disps = []
        set_of_disp2s = []
        sets_of_forces = []
        for second_atom_num in range(self._num_atom):
            disp2s = []
            if second_atom_num in unique_second_atom_nums:
                set_of_disps_disp2 = []
                sets_of_forces_disp2 = []
                for dataset_2nd in dataset_1st['second_atoms']:
                    if dataset_2nd['number'] != second_atom_num:
                        continue

                    disp2 = dataset_2nd['displacement']
                    reduced_bond_sym = self._get_reduced_bond_sym(
                        reduced_site_sym,
                        first_atom_num,
                        second_atom_num,
                        disp2)
                    d, f = self._collect_3rd_forcecs_and_disps(dataset_2nd,
                                                               reduced_bond_sym)
                    set_of_disps_disp2.append(d)
                    sets_of_forces_disp2.append(f)
                    disp2s.append(disp2)
                set_of_disps.append(set_of_disps_disp2)
                sets_of_forces.append(sets_of_forces_disp2)
                set_of_disp2s.append(disp2s)
            else:
                set_of_disps.append(None)
                sets_of_forces.append(None)
                set_of_disp2s.append(None)

        self._distribute_2nd_forces_and_disps(
            set_of_disps,
            sets_of_forces,
            set_of_disp2s,
            dataset_1st['number'],
            reduced_site_sym,
            unique_second_atom_nums)

        return set_of_disps, sets_of_forces, set_of_disp2s

    def _distribute_2nd_forces_and_disps(self,
                                         set_of_disps,
                                         sets_of_forces,
                                         set_of_disp2s,
                                         first_atom_num,
                                         reduced_site_sym,
                                         unique_second_atom_nums):
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     reduced_site_sym,
                                                     self._symprec)

        for second_atom_num in range(self._num_atom):
            if set_of_disps[second_atom_num] is not None:
                continue
            self._copy_2nd_forces_and_disps(
                second_atom_num,
                set_of_disps,
                sets_of_forces,
                set_of_disp2s,
                unique_second_atom_nums,
                reduced_site_sym,
                rot_map_syms)

    def _copy_2nd_forces_and_disps(self,
                                   second_atom_num,
                                   set_of_disps,
                                   sets_of_forces,
                                   set_of_disp2s,
                                   unique_second_atom_nums,
                                   reduced_site_sym,
                                   rot_map_syms):
        sym_cart = None
        rot_atom_map = None
        for i, sym in enumerate(reduced_site_sym):
            if rot_map_syms[i, second_atom_num] in unique_second_atom_nums:
                sym_cart = similarity_transformation(self._lattice, sym)
                rot_atom_map = rot_map_syms[i, :]
                break

        assert sym_cart is not None, "Something is wrong."

        forces = []
        disps = []
        mapped_2nd_atom = rot_atom_map[second_atom_num]
        for disps_dir, forces_dir in zip(set_of_disps[mapped_2nd_atom],
                                         sets_of_forces[mapped_2nd_atom]):
            rot_forces_dir = []
            rot_disps_dir = []
            for third_atom_num in range(self._num_atom):
                mapped_3rd_num = rot_atom_map[third_atom_num]
                rot_forces_dir.append(
                    [np.dot(f[rot_atom_map], sym_cart.T)
                     for f in forces_dir[mapped_3rd_num]])
                rot_disps_dir.append([np.dot(sym_cart, d)
                                      for d in disps_dir[mapped_3rd_num]])

            forces.append(rot_forces_dir)
            disps.append(rot_disps_dir)

        set_of_disps[second_atom_num] = disps
        sets_of_forces[second_atom_num] = forces
        set_of_disp2s[second_atom_num] = [
            np.dot(sym_cart, d) for d in set_of_disp2s[mapped_2nd_atom]]

    def _collect_3rd_forcecs_and_disps(self, dataset_2nd, reduced_bond_sym):
        third_atom_nums = [x['number'] for x in dataset_2nd['third_atoms']]
        unique_third_atom_nums = np.unique(np.array(third_atom_nums))

        set_of_disps_disp2 = []
        sets_of_forces_disp2 = []

        for third_atom_num in range(self._num_atom):
            if third_atom_num in unique_third_atom_nums:
                set_of_disps_disp3 = []
                sets_of_forces_disp3 = []
                for dataset_3rd in dataset_2nd['third_atoms']:
                    if dataset_3rd['number'] != third_atom_num:
                        continue
                    set_of_disps_disp3.append(dataset_3rd['displacement'])
                    sets_of_forces_disp3.append(dataset_3rd['forces'])
                set_of_disps_disp2.append(set_of_disps_disp3)
                sets_of_forces_disp2.append(sets_of_forces_disp3)
            else:
                set_of_disps_disp2.append(None)
                sets_of_forces_disp2.append(None)

        self._distribute_3rd_forces_and_disps(
            set_of_disps_disp2,
            sets_of_forces_disp2,
            dataset_2nd['number'],
            reduced_bond_sym,
            unique_third_atom_nums)

        return set_of_disps_disp2, sets_of_forces_disp2

    def _distribute_3rd_forces_and_disps(self,
                                         set_of_disps_disp2,
                                         sets_of_forces_disp2,
                                         second_atom_num,
                                         reduced_bond_sym,
                                         unique_third_atom_nums):
        positions = self._positions.copy() - self._positions[second_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     reduced_bond_sym,
                                                     self._symprec)

        for third_atom_num in range(self._num_atom):
            if set_of_disps_disp2[third_atom_num] is not None:
                continue
            self._copy_3rd_forces_and_disps(
                third_atom_num,
                set_of_disps_disp2,
                sets_of_forces_disp2,
                unique_third_atom_nums,
                reduced_bond_sym,
                rot_map_syms)

    def _copy_3rd_forces_and_disps(self,
                                   third_atom_num,
                                   set_of_disps,
                                   sets_of_forces,
                                   unique_third_atom_nums,
                                   reduced_bond_sym,
                                   rot_map_syms):
        sym_cart = None
        rot_atom_map = None
        for i, sym in enumerate(reduced_bond_sym):
            if rot_map_syms[i, third_atom_num] in unique_third_atom_nums:
                sym_cart = similarity_transformation(self._lattice, sym)
                rot_atom_map = rot_map_syms[i, :]
                break

        assert sym_cart is not None, "Something is wrong."

        forces = []
        for set_of_forces_orig in sets_of_forces[rot_atom_map[third_atom_num]]:
            forces.append(np.dot(set_of_forces_orig[rot_atom_map], sym_cart.T))

        disps = [np.dot(sym_cart, d)
                 for d in set_of_disps[rot_atom_map[third_atom_num]]]

        set_of_disps[third_atom_num] = disps
        sets_of_forces[third_atom_num] = forces

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

    def _show_data_structure(self, set_of_disps):
        for i in range(self._num_atom):
            print "%3d:" % (i+1),
            if set_of_disps[i] is None:
                print "None"
            else:
                print "%d ---------" % len(set_of_disps[i])
                for j in range(len(set_of_disps[i])):
                    for k in range(self._num_atom):
                        print "       %3d:" % (k+1),
                        if set_of_disps[i][j][k] is None:
                            print "None"
                        else:
                            print len(set_of_disps[i][j][k])
                    print "       ---------"

        sys.exit(1)
                

