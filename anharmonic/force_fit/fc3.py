import sys
import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv
from anharmonic.phonon3.displacement_fc3 import get_reduced_site_symmetry

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
            disp_pairs = []
            sets_of_forces = []
            for dataset_1st in dataset['first_atoms']:
                if first_atom_num != dataset_1st['number']:
                    continue
                disp_pairs_each, sets_of_forces_each = \
                    self._collect_disp_pairs_and_forces(dataset_1st)
                disp_pairs.append(disp_pairs_each)
                sets_of_forces.append(sets_of_forces_each)

            self._get_matrices_2nd(first_atom_num, disp_pairs, sets_of_forces)

    def _get_matrices_2nd(self, first_atom_num, disp_pairs, sets_of_forces):
        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)

        for second_atom_num in range(self._num_atom):
            rot_disps = []
            rot_forces = []
            rot_atom_map = rot_map_syms[:, second_atom_num]

            for disp_pairs_each, sets_of_forces_each in zip(
                disp_pairs, sets_of_forces):
                (rot_disps1,
                 rot_disps2,
                 rot_disps_pair) = self._rotate_displacements(disp_pairs_each,
                                                              site_symmetry,
                                                              rot_atom_map)
                ones = np.ones(rot_disps1.shape[0]).reshape((-1, 1))
                rot_disps.append(np.hstack(
                        (ones, rot_disps1, rot_disps2, rot_disps_pair)))
                rot_forces.append(self._create_force_matrix(sets_of_forces_each,
                                                            site_symmetry,
                                                            rot_atom_map,
                                                            rot_map_syms))

                print rot_disps[-1].shape, rot_forces[-1].shape
            
            fc = self._solve(rot_disps, rot_forces)
            
            # print "d", rot_disps.shape, "f", rot_forces.shape, "fc", fc.shape
            fc2 = fc[:, 1:4, :].reshape((self._num_atom, 3, 3))
            fc2_b = fc[:, 4:7, :].reshape((self._num_atom, 3, 3))
            self._fc2[first_atom_num] += fc2
            fc3 = fc[:, 7:, :].reshape((self._num_atom, 3, 3, 3))
            self._fc3[first_atom_num, second_atom_num] = fc3

        self._fc2[first_atom_num] /= self._num_atom
        print self._fc2[first_atom_num]

        from anharmonic.file_IO import write_fc3_dat
        write_fc3_dat(self._fc3, 'fc3-oneshot.dat')

    def _solve(self, rot_disps, rot_forces):
        fc = []

        disps = None
        for d in rot_disps:
            if disps is None:
                disps = d
            else:
                disps = np.vstack((disps, d))
        inv_disps = np.linalg.pinv(disps)

        for i in range(self._num_atom):
            forces = None
            for f in rot_forces:
                if forces is None:
                    forces = f[i]
                else:
                    forces = np.vstack((forces, f[i]))
            fc.append(-np.dot(inv_disps, forces))
        
        return np.array(fc)

    def _create_force_matrix(self,
                             sets_of_forces,
                             site_symmetry,
                             rot_atom_map,
                             rot_map_syms):
        site_syms_cart = [similarity_transformation(self._lattice, sym)
                          for sym in site_symmetry]
        force_matrix = []
        for i in range(self._num_atom):
            force_matrix_atom = []
            for map_sym, rot_atom_num, sym in zip(
                rot_map_syms, rot_atom_map, site_syms_cart):
                for forces in sets_of_forces[rot_atom_num]:
                    force_matrix_atom.append(np.dot(sym, forces[map_sym[i]]))
            force_matrix.append(force_matrix_atom)
        return np.double(force_matrix)
        
    def _rotate_displacements(self, disp_pairs, site_symmetry, rot_atom_map):
        rot_disps1 = []
        rot_disps2 = []
        rot_disps_pair = []

        for rot_atom_num, ssym in zip(rot_atom_map, site_symmetry):
            ssym_c = similarity_transformation(self._lattice, ssym)
            for (u1, u2) in disp_pairs[rot_atom_num]:
                Su1 = np.dot(ssym_c, u1)
                Su2 = np.dot(ssym_c, u2)
                rot_disps1.append(Su1)
                rot_disps2.append(Su2)
                rot_disps_pair.append(np.outer(Su1, Su2).flatten())

        return (np.double(rot_disps1),
                np.double(rot_disps2),
                np.double(rot_disps_pair))

    def _collect_disp_pairs_and_forces(self, dataset_1st):
        first_atom_num = dataset_1st['number']
        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        disp1 = dataset_1st['displacement']
        direction = np.dot(np.linalg.inv(self._lattice), disp1)
        reduced_site_sym = get_reduced_site_symmetry(
            site_symmetry, direction, self._symprec)
        second_atom_nums = [x['number'] for x in dataset_1st['second_atoms']]
        unique_second_atom_nums = np.unique(np.array(second_atom_nums))
        positions = self._positions.copy() - self._positions[first_atom_num]
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     reduced_site_sym,
                                                     self._symprec)

        set_of_disps = []
        sets_of_forces = []
        for second_atom_num in range(self._num_atom):
            if second_atom_num in unique_second_atom_nums:
                set_of_disps_atom2 = []
                sets_of_forces_atom2 = []
                for dataset_2nd in dataset_1st['second_atoms']:
                    if dataset_2nd['number'] != second_atom_num:
                        continue
                    set_of_disps_atom2.append(dataset_2nd['displacement'])
                    sets_of_forces_atom2.append(dataset_2nd['forces'])
                set_of_disps.append(set_of_disps_atom2)
                sets_of_forces.append(sets_of_forces_atom2)
            else:
                set_of_disps.append(None)
                sets_of_forces.append(None)

        disp_pairs = []
        for second_atom_num in range(self._num_atom):
            if set_of_disps[second_atom_num] is not None:
                disp_pairs.append(
                    [[disp1, d] for d in set_of_disps[second_atom_num]])
            else:
                (sets_of_forces_atom2,
                 set_of_disps_atom2) = self._copy_dataset_2nd(
                    second_atom_num,
                    set_of_disps,
                    sets_of_forces,
                    unique_second_atom_nums,
                    reduced_site_sym,
                    rot_map_syms)

                disp_pairs.append(
                    [[disp1, d] for d in set_of_disps_atom2])
                sets_of_forces[second_atom_num] = sets_of_forces_atom2

        return disp_pairs, sets_of_forces

    def _copy_dataset_2nd(self,
                          second_atom_num,
                          set_of_disps,
                          sets_of_forces,
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

        for sets_of_forces_orig_atom in sets_of_forces[rot_atom_map[second_atom_num]]:
            forces.append(np.dot(sets_of_forces_orig_atom[rot_atom_map], sym_cart.T))

        disps = [np.dot(sym_cart, d)
                 for d in set_of_disps[rot_atom_map[second_atom_num]]]

        return forces, disps
    
            
