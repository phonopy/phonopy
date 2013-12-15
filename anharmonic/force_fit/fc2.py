import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_positions_sent_by_rot_inv, distribute_force_constants
from phonopy.harmonic.dynamical_matrix import get_equivalent_smallest_vectors

class FC2Fit:
    def __init__(self,
                 supercell,
                 disp_dataset,
                 symmetry,
                 translational_invariance=False,
                 rotational_invariance=False,
                 coef_invariants=None):

        self._scell = supercell
        self._lattice = supercell.get_cell().T
        self._positions = supercell.get_scaled_positions()
        self._num_atom = len(self._positions)
        self._dataset = disp_dataset
        self._symmetry = symmetry
        self._symprec = symmetry.get_symmetry_tolerance()
        self._coef_invariants = coef_invariants
        
        self._fc2 = np.zeros((self._num_atom, self._num_atom, 3, 3),
                             dtype='double')
        self._rot_inv = rotational_invariance
        self._trans_inv = translational_invariance

    def run(self):
        self._unique_first_atom_nums = np.unique(
            [x['number'] for x in self._dataset['first_atoms']])

        if self._rot_inv or self._trans_inv:
            if self._trans_inv:
                print "Translational invariance: On"
            if self._rot_inv:
                print "Rotational invariance: On"
            self._set_fc2_displaced_atoms_one_shot()
        else:
            self._set_fc2_each_displaced_atom()
        self._distribute()

    def get_fc2(self):
        return self._fc2

    def _set_fc2_displaced_atoms_one_shot(self):
        for first_atom_num in self._unique_first_atom_nums:
            print "Atom: ", first_atom_num + 1
            disp_mat = []
            rot_disps, rot_forces = self._get_matrices(first_atom_num)
            for d in rot_disps:
                disp_mat.append(np.kron(d, np.eye(3)))
            disp_big_mat = np.kron(np.eye(self._num_atom),
                                   np.reshape(disp_mat, (-1, 9)))
            residual_force_mat = np.kron(np.ones((len(disp_big_mat) / 3, 1)),
                                         np.eye(3))
            disp_big_mat = np.hstack((residual_force_mat, disp_big_mat))
            force_mat = np.reshape(rot_forces, (-1, 1))
            if self._rot_inv:
                if self._coef_invariants is None:
                    amplitude = np.sqrt((rot_disps ** 2).sum() / len(rot_disps))
                else:
                    amplitude = self._coef_invariants
                rimat = self._get_rotational_invariance_matrix(first_atom_num)
                rimat *= amplitude
                disp_big_mat = np.vstack((disp_big_mat, rimat))
                force_mat = np.vstack((force_mat, np.zeros((9, 1))))

            if self._trans_inv:
                if self._coef_invariants is None:
                    amplitude = np.sqrt((rot_disps ** 2).sum() / len(rot_disps))
                else:
                    amplitude = self._coef_invariants
                timat = self._get_translational_invariance_matrix()
                timat *= amplitude
                disp_big_mat = np.vstack((disp_big_mat, timat))
                force_mat = np.vstack((force_mat, np.zeros((9, 1))))

            inv_disp_mat = np.linalg.pinv(disp_big_mat)
            fc2 = -np.dot(inv_disp_mat, force_mat).flatten()
            print "  Recidual force:", fc2[:3]
            self._fc2[first_atom_num] = fc2[3:].reshape(-1, 3, 3)
            
    def _get_rotational_invariance_matrix(self, patom_num):
        rimat = np.zeros((9, 9 * self._num_atom + 3), dtype='double')
        rimat[:9, :3] = [[ 0, 0, 0],
                         [-1, 0, 0],
                         [ 1, 0, 0],
                         [ 0, 1, 0],
                         [ 0, 0, 0],
                         [ 0,-1, 0],
                         [ 0, 0,-1],
                         [ 0, 0, 1],
                         [ 0, 0, 0]]
        for i in range(self._num_atom):
            vectors = get_equivalent_smallest_vectors(i,
                                                      patom_num,
                                                      self._scell,
                                                      self._lattice.T,
                                                      self._symprec)
            r_frac = np.array(vectors).sum(axis=0) / len(vectors)
            r = np.dot(self._lattice, r_frac)
            rimat_each = np.kron(np.eye(3), [[0, r[2], -r[1]],
                                             [-r[2], 0, r[0]],
                                             [r[1], -r[0], 0]])
            rimat[:, (i * 9 + 3):((i + 1) * 9 + 3)] = rimat_each
        return rimat

    def _get_translational_invariance_matrix(self):
        timat = np.zeros((9, 9 * self._num_atom + 3))
        timat[:, 3:] = np.kron(np.ones(self._num_atom), np.eye(9))
        return timat

    def _set_fc2_each_displaced_atom(self):
        for first_atom_num in self._unique_first_atom_nums:
            rot_disps, rot_forces = self._get_matrices(first_atom_num)
            ones = np.ones(len(rot_disps)).reshape((-1, 1))
            fc = self._solve(np.hstack((ones, rot_disps)), rot_forces)
            for i in range(self._num_atom):
                self._fc2[first_atom_num, i] = fc[i, 1:, :]

    def _get_matrices(self, first_atom_num):
        disps = []
        sets_of_forces = []
        for dataset_1st in self._dataset['first_atoms']:
            if first_atom_num != dataset_1st['number']:
                continue
            disps.append(dataset_1st['displacement'])
            sets_of_forces.append(dataset_1st['forces'])

        site_symmetry = self._symmetry.get_site_symmetry(first_atom_num)
        positions = (self._positions.copy() -
                     self._positions[first_atom_num])
        rot_map_syms = get_positions_sent_by_rot_inv(positions,
                                                     site_symmetry,
                                                     self._symprec)
        site_sym_cart = [similarity_transformation(self._lattice, sym)
                         for sym in site_symmetry]
        rot_disps = self._create_displacement_matrix(disps,
                                                     site_sym_cart)
        rot_forces = self._create_force_matrix(sets_of_forces,
                                               site_sym_cart,
                                               rot_map_syms)
        return rot_disps, rot_forces
        
    def _distribute(self):
        rotations = self._symmetry.get_symmetry_operations()['rotations']
        trans = self._symmetry.get_symmetry_operations()['translations']
        distribute_force_constants(self._fc2,
                                   range(self._num_atom),
                                   self._unique_first_atom_nums,
                                   self._lattice,
                                   self._positions,
                                   rotations,
                                   trans,
                                   self._symprec)

    def _create_force_matrix(self,
                             sets_of_forces,
                             site_sym_cart,
                             rot_map_syms):
        force_matrix = []
        for i in range(self._num_atom):
            for forces in sets_of_forces:
                for f, ssym_c in zip(
                    forces[rot_map_syms[:, i]], site_sym_cart):
                    force_matrix.append(np.dot(ssym_c, f))
        return np.reshape(force_matrix, (self._num_atom, -1, 3))

    def _create_displacement_matrix(self,
                                    disps,
                                    site_sym_cart):
        rot_disps = []
        for u in disps:
            for ssym_c in site_sym_cart:
                Su = np.dot(ssym_c, u)
                rot_disps.append(Su)

        return np.array(rot_disps, dtype='double')
        
    def _solve(self, rot_disps, rot_forces):
        inv_disps = np.linalg.pinv(rot_disps)
        return np.array([-np.dot(inv_disps, f) for f in rot_forces],
                        dtype='double')
