import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation, get_rotated_displacement, get_rotated_forces, get_positions_sent_by_rot_inv, distribute_force_constants

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

    def get_fc2(self):
        return self._fc2

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

        rotations = self._symmetry.get_symmetry_operations()['rotations']
        trans = self._symmetry.get_symmetry_operations()['translations']
        distribute_force_constants(self._fc2,
                                   range(self._num_atom),
                                   unique_first_atom_nums,
                                   self._lattice,
                                   self._positions,
                                   rotations,
                                   trans,
                                   self._symprec)

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
        
            
