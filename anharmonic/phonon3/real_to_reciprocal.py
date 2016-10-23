import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors

class RealToReciprocal(object):
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 mesh,
                 symprec=1e-5):
        self._fc3 = fc3
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        
        num_satom = supercell.get_number_of_atoms()
        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        # Reduce supercell atom index to primitive index
        (self._smallest_vectors,
         self._multiplicity) = get_smallest_vectors(supercell,
                                                    primitive,
                                                    symprec)
        self._fc3_reciprocal = None

    def run(self, triplet):
        self._triplet = triplet
        num_patom = self._primitive.get_number_of_atoms()
        self._fc3_reciprocal = np.zeros(
            (num_patom, num_patom, num_patom, 3, 3, 3), dtype='complex128')
        self._real_to_reciprocal()

    def get_fc3_reciprocal(self):
        return self._fc3_reciprocal

    def _real_to_reciprocal(self):
        num_patom = self._primitive.get_number_of_atoms()
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0)
        sum_q = sum_triplets.astype('double') / self._mesh
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._fc3_reciprocal[
                        i, j, k] = self._real_to_reciprocal_elements((i, j, k))

            prephase = self._get_prephase(sum_q, i)
            self._fc3_reciprocal[i] *= prephase
                
    def _real_to_reciprocal_elements(self, patom_indices):
        num_satom = self._supercell.get_number_of_atoms()
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        fc3_reciprocal = np.zeros((3, 3, 3), dtype='complex128')
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                phase = self._get_phase((j, k), pi[0])
                fc3_reciprocal += self._fc3[i, j, k] * phase
        return fc3_reciprocal

    def _get_prephase(self, sum_q, patom_index):
        r = self._primitive.get_scaled_positions()[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase(self, satom_indices, patom0_index):
        si = satom_indices
        p0 = patom0_index
        phase = 1+0j
        for i in (0, 1):
            vs = self._smallest_vectors[si[i], p0,
                                        :self._multiplicity[si[i], p0]]
            phase *= (np.exp(2j * np.pi * np.dot(
                        vs, self._triplet[i + 1].astype('double') /
                        self._mesh)).sum() / self._multiplicity[si[i], p0])
        return phase
