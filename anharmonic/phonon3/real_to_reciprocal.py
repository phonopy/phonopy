import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors

class RealToReciprocal:
    def __init__(self,
                 fc3,
                 supercell,
                 primitive,
                 triplets_address,
                 mesh,
                 symprec=1e-5):
        self._fc3 = fc3
        self._supercell = supercell
        self._primitive = primitive
        self._triplets_address = triplets_address
        self._mesh = mesh
        self._symprec = symprec
        
        num_satom = supercell.get_number_of_atoms()
        self._p2s_map = np.intc(primitive.get_primitive_to_supercell_map())
        self._s2p_map = np.intc(primitive.get_supercell_to_primitive_map())
        p2p_map = primitive.get_primitive_to_primitive_map()
        # Reduce supercell atom index to primitive index
        self._p2p_map = [p2p_map[self._s2p_map[i]] for i in range(num_satom)]
        self._smallest_vectors, self._multiplicity = \
            get_smallest_vectors(supercell, primitive, symprec)

        self._fc3_reciprocal = None

    def run(self):
        num_patom = self._primitive.get_number_of_atoms()
        num_triplets = len(self._triplets_address)
        self._fc3_reciprocal = np.zeros(
            (num_triplets, num_patom, num_patom, num_patom, 3, 3, 3),
            dtype='complex128')

        for i, triplet in enumerate(self._triplets_address):
            print "%d / %d" % (i + 1, num_triplets)
            print triplet
            self._real_to_reciprocal(triplet, i)

    def get_fc3_reciprocal(self):
        return self._fc3_reciprocal

    def _real_to_reciprocal(self, triplet, t_index):
        num_patom = self._primitive.get_number_of_atoms()
        sum_triplets = np.where(
            np.all(triplet != 0, axis=0), triplet.sum(axis=0), 0)
        sum_q = sum_triplets.astype('double') / self._mesh

        print sum_q

        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._real_to_reciprocal_elements(
                        (i, j, k), triplet, t_index)

            prephase = self._get_prephase(sum_q, i)
            self._fc3_reciprocal[t_index, i] *= prephase
                
    def _real_to_reciprocal_elements(self, patom_indices, triplet, t_index):
        num_satom = self._supercell.get_number_of_atoms()
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                phase = self._get_phase((i, j, k), pi[0], triplet)
                self._fc3_reciprocal[
                    t_index, pi[0], pi[1], pi[2]] += self._fc3[i, j, k] * phase

    def _get_prephase(self, sum_q, patom_index):
        r = self._primitive.get_scaled_positions()[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase(self,
                   satom_indices,
                   patom0_index,
                   triplet):
        si = satom_indices
        p0 = patom0_index
        phase = 1+0j
        for i in (1, 2):
            vs = self._smallest_vectors[si[i], p0,
                                        :self._multiplicity[si[i], p0]]
            phase *= (np.exp(2j * np.pi * np.dot(
                        vs, triplet[i].astype('double') / self._mesh)).sum() /
                      self._multiplicity[si[i], p0])
        return phase
