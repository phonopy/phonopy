import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors

class RealToReciprocal:
    def __init__(self,
                 fc3,
                 qpoint_triplets,
                 supercell,
                 primitive,
                 symprec=1e-5):
        self._fc3 = fc3
        self._qpoint_triplets = qpoint_triplets
        self._supercell = supercell
        self._primitive = primitive
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
        num_triplets = len(self._qpoint_triplets)
        self._fc3_reciprocal = np.zeros(
            (num_triplets, num_patom, num_patom, num_patom, 3, 3, 3),
            dtype='complex128')

        for i, triplet in enumerate(self._qpoint_triplets):
            print "%d / %d" % (i + 1, len(self._qpoint_triplets))
            print triplet
            self._real_to_reciprocal(triplet, i)

    def get_fc3_reciprocal(self):
        return self._fc3_reciprocal

    def _real_to_reciprocal(self, triplet, t_index):
        num_patom = self._primitive.get_number_of_atoms()
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._real_to_reciprocal_triplet(
                        (i, j, k), triplet, t_index)
                
    def _real_to_reciprocal_triplet(self, patom_indices, triplet, t_index):
        num_satom = self._supercell.get_number_of_atoms()
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        prephase = self._get_prephase(triplet, i)
        for j in range(num_satom):
            if self._s2p_map[j] == self._p2s_map[pi[1]]:
                for k in range(num_satom):
                    if self._s2p_map[k] == self._p2s_map[pi[2]]:
                        self._get_fc3_with_phases((i, j, k),
                                                  patom_indices,
                                                  triplet,
                                                  t_index,
                                                  prephase)
                                        
    def _get_prephase(self, triplet, satom_index):
        r = self._primitive.get_scaled_positions()[self._p2p_map[satom_index]]
        return np.exp(2j * np.pi * np.dot(triplet.sum(axis=0), r))

    def _get_fc3_with_phases(self,
                             satom_indices,
                             patom_indices,
                             triplet,
                             t_index,
                             prephase):
        si = satom_indices
        pi = patom_indices
        phase = 1+0j
        for i in (1, 2):
            vs = self._smallest_vectors[si[i], pi[i],
                                        :self._multiplicity[si[i], pi[i]]]
            phase *= (np.exp(2j * np.pi * np.dot(vs, triplet[i])).sum() /
                      self._multiplicity[si[i], pi[i]])

        self._fc3_reciprocal[t_index] = self._fc3[si] * phase * prephase
