import numpy as np
from phonopy.harmonic.dynamical_matrix import get_smallest_vectors

class RealToReciprocal:
    def __init__(self,
                 fc4,
                 supercell,
                 primitive,
                 mesh,
                 symprec=1e-5):
        self._fc4 = fc4
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec

        num_satom = supercell.get_number_of_atoms()
        self._p2s_map = primitive.get_primitive_to_supercell_map()
        self._s2p_map = primitive.get_supercell_to_primitive_map()
        (self._smallest_vectors,
         self._multiplicity) = get_smallest_vectors(supercell,
                                                    primitive,
                                                    symprec)
        self._quartet = None
        self._fc4_reciprocal = None

    def run(self, quartet, lang='py'):
        self._quartet = quartet
        num_patom = self._primitive.get_number_of_atoms()
        self._fc4_reciprocal = np.zeros(
            (num_patom,) * 4 + (3,) * 4, dtype='complex128')

        if lang=='C':
            self._real_to_reciprocal_c()
        else:
            self._real_to_reciprocal_py()

    def get_fc4_reciprocal(self):
        return self._fc4_reciprocal

    def _real_to_reciprocal_c(self):
        import anharmonic._phono4py as phono4c
        phono4c.real_to_reciprocal4(self._fc4_reciprocal,
                                    np.double(self._fc4),
                                    np.double(self._quartet /
                                              self._mesh.astype('double')),
                                    self._smallest_vectors,
                                    self._multiplicity,
                                    self._p2s_map,
                                    self._s2p_map)

    def _real_to_reciprocal_py(self):
        num_patom = self._primitive.get_number_of_atoms()
        for (i, j, k, l) in list(np.ndindex(
                num_patom, num_patom, num_patom, num_patom)):
            self._fc4_reciprocal[
                i, j, k, l] = self._real_to_reciprocal_elements((i, j, k, l))

    def _real_to_reciprocal_elements(self, patom_indices):
        num_satom = self._supercell.get_number_of_atoms()
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        fc4_reciprocal = np.zeros((3, 3, 3, 3), dtype='complex128')
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                for l in range(num_satom):
                    if self._s2p_map[l] != self._p2s_map[pi[3]]:
                        continue
                    phase = self._get_phase((j, k, l), pi[0])
                    fc4_reciprocal += self._fc4[i, j, k, l] * phase
        return fc4_reciprocal

    def _get_phase(self, satom_indices, patom0_index):
        si = satom_indices
        p0 = patom0_index
        phase = 1+0j
        for i in (0, 1, 2):
            vs = self._smallest_vectors[si[i], p0,
                                        :self._multiplicity[si[i], p0]]
            phase *= (np.exp(2j * np.pi * np.dot(
                        vs, self._quartet[i + 1].astype('double') /
                        self._mesh)).sum() / self._multiplicity[si[i], p0])
        return phase
