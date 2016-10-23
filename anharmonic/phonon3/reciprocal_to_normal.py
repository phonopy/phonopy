import numpy as np

class ReciprocalToNormal(object):
    def __init__(self,
                 primitive,
                 frequencies,
                 eigenvectors,
                 band_indices,
                 cutoff_frequency=0):
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency

        self._masses = self._primitive.get_masses()

        self._fc3_normal = None
        self._fc3_reciprocal = None

    def run(self, fc3_reciprocal, grid_triplet):
        num_band = self._primitive.get_number_of_atoms() * 3
        self._fc3_reciprocal = fc3_reciprocal
        self._fc3_normal = np.zeros(
            (len(self._band_indices), num_band, num_band), dtype='complex128')
        self._reciprocal_to_normal(grid_triplet)

    def get_reciprocal_to_normal(self):
        return self._fc3_normal

    def _reciprocal_to_normal(self, grid_triplet):
        e1, e2, e3 = self._eigenvectors[grid_triplet]
        f1, f2, f3 = self._frequencies[grid_triplet]
        num_band = len(f1)
        cutoff = self._cutoff_frequency
        for (i, j, k) in list(np.ndindex(
                len(self._band_indices), num_band, num_band)):
            bi = self._band_indices[i]
            if f1[bi] > cutoff and f2[j] > cutoff and f3[k] > cutoff:
                fc3_elem = self._sum_in_atoms((bi, j, k), (e1, e2, e3))
                fff = np.sqrt(f1[bi] * f2[j] * f3[k])
                self._fc3_normal[i, j, k] = fc3_elem / fff

    def _sum_in_atoms(self, band_indices, eigvecs):
        num_atom = self._primitive.get_number_of_atoms()
        (e1, e2, e3) = eigvecs
        (b1, b2, b3) = band_indices

        sum_fc3 = 0j
        for (i, j, k) in list(np.ndindex((num_atom,) * 3)):
            sum_fc3_cart = 0
            for (l, m, n) in list(np.ndindex((3, 3, 3))):
                sum_fc3_cart += (e1[i * 3 + l, b1] *
                                 e2[j * 3 + m, b2] *
                                 e3[k * 3 + n, b3] *
                                 self._fc3_reciprocal[i, j, k, l, m, n])
            mass_sqrt = np.sqrt(np.prod(self._masses[[i, j, k]]))
            sum_fc3 += sum_fc3_cart / mass_sqrt

        return sum_fc3
