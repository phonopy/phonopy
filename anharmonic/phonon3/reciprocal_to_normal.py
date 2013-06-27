import numpy as np

class ReciprocalToNormal:
    def __init__(self,
                 fc3_reciprocal,
                 qpoint_triplets,
                 primitive,
                 dynamical_matrix,
                 frequencies,
                 eigenvectors,
                 q_done):
        self._fc3_reciprocal = fc3_reciprocal
        self._qpoint_triplets = qpoint_triplets
        self._primitive = primitive
        self._dm = dynamical_matrix
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._q_done = q_done

    def run(self):
        pass
