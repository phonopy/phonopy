import unittest
import sys
import numpy as np
from phonopy import Phonopy
from phonopy.phonon.moment import PhononMoment
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN

class TestMoment(unittest.TestCase):
    def setUp(self):
        self._cell = read_vasp("POSCAR")
    
    def tearDown(self):
        pass
    
    def test_moment(self):
        phonon = self._get_phonon(self._cell)
        moment = phonon.set_mesh([12, 12, 12], is_eigenvectors=True)
        q, w, f, e = phonon.get_mesh()
        moment = PhononMoment(f, e, w)
        for i in range(3):
            moment.run(order=i)
            print(moment.get_moment())

    def _get_phonon(self, cell):
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[
                             [0, 0.5, 0.5],
                             [0.5, 0, 0.5],
                             [0.5, 0.5, 0]],
                         is_auto_displacements=False)
        force_sets = parse_FORCE_SETS()
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        supercell = phonon.get_supercell()
        born_elems = {'Na': [[1.08703, 0, 0],
                             [0, 1.08703, 0],
                             [0, 0, 1.08703]],
                      'Cl': [[-1.08672, 0, 0],
                             [0, -1.08672, 0],
                             [0, 0, -1.08672]]}
        born = [born_elems[s]
                for s in supercell.get_chemical_symbols()]
        epsilon = [[2.43533967, 0, 0],
                   [0, 2.43533967, 0],
                   [0, 0, 2.43533967]]
        factors = 14.400
        phonon.set_nac_params({'born': born,
                               'factor': factors,
                               'dielectric': epsilon})
        return phonon

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoment)
    unittest.TextTestRunner(verbosity=2).run(suite)
