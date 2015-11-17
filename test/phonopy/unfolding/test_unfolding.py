import unittest
import sys
import numpy as np
from phonopy.interface.phonopy_yaml import phonopyYaml
from phonopy.structure.cells import get_supercell
from phonopy.unfolding import Unfolding
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.structure.atoms import PhonopyAtoms
# from phonopy.structure.atoms import Atoms
# from phonopy.interface.vasp import write_vasp

class TestUnfolding(unittest.TestCase):

    def setUp(self):
        self._cell = read_vasp("POSCAR")
        print(PhonopyAtoms(atoms=self._cell))
    
    def tearDown(self):
        pass
    
    def test_Unfolding(self):
        nd = 10
        band = [np.array([[i, 0, 0]
                          for i in range(nd + 1)], dtype='double') / nd / 2]
        smat = np.diag([2, 2, 2])
        pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        phonon_ideal = self._get_phonon(smat, pmat, self._cell)
        supercell = phonon_ideal.get_supercell()
        phonon = self._get_phonon(np.eye(3, dtype='intc'), np.eye(3), supercell)
        mapping = range(phonon.get_supercell().get_number_of_atoms())
        unfolding = Unfolding(phonon, phonon_ideal, mapping, band)
        
        comm_points = unfolding.get_commensurate_points()
        # (nd + 1, num_atom_super / num_atom_prim, num_atom_super * 3)
        weights = unfolding.get_unfolding_weights() 
        freqs = unfolding.get_frequencies()

        print(weights.shape)

        with open("unfolding.dat", 'w') as w:
            lines = []
            for i, q in enumerate(band[0]):
                for j, f in enumerate(freqs[i]):
                    for k, G in enumerate(comm_points):
                        lines.append(
                            "%f %f %f  %f %f %f  %f  %f" %
                            (q[0], q[1], q[2], G[0], G[1], G[2], f,
                             weights[i, j, k]))
            w.write("\n".join(lines))
        
    def _get_phonon(self, smat, pmat, cell):
        print smat
        print pmat
        phonon = Phonopy(cell,
                         smat,
                         primitive_matrix=pmat,
                         is_auto_displacements=False)
        force_sets = parse_FORCE_SETS()
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        born = [[[1.08703, 0, 0],
                 [0, 1.08703, 0],
                 [0, 0, 1.08703]],
                [[-1.08672, 0, 0],
                 [0, -1.08672, 0],
                 [0, 0, -1.08672]]]
        epsilon = [[2.43533967, 0, 0],
                   [0, 2.43533967, 0],
                   [0, 0, 2.43533967]]
        factors = 14.400
        # phonon.set_nac_params({'born': born,
        #                        'factor': factors,
        #                        'dielectric': epsilon})
        return phonon

if __name__ == '__main__':
    unittest.main()
