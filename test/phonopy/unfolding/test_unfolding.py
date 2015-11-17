import unittest
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
        band = [np.array([[i, 0, 0] for i in range(21)], dtype='double') / 40]
        smat = np.diag([2, 2, 2])
        pmat = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        phonon_ideal = self._get_phonon(smat, pmat, self._cell)
        supercell = phonon_ideal.get_supercell()
        phonon = self._get_phonon(np.eye(3, dtype='intc'), np.eye(3), supercell)
        mapping = range(phonon.get_supercell().get_number_of_atoms())
        unfolding = Unfolding(phonon, phonon_ideal, mapping, band)

        for i, p in enumerate(unfolding.get_translations()):
            print("%d %s" % (i + 1, p))
        for i, p in enumerate(unfolding.get_commensurate_points()):
            print("%d %s" % (i + 1, p))
        print("%s" % unfolding.get_shifted_index_set())

        ## The following lines are for writing translations into POSCAR.
        # translations = unfolding.get_translations()
        # cell = Atoms(numbers=[1] * len(translations),
        #              scaled_positions=translations,
        #              cell=supercell.get_cell(),
        #              pbc=True)
        # write_vasp("POSCAR", cell)

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
        phonon.set_nac_params({'born': born,
                               'factor': factors,
                               'dielectric': epsilon})
        return phonon

if __name__ == '__main__':
    unittest.main()
