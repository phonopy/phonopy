import unittest

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry
from phonopy.interface.wien2k import parse_wien2k_struct
from phonopy.file_IO import parse_disp_yaml

class TestWien2k(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_parse_wien2k_struct(self):
        cell, npts, r0s, rmts = parse_wien2k_struct("BaGa2.struct")
        lattice = cell.get_cell().T
        displacements, supercell = parse_disp_yaml("disp_BaGa2.yaml",
                                                   return_cell=True)
        symmetry = Symmetry(cell)
        print(PhonopyAtoms(atoms=cell))
        sym_op = symmetry.get_symmetry_operations()
        print(symmetry.get_international_table())
        for i, (r, t) in enumerate(
                zip(sym_op['rotations'], sym_op['translations'])):
            print("--- %d ---" % (i + 1))
            print(r)
            print(t)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWien2k)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
