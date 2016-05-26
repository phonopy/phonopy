import unittest

from phonopy.structure.atoms import PhonopyAtoms, Atoms

class TestCell(unittest.TestCase):

    def setUp(self):
        symbols = ['Si'] * 2 + ['O'] * 4
        lattice = [[4.65, 0, 0],
                   [0, 4.75, 0],
                   [0, 0, 3.25]]
        points = [[0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.5],
                  [0.3, 0.3, 0.0],
                  [0.7, 0.7, 0.0],
                  [0.2, 0.8, 0.5],
                  [0.8, 0.2, 0.5]]
        
        self._cell = Atoms(cell=lattice,
                           scaled_positions=points,
                           symbols=symbols)
    
    def tearDown(self):
        pass
    
    def test_atoms(self):
        print(self._cell.get_cell())
        for s, p in zip(self._cell.get_chemical_symbols(),
                        self._cell.get_scaled_positions()):
            print("%s %s" % (s, p))

    def test_phonopy_atoms(self):
        print(PhonopyAtoms(atoms=self._cell))

if __name__ == '__main__':
    unittest.main()
