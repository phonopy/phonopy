import unittest

from phonopy.interface.phonopy_yaml import phonopyYaml
from phonopy.structure.atoms import PhonopyAtoms

class TestPhonopyYaml(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_read_poscar_yaml(self):
        filename = "POSCAR.yaml"
        phpy_yaml = phonopyYaml(filename)
        print phpy_yaml

if __name__ == '__main__':
    unittest.main()
