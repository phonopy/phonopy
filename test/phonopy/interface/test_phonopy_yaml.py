import unittest

from phonopy.interface.phonopy_yaml import PhonopyYaml

class TestPhonopyYaml(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_read_poscar_yaml(self):
        filename = "POSCAR.yaml"
        phpy_yaml = PhonopyYaml(filename)
        print(phpy_yaml)

if __name__ == '__main__':
    unittest.main()
