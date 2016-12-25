import unittest

import numpy as np
import tarfile
import os
from phonopy.interface.vasp import Vasprun, read_vasp
from phonopy.interface.phonopy_yaml import get_unitcell_from_phonopy_yaml
from phonopy.file_IO import parse_FORCE_SETS

data_dir = os.path.dirname(os.path.abspath(__file__))

class TestVASP(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_read_vasp(self):
        cell = read_vasp(os.path.join(data_dir, "../POSCAR_NaCl"))
        filename = os.path.join(data_dir, "NaCl-vasp.yaml")
        cell_ref = get_unitcell_from_phonopy_yaml(filename)
        self.assertTrue(
            (np.abs(cell.get_cell() - cell_ref.get_cell()) < 1e-5).all())
        diff_pos = cell.get_scaled_positions() - cell_ref.get_scaled_positions()
        diff_pos -= np.rint(diff_pos)
        self.assertTrue((np.abs(diff_pos) < 1e-5).all())
        for s, s_r in zip(cell.get_chemical_symbols(),
                          cell_ref.get_chemical_symbols()):
            self.assertTrue(s == s_r)

    def test_parse_vasprun_xml(self):
        filename_vasprun = os.path.join(data_dir, "vasprun.xml.tar.bz2")
        self._tar = tarfile.open(filename_vasprun)
        filename = os.path.join(data_dir, "FORCE_SETS_NaCl")
        dataset = parse_FORCE_SETS(filename=filename)
        for i, member in enumerate(self._tar.getmembers()):
            vr = Vasprun(self._tar.extractfile(member), use_expat=True)
            # for force in vr.read_forces():
            #     print("% 15.8f % 15.8f % 15.8f" % tuple(force))
            # print("")
            ref = dataset['first_atoms'][i]['forces']
            np.testing.assert_allclose(ref, vr.read_forces(), atol=1e-8)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVASP)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main()
