import unittest
import os
import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
from phonopy.phonon.random_displacements import RandomDisplacements

data_dir = os.path.dirname(os.path.abspath(__file__))

disp_str = """-0.0514654  0.1115076  0.0011100
-0.1780910 -0.0973049  0.1698372
-0.1532055 -0.1685844  0.2696392
-0.1985800 -0.1314423 -0.0496898
-0.1080178 -0.1717647 -0.0204162
 0.0936793  0.0296741 -0.1511135
-0.3356941 -0.1459993 -0.0281809
-0.2976510  0.1502010 -0.2907214
-0.2334291 -0.1556119  0.0821270
-0.1293066 -0.0674755 -0.1257030
-0.4017851 -0.1158274  0.0385163
-0.3751805  0.2183430 -0.4037309
-0.1135429 -0.0615372 -0.1399313
-0.1850432  0.1352458 -0.2158535
-0.2585097  0.0631317 -0.0843370
-0.3165835  0.1223332 -0.1733457
-0.1742562 -0.1309926  0.1222215
 0.0640609 -0.0245728 -0.1738627
-0.3140855 -0.1136079 -0.0376838
-0.3160274  0.1858955 -0.3112977
 0.1140535  0.0223505 -0.1936938
-0.0413314  0.1457069 -0.2165871
-0.2690085  0.0958113 -0.1638003
-0.3147487  0.1233289 -0.1184777
-0.1877001 -0.1533800  0.2366920
-0.0560419 -0.1331006 -0.1575333
-0.3035009 -0.0885274  0.0330783
-0.3380943  0.2472368 -0.3767877
-0.1372640 -0.0561450 -0.1189917
-0.1928859  0.1598169 -0.2433005
-0.3312110  0.1267395 -0.3016184
-0.3193205  0.1798292 -0.1289612
 0.1265522 -0.1787818  0.0009789
 0.0639243 -0.0548810 -0.0227700
-0.0470768  0.1475875 -0.2789066
-0.0783960  0.1894551 -0.0046175
-0.2270308 -0.0833323 -0.2325597
-0.2341782  0.3991952 -0.0731757
-0.3140104  0.2323930 -0.1028991
-0.5137982  0.1798655  0.0336329
 0.1871071 -0.0143198  0.1109201
 0.0259593 -0.2209235  0.1097609
 0.0934036 -0.0544112 -0.1293274
 0.0348889  0.0155738 -0.0953390
 0.0243686 -0.2053247  0.0856274
-0.0469223 -0.1280702  0.0609982
-0.1300982  0.0464706 -0.2908150
-0.1562722  0.3817999 -0.0706285
 0.2520720 -0.0446607  0.1601216
 0.0748614 -0.1583794  0.0007053
 0.0950268  0.0511400 -0.2307183
 0.0185852  0.1263185 -0.1600272
 0.0455906 -0.1628427  0.0008357
-0.0648477 -0.0435683 -0.1238486
-0.2094037  0.0993982 -0.3790105
-0.2272393  0.2552084 -0.0348276
 0.2935021 -0.0194070  0.2240964
 0.0923349 -0.2410340  0.1113123
 0.1087797 -0.0745681 -0.1396408
 0.0224714  0.0611535 -0.1814881
-0.1036679 -0.1435858  0.0360042
-0.2067574 -0.1451977 -0.0743127
-0.2753820 -0.0162736 -0.3580231
-0.2856068  0.3571327 -0.1188627"""


class TestRandomDisplacements(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _get_phonon_NaCl(self):
        cell = read_vasp(os.path.join(data_dir, "..", "POSCAR_NaCl"))
        phonon = Phonopy(cell,
                         np.diag([2, 2, 2]),
                         primitive_matrix=[[0, 0.5, 0.5],
                                           [0.5, 0, 0.5],
                                           [0.5, 0.5, 0]])
        filename = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        force_sets = parse_FORCE_SETS(filename=filename)
        phonon.set_displacement_dataset(force_sets)
        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()
        filename_born = os.path.join(data_dir, "..", "BORN_NaCl")
        nac_params = parse_BORN(phonon.get_primitive(), filename=filename_born)
        phonon.set_nac_params(nac_params)
        return phonon

    def test_NaCl(self):
        phonon = self._get_phonon_NaCl()
        rd = RandomDisplacements(phonon.supercell,
                                 phonon.primitive,
                                 phonon.force_constants,
                                 cutoff_frequency=0.01)
        rd.run(500, number_of_snapshots=1, random_seed=100)
        data = np.fromstring(disp_str.replace('\n', ' '), dtype=float, sep=' ')
        np.testing.assert_allclose(data, rd.u.ravel(), atol=1e-5)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestRandomDisplacements)
    unittest.TextTestRunner(verbosity=2).run(suite)
