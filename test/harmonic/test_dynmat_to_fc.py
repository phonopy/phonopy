import os
import unittest
import phonopy
from phonopy import Phonopy
import numpy as np
from phonopy.units import VaspToTHz
from phonopy.structure.cells import get_supercell, get_primitive
from phonopy.harmonic.dynmat_to_fc import (
    get_commensurate_points, get_commensurate_points_in_integers,
    DynmatToForceConstants, ph2fc)

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestGetCommensuratePoints(unittest.TestCase):

    def setUp(self):
        self._smat = np.dot([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], np.diag([2, 2, 2]))
        self._comm_points = get_commensurate_points(self._smat)

    def tearDown(self):
        pass

    def test_get_commensurate_points(self):
        # self._write(comm_points)
        self._compare(self._comm_points)

    def test_get_commensurate_points_in_integers(self):
        comm_points = get_commensurate_points_in_integers(self._smat)
        comm_points = comm_points / np.linalg.det(self._smat)

        all_indices = []
        for cpt in comm_points:
            diff = self._comm_points - cpt
            diff -= np.rint(diff)
            dist2 = (diff ** 2).sum(axis=1)
            indices = np.where(dist2 < 1e-5)[0]
            self.assertTrue(len(indices) == 1)
            all_indices.append(indices[0])

        all_indices.sort()
        np.testing.assert_array_equal(all_indices, np.arange(len(comm_points)))

    def _compare(self, comm_points, filename="comm_points.dat"):
        with open(os.path.join(data_dir, filename)) as f:
            comm_points_in_file = np.loadtxt(f)
            diff = comm_points_in_file[:, 1:] - comm_points
            np.testing.assert_allclose(diff, np.rint(diff), atol=1e-3)

    def _write(self, comm_points, filename="comm_points.dat"):
        with open(os.path.join(data_dir, filename), 'w') as w:
            lines = []
            for i, p in enumerate(comm_points):
                lines.append("%d %5.2f %5.2f %5.2f" % ((i + 1,) + tuple(p)))
            w.write("\n".join(lines))


class TestDynmatToForceConstants(unittest.TestCase):
    def setUp(self):
        filename = os.path.join(data_dir, "..", "POSCAR_NaCl")
        force_sets_filename = os.path.join(data_dir, "..", "FORCE_SETS_NaCl")
        born_filename = os.path.join(data_dir, "..", "BORN_NaCl")
        self.ph = phonopy.load(unitcell_filename=filename,
                               supercell_matrix=[2, 2, 2],
                               calculator='vasp',
                               force_sets_filename=force_sets_filename,
                               is_compact_fc=False)
        self.ph_nac = phonopy.load(unitcell_filename=filename,
                                   supercell_matrix=[2, 2, 2],
                                   calculator='vasp',
                                   force_sets_filename=force_sets_filename,
                                   born_filename=born_filename,
                                   is_compact_fc=False)

    def tearDown(self):
        pass

    def test_with_eigenvalues(self):
        for ph in (self.ph, self.ph_nac):
            d2f = DynmatToForceConstants(ph.primitive, ph.supercell)
            ph.run_qpoints(d2f.commensurate_points,
                           with_eigenvectors=True,
                           with_dynamical_matrices=True)
            ph_dict = ph.get_qpoints_dict()
            eigenvalues = ((ph_dict['frequencies'] / VaspToTHz) ** 2 *
                           np.sign(ph_dict['frequencies']))
            d2f.create_dynamical_matrices(eigenvalues=eigenvalues,
                                          eigenvectors=ph_dict['eigenvectors'])
            d2f.run()
            np.testing.assert_allclose(ph.force_constants, d2f.force_constants,
                                       atol=1e-5)

    def test_with_dynamical_matrices(self):
        for ph in (self.ph, ):
            d2f = DynmatToForceConstants(ph.primitive, ph.supercell)
            ph.run_qpoints(d2f.commensurate_points,
                           with_dynamical_matrices=True)
            ph_dict = ph.get_qpoints_dict()
            d2f.dynamical_matrices = ph_dict['dynamical_matrices']
            d2f.run()
            np.testing.assert_allclose(ph.force_constants, d2f.force_constants,
                                       atol=1e-5)

    def test_ph2fc(self):
        ph = self.ph_nac
        fc333 = ph2fc(ph, np.diag([3, 3, 3]))
        self._phonons_allclose(ph, fc333)

    def _phonons_allclose(self, ph, fc333):
        ph333 = Phonopy(ph.unitcell,
                        supercell_matrix=[3, 3, 3],
                        primitive_matrix=ph.primitive_matrix)
        ph333.force_constants = fc333
        ph333.nac_params = ph.nac_params
        comm_points = self._get_comm_points(ph)
        ph.run_qpoints(comm_points)
        ph333.run_qpoints(comm_points)
        np.testing.assert_allclose(ph.get_qpoints_dict()['frequencies'],
                                   ph333.get_qpoints_dict()['frequencies'],
                                   atol=1e-5)

    def _get_comm_points(self, ph):
        smat = np.dot(np.linalg.inv(ph.primitive.primitive_matrix),
                      ph.supercell_matrix)
        smat = np.rint(smat).astype(int)
        comm_points = get_commensurate_points(smat)
        return comm_points


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGetCommensuratePoints)
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestDynmatToForceConstants)
    unittest.TextTestRunner(verbosity=2).run(suite)
