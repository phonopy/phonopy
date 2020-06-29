import os
import unittest
import phonopy
from phonopy import Phonopy
import numpy as np
from phonopy.units import VaspToTHz
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.harmonic.dynmat_to_fc import (
    get_commensurate_points, DynmatToForceConstants)
from phonopy.structure.cells import get_supercell, get_primitive
from phonopy.file_IO import write_FORCE_CONSTANTS

data_dir = os.path.dirname(os.path.abspath(__file__))


class TestGetCommensuratePoints(unittest.TestCase):

    def setUp(self):
        filename = os.path.join("..", "NaCl.yaml")
        self._cell = read_cell_yaml(os.path.join(data_dir, filename))

    def tearDown(self):
        pass

    def test_get_commensurate_points(self):
        smat = np.diag([2, 2, 2])
        pmat = np.dot(np.linalg.inv(smat),
                      [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        supercell = get_supercell(self._cell, smat)
        primitive = get_primitive(supercell, pmat)
        supercell_matrix = np.linalg.inv(primitive.primitive_matrix)
        supercell_matrix = np.rint(supercell_matrix).astype('intc')
        comm_points = get_commensurate_points(supercell_matrix)
        # self._write(comm_points)
        self._compare(comm_points)

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

    def test_over_sampling(self):
        """This test shows how to over-sample commensurate q-points.

        When NAC is used, the dynamical matrices including NAC should be used.
        The force constants created should be used with NAC again.

        """

        ph = self.ph_nac
        ph444 = Phonopy(ph.unitcell,
                        supercell_matrix=[4, 4, 4],
                        primitive_matrix=ph.primitive_matrix)
        d2f = DynmatToForceConstants(ph444.primitive, ph444.supercell)
        ph.run_qpoints(d2f.commensurate_points,
                       with_dynamical_matrices=True)
        ph_dict = ph.get_qpoints_dict()
        d2f.dynamical_matrices = ph_dict['dynamical_matrices']
        d2f.run()

        ph444_nac = Phonopy(ph.unitcell,
                            supercell_matrix=[4, 4, 4],
                            primitive_matrix=ph.primitive_matrix,
                            calculator='vasp')
        ph444.force_constants = d2f.force_constants
        ph444_nac.nac_params = ph.nac_params

        smat = np.dot(np.linalg.inv(ph.primitive.primitive_matrix),
                      ph.supercell_matrix)
        smat = np.rint(smat).astype(int)
        comm_points = get_commensurate_points(smat)

        ph.run_qpoints(comm_points)
        ph444.run_qpoints(comm_points)
        np.testing.assert_allclose(ph.get_qpoints_dict()['frequencies'],
                                   ph444.get_qpoints_dict()['frequencies'],
                                   atol=1e-5)


if __name__ == '__main__':
    unittest.main()
