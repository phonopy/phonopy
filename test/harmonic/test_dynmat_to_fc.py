import os
from phonopy import Phonopy
import numpy as np
from phonopy.units import VaspToTHz
from phonopy.harmonic.dynmat_to_fc import (
    get_commensurate_points, get_commensurate_points_in_integers,
    DynmatToForceConstants, ph2fc)

data_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_commensurate_points():
    comm_points, smat = _get_commensurate_points()
    _compare(comm_points)


def test_get_commensurate_points_in_integers():
    comm_points_ref, smat = _get_commensurate_points()
    comm_points = get_commensurate_points_in_integers(smat)
    comm_points = comm_points / np.linalg.det(smat)

    all_indices = []
    for cpt in comm_points:
        diff = comm_points_ref - cpt
        diff -= np.rint(diff)
        dist2 = (diff ** 2).sum(axis=1)
        indices = np.where(dist2 < 1e-5)[0]
        assert len(indices) == 1
        all_indices.append(indices[0])

    all_indices.sort()
    np.testing.assert_array_equal(all_indices, np.arange(len(comm_points)))


def _get_commensurate_points():
    smat = np.dot([[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                  np.diag([2, 2, 2]))
    return get_commensurate_points(smat), smat


def _compare(comm_points, filename="comm_points.dat"):
    with open(os.path.join(data_dir, filename)) as f:
        comm_points_in_file = np.loadtxt(f)
        diff = comm_points_in_file[:, 1:] - comm_points
        np.testing.assert_allclose(diff, np.rint(diff), atol=1e-3)


def _write(comm_points, filename="comm_points.dat"):
    with open(os.path.join(data_dir, filename), 'w') as w:
        lines = []
        for i, p in enumerate(comm_points):
            lines.append("%d %5.2f %5.2f %5.2f" % ((i + 1,) + tuple(p)))
        w.write("\n".join(lines))


def test_ph2fc(ph_nacl, ph_nacl_nonac):
    for ph in (ph_nacl_nonac, ph_nacl):
        fc333 = ph2fc(ph, np.diag([3, 3, 3]))
        _phonons_allclose(ph, fc333)


def _phonons_allclose(ph, fc333):
    ph333 = Phonopy(ph.unitcell,
                    supercell_matrix=[3, 3, 3],
                    primitive_matrix=ph.primitive_matrix)
    ph333.force_constants = fc333
    ph333.nac_params = ph.nac_params
    comm_points = _get_comm_points(ph)
    ph.run_qpoints(comm_points)
    ph333.run_qpoints(comm_points)
    np.testing.assert_allclose(ph.get_qpoints_dict()['frequencies'],
                               ph333.get_qpoints_dict()['frequencies'],
                               atol=1e-5)


def _get_comm_points(ph):
    smat = np.dot(np.linalg.inv(ph.primitive.primitive_matrix),
                  ph.supercell_matrix)
    smat = np.rint(smat).astype(int)
    comm_points = get_commensurate_points(smat)
    return comm_points


def test_with_eigenvalues(ph_nacl, ph_nacl_nonac):
    for ph in (ph_nacl_nonac, ph_nacl):
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


def test_with_dynamical_matrices(ph_nacl, ph_nacl_nonac):
    for ph in (ph_nacl, ph_nacl_nonac):
        d2f = DynmatToForceConstants(ph.primitive, ph.supercell)
        ph.run_qpoints(d2f.commensurate_points,
                       with_dynamical_matrices=True)
        ph_dict = ph.get_qpoints_dict()
        d2f.dynamical_matrices = ph_dict['dynamical_matrices']
        d2f.run()
        np.testing.assert_allclose(ph.force_constants, d2f.force_constants,
                                   atol=1e-5)
