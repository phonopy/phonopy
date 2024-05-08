"""Tests for dynmat_to_fc, inverse phonon transformation."""

import os
from io import StringIO

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import (
    DynmatToForceConstants,
    get_commensurate_points,
    get_commensurate_points_in_integers,
    ph2ph,
)
from phonopy.units import VaspToTHz

data_dir = os.path.dirname(os.path.abspath(__file__))


comm_points_str = """1  0.00  0.00  0.00
2  0.00  0.25  0.25
3  0.00  0.50  0.50
4  0.00  0.75  0.75
5  0.25  0.00  0.25
6  0.25  0.25  0.50
7  0.25  0.50  0.75
8  0.25  0.75  0.00
9  0.50  0.00  0.50
10  0.50  0.25  0.75
11  0.50  0.50  0.00
12  0.50  0.75  0.25
13  0.75  0.00  0.75
14  0.75  0.25  0.00
15  0.75  0.50  0.25
16  0.75  0.75  0.50
17  0.25  0.25  0.00
18  0.25  0.50  0.25
19  0.25  0.75  0.50
20  0.25  0.00  0.75
21  0.50  0.25  0.25
22  0.50  0.50  0.50
23  0.50  0.75  0.75
24  0.50  0.00  0.00
25  0.75  0.25  0.50
26  0.75  0.50  0.75
27  0.75  0.75  0.00
28  0.75  0.00  0.25
29  0.00  0.25  0.75
30  0.00  0.50  0.00
31  0.00  0.75  0.25
32  0.00  0.00  0.50"""


def test_get_commensurate_points():
    """Test for getting commensurate points."""
    comm_points, _ = _get_commensurate_points()
    _compare(comm_points)


def test_get_commensurate_points_in_integers():
    """Test for getting commensurate points represented by integers."""
    comm_points_ref, smat = _get_commensurate_points()
    comm_points = get_commensurate_points_in_integers(smat)
    comm_points = comm_points / np.linalg.det(smat)

    all_indices = []
    for cpt in comm_points:
        diff = comm_points_ref - cpt
        diff -= np.rint(diff)
        dist2 = (diff**2).sum(axis=1)
        indices = np.where(dist2 < 1e-5)[0]
        assert len(indices) == 1
        all_indices.append(indices[0])

    all_indices.sort()
    np.testing.assert_array_equal(all_indices, np.arange(len(comm_points)))


def _get_commensurate_points():
    smat = np.dot([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], np.diag([2, 2, 2]))
    return get_commensurate_points(smat), smat


def _compare(comm_points):
    comm_points_ref = np.loadtxt(StringIO(comm_points_str), dtype="double")
    diff = comm_points_ref[:, 1:] - comm_points
    np.testing.assert_allclose(diff, np.rint(diff), atol=1e-3)


def _write(comm_points, filename="comm_points.dat"):
    with open(os.path.join(data_dir, filename), "w") as w:
        lines = []
        for i, p in enumerate(comm_points):
            lines.append("%d %5.2f %5.2f %5.2f" % ((i + 1,) + tuple(p)))
        w.write("\n".join(lines))


def test_ph2ph_with_nac(ph_nacl):
    """Test transformation of phonon to force constants with NAC."""
    ph = ph_nacl
    fc333 = ph2ph(ph, np.diag([3, 3, 3]), with_nac=True).force_constants
    _phonons_allclose(ph, fc333)


def test_ph2ph_without_nac(ph_nacl_nonac):
    """Test transformation of phonon to force constants without NAC."""
    ph = ph_nacl_nonac
    fc333 = ph2ph(ph, np.diag([3, 3, 3]), with_nac=False).force_constants
    _phonons_allclose(ph, fc333)


def _phonons_allclose(ph: Phonopy, fc333):
    ph333 = Phonopy(
        ph.unitcell, supercell_matrix=[3, 3, 3], primitive_matrix=ph.primitive_matrix
    )
    ph333.force_constants = fc333
    ph333.nac_params = ph.nac_params
    comm_points = _get_comm_points(ph)
    ph.run_qpoints(comm_points)
    ph333.run_qpoints(comm_points)
    np.testing.assert_allclose(
        ph.get_qpoints_dict()["frequencies"],
        ph333.get_qpoints_dict()["frequencies"],
        atol=1e-5,
    )


def _get_comm_points(ph):
    smat = np.dot(np.linalg.inv(ph.primitive.primitive_matrix), ph.supercell_matrix)
    smat = np.rint(smat).astype(int)
    comm_points = get_commensurate_points(smat)
    return comm_points


def test_with_eigenvalues(ph_nacl, ph_nacl_nonac):
    """Test transformation from eigensolutions to dynamical matrix."""
    for ph in (ph_nacl_nonac, ph_nacl):
        d2f = DynmatToForceConstants(ph.primitive, ph.supercell)
        ph.run_qpoints(
            d2f.commensurate_points,
            with_eigenvectors=True,
            with_dynamical_matrices=True,
        )
        ph_dict = ph.get_qpoints_dict()
        eigenvalues = (ph_dict["frequencies"] / VaspToTHz) ** 2 * np.sign(
            ph_dict["frequencies"]
        )
        d2f.create_dynamical_matrices(
            eigenvalues=eigenvalues, eigenvectors=ph_dict["eigenvectors"]
        )
        d2f.run()
        np.testing.assert_allclose(ph.force_constants, d2f.force_constants, atol=1e-5)


@pytest.mark.parametrize(
    "is_nac,lang", [(False, "C"), (True, "C"), (False, "Py"), (True, "Py")]
)
def test_with_dynamical_matrices(ph_nacl, ph_nacl_nonac, is_nac, lang):
    """Test transformation from dynamical matrix to force constants."""
    if is_nac:
        ph = ph_nacl
    else:
        ph = ph_nacl_nonac

    d2f = DynmatToForceConstants(ph.primitive, ph.supercell)
    ph.run_qpoints(d2f.commensurate_points, with_dynamical_matrices=True)
    ph_dict = ph.get_qpoints_dict()
    d2f.dynamical_matrices = ph_dict["dynamical_matrices"]
    d2f.run(lang=lang)
    np.testing.assert_allclose(ph.force_constants, d2f.force_constants, atol=1e-5)
