"""Tests for TetrahedronMesh."""

import os
from io import StringIO

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh

data_dir = os.path.dirname(os.path.abspath(__file__))

dos_str = """-0.672024 0.000000 0.029844 0.005522 0.731712 0.029450 1.433580 0.116998
2.135448 0.200612 2.837315 0.329781 3.539183 0.905737 4.241051 1.808068
4.942919 1.921464 5.644787 1.415087 6.346654 1.281276 7.048522 1.127877
7.750390 0.725551 8.452258 1.611979 9.154126 1.157525 9.855993 0.929781
10.557861 0.705942 11.259729 0.630369 11.961597 0.557250 12.663465 0.487940
13.365332 0.143148 14.067200 0.231518 14.769068 0.205816 15.470936 0.374575
16.172804 0.781758 16.874671 0.352501 17.576539 0.473537 18.278407 0.310505
18.980275 0.184665 19.682143 0.018812 20.384010 0.023455 21.085878 0.028958
21.787746 0.035963 22.489614 0.047177 23.191481 0.068537 23.893349 0.092864
24.595217 0.204422 25.297085 0.252177 25.998953 0.676881 26.700820 0.000000"""


@pytest.mark.parametrize("langs", [("C", "C"), ("Py", "C"), ("C", "Py")])
def test_Amm2(langs):
    """Test of DOS calculation using TetrahedronMesh for Amm2 crystal."""
    data = np.loadtxt(StringIO(dos_str))
    phonon = _get_phonon("Amm2", [3, 2, 2], [[1, 0, 0], [0, 0.5, -0.5], [0, 0.5, 0.5]])
    mesh = [11, 11, 11]
    primitive = phonon.primitive
    phonon.run_mesh([11, 11, 11])
    weights = phonon.mesh.weights
    frequencies = phonon.mesh.frequencies
    grid_address = phonon.mesh.grid_address
    ir_grid_points = phonon.mesh.ir_grid_points
    grid_mapping_table = phonon.mesh.grid_mapping_table
    th_mesh = TetrahedronMesh(
        primitive,
        frequencies,
        mesh,
        np.array(grid_address, dtype="long"),
        np.array(grid_mapping_table, dtype="long"),
        ir_grid_points,
        lang=langs[0],
    )
    th_mesh.set(value="I", division_number=40, lang=langs[1])
    freq_points = th_mesh.get_frequency_points()
    dos = np.zeros_like(freq_points)

    for i, iw in enumerate(th_mesh):
        dos += np.sum(iw * weights[i], axis=1)

    dos_comp = np.transpose([freq_points, dos]).reshape(10, 8)
    np.testing.assert_allclose(dos_comp, data, atol=1e-5)


def _show(freq_points, dos):
    data = []
    for f, d in zip(freq_points, dos):
        data.append([f, d])
    data = np.reshape(data, (10, 8))
    for row in data:
        print(("%f " * 8) % tuple(row))


def _get_phonon(spgtype, dim, pmat):
    cell = read_vasp(os.path.join(data_dir, "POSCAR_%s" % spgtype))
    phonon = Phonopy(cell, np.diag(dim), primitive_matrix=pmat)
    force_sets = parse_FORCE_SETS(
        filename=os.path.join(data_dir, "FORCE_SETS_%s" % spgtype)
    )
    phonon.dataset = force_sets
    phonon.produce_force_constants()
    return phonon
