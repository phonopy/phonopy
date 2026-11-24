# SPDX-License-Identifier: BSD-3-Clause
"""Tests for PhononMoment."""

from io import StringIO

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.phonon.moment import PhononMoment

result_full_range = """
1.000000  1.000000  1.000000
 4.063222  4.236805  3.889623
17.935854 19.412820 16.458756
 1.000000  1.000000  1.000000
 3.530039  3.621065  3.451029
12.557720 13.205191 11.995720
"""


def test_moment(ph_nacl_nofcsym: Phonopy):
    """Test PhononMoment."""
    phonon = ph_nacl_nofcsym
    data = np.loadtxt(StringIO(result_full_range), dtype="double")
    moment = phonon.run_mesh(
        [13, 13, 13], with_eigenvectors=True, is_mesh_symmetry=False
    )
    num_atom = len(phonon.primitive)
    w = phonon.mesh.weights
    f = phonon.mesh.frequencies
    e = phonon.mesh.eigenvectors
    vals = np.zeros((6, num_atom + 1), dtype="double")

    moment = PhononMoment(f, w)
    for i in range(3):
        moment.run(order=i)
        vals[i, 0] = moment.moment
        assert np.abs(moment.moment - data[i, 0]) < 1e-5

    moment = PhononMoment(f, w, eigenvectors=e)
    for i in range(3):
        moment.run(order=i)
        moms = moment.moment
        vals[i, 1:] = moms
        assert (np.abs(moms - data[i, 1:]) < 1e-5).all()

    moment = PhononMoment(f, w)
    moment.set_frequency_range(freq_min=3, freq_max=4)
    for i in range(3):
        moment.run(order=i)
        vals[i + 3, 0] = moment.moment
        assert np.abs(moment.moment - data[i + 3, 0]) < 1e-5

    moment = PhononMoment(f, w, eigenvectors=e)
    moment.set_frequency_range(freq_min=3, freq_max=4)
    for i in range(3):
        moment.run(order=i)
        moms = moment.moment
        vals[i + 3, 1:] = moms
        assert (np.abs(moms - data[i + 3, 1:]) < 1e-5).all()

    # self._show(vals)


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "order, expected", [(1, [13 / 6, 7 / 4]), (2, [47 / 9, 23 / 6])]
)
def test_projected_moment_frequency_window(rotate, order, expected):
    """Normalize each atom's PDOS after summing Cartesian projections."""
    polarizations = np.eye(3)
    if rotate:
        polarizations = np.array(
            [
                [1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(6)],
                [1 / np.sqrt(3), -1 / np.sqrt(2), 1 / np.sqrt(6)],
                [1 / np.sqrt(3), 0, -2 / np.sqrt(6)],
            ]
        )
    atom_mixing = np.array([[2, -1], [1, 2]]) / np.sqrt(5)
    eigenvectors = np.kron(atom_mixing, polarizations)
    frequencies = np.array([[1, 2, 4, 3, 5, 7], [2, 3, 6, 1, 4, 8]], dtype=float)
    moment = PhononMoment(
        frequencies,
        np.array([1, 3]),
        eigenvectors=np.array([eigenvectors, eigenvectors], dtype=complex),
    )
    moment.set_frequency_range(freq_min=0.5, freq_max=3.5)
    moment.run(order=order)

    # The selected modes have atom weights (4/5, 1/5), (4/5, 1/5),
    # and (1/5, 4/5), independently of the Cartesian polarization basis.
    np.testing.assert_allclose(moment.moment, expected, atol=1e-12)


def _show(vals):
    for v in vals:
        print(("%9.6f " * len(v)) % tuple(v))
