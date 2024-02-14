"""Tests for PhononMoment."""

from io import StringIO

import numpy as np

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


def _show(vals):
    for v in vals:
        print(("%9.6f " * len(v)) % tuple(v))
