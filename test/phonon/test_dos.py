"""Tests for DOS."""

import numpy as np

from phonopy import Phonopy
from phonopy.phonon.dos import get_pdos_indices

tp_str = """0.000000 100.000000 200.000000 300.000000 400.000000
500.000000 600.000000 700.000000 800.000000 900.000000
4.856373 3.916036 -0.276031 -6.809284 -14.961974
-24.342086 -34.708562 -45.898943 -57.796507 -70.313405
0.000000 26.328820 55.258118 74.269718 88.156943
99.054231 108.009551 115.606163 122.200204 128.024541
0.000000 36.207859 45.673598 47.838871 48.634251
49.009334 49.214960 49.339592 49.420745 49.476502"""

tdos_str = """-0.750122 0.000000
0.249878 0.000041
1.249878 0.010398
2.249878 1.363019
3.249878 1.837252
4.249878 0.991171
5.249878 0.342658
6.249878 0.106961
7.249878 0.032966"""

tdos_thm_str = """-0.750122 0.000000
0.249878 0.004508
1.249878 0.090716
2.249878 0.549237
3.249878 1.093288
4.249878 1.349766
5.249878 0.660090
6.249878 0.461219
7.249878 0.000000"""

pdos_str = """-0.750122 0.000000 0.000000
0.249878 0.000016 0.000025
1.249878 0.004469 0.005929
2.249878 0.620921 0.742098
3.249878 0.597587 1.239666
4.249878 0.501077 0.490094
5.249878 0.142162 0.200496
6.249878 0.070026 0.036935
7.249878 0.020225 0.012741"""

pdos_thm_str = """-0.750122 0.000000 0.000000
0.249878 0.001820 0.002688
1.249878 0.037419 0.053297
2.249878 0.228129 0.321109
3.249878 0.403637 0.689651
4.249878 0.756532 0.593234
5.249878 0.454573 0.205518
6.249878 0.322527 0.138692
7.249878 0.000000 0.000000"""


def testTotalDOS(ph_nacl_nofcsym: Phonopy):
    """Test of total DOS with smearing method."""
    phonon = ph_nacl_nofcsym
    phonon.run_mesh([5, 5, 5])
    phonon.run_total_dos(freq_pitch=1, use_tetrahedron_method=False)
    dos = phonon.total_dos.dos
    freqs = phonon.total_dos.frequency_points
    data_ref = np.reshape([float(x) for x in tdos_str.split()], (-1, 2))
    np.testing.assert_allclose(data_ref, np.c_[freqs, dos], atol=1e-5)
    # for f, d in zip(freqs, dos):
    #     print("%f %f" % (f, d))


def testTotalDOSTetrahedron(ph_nacl_nofcsym: Phonopy):
    """Test of total DOS with tetrahedron method."""
    phonon = ph_nacl_nofcsym
    phonon.run_mesh([5, 5, 5])
    phonon.run_total_dos(freq_pitch=1, use_tetrahedron_method=True)
    dos = phonon.total_dos.dos
    freqs = phonon.total_dos.frequency_points
    data_ref = np.reshape([float(x) for x in tdos_thm_str.split()], (-1, 2))
    np.testing.assert_allclose(data_ref, np.c_[freqs, dos], atol=1e-5)
    # for f, d in zip(freqs, dos):
    #     print("%f %f" % (f, d))


def testProjectedlDOS(ph_nacl_nofcsym: Phonopy):
    """Test projected DOS with smearing method."""
    phonon = ph_nacl_nofcsym
    phonon.run_mesh([5, 5, 5], is_mesh_symmetry=False, with_eigenvectors=True)
    phonon.run_projected_dos(freq_pitch=1, use_tetrahedron_method=False)
    pdos = phonon.projected_dos.projected_dos
    freqs = phonon.projected_dos.frequency_points
    data_ref = np.reshape([float(x) for x in pdos_str.split()], (-1, 3)).T
    np.testing.assert_allclose(data_ref, np.vstack([freqs, pdos]), atol=1e-5)
    # for f, d in zip(freqs, pdos.T):
    #     print(("%f" + " %f" * len(d)) % ((f, ) + tuple(d)))


def testPartialDOSTetrahedron(ph_nacl_nofcsym: Phonopy):
    """Test projected DOS with tetrahedron method."""
    phonon = ph_nacl_nofcsym
    phonon.run_mesh([5, 5, 5], is_mesh_symmetry=False, with_eigenvectors=True)
    phonon.run_projected_dos(freq_pitch=1, use_tetrahedron_method=True)
    pdos = phonon.projected_dos.projected_dos
    freqs = phonon.projected_dos.frequency_points
    data_ref = np.reshape([float(x) for x in pdos_thm_str.split()], (-1, 3)).T
    np.testing.assert_allclose(data_ref, np.vstack([freqs, pdos]), atol=1e-5)
    # for f, d in zip(freqs, pdos.T):
    #     print(("%f" + " %f" * len(d)) % ((f, ) + tuple(d)))


def test_get_pdos_indices(ph_tio2: Phonopy):
    """Test get_pdos_indices by TiO2."""
    indices = get_pdos_indices(ph_tio2.primitive_symmetry)
    np.testing.assert_array_equal(indices[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(indices[1], [4, 5])
