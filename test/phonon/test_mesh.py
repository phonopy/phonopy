"""Tests for phonon calculation on sampling mesh."""

import numpy as np

from phonopy import Phonopy

freqs_full_fcsym_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.509780,
    2.509780,
    4.087812,
    4.143926,
    4.143926,
    6.697105,
    2.106719,
    2.106719,
    4.571979,
    4.798841,
    4.798841,
    5.116023,
    3.115897,
    3.487080,
    4.228067,
    4.459883,
    5.037473,
    5.066083,
]

freqs_nofcsym = [
    -0.037009,
    -0.037009,
    -0.037009,
    4.608453,
    4.608453,
    4.608453,
    2.509883,
    2.509883,
    4.088001,
    4.143404,
    4.143404,
    6.696706,
    2.107450,
    2.107450,
    4.572128,
    4.799497,
    4.799497,
    5.116806,
    3.115923,
    3.486963,
    4.227909,
    4.460058,
    5.037201,
    5.066048,
]

freqs_compact_fcsym_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.509780,
    2.509780,
    4.087812,
    4.143926,
    4.143926,
    6.697105,
    2.106719,
    2.106719,
    4.571979,
    4.798841,
    4.798841,
    5.116023,
    3.115897,
    3.487080,
    4.228067,
    4.459883,
    5.037473,
    5.066083,
]

freqs_nonac_ref = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.458402,
    2.458402,
    4.146058,
    4.395223,
    4.395223,
    6.236554,
    2.106889,
    2.106889,
    4.572116,
    4.798842,
    4.798842,
    5.508652,
    2.737708,
    3.491448,
    4.231340,
    4.583972,
    5.066096,
    5.087983,
]

freqs_full_fcsym_ref_wang = [
    0.000000,
    0.000000,
    0.000000,
    4.616435,
    4.616435,
    4.616435,
    2.458402,
    2.458402,
    4.142037,
    4.395223,
    4.395223,
    6.373675,
    2.106889,
    2.106889,
    4.571823,
    4.798842,
    4.798842,
    5.116018,
    2.737708,
    3.491448,
    4.231340,
    4.613570,
    5.066096,
    5.112448,
]

freqs_full_fcsym_ref_si = [
    0.000000,
    0.000000,
    0.000000,
    15.111196,
    15.111196,
    15.111196,
    2.823122,
    2.823122,
    8.281999,
    13.784543,
    14.544795,
    14.544795,
    3.662488,
    3.662488,
    8.879957,
    13.943485,
    13.943485,
    14.047251,
    3.868017,
    6.872823,
    10.247355,
    11.783252,
    13.879338,
    14.048571,
]


def test_Mesh_nofcsym(ph_nacl_nofcsym: Phonopy):
    """Test by NaCl without symmetrizing force constants."""
    _test_IterMesh(ph_nacl_nofcsym, freqs_nofcsym)


def test_Mesh_full_fcsym(ph_nacl: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_nacl, freqs_full_fcsym_ref)


def test_Mesh_compact_fcsym(ph_nacl_compact_fcsym: Phonopy):
    """Test by NaCl with symmetrizing force constants in compact format."""
    _test_IterMesh(ph_nacl_compact_fcsym, freqs_compact_fcsym_ref)


def test_Mesh_full_fcsym_nonac(ph_nacl_nonac: Phonopy):
    """Test by NaCl without NAC."""
    _test_IterMesh(ph_nacl_nonac, freqs_nonac_ref)


def test_Mesh_full_fcsym_wang(ph_nacl_wang: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_nacl_wang, freqs_full_fcsym_ref_wang)


def test_Mesh_full_fcsym_si(ph_si: Phonopy):
    """Test by NaCl with symmetrizing force constants."""
    _test_IterMesh(ph_si, freqs_full_fcsym_ref_si)


def _test_IterMesh(ph_nacl: Phonopy, freqs_ref):
    ph_nacl.init_mesh(mesh=[3, 3, 3], with_eigenvectors=True, use_iter_mesh=True)
    freqs = []
    eigvecs = []
    for f, e in ph_nacl.mesh:
        freqs.append(f)
        eigvecs.append(e)

    # for freqs_q in freqs:
    #     print("".join(["%f, " % f for f in freqs_q]))

    np.testing.assert_allclose(freqs_ref, np.reshape(freqs, -1), atol=1e-5)
    ph_nacl.run_mesh([3, 3, 3], with_eigenvectors=True)
    mesh_obj = ph_nacl.mesh
    mesh_freqs = mesh_obj.frequencies
    np.testing.assert_allclose(mesh_freqs, freqs, atol=1e-5)
