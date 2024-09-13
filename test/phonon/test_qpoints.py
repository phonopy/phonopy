"""Tests for phonon calculation at specific q-points."""

import numpy as np

from phonopy import Phonopy
from phonopy.units import VaspToTHz


def test_Qpoints(ph_nacl_nofcsym: Phonopy):
    """Test phonon calculation at specific q-points by NaCl."""
    phonon = ph_nacl_nofcsym
    qpoints = [[0, 0, 0], [0, 0, 0.5]]
    phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
    for i, _ in enumerate(qpoints):
        dm = phonon.qpoints.dynamical_matrices[i]
        dm_eigs = np.linalg.eigvalsh(dm).real
        eigs = phonon.qpoints.eigenvalues[i]
        freqs = phonon.qpoints.frequencies[i] / VaspToTHz
        np.testing.assert_allclose(dm_eigs, eigs)
        np.testing.assert_allclose(freqs**2 * np.sign(freqs), eigs)


def test_Qpoints_with_NAC_qdirection(ph_nacl: Phonopy):
    """Test phonon calculation at specific q-points by NaCl."""
    phonon = ph_nacl
    qpoints = [[0, 0, 0]]
    phonon.run_qpoints(qpoints)
    freqs = phonon.get_qpoints_dict()["frequencies"]
    np.testing.assert_allclose(
        freqs, [[0, 0, 0, 4.61643516, 4.61643516, 4.61643516]], atol=1e-5
    )
    phonon.run_qpoints(qpoints, nac_q_direction=[1, 0, 0])
    freqs = phonon.get_qpoints_dict()["frequencies"]
    np.testing.assert_allclose(
        freqs, [[0, 0, 0, 4.61643516, 4.61643516, 7.39632718]], atol=1e-5
    )
