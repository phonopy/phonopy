"""Tests for phonon calculation at specific q-points."""
import numpy as np

from phonopy import Phonopy
from phonopy.units import VaspToTHz


def testQpoints(ph_nacl_nofcsym: Phonopy):
    """Test phonon calculation at specific q-points by NaCl."""
    phonon = ph_nacl_nofcsym
    qpoints = [[0, 0, 0], [0, 0, 0.5]]
    phonon.run_qpoints(qpoints, with_dynamical_matrices=True)
    for i, q in enumerate(qpoints):
        dm = phonon.qpoints.dynamical_matrices[i]
        dm_eigs = np.linalg.eigvalsh(dm).real
        eigs = phonon.qpoints.eigenvalues[i]
        freqs = phonon.qpoints.frequencies[i] / VaspToTHz
        np.testing.assert_allclose(dm_eigs, eigs)
        np.testing.assert_allclose(freqs**2 * np.sign(freqs), eigs)
