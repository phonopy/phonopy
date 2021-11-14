"""Tests for displacements."""
import numpy as np

from phonopy import Phonopy


def test_nacl(ph_nacl: Phonopy):
    """Test displacements of NaCl 2x2x2."""
    disp_ref = [[0, 0.01, 0.0, 0.0], [32, 0.01, 0.0, 0.0]]
    np.testing.assert_allclose(ph_nacl.displacements, disp_ref, atol=1e-8)


def test_si(ph_si: Phonopy):
    """Test displacements of Si."""
    disp_ref = [[0, 0.0, 0.0070710678118655, 0.0070710678118655]]
    np.testing.assert_allclose(ph_si.displacements, disp_ref, atol=1e-8)


def test_sno2(ph_sno2: Phonopy):
    """Test displacements of SnO2."""
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [0, 0.0, 0.0, 0.01],
        [48, 0.01, 0.0, 0.0],
        [48, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_sno2.displacements, disp_ref, atol=1e-8)


def test_tio2(ph_tio2: Phonopy):
    """Test displacements of TiO2."""
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)


def test_zr3n4(ph_zr3n4: Phonopy):
    """Test displacements of Zr3N4."""
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [16, 0.01, 0.0, 0.0],
        [16, 0.0, 0.01, 0.0],
    ]
    np.testing.assert_allclose(ph_zr3n4.displacements, disp_ref, atol=1e-8)


def test_tipn3(ph_tipn3: Phonopy):
    """Test displacements of Zr3N4."""
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [16, 0.01, 0.0, 0.0],
        [16, 0.0, 0.01, 0.0],
        [16, 0.0, 0.0, 0.01],
        [16, 0.0, 0.0, -0.01],
        [32, 0.01, 0.0, 0.0],
        [32, 0.0, 0.01, 0.0],
        [32, 0.0, -0.01, 0.0],
        [32, 0.0, 0.0, 0.01],
        [32, 0.0, 0.0, -0.01],
        [40, 0.01, 0.0, 0.0],
        [40, 0.0, 0.01, 0.0],
        [40, 0.0, 0.0, 0.01],
        [40, 0.0, 0.0, -0.01],
    ]
    np.testing.assert_allclose(ph_tipn3.displacements, disp_ref, atol=1e-8)
