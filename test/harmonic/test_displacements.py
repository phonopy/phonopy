"""Tests for displacements."""
from copy import deepcopy

import numpy as np

from phonopy import Phonopy


def test_nacl(ph_nacl: Phonopy):
    """Test displacements of NaCl 2x2x2."""
    dataset = deepcopy(ph_nacl.dataset)
    disp_ref = [[0, 0.01, 0.0, 0.0], [32, 0.01, 0.0, 0.0]]
    np.testing.assert_allclose(ph_nacl.displacements, disp_ref, atol=1e-8)
    ph_nacl.generate_displacements()
    np.testing.assert_allclose(ph_nacl.displacements, disp_ref, atol=1e-8)
    ph_nacl.dataset = dataset


def test_si(ph_si: Phonopy):
    """Test displacements of Si."""
    dataset = deepcopy(ph_si.dataset)
    disp_ref = [[0, 0.0, 0.0070710678118655, 0.0070710678118655]]
    np.testing.assert_allclose(ph_si.displacements, disp_ref, atol=1e-8)
    ph_si.generate_displacements()
    np.testing.assert_allclose(ph_si.displacements, disp_ref, atol=1e-8)
    ph_si.dataset = dataset


def test_sno2(ph_sno2: Phonopy):
    """Test displacements of SnO2."""
    dataset = deepcopy(ph_sno2.dataset)
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [0, 0.0, 0.0, 0.01],
        [48, 0.01, 0.0, 0.0],
        [48, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_sno2.displacements, disp_ref, atol=1e-8)
    ph_sno2.generate_displacements()
    disp_gen = [
        [0, 0.007032660602415084, 0.0, 0.007109267532681459],
        [0, -0.007032660602415084, 0.0, -0.007109267532681459],
        [48, 0.007032660602415084, 0.0, 0.007109267532681459],
    ]
    np.testing.assert_allclose(ph_sno2.displacements, disp_gen, atol=1e-8)
    ph_sno2.dataset = dataset


def test_tio2(ph_tio2: Phonopy):
    """Test displacements of TiO2."""
    dataset = deepcopy(ph_tio2.dataset)
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)
    ph_tio2.generate_displacements()
    disp_gen = [
        [0, 0.0060687317141537135, 0.0060687317141537135, 0.0051323474905008],
        [0, -0.0060687317141537135, -0.0060687317141537135, -0.0051323474905008],
        [72, 0.007635558297727332, 0.0, 0.006457418174627326],
        [72, -0.007635558297727332, 0.0, -0.006457418174627326],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_gen, atol=1e-8)
    ph_tio2.dataset = dataset


def test_tio2_random_disp(ph_tio2: Phonopy):
    """Test random displacements of TiO2."""
    dataset = deepcopy(ph_tio2.dataset)
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)
    ph_tio2.generate_displacements(number_of_snapshots=4, distance=0.03)
    d = ph_tio2.displacements
    np.testing.assert_allclose(np.linalg.norm(d, axis=2).ravel(), 0.03, atol=1e-8)
    ph_tio2.dataset = dataset


def test_tio2_random_disp_plusminus(ph_tio2: Phonopy):
    """Test random plus-minus displacements of TiO2.

    Note
    ----
    Displacements of last 4 supercells are minus of those of first 4 supercells.

    """
    dataset = deepcopy(ph_tio2.dataset)
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)
    ph_tio2.generate_displacements(
        number_of_snapshots=4, distance=0.03, is_plusminus=True
    )
    d = ph_tio2.displacements
    np.testing.assert_allclose(d[:4], -d[4:], atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(d, axis=2).ravel(), 0.03, atol=1e-8)
    ph_tio2.dataset = dataset


def test_zr3n4(ph_zr3n4: Phonopy):
    """Test displacements of Zr3N4."""
    dataset = deepcopy(ph_zr3n4.dataset)
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [16, 0.01, 0.0, 0.0],
        [16, 0.0, 0.01, 0.0],
    ]
    np.testing.assert_allclose(ph_zr3n4.displacements, disp_ref, atol=1e-8)
    ph_zr3n4.generate_displacements()
    disp_gen = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [16, 0.007071067811865475, 0.007071067811865475, 0.0],
        [16, -0.007071067811865475, -0.007071067811865475, 0.0],
    ]
    np.testing.assert_allclose(ph_zr3n4.displacements, disp_gen, atol=1e-8)
    ph_zr3n4.dataset = dataset


def test_tipn3(ph_tipn3: Phonopy):
    """Test displacements of Zr3N4."""
    dataset = deepcopy(ph_tipn3.dataset)
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
    ph_tipn3.generate_displacements()
    disp_gen = [
        [0, 0.006370194270018462, 0.006021020526083804, 0.00481330829956917],
        [0, -0.006370194270018462, -0.006021020526083804, -0.00481330829956917],
        [16, 0.006370194270018462, 0.006021020526083804, 0.00481330829956917],
        [16, -0.006370194270018462, -0.006021020526083804, -0.00481330829956917],
        [32, 0.007267439570389398, 0.0068690845162028965, 0.0],
        [32, -0.007267439570389398, -0.0068690845162028965, 0.0],
        [32, 0.0, 0.0, 0.01],
        [32, 0.0, 0.0, -0.01],
        [40, 0.006370194270018462, 0.006021020526083804, 0.00481330829956917],
        [40, -0.006370194270018462, -0.006021020526083804, -0.00481330829956917],
    ]
    np.testing.assert_allclose(ph_tipn3.displacements, disp_gen, atol=1e-8)
    ph_tipn3.dataset = dataset
