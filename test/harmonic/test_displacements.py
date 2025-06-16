"""Tests for displacements."""

import itertools
from typing import Literal, Optional, Union

import numpy as np
import pytest

from phonopy import Phonopy


def test_nacl(ph_nacl: Phonopy):
    """Test displacements of NaCl 2x2x2."""
    ph = ph_nacl.copy()
    disp_ref = [[0, 0.01, 0.0, 0.0], [32, 0.01, 0.0, 0.0]]
    np.testing.assert_allclose(ph_nacl.displacements, disp_ref, atol=1e-8)
    ph.generate_displacements()
    np.testing.assert_allclose(ph.displacements, disp_ref, atol=1e-8)


def test_si(ph_si: Phonopy):
    """Test displacements of Si."""
    ph = ph_si.copy()
    disp_ref = [[0, 0.0, 0.0070710678118655, 0.0070710678118655]]
    np.testing.assert_allclose(ph_si.displacements, disp_ref, atol=1e-8)
    ph.generate_displacements()
    np.testing.assert_allclose(ph.displacements, disp_ref, atol=1e-8)


def test_sno2(ph_sno2: Phonopy):
    """Test displacements of SnO2."""
    ph = ph_sno2.copy()
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [0, 0.0, 0.0, 0.01],
        [48, 0.01, 0.0, 0.0],
        [48, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_sno2.displacements, disp_ref, atol=1e-8)
    ph.generate_displacements()
    disp_gen = [
        [0, 0.007032660602415084, 0.0, 0.007109267532681459],
        [0, -0.007032660602415084, 0.0, -0.007109267532681459],
        [48, 0.007032660602415084, 0.0, 0.007109267532681459],
    ]
    np.testing.assert_allclose(ph.displacements, disp_gen, atol=1e-8)


def test_tio2(ph_tio2: Phonopy):
    """Test displacements of TiO2."""
    ph = ph_tio2.copy()
    ph.generate_displacements()
    disp_gen = [
        [0, 0.0060687317141537135, 0.0060687317141537135, 0.0051323474905008],
        [0, -0.0060687317141537135, -0.0060687317141537135, -0.0051323474905008],
        [72, 0.007635558297727332, 0.0, 0.006457418174627326],
        [72, -0.007635558297727332, 0.0, -0.006457418174627326],
    ]
    np.testing.assert_allclose(ph.displacements, disp_gen, atol=1e-8)


@pytest.mark.parametrize(
    "is_plusminus,distance,number_of_snapshots",
    itertools.product([False, True], [None, 0.03], [4, "auto"]),
)
def test_tio2_random_disp(
    ph_tio2: Phonopy,
    is_plusminus: bool,
    distance: Optional[float],
    number_of_snapshots: Union[int, Literal["auto"]],
):
    """Test random displacements of TiO2.

    Currently default displacement distance = 0.01.

    """
    pytest.importorskip("symfc")

    ph = ph_tio2.copy()
    ph.generate_displacements(
        number_of_snapshots=number_of_snapshots,
        distance=distance,
        is_plusminus=is_plusminus,
    )
    d = ph.displacements

    if number_of_snapshots == "auto":
        assert len(d) == 4 * (is_plusminus + 1)
    else:
        assert len(d) == 4 * (is_plusminus + 1)

    if distance is None:
        np.testing.assert_allclose(np.linalg.norm(d, axis=2).ravel(), 0.01, atol=1e-8)
    else:
        np.testing.assert_allclose(np.linalg.norm(d, axis=2).ravel(), 0.03, atol=1e-8)
    if is_plusminus:
        np.testing.assert_allclose(d[: len(d) // 2], -d[len(d) // 2 :], atol=1e-8)


def test_tio2_random_disp_rd_auto_estimation_factor(ph_tio2: Phonopy):
    """Test random displacements of TiO2.

    Test for Phonopy.generate_displacements(number_of_snapshots='auto',
    number_estimation_factor=NUM)

    """
    pytest.importorskip("symfc")

    ph = ph_tio2.copy()
    ph.generate_displacements(number_of_snapshots="auto")
    assert len(ph.displacements) == 4

    ph.generate_displacements(number_of_snapshots="auto", number_estimation_factor=10)
    assert len(ph.displacements) == 10

    ph.generate_displacements(number_of_snapshots="auto", max_distance=1.5)
    assert len(ph.displacements) == 8

    ph.generate_displacements(
        number_of_snapshots="auto", max_distance=1.5, number_estimation_factor=10
    )
    assert len(ph.displacements) == 10


@pytest.mark.parametrize("min_distance", [None, 0.05, 0.2])
def test_tio2_random_disp_with_random_dist(
    ph_tio2: Phonopy, min_distance: Optional[float]
):
    """Test random displacements with random distance of TiO2."""
    ph = ph_tio2.copy()

    if min_distance is not None and min_distance > 0.1:
        with pytest.raises(RuntimeError):
            ph.generate_displacements(
                number_of_snapshots=1, max_distance=0.1, distance=min_distance
            )
    else:
        n_snapshots = 100
        ph.generate_displacements(
            number_of_snapshots=n_snapshots, distance=min_distance, max_distance=0.1
        )
        d = ph.displacements
        assert len(d) == n_snapshots
        dists = np.linalg.norm(d, axis=2).ravel()
        assert (dists < 0.1 + 1e-8).all()
        if min_distance is not None:
            assert (dists > min_distance - 1e-8).all()


@pytest.mark.parametrize("max_distance", [None, 0.1])
def test_tio2_random_disp_with_random_dist_defualt(
    ph_tio2: Phonopy, max_distance: Optional[float]
):
    """Test random displacements with random distance of TiO2.

    Combination of default distance and max_distance.

    """
    ph = ph_tio2.copy()

    n_snapshots = 100
    ph.generate_displacements(
        number_of_snapshots=n_snapshots, max_distance=max_distance, distance=0.01
    )
    d = ph.displacements
    assert len(d) == n_snapshots
    dists = np.linalg.norm(d, axis=2).ravel()
    if max_distance is None:
        assert (dists < 0.01 + 1e-8).all()
    else:
        assert (dists < max_distance + 1e-8).all() and (dists > 0.01 - 1e-8).all()


def test_tio2_random_disp_with_random_max_distance(ph_tio2: Phonopy):
    """Test random displacements with random distance of TiO2.

    Combination of distance and max_distance parameters.

    """
    ph = ph_tio2.copy()

    n_snapshots = 100
    ph.generate_displacements(
        number_of_snapshots=n_snapshots, distance=0.01, max_distance=0.1
    )
    d = ph.displacements
    assert len(d) == n_snapshots
    dists = np.linalg.norm(d, axis=2).ravel()
    assert (dists < 0.1 + 1e-8).all()


def test_tio2_random_disp_plusminus(ph_tio2: Phonopy):
    """Test random plus-minus displacements of TiO2.

    Note
    ----
    Displacements of last 4 supercells are minus of those of first 4 supercells.

    """
    ph = ph_tio2.copy()
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, 0.0, 0.01, 0.0],
        [0, 0.0, 0.0, 0.01],
        [0, 0.0, 0.0, -0.01],
        [72, 0.01, 0.0, 0.0],
        [72, 0.0, 0.0, 0.01],
    ]
    np.testing.assert_allclose(ph_tio2.displacements, disp_ref, atol=1e-8)
    ph.generate_displacements(number_of_snapshots=4, distance=0.03, is_plusminus=True)
    d = ph.displacements
    np.testing.assert_allclose(d[:4], -d[4:], atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(d, axis=2).ravel(), 0.03, atol=1e-8)


def test_tio2_random_distances(ph_tio2: Phonopy):
    """Test random distance displacements with random directions of TiO2."""
    ph = ph_tio2.copy()
    n_snapshots = 100
    ph.generate_displacements(
        number_of_snapshots=n_snapshots,
        max_distance=0.1,
        distance=0.01,
    )
    d = ph.displacements
    assert len(d) == n_snapshots
    dists = np.linalg.norm(d, axis=2)
    for dist_supercell in dists:
        np.testing.assert_allclose(dist_supercell[0], dist_supercell, atol=1e-10)


def test_zr3n4(ph_zr3n4: Phonopy):
    """Test displacements of Zr3N4."""
    ph = ph_zr3n4.copy()
    disp_ref = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [16, 0.01, 0.0, 0.0],
        [16, 0.0, 0.01, 0.0],
    ]
    np.testing.assert_allclose(ph_zr3n4.displacements, disp_ref, atol=1e-8)
    ph.generate_displacements()
    disp_gen = [
        [0, 0.01, 0.0, 0.0],
        [0, -0.01, 0.0, 0.0],
        [16, 0.007071067811865475, 0.007071067811865475, 0.0],
        [16, -0.007071067811865475, -0.007071067811865475, 0.0],
    ]
    np.testing.assert_allclose(ph.displacements, disp_gen, atol=1e-8)


def test_tipn3(ph_tipn3: Phonopy):
    """Test displacements of Zr3N4."""
    ph = ph_tipn3.copy()
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
    ph.generate_displacements()
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
    np.testing.assert_allclose(ph.displacements, disp_gen, atol=1e-8)
