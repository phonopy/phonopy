"""Tests for routines in grid_points.py."""

import os

import numpy as np
import pytest

from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.grid_points import (
    GeneralizedRegularGridPoints,
    GridPoints,
    length2mesh,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

ga234 = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, -1, 0],
    [-1, -1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, -1, 1],
    [1, 2, 1],
    [0, 0, 2],
    [1, 0, 2],
    [0, 1, 2],
    [1, 1, 2],
    [0, -1, -2],
    [-1, -1, -2],
    [0, 0, -1],
    [-1, 0, -1],
    [0, 1, -1],
    [-1, -2, -1],
    [0, -1, -1],
    [-1, -1, -1],
]


def test_GridPoints():
    """Test of GridPoints."""
    gp = GridPoints([2, 3, 4], [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    assert gp.ir_grid_points.dtype == np.dtype("long")
    assert gp.weights.dtype == np.dtype("long")
    assert gp.grid_mapping_table.dtype == np.dtype("long")
    assert gp.grid_address.dtype == np.dtype("intc")
    assert gp.mesh_numbers.dtype == np.dtype("intc")
    assert gp.reciprocal_lattice.dtype == np.dtype("double")
    assert gp.qpoints.dtype == np.dtype("double")

    np.testing.assert_array_equal(gp.grid_address, ga234)


def test_GridPoints_NaCl_with_rotations(ph_nacl: Phonopy):
    """Test of GridPoints with rotations from NaCl."""
    rec_lat = np.linalg.inv(ph_nacl.primitive.cell)
    rotations = ph_nacl.primitive_symmetry.pointgroup_operations
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations)
    np.testing.assert_array_equal(gp.ir_grid_points, [0, 1, 2, 5, 6, 7, 10, 27])
    np.testing.assert_array_equal(gp.weights, [1, 8, 4, 6, 24, 12, 3, 6])
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [
            0,
            1,
            2,
            1,
            1,
            5,
            6,
            7,
            2,
            6,
            10,
            6,
            1,
            7,
            6,
            5,
            1,
            5,
            6,
            7,
            5,
            1,
            7,
            6,
            6,
            7,
            6,
            27,
            7,
            6,
            27,
            6,
            2,
            6,
            10,
            6,
            6,
            7,
            6,
            27,
            10,
            6,
            2,
            6,
            6,
            27,
            6,
            7,
            1,
            7,
            6,
            5,
            7,
            6,
            27,
            6,
            6,
            27,
            6,
            7,
            5,
            6,
            7,
            1,
        ],
    )
    np.testing.assert_allclose(
        gp.qpoints,
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.5, 0.25, 0.0],
            [-0.25, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [-0.25, 0.5, 0.25],
        ],
        atol=1e-8,
    )


@pytest.mark.parametrize("fit_in_BZ", [True, False])
def test_GridPoints_NaCl_with_rotations_fit_BZ(ph_nacl: Phonopy, fit_in_BZ):
    """Test of GridPoints with rotations from NaCl and fit_in_BZ."""
    rec_lat = np.linalg.inv(ph_nacl.primitive.cell)
    rotations = ph_nacl.primitive_symmetry.pointgroup_operations
    mesh = [5, 5, 5]
    gpts = GridPoints(mesh, rec_lat, rotations=rotations, fit_in_BZ=fit_in_BZ)
    if fit_in_BZ:
        np.testing.assert_allclose(
            gpts.qpoints,
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.2, 0.2, 0.0],
                [0.4, 0.2, 0.0],
                [-0.4, 0.2, 0.0],
                [-0.2, 0.2, 0.0],
                [0.4, 0.4, 0.0],
                [-0.4, -0.6, 0.0],
                [0.6, 0.4, 0.2],
            ],
            atol=1e-8,
        )
    else:
        np.testing.assert_allclose(
            gpts.qpoints,
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.2, 0.2, 0.0],
                [0.4, 0.2, 0.0],
                [-0.4, 0.2, 0.0],
                [-0.2, 0.2, 0.0],
                [0.4, 0.4, 0.0],
                [-0.4, 0.4, 0.0],
                [-0.4, 0.4, 0.2],
            ],
            atol=1e-8,
        )


def test_GridPoints_SnO2_with_rotations(ph_sno2: Phonopy):
    """Test of GridPoints with rotations from SnO2."""
    rec_lat = np.linalg.inv(ph_sno2.primitive.cell)
    rotations = ph_sno2.primitive_symmetry.pointgroup_operations
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations)
    np.testing.assert_array_equal(
        gp.ir_grid_points,
        [0, 1, 2, 5, 6, 10, 16, 17, 18, 21, 22, 26, 32, 33, 34, 37, 38, 42],
    )
    np.testing.assert_array_equal(
        gp.weights, [1, 4, 2, 4, 4, 1, 2, 8, 4, 8, 8, 2, 1, 4, 2, 4, 4, 1]
    )
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [
            0,
            1,
            2,
            1,
            1,
            5,
            6,
            5,
            2,
            6,
            10,
            6,
            1,
            5,
            6,
            5,
            16,
            17,
            18,
            17,
            17,
            21,
            22,
            21,
            18,
            22,
            26,
            22,
            17,
            21,
            22,
            21,
            32,
            33,
            34,
            33,
            33,
            37,
            38,
            37,
            34,
            38,
            42,
            38,
            33,
            37,
            38,
            37,
            16,
            17,
            18,
            17,
            17,
            21,
            22,
            21,
            18,
            22,
            26,
            22,
            17,
            21,
            22,
            21,
        ],
    )
    np.testing.assert_allclose(
        gp.qpoints,
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.25],
            [0.25, 0.0, 0.25],
            [0.5, 0.0, 0.25],
            [0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25],
            [0.0, 0.0, 0.5],
            [0.25, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.25, 0.25, 0.5],
            [0.5, 0.25, 0.5],
            [0.5, 0.5, 0.5],
        ],
        atol=1e-8,
    )


def test_GridPoints_SnO2_with_rotations_MP(ph_sno2: Phonopy):
    """Test of GridPoints with non-gamma-centre mesh and rotations from SnO2."""
    rec_lat = np.linalg.inv(ph_sno2.primitive.cell)
    rotations = ph_sno2.primitive_symmetry.pointgroup_operations
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations, is_gamma_center=False)
    np.testing.assert_array_equal(gp.ir_grid_points, [0, 1, 5, 16, 17, 21])
    np.testing.assert_array_equal(gp.weights, [8, 16, 8, 8, 16, 8])
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [
            0,
            1,
            1,
            0,
            1,
            5,
            5,
            1,
            1,
            5,
            5,
            1,
            0,
            1,
            1,
            0,
            16,
            17,
            17,
            16,
            17,
            21,
            21,
            17,
            17,
            21,
            21,
            17,
            16,
            17,
            17,
            16,
            16,
            17,
            17,
            16,
            17,
            21,
            21,
            17,
            17,
            21,
            21,
            17,
            16,
            17,
            17,
            16,
            0,
            1,
            1,
            0,
            1,
            5,
            5,
            1,
            1,
            5,
            5,
            1,
            0,
            1,
            1,
            0,
        ],
    )
    np.testing.assert_allclose(
        gp.qpoints,
        [
            [0.125, 0.125, 0.125],
            [0.375, 0.125, 0.125],
            [0.375, 0.375, 0.125],
            [0.125, 0.125, 0.375],
            [0.375, 0.125, 0.375],
            [0.375, 0.375, 0.375],
        ],
        atol=1e-8,
    )


@pytest.mark.parametrize("suggest", [True, False])
def test_SNF_from_GeneralizedRegularGridPoints(ph_tio2: Phonopy, suggest):
    """Test for grid rotation matrix and SNF by TiO2."""
    grgp = GeneralizedRegularGridPoints(
        ph_tio2.unitcell, 60, suggest=suggest, x_fastest=False
    )

    if suggest:
        np.testing.assert_array_equal(
            grgp.grid_matrix, [[0, 16, 16], [16, 0, 16], [6, 6, 0]]
        )
        np.testing.assert_array_equal(grgp.snf.P, [[0, -1, 3], [1, 0, 0], [-3, 3, -8]])
        np.testing.assert_array_equal(grgp.snf.D, [[2, 0, 0], [0, 16, 0], [0, 0, 96]])
        np.testing.assert_array_equal(grgp.snf.Q, [[1, 8, 17], [0, 0, -1], [0, 1, 1]])
        np.testing.assert_allclose(
            grgp.transformation_matrix,
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]],
        )
        assert (grgp.grid_address[253] == [0, 2, 61]).all()
        np.testing.assert_allclose(
            grgp.qpoints[253], [-0.19791667, 0.36458333, -0.23958333]
        )
        np.testing.assert_array_equal(
            grgp.reciprocal_operations[1], [[9, 10, 3], [8, 9, 3], [-48, -54, -17]]
        )
    else:
        np.testing.assert_array_equal(
            grgp.grid_matrix, [[16, 0, 0], [0, 16, 0], [0, 0, 6]]
        )
        np.testing.assert_array_equal(grgp.snf.P, [[1, 0, -3], [0, -1, 0], [-3, 0, 8]])
        np.testing.assert_array_equal(grgp.snf.D, [[2, 0, 0], [0, 16, 0], [0, 0, 48]])
        np.testing.assert_array_equal(
            grgp.snf.Q, [[-1, 0, -9], [0, -1, 0], [-1, 0, -8]]
        )
        np.testing.assert_allclose(grgp.transformation_matrix, np.eye(3))
        assert (grgp.grid_address[253] == [0, 5, 13]).all()
        np.testing.assert_allclose(grgp.qpoints[253], [-0.4375, -0.3125, -0.16666667])
        np.testing.assert_array_equal(
            grgp.reciprocal_operations[1], [[9, -1, 3], [-8, 0, -3], [-24, 3, -8]]
        )


@pytest.mark.parametrize("suggest", [True, False])
def test_GeneralizedRegularGridPoints_rotations_tio2(ph_tio2, suggest):
    """Test for GeneralizedRegularGridPoints by TiO2."""
    matches = _get_matches(ph_tio2, suggest, True)
    if suggest:
        matches_ref = [
            0,
            29,
            58,
            67,
            96,
            5,
            34,
            43,
            72,
            81,
            10,
            39,
            48,
            77,
            86,
            15,
            24,
            53,
            62,
            91,
            38,
            47,
            76,
            85,
            14,
            23,
            52,
            61,
            90,
            19,
            28,
            57,
            66,
            95,
            4,
            33,
            42,
            71,
            80,
            9,
            56,
            65,
            94,
            3,
            32,
            41,
            70,
            99,
            8,
            37,
            46,
            75,
            84,
            13,
            22,
            51,
            60,
            89,
            18,
            27,
            74,
            83,
            12,
            21,
            50,
            79,
            88,
            17,
            26,
            55,
            64,
            93,
            2,
            31,
            40,
            69,
            98,
            7,
            36,
            45,
            92,
            1,
            30,
            59,
            68,
            97,
            6,
            35,
            44,
            73,
            82,
            11,
            20,
            49,
            78,
            87,
            16,
            25,
            54,
            63,
        ]
    else:
        matches_ref = [
            0,
            25,
            40,
            15,
            30,
            5,
            20,
            45,
            10,
            35,
            2,
            27,
            42,
            17,
            32,
            7,
            22,
            47,
            12,
            37,
            4,
            29,
            44,
            19,
            34,
            9,
            24,
            49,
            14,
            39,
            6,
            21,
            46,
            11,
            36,
            1,
            26,
            41,
            16,
            31,
            8,
            23,
            48,
            13,
            38,
            3,
            28,
            43,
            18,
            33,
        ]
    np.testing.assert_array_equal(matches, matches_ref)


@pytest.mark.parametrize("suggest", [True, False])
@pytest.mark.parametrize("is_time_reversal", [True, False])
def test_GeneralizedRegularGridPoints_rotations_zr3n4(
    ph_zr3n4, suggest, is_time_reversal
):
    """Test for GeneralizedRegularGridPoints by non-centrosymmetric Zr3N4."""
    matches = _get_matches(ph_zr3n4, suggest, is_time_reversal)
    if suggest:
        matches_ref = [
            0,
            17,
            10,
            3,
            14,
            7,
            29,
            22,
            33,
            26,
            19,
            30,
            52,
            45,
            38,
            49,
            42,
            41,
            36,
            53,
            46,
            39,
            50,
            43,
            11,
            4,
            15,
            8,
            1,
            12,
            34,
            27,
            20,
            31,
            24,
            23,
            18,
            35,
            28,
            21,
            32,
            25,
            47,
            40,
            51,
            44,
            37,
            48,
            16,
            9,
            2,
            13,
            6,
            5,
        ]
    else:
        matches_ref = [
            0,
            2,
            1,
            9,
            11,
            10,
            18,
            20,
            19,
            6,
            8,
            7,
            15,
            17,
            16,
            24,
            26,
            25,
            3,
            5,
            4,
            12,
            14,
            13,
            21,
            23,
            22,
        ]
    np.testing.assert_array_equal(matches, matches_ref)


def _get_matches(ph, suggest, is_time_reversal):
    grgp = GeneralizedRegularGridPoints(
        ph.unitcell,
        20,
        suggest=suggest,
        is_time_reversal=is_time_reversal,
        x_fastest=False,
    )
    rot_address = np.dot(grgp.grid_address, grgp.reciprocal_operations[1])
    diag_n = np.diagonal(grgp.snf.D)
    rot_address %= diag_n
    matches = []
    for adrs in rot_address:
        d = np.abs(grgp.grid_address - adrs).sum(axis=1)
        match = np.where(d == 0)[0]
        assert len(match) == 1
        matches.append(match[0])
    return matches


def test_watch_GeneralizedRegularGridPoints(ph_tio2: Phonopy, helper_methods):
    """Test for q-points positions obtained from GeneralizedRegularGridPoints."""
    grgp = GeneralizedRegularGridPoints(ph_tio2.unitcell, 10, x_fastest=False)
    tmat = grgp.transformation_matrix
    # direct basis vectors in row vectors
    plat = np.dot(tmat.T, ph_tio2.unitcell.cell)
    # reciprocal basis vectors in row vectors (10 times magnified)
    rec_plat = np.linalg.inv(plat).T * 10
    symbols = [
        "H",
    ] * len(grgp.qpoints)
    cell = PhonopyAtoms(cell=rec_plat, scaled_positions=grgp.qpoints, symbols=symbols)
    yaml_filename = os.path.join(current_dir, "tio2_qpoints.yaml")
    cell_ref = read_cell_yaml(yaml_filename)
    helper_methods.compare_cells(cell, cell_ref)


def test_length2mesh(ph_nacl: Phonopy):
    """Test of length2mesh."""
    length = 50.0
    mesh_numbers = length2mesh(length, ph_nacl.primitive.cell)
    np.testing.assert_array_equal(mesh_numbers, [15, 15, 15])

    mesh_numbers = length2mesh(
        length, ph_nacl.primitive.cell, ph_nacl.primitive_symmetry.pointgroup_operations
    )
    np.testing.assert_array_equal(mesh_numbers, [15, 15, 15])
