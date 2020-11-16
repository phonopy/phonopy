import os
import pytest
import numpy as np
from phonopy.structure.grid_points import (
    GridPoints, GeneralizedRegularGridPoints)


current_dir = os.path.dirname(os.path.abspath(__file__))

ga234 = [[0, 0, 0],
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
         [-1, -1, -1]]


def test_GridPoints():
    gp = GridPoints([2, 3, 4], [[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    np.testing.assert_array_equal(gp.grid_address, ga234)


def test_GridPoints_NaCl_with_rotations(ph_nacl):
    rec_lat = np.linalg.inv(ph_nacl.primitive.cell)
    rotations = ph_nacl.primitive_symmetry.get_pointgroup_operations()
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations)
    np.testing.assert_array_equal(
        gp.ir_grid_points, [0, 1, 2, 5, 6, 7, 10, 27])
    np.testing.assert_array_equal(gp.weights, [1, 8, 4, 6, 24, 12, 3, 6])
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [0, 1, 2, 1, 1, 5, 6, 7, 2, 6,
         10, 6, 1, 7, 6, 5, 1, 5, 6, 7,
         5, 1, 7, 6, 6, 7, 6, 27, 7, 6,
         27, 6, 2, 6, 10, 6, 6, 7, 6, 27,
         10, 6, 2, 6, 6, 27, 6, 7, 1, 7,
         6, 5, 7, 6, 27, 6, 6, 27, 6, 7,
         5, 6, 7, 1])
    np.testing.assert_allclose(
        gp.qpoints,
        [[0.0, 0.0, 0.0],
         [0.25, 0.0, 0.0],
         [0.5, 0.0, 0.0],
         [0.25, 0.25, 0.0],
         [0.5, 0.25, 0.0],
         [-0.25, 0.25, 0.0],
         [0.5, 0.5, 0.0],
         [-0.25, 0.5, 0.25]], atol=1e-8)


def test_GridPoints_NaCl_with_rotations_fit_BZ(ph_nacl):
    rec_lat = np.linalg.inv(ph_nacl.primitive.cell)
    rotations = ph_nacl.primitive_symmetry.get_pointgroup_operations()
    mesh = [5, 5, 5]
    gpf = GridPoints(mesh, rec_lat, rotations=rotations,
                     fit_in_BZ=False)
    gpt = GridPoints(mesh, rec_lat, rotations=rotations,
                     fit_in_BZ=True)
    np.testing.assert_allclose(
        gpf.qpoints,
        [[0.0, 0.0, 0.0],
         [0.2, 0.0, 0.0],
         [0.4, 0.0, 0.0],
         [0.2, 0.2, 0.0],
         [0.4, 0.2, 0.0],
         [-0.4, 0.2, 0.0],
         [-0.2, 0.2, 0.0],
         [0.4, 0.4, 0.0],
         [-0.4, 0.4, 0.0],
         [-0.4, 0.4, 0.2]], atol=1e-8)
    np.testing.assert_allclose(
        gpt.qpoints,
        [[0.0, 0.0, 0.0],
         [0.2, 0.0, 0.0],
         [0.4, 0.0, 0.0],
         [0.2, 0.2, 0.0],
         [0.4, 0.2, 0.0],
         [-0.4, 0.2, 0.0],
         [-0.2, 0.2, 0.0],
         [0.4, 0.4, 0.0],
         [-0.4, -0.6, 0.0],
         [0.6, 0.4, 0.2]], atol=1e-8)


def test_GridPoints_SnO2_with_rotations(ph_sno2):
    rec_lat = np.linalg.inv(ph_sno2.primitive.cell)
    rotations = ph_sno2.primitive_symmetry.get_pointgroup_operations()
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations)
    np.testing.assert_array_equal(
        gp.ir_grid_points,
        [0, 1, 2, 5, 6, 10, 16, 17, 18, 21, 22, 26, 32, 33, 34, 37, 38, 42])
    np.testing.assert_array_equal(
        gp.weights,
        [1, 4, 2, 4, 4, 1, 2, 8, 4, 8, 8, 2, 1, 4, 2, 4, 4, 1])
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [0, 1, 2, 1, 1, 5, 6, 5, 2, 6,
         10, 6, 1, 5, 6, 5, 16, 17, 18, 17,
         17, 21, 22, 21, 18, 22, 26, 22, 17, 21,
         22, 21, 32, 33, 34, 33, 33, 37, 38, 37,
         34, 38, 42, 38, 33, 37, 38, 37, 16, 17,
         18, 17, 17, 21, 22, 21, 18, 22, 26, 22,
         17, 21, 22, 21])
    np.testing.assert_allclose(
        gp.qpoints,
        [[0.0, 0.0, 0.0],
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
         [0.5, 0.5, 0.5]], atol=1e-8)


def test_GridPoints_SnO2_with_rotations_MP(ph_sno2):
    rec_lat = np.linalg.inv(ph_sno2.primitive.cell)
    rotations = ph_sno2.primitive_symmetry.get_pointgroup_operations()
    gp = GridPoints([4, 4, 4], rec_lat, rotations=rotations,
                    is_gamma_center=False)
    np.testing.assert_array_equal(
        gp.ir_grid_points, [0, 1, 5, 16, 17, 21])
    np.testing.assert_array_equal(gp.weights, [8, 16, 8, 8, 16, 8])
    np.testing.assert_array_equal(
        gp.grid_mapping_table,
        [0, 1, 1, 0, 1, 5, 5, 1, 1, 5,
         5, 1, 0, 1, 1, 0, 16, 17, 17, 16,
         17, 21, 21, 17, 17, 21, 21, 17, 16, 17,
         17, 16, 16, 17, 17, 16, 17, 21, 21, 17,
         17, 21, 21, 17, 16, 17, 17, 16, 0, 1,
         1, 0, 1, 5, 5, 1, 1, 5, 5, 1,
         0, 1, 1, 0])
    np.testing.assert_allclose(
        gp.qpoints,
        [[0.125, 0.125, 0.125],
         [0.375, 0.125, 0.125],
         [0.375, 0.375, 0.125],
         [0.125, 0.125, 0.375],
         [0.375, 0.125, 0.375],
         [0.375, 0.375, 0.375]], atol=1e-8)


@pytest.mark.parametrize("suggest", [True, False])
def test_GeneralizedRegularGridPoints(ph_tio2, suggest):
    grgp = GeneralizedRegularGridPoints(ph_tio2.unitcell, 100, suggest=suggest)

    if suggest:
        np.testing.assert_array_equal(
            grgp.snf.P, [[0, -1, 3], [1, 0, 0], [-3, 3, -8]])
        np.testing.assert_array_equal(
            grgp.snf.D, [[2, 0, 0], [0, 16, 0], [0, 0, 96]])
        np.testing.assert_array_equal(
            grgp.snf.Q, [[1, 8, 17], [0, 0, -1], [0, 1, 1]])
        np.testing.assert_allclose(
            grgp.matrix_to_primitive,
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
        np.testing.assert_array_equal(
            grgp.grid_matrix, [[0, 16, 16], [16, 0, 16], [6, 6, 0]])
        assert (grgp.grid_address[253] == [0, 2, 61]).all()
        np.testing.assert_allclose(grgp.qpoints[253],
                                   [-0.19791667, 0.36458333, -0.23958333])
    else:
        np.testing.assert_array_equal(
            grgp.snf.P, [[1, 0, -3], [0, -1, 0], [-3, 0, 8]])
        np.testing.assert_array_equal(
            grgp.snf.D, [[2, 0, 0], [0, 16, 0], [0, 0, 48]])
        np.testing.assert_array_equal(
            grgp.snf.Q, [[-1, 0, -9], [0, -1, 0], [-1, 0, -8]])
        assert grgp.matrix_to_primitive is None
        np.testing.assert_array_equal(
            grgp.grid_matrix, [[16, 0, 0], [0, 16, 0], [0, 0, 6]])
        assert (grgp.grid_address[253] == [0, 5, 13]).all()
        np.testing.assert_allclose(grgp.qpoints[253],
                                   [-0.4375, -0.3125, -0.16666667])


def test_watch_GeneralizedRegularGridPoints(ph_tio2, helper_methods):
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.interface.phonopy_yaml import read_cell_yaml
    grgp = GeneralizedRegularGridPoints(ph_tio2.unitcell, 10)
    tmat = grgp.matrix_to_primitive
    # direct basis vectors in row vectors
    plat = np.dot(tmat.T, ph_tio2.unitcell.cell)
    # reciprocal basis vectors in row vectors (10 times magnified)
    rec_plat = np.linalg.inv(plat).T * 10
    nums = [1, ] * len(grgp.qpoints)
    cell = PhonopyAtoms(cell=rec_plat,
                        scaled_positions=grgp.qpoints,
                        numbers=nums)
    yaml_filename = os.path.join(current_dir, "tio2_qpoints.yaml")
    cell_ref = read_cell_yaml(yaml_filename)
    helper_methods.compare_cells(cell, cell_ref)
