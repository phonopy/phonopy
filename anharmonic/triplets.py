import numpy as np
import phonopy.structure.spglib as spg

def get_triplets_at_q(gp,
                      mesh,
                      primitive_lattice,
                      rotations,
                      is_time_reversal=True):

    (weights,
     third_q,
     grid_address) = spg.get_triplets_reciprocal_mesh_at_q(gp,
                                                           mesh,
                                                           rotations,
                                                           is_time_reversal)
    weights_at_q = []
    triplets_at_q = []
    # A pair of q-points determins the third q-point by conservatoin law.
    # If q-point triplet is independent, it is set as w > 0.
    # Sum of w has to be prod(mesh).
    for i, (w, q) in enumerate(zip(weights, third_q)):
        if w > 0:
            weights_at_q.append(w)
            triplets_at_q.append([gp, i, q])

    return np.array(triplets_at_q), np.array(weights_at_q), grid_address

def get_nosym_triplets(mesh, grid_point0):
    grid_address = get_grid_address(mesh)
    triplets = np.zeros((len(grid_address), 3), dtype=int)
    weights = np.ones(len(grid_address), dtype=int)
    for i, g1 in enumerate(grid_address):
        g2 = - (grid_address[grid_point0] + g1)
        q = get_address(g2, mesh)
        triplets[i] = [grid_point0, i, q]

    return triplets, weights, grid_address

def get_grid_address(mesh):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=int))

    return grid_address

def get_address(grid, mesh):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def get_ir_grid_points(mesh, primitive):
    grid_mapping_table, grid_address = spg.get_ir_reciprocal_mesh(mesh,
                                                                  primitive)
    ir_grid_points = np.unique(grid_mapping_table)
    weights = np.zeros_like(grid_mapping_table)
    for g in grid_mapping_table:
        weights[g] += 1
    ir_grid_weights = weights[ir_grid_points]

    return ir_grid_points, ir_grid_weights

def reduce_grid_points(mesh_divisors,
                       grid_address,
                       dense_grid_points,
                       dense_grid_weights=None):
    divisors = np.array(mesh_divisors, dtype=int)
    if (divisors == 1).all():
        grid_points = np.array(dense_grid_points, dtype=int)
        if dense_grid_weights is not None:
            grid_weights = np.array(dense_grid_weights, dtype=int)
    else:
        grid_points = []
        grid_weights = []

        for i, modulo in enumerate(
            grid_address[dense_grid_points] % divisors):
            if (modulo == 0).all():
                grid_points.append(dense_grid_points[i])
                if dense_grid_weights is not None:
                    grid_weights.append(dense_grid_weights[i])

        grid_points = np.array(grid_points, dtype=int)
        grid_weights = np.array(grid_weights, dtype=int)

    if dense_grid_weights is None:
        return grid_points
    else:
        return grid_points, grid_weights
