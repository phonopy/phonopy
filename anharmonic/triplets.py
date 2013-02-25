import numpy as np
import phonopy.structure.spglib as spg

def get_triplets_at_q(gp,
                      mesh,
                      primitive_lattice,
                      rotations,
                      is_time_reversal=True,
                      symprec=1e-5):

    (weights,
     third_q,
     grid_address) = spg.get_triplets_reciprocal_mesh_at_q(gp,
                                                           mesh,
                                                           rotations,
                                                           is_time_reversal,
                                                           symprec)

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
    # XYZ where X runs first.
    # This has to match with get_ir_reciprocal_mesh.
    grid_address = np.zeros((np.prod(mesh), 3), dtype=int)
    count = 0
    for i in range(mesh[2]):
        for j in range(mesh[1]):
            for k in range(mesh[0]):
                grid_address[count] = [k - (k > (mesh[2] // 2)) * mesh[0],
                                       j - (j > (mesh[1] // 2)) * mesh[1],
                                       i - (i > (mesh[0] // 2)) * mesh[2]]

                count += 1
    
    return np.array(grid_address)

def get_address(grid, mesh):
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def get_ir_grid_points(mesh, primitive):
    grid_mapping_table, grid_address = spg.get_ir_reciprocal_mesh(mesh,
                                                                  primitive)
    ir_grid_indices = np.unique(grid_mapping_table)
    ir_weights = np.array([np.sum(grid_mapping_table == g)
                           for g in ir_grid_indices])
    return ir_grid_indices, ir_weights

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
        for i, dgp in enumerate(dense_grid_points):
            if (grid_address[dgp] % divisors == 0).all():
                grid_points.append(dgp)
                if dense_grid_weights is not None:
                    grid_weights.append(dense_grid_weights[i])

        grid_points = np.array(grid_points, dtype=int)
        grid_weights = np.array(grid_weights, dtype=int)

    if dense_grid_weights is None:
        return grid_points
    else:
        return grid_points, grid_weights
