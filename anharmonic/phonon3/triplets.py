import numpy as np
import phonopy.structure.spglib as spg

def get_triplets_at_q(grid_point,
                      mesh,
                      point_group, # real space point group of space group
                      is_time_reversal=True):
    weights, third_q, grid_address = spg.get_triplets_reciprocal_mesh_at_q(
        grid_point,
        mesh,
        point_group,
        is_time_reversal)
    # weights_at_q = []
    # triplets_at_q = []
    # for i, (w, q) in enumerate(zip(weights, third_q)):
    #     if w > 0:
    #         weights_at_q.append(w)
    #         triplets_at_q.append([grid_point, i, q])

    # weights_at_q = np.intc(weights_at_q)
    # triplets_at_q = np.intc(triplets_at_q)
            
    # assert np.prod(mesh) == weights_at_q.sum(), \
    #     "Num grid points %d, sum of weight %d" % (
    #                 np.prod(mesh), weights_at_q.sum())

    triplets_at_q = spg.get_grid_triplets_at_q(
        grid_point,
        grid_address,
        third_q,
        mesh)
    weights_at_q = np.extract(weights > 0, weights)
    print np.extract(third_q >= 0, third_q), len(np.extract(third_q >= 0, third_q))
    print weights_at_q, len(weights_at_q), weights_at_q.sum()

    return triplets_at_q, weights_at_q, grid_address

def get_nosym_triplets_at_q(grid_point, mesh):
    grid_address = get_grid_address(mesh)

    triplets = np.zeros((len(grid_address), 3), dtype='intc')
    weights = np.ones(len(grid_address), dtype='intc')
    for i, g1 in enumerate(grid_address):
        g2 = - (grid_address[grid_point] + g1)
        q = get_grid_point_from_address(g2, mesh)
        triplets[i] = [grid_point, i, q]

    return triplets, weights, grid_address

def get_grid_address(mesh):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        is_time_reversal=False)

    return grid_address

def get_grid_point_from_address(grid, mesh):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def invert_grid_point(grid_point, grid_address, mesh):
    # gp --> [address] --> [-address] --> inv_gp
    address = grid_address[grid_point]
    return get_grid_point_from_address(-address, mesh)

def get_ir_grid_points(mesh, primitive, mesh_shifts=[False, False, False]):
    grid_mapping_table, grid_address = spg.get_ir_reciprocal_mesh(
        mesh,
        primitive,
        is_shift=np.where(mesh_shifts, 1, 0))
    ir_grid_points = np.unique(grid_mapping_table)
    weights = np.zeros_like(grid_mapping_table)
    for g in grid_mapping_table:
        weights[g] += 1
    ir_grid_weights = weights[ir_grid_points]

    return ir_grid_points, ir_grid_weights, grid_address

def reduce_grid_points(mesh_divisors,
                       grid_address,
                       dense_grid_points,
                       dense_grid_weights=None,
                       coarse_mesh_shifts=None):
    divisors = np.array(mesh_divisors, dtype=int)
    if (divisors == 1).all():
        coarse_grid_points = np.array(dense_grid_points, dtype=int)
        if dense_grid_weights is not None:
            coarse_grid_weights = np.array(dense_grid_weights, dtype=int)
    else:
        grid_weights = []
        if coarse_mesh_shifts is None:
            shift = [0, 0, 0]
        else:
            shift = np.where(coarse_mesh_shifts, divisors / 2, [0, 0, 0])
        modulo = grid_address[dense_grid_points] % divisors
        condition = (modulo == shift).all(axis=1)
        coarse_grid_points = np.extract(condition, dense_grid_points)
        if dense_grid_weights is not None:
            coarse_grid_weights = np.extract(condition, dense_grid_weights)

    if dense_grid_weights is None:
        return coarse_grid_points
    else:
        return coarse_grid_points, coarse_grid_weights

def from_coarse_to_dense_grid_points(dense_mesh,
                                     mesh_divisors,
                                     coarse_grid_points,
                                     coarse_grid_address,
                                     coarse_mesh_shifts=[False, False, False]):
    shifts = np.where(coarse_mesh_shifts, 1, 0)
    dense_grid_points = []
    for cga in coarse_grid_address[coarse_grid_points]:
        dense_address = cga * mesh_divisors + shifts * (mesh_divisors / 2)
        dense_grid_points.append(get_grid_point_from_address(dense_address,
                                                             dense_mesh))
    return np.intc(dense_grid_points)

def get_coarse_ir_grid_points(primitive, mesh, mesh_divs, coarse_mesh_shifts):
    if mesh_divs is None:
        mesh_divs = [1, 1, 1]
    mesh = np.intc(mesh)
    mesh_divs = np.intc(mesh_divs)
    coarse_mesh = mesh / mesh_divs
    if coarse_mesh_shifts is None:
        coarse_mesh_shifts = [False, False, False]
    (coarse_grid_points,
     coarse_grid_weights,
     coarse_grid_address) = get_ir_grid_points(
        coarse_mesh,
        primitive,
        mesh_shifts=coarse_mesh_shifts)
    grid_points = from_coarse_to_dense_grid_points(
        mesh,
        mesh_divs,
        coarse_grid_points,
        coarse_grid_address,
        coarse_mesh_shifts=coarse_mesh_shifts)
    grid_address = get_grid_address(mesh)

    return grid_points, coarse_grid_weights, grid_address


if __name__ == '__main__':
    # This checks if ir_grid_points.yaml gives correct dense grid points
    # that are converted from coase grid points by comparing with 
    # mesh.yaml.
    
    import yaml
    import sys
    
    data1 = yaml.load(open(sys.argv[1]))['ir_grid_points'] # ir_grid_points.yaml
    data2 = yaml.load(open(sys.argv[2]))['phonon'] # phonpy mesh.yaml
    
    weights1 = np.array([x['weight'] for x in data1])
    weights2 = np.array([x['weight'] for x in data2])
    print (weights1 == weights2).sum()
    q1 = np.array([x['q-point'] for x in data1])
    q2 = np.array([x['q-position'] for x in data2])
    print (q1 == q2).all(axis=1).sum()
    for d in (q1 - q2):
        print d
