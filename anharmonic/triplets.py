def get_grid_points(mesh):
    grid_points = np.zeros((np.prod(mesh), 3), dtype=int)
    count = 0
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                grid_points[count] = [k - (k > (mesh[2] // 2)) * mesh[2],
                                      j - (j > (mesh[1] // 2)) * mesh[1],
                                      i - (i > (mesh[0] // 2)) * mesh[0]]
                count += 1
    
    return grid_points

def get_nosym_triplets(mesh, grid_point0):
    grid_points = get_grid_points(mesh)
    triplets = np.zeros((len(grid_points), 3), dtype=int)
    for i, g1 in enumerate(grid_points):
        g2 = - (grid_points[grid_point0] + g1)
        triplets[i] = [grid_point0, i, get_address(g2, mesh)]
    weights = np.ones(len(grid_points), dtype=int)

    return triplets, weights, grid_points

def get_address(grid, mesh):
    return ((grid[0] + mesh[0]) % mesh[0] +
            ((grid[1] + mesh[1]) % mesh[1]) * mesh[0] +
            ((grid[2] + mesh[2]) % mesh[2]) * mesh[0] * mesh[1])

def get_triplets_at_q(gp,
                      mesh,
                      primitive_lattice,
                      rotations,
                      is_time_reversal=True,
                      symprec=1e-5):

    (weights,
     third_q,
     grid_points) = spg.get_triplets_reciprocal_mesh_at_q(gp,
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

    return np.array(triplets_at_q), np.array(weights_at_q), grid_points

