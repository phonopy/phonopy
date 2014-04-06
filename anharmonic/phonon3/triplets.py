import numpy as np
from phonopy.units import THzToEv, Kb
import phonopy.structure.spglib as spg
from phonopy.structure.symmetry import Symmetry
from phonopy.structure.tetrahedron_method import TetrahedronMethod

def gaussian(x, sigma):
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-x**2 / 2 / sigma**2)

def occupation(x, t):
    return 1.0 / (np.exp(THzToEv * x / (Kb * t)) - 1)


def get_triplets_at_q(grid_point,
                      mesh,
                      point_group, # real space point group of space group
                      primitive_lattice, # column vectors
                      is_time_reversal=True,
                      with_bz_map=False):
    weights, third_q, grid_address = spg.get_triplets_reciprocal_mesh_at_q(
        grid_point,
        mesh,
        point_group,
        is_time_reversal)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           primitive_lattice)
    triplets_at_q = spg.get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        weights,
        mesh)
    ir_weights = np.extract(weights > 0, weights)

    assert np.prod(mesh) == ir_weights.sum(), \
        "Num grid points %d, sum of weight %d" % (
                    np.prod(mesh), ir_weights.sum())

    if with_bz_map:
        return triplets_at_q, ir_weights, bz_grid_address, bz_map
    else:
        return triplets_at_q, ir_weights, bz_grid_address

def get_nosym_triplets_at_q(grid_point,
                            mesh,
                            primitive_lattice,
                            with_bz_map=False):
    grid_address = get_grid_address(mesh)
    weights = np.ones(len(grid_address), dtype='intc')
    third_q = np.zeros_like(weights)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           primitive_lattice)
    triplets_at_q = spg.get_BZ_triplets_at_q(
        grid_point,
        bz_grid_address,
        bz_map,
        weights,
        mesh)

    if with_bz_map:
        return triplets_at_q, weights, bz_grid_address, bz_map
    else:    
        return triplets_at_q, weights, bz_grid_address

def get_grid_address(mesh):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        is_time_reversal=False)

    return grid_address

def get_bz_grid_address(mesh, primitive_lattice, with_boundary=False):
    grid_address = get_grid_address(mesh)
    bz_grid_address, bz_map = spg.relocate_BZ_grid_address(grid_address,
                                                           mesh,
                                                           primitive_lattice)
    if with_boundary:
        return bz_grid_address, bz_map
    else:
        return bz_grid_address[:np.prod(mesh)]

def get_grid_point_from_address(address, mesh):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    m = mesh
    return (address[0] % m[0] +
            (address[1] % m[1]) * m[0] +
            (address[2] % m[2]) * m[0] * m[1])

def get_bz_grid_point_from_address(address, mesh, bz_map):
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    # 2m is defined in kpoint.c of spglib.
    m = 2 * np.array(mesh, dtype='intc')
    return bz_map[get_grid_point_from_address(address, m)]

def invert_grid_point(grid_point, mesh, grid_address, bz_map):
    # gp --> [address] --> [-address] --> inv_gp
    address = grid_address[grid_point]
    return get_bz_grid_point_from_address(-address, mesh, bz_map)

def get_ir_grid_points(mesh, rotations, mesh_shifts=[False, False, False]):
    grid_mapping_table, grid_address = spg.get_stabilized_reciprocal_mesh(
        mesh,
        rotations,
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
    divisors = np.array(mesh_divisors, dtype='intc')
    if (divisors == 1).all():
        coarse_grid_points = np.array(dense_grid_points, dtype='intc')
        if dense_grid_weights is not None:
            coarse_grid_weights = np.array(dense_grid_weights, dtype='intc')
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
    return np.array(dense_grid_points, dtype='intc')

def get_coarse_ir_grid_points(primitive,
                              mesh,
                              mesh_divisors,
                              coarse_mesh_shifts,
                              is_nosym=False,
                              symprec=1e-5):
    if mesh_divisors is None:
        mesh_divs = [1, 1, 1]
    else:
        mesh_divs = mesh_divisors
    mesh = np.array(mesh, dtype='intc')
    mesh_divs = np.array(mesh_divs, dtype='intc')
    coarse_mesh = mesh / mesh_divs
    if coarse_mesh_shifts is None:
        coarse_mesh_shifts = [False, False, False]

    if is_nosym:
        coarse_grid_address = get_grid_address(coarse_mesh)
        coarse_grid_points = np.arange(np.prod(coarse_mesh), dtype='intc')
        coarse_grid_weights = np.ones(len(coarse_grid_points), dtype='intc')
    else:
        symmetry = Symmetry(primitive, symprec)
        (coarse_grid_points,
         coarse_grid_weights,
         coarse_grid_address) = get_ir_grid_points(
            coarse_mesh,
            symmetry.get_pointgroup_operations(),
            mesh_shifts=coarse_mesh_shifts)
    grid_points = from_coarse_to_dense_grid_points(
        mesh,
        mesh_divs,
        coarse_grid_points,
        coarse_grid_address,
        coarse_mesh_shifts=coarse_mesh_shifts)
    grid_address = get_grid_address(mesh)
    primitive_lattice = np.linalg.inv(primitive.get_cell())
    spg.relocate_BZ_grid_address(grid_address,
                                 mesh,
                                 primitive_lattice)

    return grid_points, coarse_grid_weights, grid_address

def get_grid_points_in_Brillouin_zone(primitive_vectors, # column vectors
                                      mesh,
                                      grid_address,
                                      grid_points,
                                      with_boundary=False):
    gbz = GridBrillouinZone(primitive_vectors,
                            mesh,
                            grid_address,
                            with_boundary=with_boundary)
    gbz.run(grid_points)
    return gbz.get_shortest_addresses()

def get_triplets_integration_weights(interaction,
                                     frequency_points,
                                     sigma,
                                     lang='C'):
    triplets = interaction.get_triplets_at_q()[0]
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    g = np.zeros((2, len(triplets), len(frequency_points), num_band, num_band),
                 dtype='double')

    if sigma:
        if lang == 'C':
            import anharmonic._phono3py as phono3c
            phono3c.triplets_integration_weights_with_sigma(
                g,
                frequency_points,
                triplets,
                frequencies,
                sigma)
        else:        
            for i, tp in enumerate(triplets):
                f1s = frequencies[tp[1]]
                f2s = frequencies[tp[2]]
                for j, k in list(np.ndindex((num_band, num_band))):
                    f1 = f1s[j]
                    f2 = f2s[k]
                    g0 = gaussian(frequency_points - f1 - f2, sigma)
                    g[0, i, :, j, k] = g0
                    g1 = gaussian(frequency_points + f1 - f2, sigma)
                    g2 = gaussian(frequency_points - f1 + f2, sigma)
                    g[1, i, :, j, k] = g1 - g2
                    if len(g) == 3:
                        g[2, i, :, j, k] = g0 + g1 + g2
    else:
        if lang == 'C':
            _set_triplets_integration_weights_c(
                g, interaction, frequency_points)
        else:
            _set_triplets_integration_weights_py(
                g, interaction, frequency_points)

    return g

def get_tetrahedra_vertices(relative_address,
                            mesh,
                            triplets_at_q,
                            bz_grid_address,
                            bz_map):
    bzmesh = mesh * 2
    grid_order = [1, mesh[0], mesh[0] * mesh[1]]
    bz_grid_order = [1, bzmesh[0], bzmesh[0] * bzmesh[1]]
    num_triplets = len(triplets_at_q)
    vertices = np.zeros((num_triplets, 2, 24, 4), dtype='intc')
    for i, tp in enumerate(triplets_at_q):
        for j, adrs_shift in enumerate(
                (relative_address, -relative_address)):
            adrs = bz_grid_address[tp[j + 1]] + adrs_shift
            bz_gp = np.dot(adrs % bzmesh, bz_grid_order)
            gp = np.dot(adrs % mesh, grid_order)
            vgp = bz_map[bz_gp]
            vertices[i, j] = vgp + (vgp == -1) * (gp + 1)
    return vertices

def _set_triplets_integration_weights_c(g, interaction, frequency_points):
    import anharmonic._phono3py as phono3c

    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    triplets_at_q = interaction.get_triplets_at_q()[0]
    unique_vertices = thm.get_unique_tetrahedra_vertices()
    
    for i, j in zip((1, 2), (1, -1)):
        neighboring_grid_points = np.zeros(
            len(unique_vertices) * len(triplets_at_q), dtype='intc')
        phono3c.neighboring_grid_points(
            neighboring_grid_points,
            triplets_at_q[:, i].flatten(),
            j * unique_vertices,
            mesh,
            grid_address,
            bz_map)
        interaction.set_phonon(np.unique(neighboring_grid_points))

    phono3c.triplets_integration_weights(
        g,
        np.array(frequency_points, dtype='double'),
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        interaction.get_phonons()[0],
        grid_address,
        bz_map)

def _set_triplets_integration_weights_py(g, interaction, frequency_points):
    reciprocal_lattice = np.linalg.inv(interaction.get_primitive().get_cell())
    mesh = interaction.get_mesh_numbers()
    thm = TetrahedronMethod(reciprocal_lattice, mesh=mesh)
    grid_address = interaction.get_grid_address()
    bz_map = interaction.get_bz_map()
    triplets_at_q = interaction.get_triplets_at_q()[0]
    unique_vertices = thm.get_unique_tetrahedra_vertices()
    tetrahedra_vertices = get_tetrahedra_vertices(
        thm.get_tetrahedra(),
        mesh,
        triplets_at_q,
        grid_address,
        bz_map)
    interaction.set_phonon(np.unique(tetrahedra_vertices))
    frequencies = interaction.get_phonons()[0]
    num_band = frequencies.shape[1]
    for i, vertices in enumerate(tetrahedra_vertices):
        for j, k in list(np.ndindex((num_band, num_band))):
            f1_v = frequencies[vertices[0], j]
            f2_v = frequencies[vertices[1], k]
            thm.set_tetrahedra_omegas(f1_v + f2_v)
            thm.run(frequency_points)
            g0 = thm.get_integration_weight()
            g[0, i, :, j, k] = g0
            thm.set_tetrahedra_omegas(-f1_v + f2_v)
            thm.run(frequency_points)
            g1 = thm.get_integration_weight()
            thm.set_tetrahedra_omegas(f1_v - f2_v)
            thm.run(frequency_points)
            g2 = thm.get_integration_weight()
            g[1, i, :, j, k] = g1 - g2
            if len(g) == 3:
                g[2, i, :, j, k] = g0 + g1 + g2
                    
    # if len(tetrahedra) == 4:
    #     g /= 4
    

class GridBrillouinZone:
    search_space = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1]], dtype='intc', order='C')
    
    def __init__(self,
                 primitive_vectors,
                 mesh,
                 grid_address,
                 with_boundary=False): # extended grid if True
        self._primitive_vectors = np.array(primitive_vectors) # column vectors
        self._mesh = mesh
        self._grid_address = grid_address
        self._with_boundary = with_boundary

        self._tolerance = min(np.sum(self._primitive_vectors ** 2, axis=0)) / 10
        self._primitive_vectors_inv = np.linalg.inv(self._primitive_vectors)
        self._search_space = search_space * mesh

        self._shortest_addresses = None

    def run(self, grid_points):
        self._shortest_addresses = []
        for address in self._grid_address[grid_points]:
            distances = np.array(
                [(np.dot(self._primitive_vectors, address + g) ** 2).sum()
                 for g in self._search_space], dtype='double')
            min_dist = min(distances)
            shortest_indices = [i for i, d in enumerate(distances - min_dist)
                                if abs(d) < self._tolerance]
            self._shortest_addresses.append(
                self._search_space[shortest_indices] + address)

    def get_shortest_addresses(self):
        return self._shortest_addresses
    

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
