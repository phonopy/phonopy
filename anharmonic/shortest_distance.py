import numpy as np
from phonopy.units import Bohr, Rydberg
from phonopy.structure.cells import get_reduced_bases

#             
# The functions:
# - get_equivalent_shortest_vectors
#   originally get_equivalent_smallest_vectors
# - get_shortest_vectors
#   originally __smallest_vectors
#
def get_shortest_vectors(supercell, primitive, symprec=1e-5):
    """
    shortest_vectors:

      Shortest vectors between two atoms with periodic boundary
      condition by supercell, which is given in the fractional
      coordinates. If one atom is on the border of the supercell
      mesured by another atom, several r's are stored. The
      multiplicity is stored in another file.
      The array is given as
      ( atom_super, atom_primitive, multiplicity, 3 )

      
    multiplicity:
      The multiplicities of shortest_vectors are stored.
    """
    
    size_super = supercell.get_number_of_atoms()
    size_prim = primitive.get_number_of_atoms()
    p2s_map = primitive.get_primitive_to_supercell_map()

    shortest_vectors = np.zeros((size_super, size_prim, 27, 3), dtype=float)
    multiplicity = np.zeros((size_super, size_prim), dtype=int)

    for i in range(size_super):           # loop over supercell
        for j, s_j in enumerate(p2s_map): # loop over primitive
            vectors = get_equivalent_shortest_vectors(supercell,
                                                      primitive,
                                                      i,
                                                      s_j,
                                                      symprec)
            multiplicity[i][j] = len(vectors)
            for k, elem in enumerate(vectors):
                shortest_vectors[i][j][k] = elem

    return shortest_vectors, multiplicity

def get_equivalent_shortest_vectors(supercell,
                                    primitive,
                                    atom_number_supercell,
                                    atom_number_primitive,
                                    symprec=1e-5):
    distances = []
    differences = []

    # Reduced bases for the supercell (smallest lattice)
    reduced_bases = get_reduced_bases(supercell.get_cell(), symprec)

    # Atomic positions are confined into the lattice made of reduced bases.
    positions = np.dot(supercell.get_positions(),
                       np.linalg.inv(reduced_bases))

    for pos in positions:
        pos -= pos.round()

    p_pos = positions[atom_number_primitive]
    s_pos = positions[atom_number_supercell]
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                # The vector arrow is from the atom in primitive cell
                # to the atom in supercell. This relates to determine
                # the phase convension. In harmonic phonopy, the phase
                # convension is same but the arrow direction is
                # inverted because summation order is different.
                diff = s_pos + np.array([i, j, k]) - p_pos
                differences.append(diff)
                vec = np.dot(diff, reduced_bases)
                distances.append(np.dot(vec, vec))

    minimum = min(distances)
    shortest_vectors = []
    for i in range(27):
        if abs( minimum - distances[i] ) < symprec:
            relative_scale = np.dot(reduced_bases,
                                    np.linalg.inv(primitive.get_cell()))
            shortest_vectors.append(np.dot(differences[i], relative_scale))

    return shortest_vectors

