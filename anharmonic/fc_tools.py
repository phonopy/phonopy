import numpy as np
from phonopy.structure.symmetry import Symmetry
from anharmonic.fc3 import get_atom_mapping_by_symmetry, get_atom_by_symmetry 
from anharmonic.fc3 import distribute_fc3
from phonopy.harmonic.force_constants import similarity_transformation

# In this file, force constant manipulations are implemented.  If
# translational invariance is assumed, one atomic index is reduced
# from supercell to primitive cell. In the methods below, the
# reduction and expansion of the force constants, and index exchange
# are implemented. Keep in mind that some are re-implementation of the
# methods in dyncamial_matrix.py and force_constant.py.

def expand_fc2(fc2_reduced,
               supercell,
               primitive,
               symmetry=None,
               symprec=1e-5):
    """
    fc2_reduced[num_atom_prim, num_atom_super, 3, 3]
    """

    if symmetry == None:
        symmetry = Symmetry(supercell, symprec)
        
    pos = supercell.get_scaled_positions()
    num_atom_super = supercell.get_number_of_atoms()
    L = supercell.get_cell()
    rot = symmetry.get_symmetry_operations()['rotation']
    trans = symmetry.get_symmetry_operations()['translation']
    
    p2p = primitive.get_primitive_to_primitive_map()
    s2p = primitive.get_supercell_to_primitive_map()

    fc2 = np.zeros((num_atom_super, num_atom_super, 3, 3), dtype=float)

    for s, p in enumerate( s2p ):
        if s==p:
            fc2[s,:] = fc2_reduced[p2p[p],:]
            continue

        sym_index = get_atom_mapping_by_symmetry(pos, p, s, symmetry)

        # L R L^-1 (L and R in spglib style)
        rot_cartesian = similarity_transformation(L.T, rot[sym_index])

        for i in range(num_atom_super):
            j = get_atom_by_symmetry(pos,
                                     rot[sym_index],
                                     trans[sym_index],
                                     i,
                                     symprec)
            # R^-1 P R (rotation is inverse)
            fc2[s, i] = similarity_transformation(
                rot_cartesian, fc2_reduced[p2p[p], j])
    
    return fc2

def expand_fc3(fc3_reduced,
               supercell,
               primitive,
               symmetry=None,
               symprec=1e-5):
    """
    fc3_reduced[num_atom_prim, num_atom_super, num_atom_super, 3, 3, 3]
    """

    if symmetry == None:
        symmetry = Symmetry(supercell, symprec)
        
    num_atom_super = supercell.get_number_of_atoms()
    p2p = primitive.get_primitive_to_primitive_map()
    s2p = primitive.get_supercell_to_primitive_map()
    p2s = primitive.get_primitive_to_supercell_map()

    fc3 = np.zeros((num_atom_super, num_atom_super, num_atom_super, 3, 3, 3 ),
                   dtype=float)
        
    for p, s in enumerate(p2s):
        fc3[s] = fc3_reduced[p]

    fc3_x = fc3_index_exchange(fc3)
    distribute_fc3(fc3_x,
                   range(num_atom_super),
                   p2s,
                   supercell,
                   symmetry,
                   verbose=True)

    return fc3_index_exchange(fc3_x)

# Indices 1 and 3 are exchanged
def fc3_index_exchange(fc3):
    fc3_x = np.zeros(fc3.shape, dtype=float)
    for M in range(fc3.shape[0]):
        for P in range(fc3.shape[0]):
            for i in range(3):
                for k in range(3):
                    fc3_x[P, :, M, k, :, i] = fc3[M, :, P, i, :, k]

    return fc3_x
