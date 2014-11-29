from anharmonic.phonon3.triplets import get_triplets_at_q
from phonopy.interface.vasp import read_vasp 
from phonopy.structure.symmetry import Symmetry
import numpy as np

cell = read_vasp("POSCAR-unitcell")
symmetry = Symmetry(cell, 1e-2)
print symmetry.get_international_table()
reciprocal_lattice = np.linalg.inv(cell.get_cell())
mesh = [7, 7, 7]

print reciprocal_lattice

(triplets_at_q,
 weights_at_q,
 grid_address,
 bz_map,
 triplets_map_at_q,
 ir_map_at_q)= get_triplets_at_q(74,
                                 mesh,
                                 symmetry.get_pointgroup_operations(),
                                 reciprocal_lattice,
                                 stores_triplets_map=True)

for triplet in triplets_at_q:
    sum_q = (grid_address[triplet]).sum(axis=0)
    if (sum_q % mesh != 0).any():
        print "============= Warning =================="
        print triplet
        for tp in triplet:
            print grid_address[tp], np.linalg.norm(np.dot(reciprocal_lattice, grid_address[tp] / mesh))
        print sum_q
        print "============= Warning =================="
