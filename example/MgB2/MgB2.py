from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.hphonopy.file_IO import parse_FORCE_SETS, parse_BORN
import numpy as np

cell = read_vasp("POSCAR")

# Initialize phonon. Supercell matrix has to have the shape of (3, 3)
phonon = Phonopy(cell, np.diag([3, 3, 2]))

symmetry = phonon.get_symmetry()
print "Space group:", symmetry.get_international_table()

# Read and convert forces and displacements
force_sets = parse_FORCE_SETS(cell.get_number_of_atoms() * 18)
sets_of_forces = []
displacements = []
for force in force_sets:
    sets_of_forces.append(force.get_forces())
    disp = force.get_displacement()
    atom_number = force.get_atom_number()
    displacements.append([atom_number,
                          disp[0], disp[1], disp[2]])

# A set of Displacement has to be set before phonon.set_post_process
phonon.set_displacements(displacements)

# Sets of forces have to be set before phonon.set_post_process or
# at phonon.set_post_process(..., sets_of_forces=sets_of_forces, ...).
phonon.set_forces(sets_of_forces)
phonon.set_post_process()

# Character table
phonon.set_character_table([1./3, 1./3, 0], 1e-4)
phonon.show_character_table()
ct = phonon.get_character_table()
eigvecs = ct.get_eigenvectors()
ops = ct.get_projection_operators()
print ops.shape

print ops[2]
for vec in ops[2]:
    print "[",
    for i, x in enumerate(vec):
        print "(%6.3f %6.3f)" % (x.real, x.imag),
        if i % 4 == 3:
            print
    print "]"

for i in (3, 4):
    print
    for x, y in zip(eigvecs[:,i], np.dot(ops[2], eigvecs[:,i])):
        print "(%7.3f %7.3fi) (%7.3f %7.3fi)" % (x.real, x.imag,
                                                 y.real, y.imag)
                                           
