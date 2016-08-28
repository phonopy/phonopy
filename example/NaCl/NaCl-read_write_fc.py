from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS

cell = read_vasp("POSCAR")
phonon = Phonopy(cell,
                 [[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]])
force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()
write_FORCE_CONSTANTS(phonon.get_force_constants(),
                      filename="FORCE_CONSTANTS")

force_constants = parse_FORCE_CONSTANTS()
phonon.set_force_constants(force_constants)
phonon.symmetrize_force_constants(iteration=1)
write_FORCE_CONSTANTS(phonon.get_force_constants(),
                      filename="FORCE_CONSTANTS_NEW")
