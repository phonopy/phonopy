"""Example to read and write FORCE_CONSTANTS file."""

import phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS

phonon = phonopy.load(
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    unitcell_filename="POSCAR-unitcell",
    force_sets_filename="FORCE_SETS",
    born_filename="BORN",
)
write_FORCE_CONSTANTS(phonon.force_constants, filename="FORCE_CONSTANTS")

force_constants = parse_FORCE_CONSTANTS()
phonon.force_constants = force_constants
phonon.symmetrize_force_constants()
write_FORCE_CONSTANTS(phonon.force_constants, filename="FORCE_CONSTANTS_NEW")
