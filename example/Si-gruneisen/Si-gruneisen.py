import numpy as np
from phonopy import Phonopy, PhonopyGruneisen
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS

phonons = {}
for vol in ("orig", "plus", "minus"):
    unitcell = read_vasp("%s/POSCAR-unitcell" % vol)
    phonon = Phonopy(unitcell,
                     [[2, 0, 0],
                      [0, 2, 0],
                      [0, 0, 2]],
                     primitive_matrix=[[0, 0.5, 0.5],
                                       [0.5, 0, 0.5],
                                       [0.5, 0.5, 0]])
    force_sets = parse_FORCE_SETS(filename="%s/FORCE_SETS" % vol)
    phonon.set_displacement_dataset(force_sets)
    phonon.produce_force_constants()
    phonons[vol] = phonon

gruneisen = PhonopyGruneisen(phonons["orig"],
                             phonons["plus"],
                             phonons["minus"])

gruneisen.set_mesh([2, 2, 2])
gruneisen_params_on_mesh = gruneisen.get_mesh().get_gruneisen()
print(gruneisen_params_on_mesh)
