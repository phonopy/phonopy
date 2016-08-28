#!/usr/bin/env python

from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import yaml
import numpy as np

unitcell = read_vasp("POSCAR")
phonon = Phonopy(unitcell,
                 [[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]],
                 primitive_matrix=[[0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]])
force_sets = parse_FORCE_SETS()
phonon.set_displacement_dataset(force_sets)
phonon.produce_force_constants()

q = [0.1, 0.1, 0.1]
dynmat = phonon.get_dynamical_matrix_at_q(q)
print(dynmat)
phonon.set_qpoints_phonon(q, write_dynamical_matrices=True)
print(phonon.get_qpoints_phonon()[0][0])
phonon.write_yaml_qpoints_phonon()

data = yaml.load(open("qpoints.yaml"))
dynmat_from_yaml = []
dynmat_data = data['phonon'][0]['dynamical_matrix']
for row in dynmat_data:
    vals = np.reshape(row, (-1, 2))
    dynmat_from_yaml.append(vals[:, 0] + vals[:, 1] * 1j)
dynmat_from_yaml = np.array(dynmat_from_yaml)
print(dynmat_from_yaml)
eigvals, eigvecs, = np.linalg.eigh(dynmat)
frequencies = np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)
conversion_factor_to_THz = 15.633302
print frequencies * conversion_factor_to_THz


