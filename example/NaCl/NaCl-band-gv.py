import numpy as np
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS, parse_BORN
import matplotlib.pyplot as plt

def append_band(bands, q_start, q_end):
    band = []
    nq = 51
    for i in range(nq):
        band.append(np.array(q_start) +
                    (np.array(q_end) - np.array(q_start)) / (nq - 1) * i)
    bands.append(band)

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
primitive = phonon.get_primitive()
nac_params = parse_BORN(primitive, filename="BORN")
phonon.set_nac_params(nac_params)
phonon.set_group_velocity()
bands = []
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0])
append_band(bands, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0])
append_band(bands, [0.5, 0.5, 0.0], [0.0, 0.0, 0.0])
append_band(bands, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
phonon.set_band_structure(bands)
q_points, distances, frequencies, eigvecs = phonon.get_band_structure()
group_velocities = phonon.get_group_velocities_on_bands()
print(len(frequencies))
print(frequencies[0].shape)
print(len(group_velocities))
print(group_velocities[0].shape)
